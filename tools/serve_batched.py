#!/usr/bin/env python3
"""Roadmap E milestone 3.2 + 3.3: continuous-batching scheduler + multi-client OpenAI server.

A single engine thread owns the CUDA engine (paged Q4 KV pool). HTTP handler threads only
tokenize, submit a Request, and wait on its completion event. The engine thread runs a
continuous-batching loop:
  - ADMIT queued requests into free slots (prefill the prompt single-stream, snapshot into the
    slot's paged blocks via qwn_paged_load_client = the validated bring-up path), subject to
    admission control (free slot + enough free pool blocks).
  - DECODE one paged step over ALL active slots (qwn_paged_decode_step), push each slot's new
    token to its request, advance positions, and DETACH finished requests (EOS / max_tokens),
    returning their blocks to the pool.

Decodes are batched (the throughput win); prefills are serialized in the engine thread (MVP --
chunked/in-batch prefill is a later optimization). Default-off relative to the single-stream
serve_openai.py; this is a SEPARATE server.

Modes:
  --selftest     : staggered join/leave correctness vs single-stream + aggregate tok/s (no HTTP)
  (default)      : run the OpenAI server on --port

Run (GPU7, prod Q4):
  CUDA_VISIBLE_DEVICES=7 TQ_CTX=8192 TQ_KV_Q4=1 python3 -u tools/serve_batched.py --selftest
  CUDA_VISIBLE_DEVICES=7 TQ_CTX=131072 TQ_KV_Q4=1 python3 -u tools/serve_batched.py --port 8100
"""
from __future__ import annotations
import argparse, ctypes, json, os, threading, time, queue, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Prefill wave column cap. NEVER hardcode this: it is the engine's tiled-GEMM column
# tile and it has moved (128 -> 256). This file's stale 128 silently halved every wave
# and cost ~1.2x of prefill even when the operator passed a larger --prefill-budget.
# load_lib() overwrites both from qwn_wave_cap(); these are only the pre-load fallback.
WAVE_MAX_RUNTIME = 128
WAVE_MAX = WAVE_MAX_RUNTIME


def load_lib(path):
    L = ctypes.CDLL(path)
    L.qwn_init.argtypes = [ctypes.c_char_p]; L.qwn_init.restype = ctypes.c_int
    L.qwn_hidden_size.restype = ctypes.c_int
    L.qwn_reset_state.restype = ctypes.c_int
    L.qwn_decode.argtypes = [ctypes.c_int, ctypes.c_int]; L.qwn_decode.restype = ctypes.c_int
    L.qwn_paged_init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]; L.qwn_paged_init.restype = ctypes.c_int
    L.qwn_paged_free.restype = ctypes.c_int
    L.qwn_paged_reset_slot.argtypes = [ctypes.c_int]; L.qwn_paged_reset_slot.restype = ctypes.c_int
    L.qwn_paged_set_sampling.argtypes = [ctypes.c_int, ctypes.c_float, ctypes.c_ulonglong]
    L.qwn_paged_set_sampling.restype = ctypes.c_int
    L.qwn_paged_spec_round.argtypes = [ctypes.POINTER(ctypes.c_int)] * 5 + [ctypes.c_int, ctypes.c_int] + [ctypes.POINTER(ctypes.c_int)] * 2
    L.qwn_paged_spec_round.restype = ctypes.c_int
    L.qwn_paged_load_client.argtypes = [ctypes.c_int, ctypes.c_int]; L.qwn_paged_load_client.restype = ctypes.c_int
    L.qwn_paged_ckpt_save.argtypes = [ctypes.c_int, ctypes.c_int]; L.qwn_paged_ckpt_save.restype = ctypes.c_int
    L.qwn_paged_ckpt_adopt.argtypes = [ctypes.c_int, ctypes.c_int]; L.qwn_paged_ckpt_adopt.restype = ctypes.c_int
    L.qwn_paged_ckpt_free.argtypes = [ctypes.c_int]; L.qwn_paged_ckpt_free.restype = ctypes.c_int
    L.qwn_paged_ckpt_demote.argtypes = [ctypes.c_int]; L.qwn_paged_ckpt_demote.restype = ctypes.c_int
    L.qwn_paged_ckpt_promote.argtypes = [ctypes.c_int]; L.qwn_paged_ckpt_promote.restype = ctypes.c_int
    L.qwn_paged_ckpt_tier.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    L.qwn_paged_ckpt_tier.restype = ctypes.c_int
    L.qwn_paged_fork.argtypes = [ctypes.c_int] * 3; L.qwn_paged_fork.restype = ctypes.c_int
    L.qwn_paged_decode_step.argtypes = [ctypes.POINTER(ctypes.c_int)] * 3 + [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    L.qwn_paged_decode_step.restype = ctypes.c_int
    L.qwn_paged_stats.argtypes = [ctypes.POINTER(ctypes.c_int)] * 4; L.qwn_paged_stats.restype = ctypes.c_int
    L.qwn_paged_prefill_batch.argtypes = [ctypes.POINTER(ctypes.c_int)] * 7 + [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    L.qwn_paged_prefill_batch.restype = ctypes.c_int
    L.qwn_prefill_chunk.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    L.qwn_prefill_chunk.restype = ctypes.c_int
    L.qwn_prefill_wide.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int,
                                   ctypes.POINTER(ctypes.c_int)]
    L.qwn_prefill_wide.restype = ctypes.c_int
    L.qwn_free.restype = ctypes.c_int
    L.qwn_wave_cap.restype = ctypes.c_int
    # Take the wave cap from the engine so the two can never disagree.
    global WAVE_MAX_RUNTIME, WAVE_MAX
    cap = L.qwn_wave_cap()
    if cap >= 1:
        WAVE_MAX_RUNTIME = WAVE_MAX = cap
    return L


def fast_prefill(L, ids, chunk=16):
    """Single-stream prefill in <=16-token spec chunks (qwn_prefill_chunk) so long
    contexts (e.g. 64k) prefill in seconds instead of token-by-token minutes.
    Returns (final_argmax, last_pos). Timing-grade: the committed KV is what
    qwn_paged_load_client snapshots into each paged slot."""
    L.qwn_reset_state()
    n = len(ids); pos = 0; am = 0
    out = (ctypes.c_int * 1)()
    while pos < n:
        c = min(chunk, n - pos)
        buf = (ctypes.c_int * c)(*[int(x) for x in ids[pos:pos + c]])
        rc = L.qwn_prefill_chunk(buf, pos - 1, c, out)
        if rc != 0:
            raise RuntimeError(f"qwn_prefill_chunk rc={rc} at pos {pos}")
        am = out[0]; pos += c
    return am, n - 1


def paged_prefill_slot(L, slot, ids, page, wave_max=128):
    """Prefill one paged slot to len(ids) tokens via qwn_paged_prefill_batch in
    <=wave_max-col chunks (the server's _prefill_long path; allocates blocks from the
    pool, no single-stream/spec scratch). Returns the seed (argmax of the final chunk).
    Raises on -4 (pool exhausted = this worker count does not fit)."""
    n = len(ids); pos = 0; seed = 0
    oseed = (ctypes.c_int * 1)()
    while pos < n:
        c = min(wave_max, n - pos)
        final = 1 if pos + c >= n else 0
        rc = L.qwn_paged_prefill_batch(
            _ci([int(x) for x in ids[pos:pos + c]]), _ci([slot] * c),
            _ci(list(range(pos, pos + c))), _ci([slot]), _ci([0]), _ci([c]),
            _ci([final]), 1, c, oseed)
        if rc != 0:
            raise RuntimeError(f"qwn_paged_prefill_batch rc={rc} at pos {pos} slot {slot}")
        seed = oseed[0]; pos += c
    return seed


def _ci(a):
    return (ctypes.c_int * len(a))(*a)


def ck(r, what):
    if isinstance(r, int) and r < 0:
        raise RuntimeError(f"{what} failed: {r}")
    return r


class Request:
    __slots__ = ("ids", "max_new", "eos", "out", "done", "slot", "pos", "next_tok",
                 "started", "t_admit", "t_first", "t_done", "t_tok", "n_prompt", "err",
                 "progress", "cancel", "temp", "seed", "ng", "ng_n", "acc_ema", "seq")

    def __init__(self, ids, max_new, eos, temp=0.0, seed=0):
        self.ids = ids; self.max_new = max_new; self.eos = set(eos)
        self.temp = float(temp); self.seed = int(seed) & 0xFFFFFFFFFFFFFFFF
        self.ng = None; self.ng_n = 0; self.acc_ema = 1.0; self.seq = None
        self.out = []; self.done = threading.Event(); self.slot = -1
        self.pos = 0; self.next_tok = 0; self.started = False
        self.t_admit = self.t_first = self.t_done = self.t_tok = 0.0; self.n_prompt = len(ids); self.err = None
        # progress: set by the engine when out grows, so a streaming handler wakes
        # per committed token instead of only at completion.
        # cancel: set by the handler when the client is gone -- the engine detaches
        # the slot at the next step instead of decoding into a dead socket.
        self.progress = threading.Event(); self.cancel = False


class BatchedEngine:
    def __init__(self, lib, tqf, max_slots, num_blocks, page, wave_cols=WAVE_MAX,
                 max_prefill=2, fuse=True, fuse_ratio=0.0, fuse_idle_ms=125.0,
                 decode_every=0, prefix_cache=True, prefix_cache_min=256,
                 decode_min_rows=8, decode_max_idle_ms=250.0, prefill_budget=96):
        self.L = lib
        ck(self.L.qwn_init(tqf.encode()), "init")
        self.page = page; self.max_slots = max_slots
        self.wave_cols = max(1, min(WAVE_MAX, wave_cols))
        self.max_prefill = max(1, max_prefill)
        self.fuse = fuse; self.fuse_ratio = fuse_ratio
        self.decode_every = max(0, decode_every); self.starve = 0
        self.decode_min_rows = max(1, decode_min_rows)
        self.prefill_budget = max(1, prefill_budget)
        self.fuse_idle_ms = max(0.0, fuse_idle_ms)
        self.decode_max_idle_ms = max(1.0, decode_max_idle_ms)
        self.last_decode = time.time()
        self.pc_enabled = prefix_cache; self.pc_min = max(1, prefix_cache_min)
        self.pc_hits = self.pc_misses = self.pc_builds = self.pc_saved = 0
        # checkpoint registry (APC phase 2): engine ckpt id -> prefix ids + stats
        self.cks = []               # [{id, pos, ids, t_hit}]
        # With a host tier the registry is no longer bounded by the VRAM slab pool:
        # entries beyond it live demoted in pinned host RAM (engine cap is 24).
        _hgb = float(os.environ.get("TQ_CKPT_HOST_GB", "8"))
        self.ck_max = int(os.environ.get("TQ_CKPT_MAX", "16" if _hgb > 0 else "6"))
        self.ck_trim = max(0, int(os.environ.get("TQ_CKPT_TRIM", "8")))
        self.ck_last = None         # previous admitted prompt (LCP checkpoint candidate)
        self.pref = {}              # slot -> [Request, prefill cursor] (chunked prefill)
        ck(self.L.qwn_paged_init(max_slots, num_blocks, page), "paged_init")
        fb, tb, pg, mb = self._stats()
        self.num_blocks = tb; self.max_blocks_per_seq = mb
        self.free_slots = list(range(max_slots))
        self.active = {}            # slot -> Request
        self.q = []                 # pending Requests (FIFO)
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.running = True
        self.steps = 0; self.decoded_tokens = 0
        self.prefill_waves = 0; self.prefilled_tokens = 0
        # paged speculative decoding (chain verify): CORRECT (greedy bit-exact,
        # APC-safe, NVFP4-native) but not yet PROFITABLE -- the verify wave runs
        # chunk-256 DeltaNet scans + MMA-prefill attention on ~9-column chains,
        # costing 4-8x a decode step (measured 2026-09-01: slower at every
        # (ctx, n) cell despite 2.8-8.7 accepted tokens/round). Opt-in until the
        # wave core grows a small-T spec path (decode-attention batching + the
        # spec-class fused DeltaNet); break-even needs round <= step * tok/round.
        self.spec_on = os.environ.get("TQ_PAGED_SPEC", "0") == "1"
        self.spec_slots = max(1, int(os.environ.get("TQ_PG_SPEC_SLOTS", "4")))
        self.spec_maxd = max(0, min(15, int(os.environ.get("TQ_PAGED_SPEC_D", "8"))))
        self.spec_rounds = 0; self.spec_committed = 0; self.spec_drafted = 0
        self.spec_rounds_by_n = {}
        # wave cost accounting (see _wave / _loop)
        self.t_marshal = 0.0; self.t_engine = 0.0; self.t_loop = 0.0; self.t_idle = 0.0
        # per-wave timeline for scheduler diagnosis: (t_end, engine_ms, pref_cols,
        # dec_rows, segs, kind 0=fused wave 1=pure decode step). /v1/wavelog dumps it.
        self.wavelog = []
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _stats(self):
        fb, tb, pg, mb = (ctypes.c_int(), ctypes.c_int(), ctypes.c_int(), ctypes.c_int())
        ck(self.L.qwn_paged_stats(ctypes.byref(fb), ctypes.byref(tb), ctypes.byref(pg), ctypes.byref(mb)), "stats")
        return fb.value, tb.value, pg.value, mb.value

    def submit(self, ids, max_new, eos, temp=0.0, seed=0):
        req = Request(ids, max_new, eos, temp, seed)
        with self.cv:
            self.q.append(req)
            self.cv.notify()
        return req

    def _free_blocks(self):
        return self._stats()[0]

    def _activate(self, req, slot, seed):
        req.slot = slot; req.pos = req.n_prompt; req.next_tok = seed
        req.out.append(seed); req.started = True; req.t_admit = time.time(); req.t_first = req.t_admit
        req.t_tok = req.t_admit
        self.active[slot] = req
        if seed in req.eos or len(req.out) >= req.max_new:
            self._detach(slot)

    # ------------------- prefix checkpoints: the hybrid's APC (phase 2) -------------------
    # A checkpoint = shared refs to the full KV blocks of [0, pos) + COPIES of the tail
    # rows and the O(1) DeltaNet state (engine ABI, ckpt_smoke-gated bit-exact). The
    # scheduler saves one per distinct prompt at (n_prompt - trim): the trim stops a few
    # tokens short of the generation opener so the NEXT append-only turn -- whose history
    # replaces "<think>" with the rendered assistant message -- still prefix-matches. The
    # save happens MID-PREFILL at exactly that cursor (the DeltaNet state cannot be
    # rewound, so position and state must agree by construction). Admission adopts the
    # deepest matching checkpoint and prefills only the suffix. N-way, LRU, VRAM-capped;
    # every failure path degrades to a plain full prefill.
    def _ck_match(self, req):
        best = None
        for c in self.cks:
            if c["pos"] < req.n_prompt and (best is None or c["pos"] > best["pos"]) \
               and req.ids[:c["pos"]] == c["ids"]:
                best = c
        return best

    def _ck_evict_one(self):
        """Make room. Demoting the LRU RESIDENT entry hands back its state slab and
        every KV block ref while keeping the entry matchable from host RAM, so
        capacity stops being bounded by the slab pool. Only when nothing can be
        demoted (host budget spent, or everything already demoted) do we destroy the
        LRU outright."""
        if not self.cks:
            return False
        mb = ctypes.c_int(0)
        res = [c for c in self.cks
               if self.L.qwn_paged_ckpt_tier(c["id"], ctypes.byref(mb)) == 0]
        if res:
            lru = min(res, key=lambda c: c["t_hit"])
            if self.L.qwn_paged_ckpt_demote(lru["id"]) == 0:
                self.L.qwn_paged_ckpt_tier(lru["id"], ctypes.byref(mb))
                print(f"[ckpt] demote id={lru['id']} pos={lru['pos']} host={mb.value}MB", flush=True)
                return True
        lru = min(self.cks, key=lambda c: c["t_hit"])
        self.cks.remove(lru)
        rc = self.L.qwn_paged_ckpt_free(lru["id"])
        print(f"[ckpt] evict id={lru['id']} pos={lru['pos']} rc={rc}", flush=True)
        return True

    def _ck_targets(self, req, cursor):
        """Checkpoint positions for this prompt's prefill (sorted). Two candidates:
        the LCP with the PREVIOUS prompt (= where the shared history ends -- the
        boundary the next append-only turn will extend), and n_prompt - trim (the
        exact-resend / parallel-fan-out boundary)."""
        if not self.pc_enabled:
            return []
        cands = set()
        cands.add(req.n_prompt - self.ck_trim)
        prev = self.ck_last
        if prev:
            m = min(len(prev), req.n_prompt)
            i = 0
            while i < m and prev[i] == req.ids[i]:
                i += 1
            cands.add(i)
        out = []
        for n in sorted(cands):
            if n < self.pc_min or n <= cursor:
                continue
            # Near-dedup within 2*trim: a boundary within 16 tokens BELOW serves
            # the same adopters at <=16 extra suffix tokens, so saving both burns
            # TWO registry slots per turn (the LCP and n-8 land 8 apart, halving
            # the LRU's effective turn depth -- measured: resends of turn 1 missed
            # because turns 4-5's pairs evicted it).
            near = n - 2 * self.ck_trim
            if any(near <= c["pos"] <= n and c["ids"] == req.ids[:c["pos"]] for c in self.cks):
                continue
            if out and out[-1] > near:
                continue
            out.append(n)
        return out

    def _ck_save(self, req, slot, pos):
        while len(self.cks) >= self.ck_max:
            if not self._ck_evict_one():
                return
        cid = self.L.qwn_paged_ckpt_save(slot, pos)
        if cid < 0 and self._ck_evict_one():     # registry/VRAM pressure: retry once
            cid = self.L.qwn_paged_ckpt_save(slot, pos)
        if cid < 0:
            print(f"[ckpt] save pos={pos} rc={cid} FAILED", flush=True)
            return                               # optimization only; never fatal
        print(f"[ckpt] save pos={pos} -> id={cid}", flush=True)
        self.cks.append({"id": cid, "pos": pos, "ids": list(req.ids[:pos]),
                         "t_hit": time.time()})
        self.pc_builds += 1

    def _admit(self):
        """Move queued requests into the PREFILLING set (slot + blocks reserved).
        No prefill work happens here: _work() spends a bounded column budget per
        iteration, so an admission never stalls the active slots' decode."""
        free_blk = self._free_blocks()
        for st in self.pref.values():                    # blocks the in-flight prefills still need
            free_blk -= (st[0].n_prompt - st[1] + self.page - 1) // self.page
        qi = 0
        while qi < len(self.q) and self.free_slots:
            req = self.q[qi]
            # Cache-aware admission: if an IN-FLIGHT prefill is about to checkpoint a
            # boundary this request shares, wait the few waves for it instead of
            # re-prefilling the whole shared prefix (the concurrent fan-out race:
            # 6 subagents arriving together all missed and paid 24k tokens each).
            # No deadlock: the donor always progresses or is torn down, and the
            # hold lasts only while the target is ahead of the donor's cursor.
            # Held requests are SKIPPED (qi advances): no head-of-line blocking.
            if self._ck_match(req) is None:
                waiting = False
                for st in self.pref.values():
                    for t in (st[2] if len(st) > 2 else []):
                        if st[1] < t and req.n_prompt > t and req.ids[:t] == st[0].ids[:t]:
                            waiting = True
                            break
                    if waiting:
                        break
                if waiting:
                    qi += 1
                    continue
            if req.n_prompt < 1:
                self.q.pop(qi); req.err = "empty prompt"; req.done.set(); continue
            need = (req.n_prompt + self.page - 1) // self.page
            if need > self.max_blocks_per_seq:
                self.q.pop(qi); req.err = "prompt exceeds context"; req.done.set(); continue
            if need > free_blk:
                break                                   # pool full -> wait (admission control)
            if len(self.pref) >= self.max_prefill:
                break                                   # cap concurrent prefills (TTFT fairness)
            self.q.pop(qi); slot = self.free_slots.pop()
            try:
                ck(self.L.qwn_paged_reset_slot(slot), "reset_slot")
                if req.temp > 0.0:
                    ck(self.L.qwn_paged_set_sampling(slot, req.temp, req.seed), "set_sampling")
                free_blk -= need
                cursor = 0
                hit = self._ck_match(req)
                if hit is not None:
                    if self.L.qwn_paged_ckpt_tier(hit["id"], None) == 1:
                        pr = self.L.qwn_paged_ckpt_promote(hit["id"])
                        if pr == -3 and self._ck_evict_one():   # freed a slab; retry once
                            pr = self.L.qwn_paged_ckpt_promote(hit["id"])
                        if pr != 0:                             # adopt below returns -5
                            print(f"[ckpt] promote id={hit['id']} rc={pr}", flush=True)
                    pos = self.L.qwn_paged_ckpt_adopt(slot, hit["id"])
                    if pos == hit["pos"]:
                        cursor = pos                     # prefill only the suffix
                        hit["t_hit"] = time.time()
                        self.pc_hits += 1; self.pc_saved += cursor
                    else:
                        ck(self.L.qwn_paged_reset_slot(slot), "reset_slot")
                        print(f"[engine] ckpt adopt rc={pos}, full prefill instead", flush=True)
                        self.pc_misses += 1
                else:
                    self.pc_misses += 1
                self.pref[slot] = [req, cursor, self._ck_targets(req, cursor)]
                self.ck_last = list(req.ids)
            except Exception as e:
                # a failed admission must cost ONE request, not the slot and not the
                # server: before this guard a raise here leaked the slot and left the
                # request neither queued nor prefilling (client hung forever)
                self.free_slots.append(slot)
                req.err = f"admission failed: {e}"
                req.done.set()
                print(f"[engine] admit failed, request errored: {e}", flush=True)

    def _wave(self, dec_slots, pref_plan, cols_tok, cols_slot, cols_pos,
              seg_slot, seg_off, seg_len, seg_fin):
        """One qwn_paged_prefill_batch call carrying decode rows AND prompt chunks:
        a decode row is a 1-column final segment, so both ride ONE pass over the
        weights (chunked prefill). out_seed[k] = that segment's argmax."""
        # Split the wave cost three ways so it is knowable whether the scheduler
        # language matters: marshalling (pure Python), the engine call (GPU + engine
        # host work), and the rest of the loop. Reported by /v1/waveprof.
        K = len(seg_slot); T = len(cols_tok)
        t0 = time.perf_counter()
        oseed = (ctypes.c_int * K)()
        a_tok, a_slot, a_pos = _ci(cols_tok), _ci(cols_slot), _ci(cols_pos)
        a_ss, a_so, a_sl, a_sf = _ci(seg_slot), _ci(seg_off), _ci(seg_len), _ci(seg_fin)
        t1 = time.perf_counter()
        rc = self.L.qwn_paged_prefill_batch(a_tok, a_slot, a_pos,
                                            a_ss, a_so, a_sl, a_sf, K, T, oseed)
        t2 = time.perf_counter()
        self.t_marshal += t1 - t0
        self.t_engine += t2 - t1
        if len(self.wavelog) < 65536:
            self.wavelog.append((t2, (t2 - t1) * 1e3, T - len(dec_slots), len(dec_slots), K, 0))
        if rc != 0:
            raise RuntimeError(f"fused wave rc={rc} (K={K} T={T})")
        self.prefill_waves += 1
        self.prefilled_tokens += T - len(dec_slots)
        self.decoded_tokens += len(dec_slots)
        self.steps += 1
        finished = []
        _tnow = time.time()
        for j, s in enumerate(dec_slots):                # decode segments come first
            req = self.active[s]; o = oseed[j]
            req.out.append(o); req.pos += 1; req.next_tok = o; req.t_tok = _tnow
            req.progress.set()
            if req.cancel or o in req.eos or len(req.out) >= req.max_new:
                finished.append(s)
        for s in finished:
            self._detach(s)
        for slot, c, final, k in pref_plan:              # then the prompt segments
            st = self.pref.get(slot)
            if st is None:
                continue
            st[1] += c
            while len(st) > 2 and st[2] and st[2][0] == st[1]:
                self._ck_save(st[0], slot, st[1])        # state == cursor by construction
                st[2].pop(0)
            if final:
                del self.pref[slot]
                self._activate(st[0], slot, oseed[k])

    def _work(self):
        """One scheduler iteration.

        Two measured facts drive the policy:

        1. A decode step is ONE pass over ~20 GiB of FP6 weights whatever the row
           count (17.7 ms @ N=1, 29.2 ms @ N=32), so throughput wants FEW decode
           steps with MANY rows. Interleaving a decode step between every prefill
           wave burns whole weight reads at tiny batch size -- measured 285 vs 343
           tok/s agg @ N=32. Hence prefill-priority: while prompts are pending the
           GPU goes to prefill and rows accumulate. --decode-every N bounds the
           starvation (one decode step per N prefill waves; 0 = pure priority).
        2. The paged prefill wave is cheapest with FEW segments and MANY columns
           each (128 cols as 1 segment = 42.8 ms, as 32 segments = 73.5 ms), and
           the attention kernel charges every column of a wave the wave's DEEPEST
           position. So prompts prefill in fat chunks (small --max-prefill) and
           decode rows only ride along (--fuse) when their depth is comparable --
           fusing pos-0 columns into rows at pos ~700 measured a net loss."""
        if not self.pref:
            self.starve = 0
            if self.active:
                # depth gate: the verify round costs step x 1.23@1.5k / 1.47@8k /
                # 2.0@24k (wavelog microbench 2026-09-01) -- spec only where the
                # accept rate can pay. TQ_PAGED_SPEC_MAXPOS=0 removes the gate.
                spec_maxpos = int(os.environ.get("TQ_PAGED_SPEC_MAXPOS", "65536"))
                if (self.spec_on and len(self.active) <= self.spec_slots
                        and (spec_maxpos <= 0
                             or all(r.pos <= spec_maxpos for r in self.active.values()))):
                    self._spec_step()
                else:
                    self._step()
                self.last_decode = time.time()
            return
        cols_tok = []; cols_slot = []; cols_pos = []
        seg_slot = []; seg_off = []; seg_len = []; seg_fin = []
        dec_slots = []
        dec_maxpos = max((self.active[s].pos for s in self.active), default=0)
        pref_maxpos = 0
        for st in self.pref.values():
            pref_maxpos = max(pref_maxpos, st[1] + min(self.wave_cols, st[0].n_prompt - st[1]))
        # Riding is NOT free: measured 1.2 ms/row/wave inflation (N=8 sync batch,
        # 225 ridden tokens = 0.27 s) while the tail step count is set by the LAST
        # client and a step's cost is row-invariant -- so rides that merely give
        # early finishers a head start buy nothing. Ride ONLY rows that are
        # starving (no token for fuse_idle_ms); that keeps the ITL bound for
        # continuous arrival at ~6x less inflation for synchronized batches.
        now = time.time()
        ride = []
        if self.fuse and self.active and (self.fuse_ratio <= 0.0 or pref_maxpos * self.fuse_ratio >= dec_maxpos):
            ride = [s for s in self.active
                    if (now - self.active[s].t_tok) * 1000.0 >= self.fuse_idle_ms]
        fuse = bool(ride)
        self.starve += 1
        # A decode step costs one full pass over ~20 GiB of weights whatever the row
        # count, so running it at low occupancy WASTES a weight read. Measured under
        # guidellm continuous arrival: a fixed 'every N waves' tick produced 440 steps
        # carrying 4445 tokens = 10 rows/step. Gate on amortization instead, with a
        # wall-clock starvation deadline so ITL stays bounded for admitted requests.
        rows = len(self.active)
        idle_ms = (time.time() - self.last_decode) * 1000.0
        # With fuse on, gated rides ARE the starvation bound (1.2 ms/row vs a 17 ms
        # row-invariant weight read), so the min-rows and every-N clauses only apply
        # when riding is disabled; the wall-clock deadline stays as the safety net.
        due = bool(self.active) and (
            (not self.fuse and rows >= self.decode_min_rows)
            or idle_ms >= self.decode_max_idle_ms
            or (not self.fuse and self.decode_every > 0 and self.starve >= self.decode_every))
        if fuse:
            dec_slots = ride
            for s in dec_slots:
                req = self.active[s]
                seg_off.append(len(cols_tok)); seg_slot.append(s); seg_len.append(1); seg_fin.append(1)
                cols_tok.append(req.next_tok); cols_slot.append(s); cols_pos.append(req.pos)
            self.starve = 0; self.last_decode = time.time()
        elif due:
            self._step()                                 # fat decode step, fast path
            self.starve = 0; self.last_decode = time.time()
        # Decode rows must NOT eat the prompt budget. vLLM schedules running
        # (decode) requests first and lets prefill fill the remaining token budget
        # (v1/core/sched/scheduler.py: token_budget -> running loop -> waiting loop),
        # but its budget is 8192 so the decode rows are noise. Ours was 128, so at
        # N=32 the rows consumed a quarter of every wave and prefill progress fell
        # with concurrency. Wave cost is linear in columns (measured 0.417/0.418/
        # 0.433 ms per column at 128/256/512, i.e. no per-wave fixed cost to
        # amortize), so the prompt budget can simply ride ON TOP of the rows --
        # which only became expressible once the quantizer stopped capping a wave
        # at 128 columns.
        room = self.prefill_budget
        # Depth-adaptive wave width: past 16k the wave is attention-dominated and
        # wider waves amortize the weight read (64k prefill 4141 -> 4567 tok/s at
        # 2048 cols), so widen to the engine cap for deep clients.
        for st_deep in self.pref.values():
            if st_deep[1] >= 16384:
                room = max(room, WAVE_MAX_RUNTIME)
                break
        if len(cols_tok) + room > WAVE_MAX_RUNTIME:
            room = max(0, WAVE_MAX_RUNTIME - len(cols_tok))
        pref_plan = []
        for slot, st in list(self.pref.items()):
            if room <= 0:
                break
            req, pos = st[0], st[1]
            c = min(room, req.n_prompt - pos)
            ckts = st[2] if len(st) > 2 else []
            for t in ckts:
                if pos < t < pos + c:
                    c = t - pos                  # land the cursor ON the next checkpoint
                    break
            final = 1 if pos + c >= req.n_prompt else 0
            seg_off.append(len(cols_tok)); seg_slot.append(slot); seg_len.append(c); seg_fin.append(final)
            cols_tok += req.ids[pos:pos + c]
            cols_slot += [slot] * c
            cols_pos += list(range(pos, pos + c))
            pref_plan.append((slot, c, final, len(seg_slot) - 1))
            room -= c
        if not cols_tok:
            return
        self._wave(dec_slots, pref_plan, cols_tok, cols_slot, cols_pos,
                   seg_slot, seg_off, seg_len, seg_fin)

    def _detach(self, slot):
        req = self.active.pop(slot)
        ck(self.L.qwn_paged_reset_slot(slot), "reset_slot")
        self.free_slots.append(slot)
        req.t_done = time.time()
        req.done.set()
        req.progress.set()

    def _draft(self, req):
        """Recency n-gram draft from the request's OWN history, longest-suffix
        cascade (5-gram, then 4, then 3 -- the vLLM prompt-lookup precision
        order): the draft is whatever followed the latest PRIOR occurrence of
        the current tail. The materialized token list lives on the request and
        grows incrementally (req.seq): rebuilding ids+out per round was an
        O(context) copy that ate the engine's win at depth. Gated by a
        per-request accept EMA with a slow upward re-probe."""
        if req.acc_ema < 0.2:
            req.acc_ema = min(1.0, req.acc_ema + 0.01)   # re-probe after ~20 rounds
            return []
        if req.seq is None or len(req.seq) != req.n_prompt + len(req.out):
            req.seq = req.ids + req.out                  # one rebuild, then extended in place
        seq = req.seq
        n = len(seq)
        if req.ng is None:
            req.ng = ({}, {}, {}); req.ng_n = 0
        g5, g4, g3 = req.ng
        for i in range(max(req.ng_n, 4), n - 1):   # exclude the live tail gram
            g5[(seq[i - 4], seq[i - 3], seq[i - 2], seq[i - 1], seq[i])] = i + 1
            g4[(seq[i - 3], seq[i - 2], seq[i - 1], seq[i])] = i + 1
            g3[(seq[i - 2], seq[i - 1], seq[i])] = i + 1
        req.ng_n = max(req.ng_n, n - 1)
        if n < 5:
            return []
        j = (g5.get((seq[n - 5], seq[n - 4], seq[n - 3], seq[n - 2], seq[n - 1]))
             or g4.get((seq[n - 4], seq[n - 3], seq[n - 2], seq[n - 1]))
             or g3.get((seq[n - 3], seq[n - 2], seq[n - 1])))
        if not j or j >= n:
            return []
        return seq[j:j + self.spec_maxd]

    def _spec_step(self):
        """One fused speculative round over every active slot (chain verify;
        dlen=0 slots are plain decode inside the same wave). Greedy outputs are
        token-exact vs _step; sampled outputs are seed-replayable within spec
        mode and eps-equivalent across modes (wave-path logits differ from the
        decode step at float eps; a Gumbel draw can flip a near-tie)."""
        slots = list(self.active.keys())
        n = len(slots)
        # v2 verify core: total chain nodes (n slots x (draft+seed)) ride the
        # TQ_PG_SPEC_NODES archive (default 8) -- clamp per-round depth to fit.
        maxd = max(0, min(self.spec_maxd, int(os.environ.get("TQ_PG_SPEC_NODES", "8")) // n - 1))
        sl = (ctypes.c_int * n)(*slots)
        seeds = (ctypes.c_int * n)(*[self.active[s].next_tok for s in slots])
        pos = (ctypes.c_int * n)(*[self.active[s].pos for s in slots])
        dl = (ctypes.c_int * n)()
        dr = (ctypes.c_int * max(1, n * maxd))()
        for j, s in enumerate(slots):
            d = self._draft(self.active[s])[:maxd] if maxd > 0 else []
            dl[j] = len(d)
            for k2, t in enumerate(d):
                dr[j * maxd + k2] = t
            self.spec_drafted += len(d)
        if not any(dl[j] for j in range(n)):
            return self._step()                    # nothing drafted: zero-overhead round
        out = (ctypes.c_int * (n * (maxd + 1)))()
        om = (ctypes.c_int * n)()
        _t1 = time.perf_counter()
        rc = self.L.qwn_paged_spec_round(sl, seeds, pos, dr, dl, n, maxd, out, om)
        _t2 = time.perf_counter()
        if rc in (-111, -3):
            # Archive/pool unavailable (VRAM): the round fails BEFORE touching
            # any slot state, so plain decode is safe. Degrade permanently
            # instead of erroring requests (vLLM semantics: never 500 a request
            # because an optional accelerator could not allocate).
            self.spec_maxd = 0
            print(f"[engine] spec unavailable rc={rc}; permanent plain-decode fallback", flush=True)
            return self._step()
        if rc != 0:
            # a failed round can leave a partial slot rewound-but-unreplayed:
            # fail the participants (the _loop wave-error contract handles it)
            raise RuntimeError(f"paged spec round rc={rc} (N={n} maxd={maxd})")
        if len(self.wavelog) < 65536:
            self.wavelog.append((_t2, (_t2 - _t1) * 1e3, 0, n, n, 2))
        self.spec_rounds += 1
        self.spec_rounds_by_n[n] = self.spec_rounds_by_n.get(n, 0) + 1
        finished = []
        _tnow = time.time()
        for j, s in enumerate(slots):
            req = self.active[s]
            m = om[j]
            self.spec_committed += m
            self.decoded_tokens += m
            if dl[j] > 0:                          # accept EMA drives the draft gate
                req.acc_ema = 0.7 * req.acc_ema + 0.3 * ((m - 1) / dl[j])
            fin = False
            for k2 in range(m):
                o = out[j * (maxd + 1) + k2]
                req.out.append(o); req.pos += 1; req.next_tok = o
                if req.seq is not None:
                    req.seq.append(o)
                if req.cancel or o in req.eos or len(req.out) >= req.max_new:
                    fin = True
                    break
            req.t_tok = _tnow
            req.progress.set()
            if fin or req.cancel:
                finished.append(s)
        self.steps += 1
        for s in finished:
            self._detach(s)

    def _step(self):
        slots = list(self.active.keys())
        n = len(slots)
        toks = (ctypes.c_int * n)(*[self.active[s].next_tok for s in slots])
        sid = (ctypes.c_int * n)(*slots)
        pos = (ctypes.c_int * n)(*[self.active[s].pos for s in slots])
        out = (ctypes.c_int * n)()
        _t1 = time.perf_counter()
        ck(self.L.qwn_paged_decode_step(toks, sid, pos, n, out), "paged_step")
        _t2 = time.perf_counter()
        if len(self.wavelog) < 65536:
            self.wavelog.append((_t2, (_t2 - _t1) * 1e3, 0, n, n, 1))
        self.steps += 1; self.decoded_tokens += n
        finished = []
        _tnow = time.time()
        for j, s in enumerate(slots):
            req = self.active[s]
            o = out[j]
            req.out.append(o); req.pos += 1; req.next_tok = o; req.t_tok = _tnow
            req.progress.set()
            if req.cancel or o in req.eos or len(req.out) >= req.max_new:
                finished.append(s)
        for s in finished:
            self._detach(s)

    def _loop(self):
        while True:
            with self.cv:
                while self.running and not self.q and not self.active and not self.pref:
                    self.cv.wait(timeout=0.5)
                if not self.running and not self.active and not self.q and not self.pref:
                    return
                try:
                    self._admit()
                except Exception as e:
                    print(f"[engine] admit error: {e}", flush=True)
                have = bool(self.active) or bool(self.pref)
            if have:
                _t = time.perf_counter()
                try:
                    self._work()
                    self.err_streak = 0
                except Exception as e:                  # never let the engine thread die
                    print(f"[engine] step error: {e}", flush=True)
                    self.err_streak = getattr(self, "err_streak", 0) + 1
                    if self.err_streak >= 8:
                        # 8 consecutive failed steps = the CUDA context is gone
                        # (sticky error): every future wave fails identically.
                        # Die loudly so a supervisor restart yields a live engine
                        # instead of a zombie serving 100% errors.
                        print("[engine] FATAL: 8 consecutive step errors, exiting", flush=True)
                        os._exit(70)
                    with self.cv:
                        for s in list(self.active.keys()):
                            r = self.active.pop(s)
                            try:
                                self.L.qwn_paged_reset_slot(s)
                            except Exception:
                                pass
                            self.free_slots.append(s)
                            r.err = f"step failed: {e}"; r.done.set()
                        for s in list(self.pref.keys()):
                            r = self.pref.pop(s)[0]
                            try:
                                self.L.qwn_paged_reset_slot(s)
                            except Exception:
                                pass
                            self.free_slots.append(s)
                            r.err = f"prefill failed: {e}"; r.done.set()
                self.t_loop += time.perf_counter() - _t

    def shutdown(self):
        with self.cv:
            self.running = False
            self.cv.notify_all()
        self.thread.join(timeout=5)
        self.L.qwn_paged_free(); self.L.qwn_free()


# ----------------------------- self test (milestone 3.2) -----------------------------
def selftest(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    eos = [tok.eos_token_id] if tok.eos_token_id is not None else []
    prompts_txt = [
        "The history of cartography is the study of how maps",
        "In quantum mechanics, the wave function describes",
        "def fibonacci(n):\n    # return the nth Fibonacci number\n",
        "The recipe for a classic margherita pizza starts with",
        "Climate scientists measure global temperature using",
        "The French Revolution began in 1789 when",
        "To train a neural network you first need to",
        "The mitochondria is the powerhouse of",
    ][:args.clients]
    ids_list = [tok(t, add_special_tokens=False).input_ids for t in prompts_txt]
    MAXNEW = args.gen

    eng = BatchedEngine(load_lib(args.lib), args.tqf, args.max_slots, args.num_blocks, args.page)
    fb, tb, pg, mb = eng._stats()
    print(f"engine up: pool blocks={tb} page={pg} max_slots={eng.max_slots} free={fb}", flush=True)

    # single-stream reference (greedy) for each prompt
    refs = []
    for ids in ids_list:
        eng.L.qwn_reset_state()
        am = 0
        for t, tk in enumerate(ids):
            am = ck(eng.L.qwn_decode(int(tk), t), "ref")
        out = [am]; p = len(ids) - 1
        for _ in range(MAXNEW - 1):
            am = ck(eng.L.qwn_decode(am, p + 1), "ref"); p += 1
            out.append(am)
            if am in set(eos):
                break
        refs.append(out)

    # staggered submission: client i joins after i*stagger seconds
    reqs = []
    t0 = time.time()

    def submit_staggered():
        for i, ids in enumerate(ids_list):
            time.sleep(args.stagger)
            reqs.append((i, eng.submit(ids, MAXNEW, eos)))

    th = threading.Thread(target=submit_staggered, daemon=True); th.start()
    th.join()
    for _, r in reqs:
        r.done.wait(timeout=120)
    elapsed = time.time() - t0

    print("\n" + "=" * 72, flush=True)
    print(f"  3.2 CONTINUOUS-BATCHING SELFTEST: {len(ids_list)} clients, stagger={args.stagger}s, gen={MAXNEW}", flush=True)
    print("=" * 72, flush=True)
    total_tok = 0
    for i, r in reqs:
        ref = refs[i]
        n = min(len(ref), len(r.out))
        match = 0
        for a, b in zip(r.out[:n], ref[:n]):
            if a == b:
                match += 1
            else:
                break
        total_tok += len(r.out)
        cont = tok.decode(r.out[1:]) if len(r.out) > 1 else ""
        print(f"  client {i}: gen={len(r.out)} leading-match-vs-single={match}/{n}  "
              f"text={cont[:54]!r}", flush=True)
    agg = total_tok / elapsed
    print(f"\n  aggregate: {total_tok} tokens / {elapsed:.2f}s = {agg:.1f} tok/s across "
          f"{len(ids_list)} staggered clients", flush=True)
    print(f"  engine steps={eng.steps} batched-decode-tokens={eng.decoded_tokens}", flush=True)
    eng.shutdown()


# ----------------------------- tool-calling (ported from serve_openai.py) -----------------
TOOL_OPEN, TOOL_CLOSE = "<tool_call>", "</tool_call>"


def _coerce_arg(val, typ):
    """Coerce an XML-ish <parameter> string to its JSON-schema type (Qwen3.6 template
    serializes every value as text). With no declared type, best-effort JSON literal."""
    s = val.strip() if isinstance(val, str) else val
    if typ in ("integer", "number"):
        try:
            return int(s)
        except (ValueError, TypeError):
            try:
                return float(s)
            except (ValueError, TypeError):
                return val
    if typ == "boolean":
        if isinstance(s, str) and s.lower() in ("true", "false"):
            return s.lower() == "true"
        return val
    if typ in ("array", "object"):
        try:
            return json.loads(s)
        except Exception:
            return val
    if typ == "string":
        return val
    try:
        j = json.loads(s)
        return j if isinstance(j, (int, float, bool, list, dict)) or j is None else val
    except Exception:
        return val


def parse_tool_calls(text, tools=None):
    """Qwen tool-call formats: (a) JSON <tool_call>{"name":..,"arguments":{..}}</tool_call>
    (b) XML-ish <tool_call><function=NAME><parameter=KEY>VALUE</parameter>..</function></tool_call>.
    Returns (clean_text, tool_calls_list_or_None)."""
    import re as _re
    types = {}
    for t in (tools or []):
        fn = t.get("function", t) if isinstance(t, dict) else {}
        props = (((fn.get("parameters") or {}).get("properties")) or {})
        types[fn.get("name", "")] = {k: (v or {}).get("type") for k, v in props.items()}
    calls = []

    def _emit(name, args):
        calls.append({"id": "call_" + uuid.uuid4().hex[:16], "type": "function",
                      "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})

    def _take(m):
        raw = m.group(1).strip()
        try:
            obj = json.loads(raw)
            _emit(obj.get("name", ""), obj.get("arguments", {}))
            return ""
        except Exception:
            pass
        fm = _re.search(r"<function=([^>\s]+)>(.*?)(?:</function>|$)", raw, _re.S)
        if fm:
            name = fm.group(1)
            ptypes = types.get(name, {})
            args = {}
            for pm in _re.finditer(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", fm.group(2), _re.S):
                args[pm.group(1)] = _coerce_arg(pm.group(2), ptypes.get(pm.group(1)))
            _emit(name, args)
            return ""
        return m.group(0)

    clean = _re.sub(_re.escape(TOOL_OPEN) + r"(.*?)" + _re.escape(TOOL_CLOSE), _take, text, flags=_re.S)
    return clean.strip(), (calls or None)


# ----------------------------- OpenAI server (milestone 3.3) -----------------------------
def make_handler(eng, tok, args):
    # EOS: eos_token_id + im_end/endoftext so tool-call turns terminate cleanly
    eos = set(int(t) for t in [tok.eos_token_id] if t is not None)
    for _name in ("<|im_end|>", "<|endoftext|>"):
        try:
            _t = tok.convert_tokens_to_ids(_name)
            if _t is not None and _t >= 0:
                eos.add(int(_t))
        except Exception:
            pass
    eos = list(eos)

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/v1/models"):
                ctx_max = int(os.environ.get("TQ_CTX", "262144"))
                self._json(200, {"object": "list", "data": [{
                    "id": args.model_name, "object": "model", "created": int(time.time()),
                    "owned_by": "knivesysl", "root": args.model_name, "parent": None,
                    "max_model_len": ctx_max,
                    "permission": [{"id": "modelperm-ksl", "object": "model_permission",
                                    "created": int(time.time()), "allow_create_engine": False,
                                    "allow_sampling": True, "allow_logprobs": False,
                                    "allow_search_indices": False, "allow_view": True,
                                    "allow_fine_tuning": False, "organization": "*",
                                    "group": None, "is_blocking": False}]}]})
            elif self.path.startswith("/health"):
                fb, tb, _, _ = eng._stats()
                self._json(200, {"status": "ok", "free_blocks": fb, "total_blocks": tb,
                                 "active": len(eng.active), "prefilling": len(eng.pref),
                                 "queued": len(eng.q), "steps": eng.steps,
                                 "decoded_tokens": eng.decoded_tokens,
                                 "prefilled_tokens": eng.prefilled_tokens,
                                 "prefix_cache": {"enabled": eng.pc_enabled,
                                                  "prefix_tokens": sum(c["pos"] for c in eng.cks),
                                                  "checkpoints": len(eng.cks),
                                                  "hits": eng.pc_hits, "misses": eng.pc_misses,
                                                  "builds": eng.pc_builds,
                                                  "tokens_saved": eng.pc_saved},
                                 "spec": {"enabled": eng.spec_on,
                                          "rounds": eng.spec_rounds,
                                          "committed": eng.spec_committed,
                                          "drafted": eng.spec_drafted,
                                          "rounds_by_n": eng.spec_rounds_by_n,
                                          "tokens_per_round": (eng.spec_committed / eng.spec_rounds)
                                                              if eng.spec_rounds else 0.0}})
            elif self.path.startswith("/waveprof"):
                # Where a wave's wall time actually goes. `engine_inside` is measured by
                # the engine itself; `engine` is what Python sees around the ctypes call,
                # so their difference is interpreter cost (GIL re-acquisition) that no
                # amount of scheduler tuning can remove.
                w = max(eng.prefill_waves, 1)
                other = max(eng.t_loop - eng.t_marshal - eng.t_engine, 0.0)
                try:
                    eng.L.qwn_wave_ms.restype = ctypes.c_double
                    eng.L.qwn_wave_count.restype = ctypes.c_long
                    e_ms, e_n = eng.L.qwn_wave_ms(), max(eng.L.qwn_wave_count(), 1)
                except Exception:
                    e_ms, e_n = 0.0, 1
                self._json(200, {"waves": eng.prefill_waves, "engine_waves": e_n,
                                 "ms_per_wave": {
                                     "marshal": 1000.0 * eng.t_marshal / w,
                                     "engine_seen_by_python": 1000.0 * eng.t_engine / w,
                                     "engine_inside": e_ms / e_n,
                                     "scheduler_other": 1000.0 * other / w,
                                     "loop_total": 1000.0 * eng.t_loop / w}})
            elif self.path.startswith("/v1/wavelog"):
                # Raw wave timeline. Host gap between consecutive engine calls is
                # rec[i].t_end - rec[i-1].t_end - rec[i].engine_ms.
                self._json(200, {"log": [list(r) for r in eng.wavelog]})
            elif self.path.startswith("/v1/wavereset"):
                eng.wavelog = []
                self._json(200, {"ok": True})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if not self.path.startswith("/v1/chat/completions") and not self.path.startswith("/v1/completions"):
                self._json(404, {"error": "not found"}); return
            n = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(n) or b"{}")
            # OpenAI chat sends max_completion_tokens (guidellm's chat handler does);
            # /v1/completions sends max_tokens; Responses-style clients send
            # max_output_tokens. Reading only one silently halves the requested length.
            # No field at all -> vLLM semantics: generate into the remaining window
            # (capped at 16k; the old default of 128 truncated every agent reply).
            mt_raw = (body.get("max_tokens") or body.get("max_completion_tokens")
                      or body.get("max_output_tokens"))
            stream = bool(body.get("stream", False))
            is_chat = self.path.startswith("/v1/chat")
            tools = body.get("tools") or None
            if is_chat:
                msgs = body.get("messages", [])
                # OpenAI carries tool_call arguments as a JSON string; the Qwen
                # template iterates them as a mapping -> parse in place.
                for m in msgs:
                    for tc in (m.get("tool_calls") or []):
                        fn = tc.get("function") or {}
                        if isinstance(fn.get("arguments"), str):
                            try:
                                fn["arguments"] = json.loads(fn["arguments"])
                            except Exception:
                                pass
                # Default ON (TQ_THINK=0 to flip). vLLM's actual API for this is
                # chat_template_kwargs={"enable_thinking": ...}; the bare top-level
                # field is kept as a convenience alias.
                ctk = body.get("chat_template_kwargs") or {}
                think = bool(ctk.get("enable_thinking",
                                     body.get("enable_thinking",
                                              os.environ.get("TQ_THINK", "1") != "0")))
                try:
                    tmpl = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True,
                                                   tokenize=False, enable_thinking=think)
                except TypeError:
                    tmpl = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True,
                                                   tokenize=False)
                ids = tok(tmpl, add_special_tokens=False).input_ids
            else:
                ids = tok(body.get("prompt", ""), add_special_tokens=False).input_ids
            ctx_max = int(os.environ.get("TQ_CTX", "262144"))
            win = ctx_max - len(ids) - 8
            # Interactive floor (TQ_MIN_OUT, default 8192): coding CLIs compute
            # max_tokens from THEIR configured window with THEIR tokenizer and
            # routinely send tiny caps (observed: 36 on a 26K prompt) that cut
            # replies mid-word with finish_reason=length. Lift explicit caps to
            # the floor unless the client pinned length for benchmarking
            # (ignore_eos + max_tokens) -- EOS still ends generation naturally,
            # so the floor only removes artificial truncation. No field at all
            # -> vLLM semantics: generate into the remaining window
            # (TQ_DEF_OUT cap, default 16384; the old default of 128 truncated
            # every agent reply).
            ignore = bool(body.get("ignore_eos", False))
            if mt_raw:
                max_new = int(mt_raw)
                if not ignore:
                    max_new = max(max_new, int(os.environ.get("TQ_MIN_OUT", "8192")))
            else:
                max_new = int(os.environ.get("TQ_DEF_OUT", "16384"))
            max_new = max(16, min(max_new, win))
            # stop strings, vLLM semantics: applied to the RAW generation (thinking
            # included), earliest match truncates and finishes with "stop".
            stop_raw = body.get("stop")
            stops = ([stop_raw] if isinstance(stop_raw, str)
                     else [s0 for s0 in (stop_raw or []) if isinstance(s0, str) and s0])
            # ignore_eos: benchmark/eval harnesses (guidellm) pin the output length by
            # sending max_tokens + ignore_eos, so every request does equal work.
            req_eos = [] if ignore else eos
            want_usage = bool((body.get("stream_options") or {}).get("include_usage"))
            # Sampling: an omitted temperature keeps the engine's greedy default
            # (agentic clients here want determinism + APC-friendly replays);
            # explicit temperature>0 samples engine-side with the spec-sampler
            # semantics (temp-scaled + TQ_MIN_P tail floor + replayable seed).
            # top_p/top_k are not implemented by the v1 sampler (min-p governs the
            # tail instead) and are accepted-but-ignored, like serve_openai.
            temp = max(0.0, float(body.get("temperature") or 0.0))
            seed = int(body.get("seed") or int.from_bytes(os.urandom(8), "little"))
            req = eng.submit(list(ids), max_new, req_eos, temp, seed)
            cid = f"chatcmpl-{int(time.time()*1000)}"

            if stream:
                # TRUE token streaming: the engine sets req.progress as each token
                # commits, so deltas leave as they are produced (TTFT/ITL are real).
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                base = {"id": cid, "object": "chat.completion.chunk" if is_chat else "text_completion",
                        "model": args.model_name}

                def _sse(o):
                    self.wfile.write(("data: " + json.dumps(o) + "\n\n").encode())
                    self.wfile.flush()

                try:
                    if is_chat:
                        _sse({**base, "choices": [{"index": 0, "delta": {"role": "assistant"},
                                                   "finish_reason": None}]})
                    # The completion STARTS inside <think> (the template appends the
                    # opener), so text up to </think> streams as delta.reasoning_content
                    # and the rest as delta.content -- the split vLLM's qwen3 reasoning
                    # parser performs. A 7-char holdback avoids emitting a partial
                    # "</think"; stop strings get the same holdback treatment.
                    in_think = is_chat and think
                    up, stopped, content_open = 0, False, not (is_chat and think)
                    tool_start = -1                     # index of first <tool_call> in full
                    sent_txt, sent_tok, deadline = "", 0, time.time() + args.timeout
                    while True:
                        done = req.done.is_set()
                        n_out = len(req.out)
                        if n_out > sent_tok:
                            full = tok.decode(req.out[:n_out], skip_special_tokens=True)
                            for st0 in stops:
                                i2 = full.find(st0)
                                if i2 >= 0:
                                    full = full[:i2]; stopped = True; req.cancel = True
                            fin = done or stopped
                            if in_think:
                                b = full.find("</think>")
                                if b < 0:
                                    safe = len(full) if fin else max(up, len(full) - 7)
                                    if safe > up:
                                        _sse({**base, "choices": [{"index": 0, "delta": {"reasoning_content": full[up:safe]}, "finish_reason": None}]})
                                        up = safe
                                else:
                                    if b > up:
                                        _sse({**base, "choices": [{"index": 0, "delta": {"reasoning_content": full[up:b]}, "finish_reason": None}]})
                                    up = b + 8
                                    in_think = False
                            if not in_think:
                                # swallow the newlines that follow </think> even when they
                                # arrive in a LATER decode step than the tag itself
                                if not content_open:
                                    while up < len(full) and full[up] == "\n": up += 1
                                # auto tool choice: content stops streaming at the first
                                # <tool_call>; the XML buffers silently and is emitted as
                                # a delta.tool_calls chunk when generation ends (the
                                # qwen3_coder-parser behavior vLLM has).
                                if is_chat and tools and tool_start < 0:
                                    ti = full.find(TOOL_OPEN, up)
                                    if ti >= 0:
                                        tool_start = ti
                                lim = tool_start if tool_start >= 0 else len(full)
                                hb = 0
                                if not fin:
                                    if stops: hb = max(hb, max(len(s0) for s0 in stops) - 1)
                                    if is_chat and tools and tool_start < 0:
                                        hb = max(hb, len(TOOL_OPEN) - 1)
                                safe = max(up, min(lim, len(full) - hb))
                                if safe > up:
                                    content_open = True
                                    ch = ({"index": 0, "delta": {"content": full[up:safe]}, "finish_reason": None}
                                          if is_chat else
                                          {"index": 0, "text": full[up:safe], "finish_reason": None})
                                    _sse({**base, "choices": [ch]})
                                    up = safe
                            sent_txt, sent_tok = full, n_out
                            if stopped:
                                break
                        elif done:
                            break
                        else:
                            if time.time() > deadline:
                                break
                            req.progress.wait(0.05)
                            req.progress.clear()
                    gen = len(req.out)
                    tcs = None
                    if is_chat and tools and tool_start >= 0:
                        _clean, tcs = parse_tool_calls(sent_txt[tool_start:], tools)
                        if tcs:
                            _sse({**base, "choices": [{"index": 0, "delta": {"tool_calls": [
                                dict(tc, index=i) for i, tc in enumerate(tcs)]},
                                "finish_reason": None}]})
                    finish = ("tool_calls" if tcs else
                              ("stop" if stopped else ("length" if gen >= max_new else "stop")))
                    ch = ({"index": 0, "delta": {}, "finish_reason": finish} if is_chat
                          else {"index": 0, "text": "", "finish_reason": finish})
                    _sse({**base, "choices": [ch]})
                    if want_usage:
                        _sse({**base, "choices": [],
                              "usage": {"prompt_tokens": req.n_prompt, "completion_tokens": gen,
                                        "total_tokens": req.n_prompt + gen}})
                    self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    # client walked away: stop burning a slot on a dead socket
                    req.cancel = True
                return

            if not req.done.wait(timeout=args.timeout):
                req.cancel = True
                self._json(504, {"error": "generation timeout"}); return
            if getattr(req, "err", None):
                self._json(500, {"error": req.err}); return
            # include the full committed continuation (out[0] is the first generated token)
            text = tok.decode(req.out if len(req.out) else [], skip_special_tokens=True)
            gen = len(req.out)
            stopped = False
            for st0 in stops:
                i2 = text.find(st0)
                if i2 >= 0:
                    text = text[:i2]; stopped = True
            reasoning = None
            if is_chat and think:
                j = text.find("</think>")
                if j >= 0:
                    reasoning, text = text[:j], text[j + 8:].lstrip("\n")
                else:                       # budget exhausted inside the think block
                    reasoning, text = text, ""
            tool_calls = None
            if is_chat and tools and TOOL_OPEN in text:
                text, tool_calls = parse_tool_calls(text, tools)
            finish = ("tool_calls" if tool_calls else
                      ("stop" if stopped else ("length" if gen >= max_new else "stop")))
            usage = {"prompt_tokens": req.n_prompt, "completion_tokens": gen,
                     "total_tokens": req.n_prompt + gen}
            if is_chat:
                msg = {"role": "assistant", "content": text or None}
                if reasoning:
                    msg["reasoning_content"] = reasoning
                if tool_calls:
                    msg["tool_calls"] = tool_calls
                resp = {"id": cid, "object": "chat.completion", "model": args.model_name,
                        "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                        "usage": usage}
            else:
                resp = {"id": cid, "object": "text_completion", "model": args.model_name,
                        "choices": [{"index": 0, "text": text, "finish_reason": finish}],
                        "usage": usage}
            self._json(200, resp)

    return H


def serve(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    L = load_lib(args.lib)          # this also publishes the engine's wave cap
    # argparse defaults are frozen at parse time, i.e. BEFORE the library is loaded, so
    # wave-derived options must be resolved here or they keep the stale fallback.
    if args.wave_cols is None:
        args.wave_cols = WAVE_MAX
    if args.prefill_budget is None:
        # Full-width default: post GQA-attention + wide-wave ks=1, chunk 2048 wins at
        # EVERY depth (2k +12%, 16k +11%, 64k +6% over 512-col waves). Latency-sensitive
        # multi-client setups can pass --prefill-budget 512 to trade throughput for ITL.
        args.prefill_budget = WAVE_MAX
    print(f"batched server: engine wave cap={WAVE_MAX} "
          f"wave_cols={args.wave_cols} prefill_budget={args.prefill_budget}", flush=True)
    eng = BatchedEngine(L, args.tqf, args.max_slots, args.num_blocks, args.page,
                        wave_cols=args.wave_cols, max_prefill=args.max_prefill,
                        fuse=args.fuse, fuse_ratio=args.fuse_ratio, fuse_idle_ms=args.fuse_idle_ms,
                        decode_every=args.decode_every,
                        prefix_cache=args.prefix_cache, prefix_cache_min=args.prefix_cache_min,
                        decode_min_rows=args.decode_min_rows,
                        decode_max_idle_ms=args.decode_max_idle_ms,
                        prefill_budget=args.prefill_budget)
    fb, tb, pg, mb = eng._stats()
    print(f"batched server: pool blocks={tb} page={pg} max_slots={eng.max_slots} free={fb}", flush=True)
    # BaseHTTPServer's default backlog is 5: with N clients connecting at once the
    # kernel resets the surplus SYNs (ConnectionReset at the client) long before the
    # scheduler is the limit. This is a many-client server -- size it to the slots.
    class _Server(ThreadingHTTPServer):
        daemon_threads = True
        request_queue_size = max(128, 4 * args.max_slots)
    httpd = _Server((args.host, args.port), make_handler(eng, tok, args))
    print(f"listening on {args.host}:{args.port}  (model={args.model_name})", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        eng.shutdown()


# ----------------------------- steady-state throughput bench -----------------------------
def bench(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    L = load_lib(args.lib)
    ck(L.qwn_init(args.tqf.encode()), "init")
    text = "The history of cartography is the study of maps and "
    base_ids = tok(text, add_special_tokens=False).input_ids
    # tile the base text up to bench_p so we can measure decode at long contexts (e.g. 64k)
    reps = (args.bench_p + len(base_ids) - 1) // max(1, len(base_ids))
    ids = (base_ids * reps)[: args.bench_p]
    Ns = [int(x) for x in args.bench_ns.split(",")]
    Nmax = max(Ns)
    ck(L.qwn_paged_init(max(args.max_slots, Nmax), args.num_blocks, args.page), "paged_init")
    # Prefill all Nmax slots DIRECTLY via the paged path (qwn_paged_prefill_batch,
    # <=128-col chunks) -- the real server prefill. No single-stream spec scratch, so the
    # KV pool can be sized to the actual worker count at long contexts (e.g. 64k). A slot
    # prefill that runs the pool dry returns -4 (= that worker count does not fit).
    seed = 0
    for s in range(Nmax):
        ck(L.qwn_paged_reset_slot(s), "reset")
        seed = paged_prefill_slot(L, s, ids, args.page)
    p = len(ids) - 1
    M = args.bench_iters
    fb = ctypes.c_int(); tb = ctypes.c_int(); pgv = ctypes.c_int(); mb = ctypes.c_int()
    L.qwn_paged_stats(ctypes.byref(fb), ctypes.byref(tb), ctypes.byref(pgv), ctypes.byref(mb))
    print("\n" + "=" * 72, flush=True)
    print(f"  3.x STEADY-STATE THROUGHPUT (paged decode) @ ctx={len(ids)} tok, "
          f"{Nmax} slots prefilled, KV blocks used={tb.value - fb.value}/{tb.value}", flush=True)
    print("=" * 72, flush=True)
    print(f"  {'N':>4} {'ms/step':>9} {'agg tok/s':>11} {'per-cli':>9}", flush=True)
    for N in Ns:
        toks = (ctypes.c_int * N)(*([seed] * N))
        sid = (ctypes.c_int * N)(*list(range(N)))
        pos = (ctypes.c_int * N)(*([p + 1] * N))
        out = (ctypes.c_int * N)()
        ck(L.qwn_paged_decode_step(toks, sid, pos, N, out), "warm")
        t0 = time.time()
        for _ in range(M):
            ck(L.qwn_paged_decode_step(toks, sid, pos, N, out), "step")
        t_step = (time.time() - t0) / M
        agg = N / t_step
        print(f"  {N:>4} {1000*t_step:>9.3f} {agg:>11.1f} {1.0/t_step:>9.1f}", flush=True)
    L.qwn_paged_free(); L.qwn_free()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tqf", default="/workspace/models/Qwen3.6-27B/qwen3_6-27b-e2m3-mtp.tqf")
    ap.add_argument("--model-dir", default="/workspace/models/Qwen3.6-27B")
    ap.add_argument("--lib", default=os.path.join(HERE, "build-qwen", "libforward_qwen.so"))
    ap.add_argument("--model-name", default="knivesysl-qwen3.6-27b-batched")
    ap.add_argument("--page", type=int, default=128)
    ap.add_argument("--max-slots", type=int, default=12)
    ap.add_argument("--num-blocks", type=int, default=1024)
    ap.add_argument("--wave-cols", type=int, default=None,
                    help="columns per scheduler iteration (decode rows + prompt chunks share "
                         "one weight read); default = the engine's own wave cap")
    ap.add_argument("--max-prefill", type=int, default=2,
                    help="max concurrently prefilling requests (TTFT fairness vs decode share)")
    ap.add_argument("--fuse", dest="fuse", action="store_true", default=True,
                    help="let decode rows ride the prompt-chunk wave (one weight read) when the "
                         "prompt depth is comparable to the decode depth (default ON)")
    ap.add_argument("--no-fuse", dest="fuse", action="store_false")
    ap.add_argument("--fuse-ratio", type=float, default=0.0,
                    help="fuse only if prompt_chunk_depth * RATIO >= decode_depth; the paged attn "
                         "kernel charges every column the wave's deepest position")
    ap.add_argument("--fuse-idle-ms", type=float, default=125.0,
                    help="ride a decode row on a prompt wave only when that client has had no "
                         "token for this long; riding costs ~1.2 ms/row/wave, a tail step is "
                         "row-invariant, so eager riding buys nothing for synchronized batches "
                         "(0 = ride every wave, the old behavior)")
    ap.add_argument("--decode-every", type=int, default=4,
                    help="while prompts are pending, force a decode step every N prefill waves. "
                         "0 = pure prefill-priority, which STARVES decode under continuous "
                         "arrival (there is always a pending prompt): measured 9/41 requests "
                         "completed in a 20s guidellm window. 4 bounds the starvation while "
                         "keeping most of the fat-wave prefill throughput")
    ap.add_argument("--prefix-cache", dest="prefix_cache", action="store_true", default=True,
                    help="materialize a shared prompt prefix once and snapshot it into each new "
                         "slot (qwn_paged_load_client) instead of re-prefilling it (default ON)")
    ap.add_argument("--no-prefix-cache", dest="prefix_cache", action="store_false")
    ap.add_argument("--prefill-budget", type=int, default=None,
                    help="prompt columns per wave, ON TOP of the decode rows (which are "
                         "scheduled first). Needs TQ_WAVE_MAX >= rows + budget")
    ap.add_argument("--decode-min-rows", type=int, default=8,
                    help="run a decode step while prompts are pending only once this many slots "
                         "are active (amortizes the weight read); ignored at the idle deadline")
    ap.add_argument("--decode-max-idle-ms", type=float, default=250.0,
                    help="force a decode step if none has run for this long (bounds ITL for "
                         "already-admitted requests under continuous arrival)")
    ap.add_argument("--prefix-cache-min", type=int, default=256,
                    help="min shared prefix tokens before the cache bothers")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--bench-ns", default="1,2,4,8,16,32")
    ap.add_argument("--bench-p", type=int, default=48)
    ap.add_argument("--bench-iters", type=int, default=64)
    ap.add_argument("--clients", type=int, default=8)
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--stagger", type=float, default=0.15)
    args = ap.parse_args()
    if args.bench:
        bench(args)
    elif args.selftest:
        selftest(args)
    else:
        serve(args)


if __name__ == "__main__":
    main()
