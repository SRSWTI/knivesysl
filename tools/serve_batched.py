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
    L.qwn_paged_load_client.argtypes = [ctypes.c_int, ctypes.c_int]; L.qwn_paged_load_client.restype = ctypes.c_int
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
                 "progress", "cancel")

    def __init__(self, ids, max_new, eos):
        self.ids = ids; self.max_new = max_new; self.eos = set(eos)
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
        self.pc_ids = []; self.pc_last = None
        self.pc_hits = self.pc_misses = self.pc_builds = self.pc_saved = 0
        self.pc_streak = 0          # consecutive misses (lets a stale prefix be replaced)
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

    def submit(self, ids, max_new, eos):
        req = Request(ids, max_new, eos)
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

    # ---------------------------- shared-prefix cache ----------------------------
    # qwn_paged_load_client(slot, upto_pos) copies the single-stream Q4 KV rows
    # [0..upto_pos] into a slot's pool blocks AND the DeltaNet recurrent/conv state
    # into that slot. That recurrent state is the state AFTER the whole single-stream
    # prefill (it is O(1), not per-position), so a slot may reuse it only when its
    # prompt starts with EXACTLY the materialized prefix -- there is no partial
    # rewind for the 48 gated-DeltaNet layers. Hence ONE materialized prefix (the
    # shared system prompt of an agent fleet), not vLLM's N-way hash-addressed APC.
    # This dedups the prefix COMPUTE, not its memory: every slot still gets its own
    # physical copy of the prefix blocks (block sharing needs refcounted block
    # tables in the engine).
    def _prefix_hit(self, req):
        n = len(self.pc_ids)
        return (self.pc_enabled and n >= self.pc_min and req.n_prompt > n
                and req.ids[:n] == self.pc_ids)

    def _materialize_prefix(self, ids):
        """Prefill `ids` into the single-stream state (wide path, ~2500 tok/s) so
        later admissions snapshot it into their slot instead of re-prefilling."""
        n = len(ids)
        ck(self.L.qwn_reset_state(), "reset_state")
        am = ctypes.c_int(0)
        pos = 0
        while pos < n:
            c = min(WAVE_MAX, n - pos)
            last = (pos + c >= n)
            rc = self.L.qwn_prefill_wide(_ci(ids[pos:pos + c]), pos - 1, c,
                                         ctypes.byref(am) if last else None)
            if rc != 0:
                self.pc_ids = []
                print(f"[engine] prefix materialize rc={rc} at {pos}", flush=True)
                return
            pos += c
        self.pc_ids = list(ids)
        self.pc_builds += 1

    def _prefix_candidate(self, req):
        """Longest common prefix with the last admitted prompt: a shared system
        prompt shows up here on the second request that carries it.

        A cached prefix that stopped hitting must be replaceable, even by a SHORTER
        candidate -- otherwise the first (possibly over-long) prefix wins forever
        and the fleet's real shared prompt never gets cached."""
        prev = self.pc_last
        self.pc_last = req.ids
        if not self.pc_enabled or not prev:
            return None
        m = min(len(prev), req.n_prompt - 1)
        i = 0
        while i < m and prev[i] == req.ids[i]:
            i += 1
        if i < self.pc_min:
            return None
        if i > len(self.pc_ids) or self.pc_streak >= 2:
            return req.ids[:i]
        return None

    def _admit(self):
        """Move queued requests into the PREFILLING set (slot + blocks reserved).
        No prefill work happens here: _work() spends a bounded column budget per
        iteration, so an admission never stalls the active slots' decode."""
        free_blk = self._free_blocks()
        for st in self.pref.values():                    # blocks the in-flight prefills still need
            free_blk -= (st[0].n_prompt - st[1] + self.page - 1) // self.page
        while self.q and self.free_slots:
            req = self.q[0]
            if req.n_prompt < 1:
                self.q.pop(0); req.err = "empty prompt"; req.done.set(); continue
            need = (req.n_prompt + self.page - 1) // self.page
            if need > self.max_blocks_per_seq:
                self.q.pop(0); req.err = "prompt exceeds context"; req.done.set(); continue
            if need > free_blk:
                break                                   # pool full -> wait (admission control)
            if len(self.pref) >= self.max_prefill:
                break                                   # cap concurrent prefills (TTFT fairness)
            self.q.pop(0); slot = self.free_slots.pop()
            ck(self.L.qwn_paged_reset_slot(slot), "reset_slot")
            free_blk -= need
            cursor = 0
            if self._prefix_hit(req):
                rc = self.L.qwn_paged_load_client(slot, len(self.pc_ids) - 1)
                if rc == 0:
                    cursor = len(self.pc_ids)            # prefill only the suffix
                    self.pc_hits += 1; self.pc_saved += cursor; self.pc_streak = 0
                else:
                    ck(self.L.qwn_paged_reset_slot(slot), "reset_slot")
                    print(f"[engine] load_client rc={rc}, full prefill instead", flush=True)
            else:
                self.pc_misses += 1; self.pc_streak += 1
                cand = self._prefix_candidate(req)
                if cand is not None:
                    self._materialize_prefix(cand)
                    if self._prefix_hit(req) and self.L.qwn_paged_load_client(slot, len(cand) - 1) == 0:
                        cursor = len(cand)
                        self.pc_hits += 1; self.pc_saved += cursor; self.pc_streak = 0
            self.pref[slot] = [req, cursor]              # [request, prefill cursor]

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
                self._step(); self.last_decode = time.time()
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
        if len(cols_tok) + room > WAVE_MAX_RUNTIME:
            room = max(0, WAVE_MAX_RUNTIME - len(cols_tok))
        pref_plan = []
        for slot, st in list(self.pref.items()):
            if room <= 0:
                break
            req, pos = st
            c = min(room, req.n_prompt - pos)
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
                except Exception as e:                  # never let the engine thread die
                    print(f"[engine] step error: {e}", flush=True)
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
                self._json(200, {"object": "list", "data": [{"id": args.model_name, "object": "model"}]})
            elif self.path.startswith("/health"):
                fb, tb, _, _ = eng._stats()
                self._json(200, {"status": "ok", "free_blocks": fb, "total_blocks": tb,
                                 "active": len(eng.active), "prefilling": len(eng.pref),
                                 "queued": len(eng.q), "steps": eng.steps,
                                 "decoded_tokens": eng.decoded_tokens,
                                 "prefilled_tokens": eng.prefilled_tokens,
                                 "prefix_cache": {"enabled": eng.pc_enabled,
                                                  "prefix_tokens": len(eng.pc_ids),
                                                  "hits": eng.pc_hits, "misses": eng.pc_misses,
                                                  "builds": eng.pc_builds,
                                                  "tokens_saved": eng.pc_saved}})
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
            max_new = int(body.get("max_tokens") or body.get("max_completion_tokens")
                          or body.get("max_output_tokens") or 128)
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
                think = bool(body.get("enable_thinking", False))
                try:
                    tmpl = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True,
                                                   tokenize=False, enable_thinking=think)
                except TypeError:
                    tmpl = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True,
                                                   tokenize=False)
                ids = tok(tmpl, add_special_tokens=False).input_ids
            else:
                ids = tok(body.get("prompt", ""), add_special_tokens=False).input_ids
            # ignore_eos: benchmark/eval harnesses (guidellm) pin the output length by
            # sending max_tokens + ignore_eos, so every request does equal work.
            req_eos = [] if bool(body.get("ignore_eos", False)) else eos
            want_usage = bool((body.get("stream_options") or {}).get("include_usage"))
            req = eng.submit(list(ids), max_new, req_eos)
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
                    sent_txt, sent_tok, deadline = "", 0, time.time() + args.timeout
                    while True:
                        done = req.done.is_set()
                        n_out = len(req.out)
                        if n_out > sent_tok:
                            full = tok.decode(req.out[:n_out], skip_special_tokens=True)
                            delta = full[len(sent_txt):]
                            sent_txt, sent_tok = full, n_out
                            if delta:
                                ch = ({"index": 0, "delta": {"content": delta}, "finish_reason": None}
                                      if is_chat else
                                      {"index": 0, "text": delta, "finish_reason": None})
                                _sse({**base, "choices": [ch]})
                        elif done:
                            break
                        else:
                            if time.time() > deadline:
                                break
                            req.progress.wait(0.05)
                            req.progress.clear()
                    gen = len(req.out)
                    finish = "length" if gen >= max_new else "stop"
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
            tool_calls = None
            if is_chat and tools and TOOL_OPEN in text:
                text, tool_calls = parse_tool_calls(text, tools)
            finish = "tool_calls" if tool_calls else ("length" if gen >= max_new else "stop")
            usage = {"prompt_tokens": req.n_prompt, "completion_tokens": gen,
                     "total_tokens": req.n_prompt + gen}
            if is_chat:
                msg = {"role": "assistant", "content": text or None}
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
