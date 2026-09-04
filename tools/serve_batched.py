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
import argparse, ctypes, faulthandler, hmac, json, math, os, select, signal, socket, threading, time, traceback, queue, uuid
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


class RequestError(Exception):
    """A client-visible OpenAI request error."""
    def __init__(self, status, message, error_type="invalid_request_error",
                 code=None, param=None):
        super().__init__(message)
        self.status = int(status)
        self.message = str(message)
        self.error_type = str(error_type)
        self.code = code
        self.param = param


class EngineFatal(RuntimeError):
    """Native state ownership is no longer trustworthy; restart the process."""


def _error_body(message, error_type="server_error", code=None, param=None):
    return {"error": {"message": str(message), "type": str(error_type),
                      "param": param, "code": code}}


class Request:
    __slots__ = ("id", "ids", "max_new", "eos", "out", "done", "slot", "pos",
                 "next_tok", "started", "t_admit", "t_first", "t_done", "t_tok",
                 "t_submit_mono", "queue_deadline_mono", "deadline_mono",
                 "n_prompt", "err", "err_status", "err_type", "err_code",
                 "progress", "cancel", "cancel_lock", "cancel_reason",
                 "cancel_status", "cancel_kind", "temp", "seed", "priority",
                 "ng", "ng_n", "acc_ema", "seq", "ck_epoch", "ck_hit",
                 "state", "terminal_counted")

    def __init__(self, rid, ids, max_new, eos, temp=0.0, seed=0, priority=0,
                 request_timeout=300.0, queue_timeout=300.0):
        self.id = rid
        self.ids = ids; self.max_new = max_new; self.eos = set(eos)
        self.temp = float(temp); self.seed = int(seed) & 0xFFFFFFFFFFFFFFFF
        self.priority = int(priority)
        self.ng = None; self.ng_n = 0; self.acc_ema = 1.0; self.seq = None
        self.out = []; self.done = threading.Event(); self.slot = -1
        self.pos = 0; self.next_tok = 0; self.started = False
        self.t_admit = self.t_first = self.t_done = self.t_tok = 0.0
        self.t_submit_mono = time.monotonic()
        self.queue_deadline_mono = self.t_submit_mono + max(0.001, queue_timeout)
        self.deadline_mono = self.t_submit_mono + max(0.001, request_timeout)
        self.n_prompt = len(ids); self.err = None; self.err_status = 500
        self.err_type = "server_error"; self.err_code = None
        self.ck_epoch = -1; self.ck_hit = None
        self.progress = threading.Event()
        self.cancel_lock = threading.Lock()
        self.cancel = False; self.cancel_reason = None; self.cancel_status = 499
        self.cancel_kind = "cancelled"
        self.state = "queued"; self.terminal_counted = False


class BatchedEngine:
    def __init__(self, lib, tqf, max_slots, num_blocks, page, wave_cols=WAVE_MAX,
                 max_prefill=2, fuse=True, fuse_ratio=0.0, fuse_idle_ms=125.0,
                 decode_every=0, prefix_cache=True, prefix_cache_min=256,
                 decode_min_rows=8, decode_max_idle_ms=250.0, prefill_budget=96,
                 max_queue=128, queue_timeout=300.0):
        self.L = lib
        self.phase_lock = threading.Lock()
        self.phase = "startup"
        self.phase_since_mono = time.monotonic()
        self.current_request_id = None
        self.last_progress_mono = None
        self.busy_since_mono = None
        self.last_engine_error = None
        self._metric_lock = threading.Lock()
        self.metrics = {
            "requests_total": 0, "requests_completed": 0,
            "requests_cancelled": 0, "requests_failed": 0,
            "requests_rejected": 0, "queue_timeouts": 0,
            "request_timeouts": 0, "admission_recoveries": 0,
            "admission_failures": 0, "native_failures": 0,
        }
        ck(self._native("startup_model", self.L.qwn_init, tqf.encode()), "init")
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
        self.ck_epoch = 0           # invalidates per-request checkpoint-match caches
        # With a host tier the registry is no longer bounded by the VRAM slab pool:
        # entries beyond it live demoted in pinned host RAM (engine cap is 24).
        _hgb = float(os.environ.get("TQ_CKPT_HOST_GB", "0"))
        self.ck_max = int(os.environ.get("TQ_CKPT_MAX", "16" if _hgb > 0 else "6"))
        self.ck_trim = max(0, int(os.environ.get("TQ_CKPT_TRIM", "8")))
        self.ck_last = None         # previous admitted prompt (LCP checkpoint candidate)
        self.pref = {}              # slot -> [Request, prefill cursor] (chunked prefill)
        self._cached_free_blocks = num_blocks
        self._cached_total_blocks = num_blocks
        self._cached_stats_mono = time.monotonic()
        self.max_queue = max(1, int(max_queue))
        self.queue_timeout = max(0.001, float(queue_timeout))
        ck(self._native("startup_pool", self.L.qwn_paged_init,
                        max_slots, num_blocks, page), "paged_init")
        fb, tb, pg, mb = self._stats()
        self.num_blocks = tb; self.max_blocks_per_seq = mb
        self.free_slots = list(range(max_slots))
        self.active = {}            # slot -> Request
        self.q = []                 # pending Requests (FIFO)
        self.lock = threading.Lock()
        self._owned_lock = threading.Lock()
        self._owned = {}            # request id -> Request until terminal outcome
        self.cv = threading.Condition(self.lock)
        self.running = True
        self.steps = 0; self.decoded_tokens = 0
        self.prefill_waves = 0; self.prefilled_tokens = 0
        self.started_mono = time.monotonic()
        self.started_unix = time.time()
        self.supervisor_restarts = int(os.environ.get("KSL_SUPERVISOR_RESTARTS", "0"))
        self.supervisor_last_exit_code = int(
            os.environ.get("KSL_SUPERVISOR_LAST_EXIT_CODE", "0"))
        self.supervisor_last_exit_unix = int(
            os.environ.get("KSL_SUPERVISOR_LAST_EXIT_TIME_SECONDS", "0"))
        self.admission_recoveries = 0; self.admission_failures = 0
        self.admission_failure_streak = 0
        self._set_phase("idle")
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

    def _metric(self, name, delta=1):
        with self._metric_lock:
            self.metrics[name] = self.metrics.get(name, 0) + delta
    def metrics_snapshot(self):
        with self._metric_lock:
            return dict(self.metrics)


    def _set_phase(self, phase, req=None):
        now = time.monotonic()
        with self.phase_lock:
            if phase != self.phase or req != self.current_request_id:
                self.phase = phase
                self.current_request_id = req
                self.phase_since_mono = now

    def _progress(self):
        self.last_progress_mono = time.monotonic()

    def _native(self, phase, fn, *args, req=None):
        """Bracket every native call so a watchdog can identify a wedged phase."""
        previous = (self.phase, self.current_request_id)
        self._set_phase(phase, req)
        try:
            result = fn(*args)
        except BaseException as exc:
            self.last_engine_error = f"{phase}: {type(exc).__name__}: {exc}"
            self._metric("native_failures")
            raise
        finally:
            self._set_phase(*previous)
        return result

    def _finish(self, req, state, error=None, status=500,
                error_type="server_error", error_code=None):
        # Cancellation and native completion race at token boundaries. Serialize
        # the terminal transition so exactly one outcome and one metric wins.
        with req.cancel_lock:
            if req.done.is_set():
                return
            if req.cancel:
                state = req.cancel_kind
                error = req.cancel_reason
                status = req.cancel_status
                error_type = req.err_type
                error_code = req.err_code
            req.state = state
            req.t_done = time.time()
            if error is not None:
                req.err = str(error)
                req.err_status = int(status)
                req.err_type = str(error_type)
                req.err_code = error_code
            if not req.terminal_counted:
                req.terminal_counted = True
                duration = max(0.0, time.monotonic() - req.t_submit_mono)
                terminal_metric = (
                    "requests_completed" if state == "completed" else
                    "requests_cancelled" if state == "cancelled" else
                    "requests_failed")
                with self._metric_lock:
                    m = self.metrics
                    m[terminal_metric] = m.get(terminal_metric, 0) + 1
                    m["request_duration_seconds"] = (
                        m.get("request_duration_seconds", 0) + duration)
                    m["prompt_tokens"] = m.get("prompt_tokens", 0) + req.n_prompt
                    m["generation_tokens"] = (
                        m.get("generation_tokens", 0) + len(req.out))
                    if error_code == "request_timeout":
                        m["request_timeouts"] = m.get("request_timeouts", 0) + 1
                    elif error_code == "queue_timeout":
                        m["queue_timeouts"] = m.get("queue_timeouts", 0) + 1
                    if req.t_admit:
                        m["queue_duration_seconds"] = (
                            m.get("queue_duration_seconds", 0) +
                            max(0.0, req.t_admit - req.t_submit_mono))
                        m["queue_duration_count"] = (
                            m.get("queue_duration_count", 0) + 1)
                    if req.t_first:
                        m["time_to_first_token_seconds"] = (
                            m.get("time_to_first_token_seconds", 0) +
                            max(0.0, req.t_first - req.t_submit_mono))
                        m["time_to_first_token_count"] = (
                            m.get("time_to_first_token_count", 0) + 1)
            with self._owned_lock:
                if self._owned.get(req.id) is req:
                    del self._owned[req.id]
            req.done.set()
        req.progress.set()

    def cancel(self, req, reason="client disconnected", status=499,
               state="cancelled", error_type="request_cancelled",
               error_code=None):
        with req.cancel_lock:
            if req.done.is_set() or req.cancel:
                return False
            req.cancel_reason = None if reason is None else str(reason)
            req.cancel_status = int(status)
            req.cancel_kind = state
            req.err_type = error_type
            req.err_code = error_code
            if state != "completed" and reason is not None:
                req.err = str(reason)
                req.err_status = int(status)
            req.cancel = True
        req.progress.set()
        # The engine owns this lock across native work. Cancellation must never
        # wait behind a wedged CUDA call; an idle engine is woken when possible.
        if self.cv.acquire(blocking=False):
            try:
                self.cv.notify()
            finally:
                self.cv.release()
        return True

    def _fatal(self, reason):
        self.last_engine_error = str(reason)
        self._set_phase("fatal")
        print(f"[Engine] FATAL: {reason}", flush=True)
        faulthandler.dump_traceback(all_threads=True)
        os._exit(70)

    def _stats(self):
        fb, tb, pg, mb = (ctypes.c_int(), ctypes.c_int(), ctypes.c_int(), ctypes.c_int())
        ck(self._native("stats", self.L.qwn_paged_stats,
                        ctypes.byref(fb), ctypes.byref(tb),
                        ctypes.byref(pg), ctypes.byref(mb)), "stats")
        # Health handlers must never enter CUDA: if the engine stream is wedged,
        # a native stats call would wedge the watchdog too. Publish the latest
        # engine-thread sample instead.
        self._cached_free_blocks = fb.value
        self._cached_total_blocks = tb.value
        self._cached_stats_mono = time.monotonic()
        return fb.value, tb.value, pg.value, mb.value

    def submit(self, ids, max_new, eos, temp=0.0, seed=0, priority=0,
               request_id=None, request_timeout=300.0, queue_timeout=None):
        if not ids:
            raise RequestError(400, "prompt must contain at least one token",
                               param="prompt")
        if max_new < 1:
            raise RequestError(400, "max_tokens must be at least 1",
                               param="max_tokens")
        rid = request_id or f"req-{uuid.uuid4().hex}"
        req = Request(rid, list(ids), int(max_new), eos, temp, seed,
                      priority=priority, request_timeout=request_timeout,
                      queue_timeout=self.queue_timeout if queue_timeout is None
                      else queue_timeout)
        with self.cv:
            if not self.running or not self.thread.is_alive():
                self._metric("requests_rejected")
                raise RequestError(503, "engine is not available", "server_error",
                                   "engine_unavailable")
            if request_id is not None:
                with self._owned_lock:
                    duplicate = rid in self._owned
                if duplicate:
                    self._metric("requests_rejected")
                    raise RequestError(409, "X-Request-ID is already in flight",
                                       "invalid_request_error",
                                       "duplicate_request_id")
            if len(self.q) >= self.max_queue:
                self._metric("requests_rejected")
                raise RequestError(429, "request queue is full",
                                   "server_overloaded", "queue_full")
            if not self.q and not self.active and not self.pref:
                self.busy_since_mono = time.monotonic()
            self.q.append(req)
            with self._owned_lock:
                self._owned[rid] = req
            self.q.sort(key=lambda r: (r.priority, r.t_submit_mono))
            self._metric("requests_total")
            self.cv.notify()
        return req

    def _free_blocks(self):
        return self._stats()[0]

    def _activate(self, req, slot, seed):
        req.slot = slot; req.pos = req.n_prompt; req.next_tok = seed
        req.out.append(seed); req.started = True; req.state = "active"
        req.t_first = time.monotonic()
        req.t_tok = time.time()
        self.active[slot] = req
        terminal = self._terminal(req, seed)
        if terminal is not None:
            self._detach(slot, terminal)

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
        # A capacity-blocked request may revisit admission many times. Cache the
        # O(prefix length) comparison until the checkpoint registry changes.
        if req.ck_epoch == self.ck_epoch:
            return req.ck_hit
        best = None
        for c in self.cks:
            if c["pos"] < req.n_prompt and (best is None or c["pos"] > best["pos"]) \
               and req.ids[:c["pos"]] == c["ids"]:
                best = c
        req.ck_epoch = self.ck_epoch
        req.ck_hit = best
        return best

    def _ck_evict_one(self, protect_id=None):
        """Make room without evicting a checkpoint selected for this admission.

        Demoting a resident entry returns its state slab and KV refs. If host
        demotion is unavailable, destroy that same resident entry; removing an
        already-demoted entry would reclaim no GPU capacity.
        """
        cands = [c for c in self.cks if c["id"] != protect_id]
        if not cands:
            return False
        mb = ctypes.c_int(0)
        res = []
        for c in cands:
            tier = self._native("ckpt_tier", self.L.qwn_paged_ckpt_tier,
                                c["id"], ctypes.byref(mb))
            if tier == 0:
                res.append(c)
            elif tier != 1:
                raise EngineFatal(f"checkpoint tier id={c['id']} rc={tier}")
        if res:
            lru = min(res, key=lambda c: c["t_hit"])
            drc = self._native("ckpt_demote", self.L.qwn_paged_ckpt_demote,
                               lru["id"])
            if drc == 0:
                self._native("ckpt_tier", self.L.qwn_paged_ckpt_tier,
                             lru["id"], ctypes.byref(mb))
                print(f"[ckpt] demote id={lru['id']} pos={lru['pos']} "
                      f"host={mb.value}MB", flush=True)
                return True
        else:
            lru = min(cands, key=lambda c: c["t_hit"])
        rc = self._native("ckpt_free", self.L.qwn_paged_ckpt_free, lru["id"])
        if rc != 0:
            raise EngineFatal(f"checkpoint free id={lru['id']} rc={rc}")
        self.cks.remove(lru)
        self.ck_epoch += 1
        print(f"[ckpt] evict id={lru['id']} pos={lru['pos']} rc=0", flush=True)
        return True

    def _ck_flush(self, reason):
        """Drop every checkpoint or fail closed on an ownership mismatch."""
        entries = list(self.cks)
        for c in entries:
            rc = self._native("ckpt_free", self.L.qwn_paged_ckpt_free, c["id"])
            if rc != 0:
                raise EngineFatal(f"checkpoint flush id={c['id']} rc={rc}")
            self.cks.remove(c)
            self.ck_epoch += 1
        print(f"[ckpt] flush reason={reason} count={len(entries)} failed=0",
              flush=True)
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
        cid = self._native("ckpt_save", self.L.qwn_paged_ckpt_save, slot, pos,
                           req=req.id)
        if cid < 0 and self._ck_evict_one():
            cid = self._native("ckpt_save", self.L.qwn_paged_ckpt_save,
                               slot, pos, req=req.id)
        if cid < 0:
            print(f"[ckpt] save pos={pos} rc={cid} FAILED", flush=True)
            return                               # optimization only; never fatal
        print(f"[ckpt] save pos={pos} -> id={cid}", flush=True)
        self.cks.append({"id": cid, "pos": pos, "ids": list(req.ids[:pos]),
                         "t_hit": time.time()})
        self.ck_epoch += 1
        self.pc_builds += 1

    def _request_blocks(self, req):
        # The final emitted token is never fed back into KV. Reserve the exact
        # worst-case footprint so two long requests cannot both admit and then
        # deadlock at a later page boundary.
        return (req.n_prompt + max(0, req.max_new - 1) + self.page - 1) // self.page

    def _remaining_blocks(self, req, cursor):
        return max(0, self._request_blocks(req) -
                      (cursor + self.page - 1) // self.page)

    def _admission_free_blocks(self):
        """Pool blocks not already promised to an admitted request."""
        free_blk = self._free_blocks()
        for req in self.active.values():
            free_blk -= self._remaining_blocks(req, req.pos)
        for st in self.pref.values():
            free_blk -= self._remaining_blocks(st[0], st[1])
        return free_blk

    def _reset_slot(self, slot, req=None):
        rc = self._native("reset_slot", self.L.qwn_paged_reset_slot, slot,
                          req=req.id if req is not None else None)
        if rc != 0:
            raise EngineFatal(f"reset slot={slot} rc={rc}")

    def _expire_queued(self):
        """Finish cancelled or expired requests before considering capacity."""
        now = time.monotonic()
        qi = 0
        while qi < len(self.q):
            req = self.q[qi]
            if req.cancel:
                self.q.pop(qi)
                self._finish(req, req.cancel_kind, req.cancel_reason,
                             req.cancel_status, req.err_type, req.err_code)
            elif now >= req.deadline_mono:
                self.q.pop(qi)
                self._finish(req, "failed", "request deadline exceeded", 504,
                             "timeout_error", "request_timeout")
            elif now >= req.queue_deadline_mono:
                self.q.pop(qi)
                self._finish(req, "failed", "request timed out in queue", 503,
                             "server_overloaded", "queue_timeout")
            else:
                qi += 1

    def _admit(self):
        """Move queued requests into the PREFILLING set (slot + blocks reserved).
        No prefill work happens here: _work() spends a bounded column budget per
        iteration, so an admission never stalls the active slots' decode."""
        self._expire_queued()
        free_blk = self._admission_free_blocks()
        qi = 0
        while qi < len(self.q) and self.free_slots:
            req = self.q[qi]
            # Queue cancellation/deadline handling runs before the capacity loop.
            # Include the whole requested decode budget, not just prompt pages.
            total_need = self._request_blocks(req)
            if total_need > self.max_blocks_per_seq:
                self.q.pop(qi)
                self._finish(req, "failed", "request exceeds model context", 400,
                             "invalid_request_error", "context_length_exceeded")
                continue
            hit = self._ck_match(req)
            hit_tier = (self._native("ckpt_tier", self.L.qwn_paged_ckpt_tier,
                                     hit["id"], None, req=req.id)
                        if hit is not None else -1)
            if hit is not None and hit_tier not in (0, 1):
                raise EngineFatal(f"checkpoint tier id={hit['id']} rc={hit_tier}")
            need = (total_need - hit["pos"] // self.page) if hit_tier == 0 else \
                   (total_need + (1 if hit is not None and hit["pos"] % self.page else 0))
            # Cache state is expendable. Evict non-selected entries before
            # declaring capacity unavailable.
            while need > free_blk and self._ck_evict_one(
                    hit["id"] if hit is not None else None):
                free_blk = self._admission_free_blocks()
            if need > free_blk:
                qi += 1                              # bounded HOL bypass
                continue
            if len(self.pref) >= self.max_prefill:
                break
            # Cache-aware admission: if an IN-FLIGHT prefill is about to checkpoint a
            # boundary this request shares, wait the few waves for it instead of
            # re-prefilling the whole shared prefix (the concurrent fan-out race).
            if hit is None:
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
            self.q.pop(qi); slot = self.free_slots.pop()
            req.t_admit = time.monotonic()
            req.state = "prefilling"
            try:
                self._reset_slot(slot, req)
                if req.temp > 0.0:
                    src = self._native("set_sampling",
                                       self.L.qwn_paged_set_sampling,
                                       slot, req.temp, req.seed, req=req.id)
                    if src != 0:
                        raise EngineFatal(f"set sampling slot={slot} rc={src}")
                free_blk -= need
                cursor = 0
                if hit is not None:
                    if self._native("ckpt_tier", self.L.qwn_paged_ckpt_tier,
                                    hit["id"], None, req=req.id) == 1:
                        pr = self._native("ckpt_promote",
                                          self.L.qwn_paged_ckpt_promote,
                                          hit["id"], req=req.id)
                        if pr == -3 and self._ck_evict_one(hit["id"]):
                            pr = self._native("ckpt_promote",
                                              self.L.qwn_paged_ckpt_promote,
                                              hit["id"], req=req.id)
                        if pr != 0:
                            print(f"[ckpt] promote id={hit['id']} rc={pr}",
                                  flush=True)
                    pos = self._native("ckpt_adopt",
                                       self.L.qwn_paged_ckpt_adopt,
                                       slot, hit["id"], req=req.id)
                    if pos == hit["pos"]:
                        cursor = pos                     # prefill only the suffix
                        hit["t_hit"] = time.time()
                        self.pc_hits += 1; self.pc_saved += cursor
                    else:
                        # A stale/corrupt hit must not turn the suffix reservation
                        # into an unaccounted full prefill. Discard it, re-plan the
                        # cold request, and either reserve that plan or wait.
                        self._reset_slot(slot, req)
                        frc = self._native("ckpt_free",
                                           self.L.qwn_paged_ckpt_free,
                                           hit["id"], req=req.id)
                        if frc != 0:
                            raise EngineFatal(
                                f"stale checkpoint free id={hit['id']} rc={frc}")
                        self.cks.remove(hit)
                        self.ck_epoch += 1
                        print(f"[engine] ckpt adopt rc={pos}; evict id={hit['id']} "
                              "rc=0, full prefill instead", flush=True)
                        self.pc_misses += 1
                        free_blk = self._admission_free_blocks()
                        while total_need > free_blk and self._ck_evict_one():
                            free_blk = self._admission_free_blocks()
                        if total_need > free_blk:
                            self.q.insert(qi, req)
                            self.free_slots.append(slot)
                            break
                        free_blk -= total_need
                else:
                    self.pc_misses += 1
                self.pref[slot] = [req, cursor, self._ck_targets(req, cursor)]
                self.ck_last = list(req.ids)
                self._progress()
            except EngineFatal:
                raise
            except Exception as e:
                try:
                    self._reset_slot(slot, req)
                except Exception as reset_exc:
                    raise EngineFatal(
                        f"admission cleanup slot={slot}: {reset_exc}") from reset_exc
                self.free_slots.append(slot)
                self.admission_failures += 1
                self.admission_failure_streak += 1
                self._metric("admission_failures")
                self._finish(req, "failed", f"admission failed: {e}", 500,
                             "server_error", "admission_failed")
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
        rc = self._native("prefill_batch", self.L.qwn_paged_prefill_batch,
                          a_tok, a_slot, a_pos, a_ss, a_so, a_sl, a_sf,
                          K, T, oseed)
        t2 = time.perf_counter()
        self.t_marshal += t1 - t0
        self.t_engine += t2 - t1
        if len(self.wavelog) < 65536:
            self.wavelog.append((t2, (t2 - t1) * 1e3, T - len(dec_slots), len(dec_slots), K, 0))
        if rc != 0:
            raise EngineFatal(f"fused wave rc={rc} (K={K} T={T})")
        self.last_progress_mono = t2
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
            terminal = self._terminal(req, o)
            if terminal is not None:
                finished.append((s, terminal))
        for s, terminal in finished:
            self._detach(s, terminal)
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

    def _expire_admitted(self):
        """Cancel/deadline checks for requests that already own a slot.

        HTTP handlers only publish cancellation; the engine thread owns every
        native reset and slot transition.
        """
        for slot in list(self.active):
            req = self.active.get(slot)
            if req is None:
                continue
            terminal = self._terminal(req)
            if terminal is not None:
                self._detach(slot, terminal)
        for slot in list(self.pref):
            st = self.pref.get(slot)
            if st is None:
                continue
            req = st[0]
            terminal = self._terminal(req)
            if terminal is not None:
                self._abort_prefill(slot, terminal)

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
        self._expire_admitted()
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

    def _terminal(self, req, token=None):
        with req.cancel_lock:
            if req.cancel:
                return (req.cancel_kind, req.cancel_reason, req.cancel_status,
                        req.err_type, req.err_code)
        if time.monotonic() >= req.deadline_mono:
            return ("failed", "request deadline exceeded", 504,
                    "timeout_error", "request_timeout")
        if token is not None and token in req.eos:
            return ("completed", None, 200, None, None)
        if len(req.out) >= req.max_new:
            return ("completed", None, 200, None, None)
        return None

    def _detach(self, slot, terminal=None):
        req = self.active[slot]
        if terminal is None:
            terminal = ("completed", None, 200, None, None)
        # Retain ownership until native cleanup succeeds. On failure the outer
        # EngineFatal path terminates the process with the request still owned.
        self._reset_slot(slot, req)
        del self.active[slot]
        self.free_slots.append(slot)
        self._finish(req, *terminal)

    def _abort_prefill(self, slot, terminal):
        req = self.pref[slot][0]
        self._reset_slot(slot, req)
        del self.pref[slot]
        self.free_slots.append(slot)
        self._finish(req, *terminal)


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
        rc = self._native("spec_round", self.L.qwn_paged_spec_round,
                          sl, seeds, pos, dr, dl, n, maxd, out, om)
        _t2 = time.perf_counter()
        if rc in (-111, -3):
            # Archive/pool unavailable (VRAM): the round fails BEFORE touching
            # any slot state, so plain decode is safe. Degrade permanently
            # instead of erroring requests (vLLM semantics: never 500 a request
            # because an optional accelerator could not allocate).
            self.spec_maxd = 0
            self._metric("spec_fallbacks")
            print(f"[engine] spec unavailable rc={rc}; permanent plain-decode fallback", flush=True)
            return self._step()
        if rc != 0:
            # Other native failures may have partially mutated slots or poisoned
            # CUDA. Never retry participant state whose atomicity is unknown.
            raise EngineFatal(f"paged spec round rc={rc} (N={n} maxd={maxd})")
        self.last_progress_mono = _t2
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
            terminal = None
            for k2 in range(m):
                o = out[j * (maxd + 1) + k2]
                req.out.append(o); req.pos += 1; req.next_tok = o
                if req.seq is not None:
                    req.seq.append(o)
                terminal = self._terminal(req, o)
                if terminal is not None:
                    break
            req.t_tok = _tnow
            req.progress.set()
            if terminal is not None:
                finished.append((s, terminal))
        self.steps += 1
        for s, terminal in finished:
            self._detach(s, terminal)

    def _step(self):
        slots = list(self.active.keys())
        n = len(slots)
        toks = (ctypes.c_int * n)(*[self.active[s].next_tok for s in slots])
        sid = (ctypes.c_int * n)(*slots)
        pos = (ctypes.c_int * n)(*[self.active[s].pos for s in slots])
        out = (ctypes.c_int * n)()
        _t1 = time.perf_counter()
        rc = self._native("decode_step", self.L.qwn_paged_decode_step,
                          toks, sid, pos, n, out)
        if rc != 0:
            raise EngineFatal(f"paged decode step rc={rc} (N={n})")
        _t2 = time.perf_counter()
        if len(self.wavelog) < 65536:
            self.wavelog.append((_t2, (_t2 - _t1) * 1e3, 0, n, n, 1))
        self.last_progress_mono = _t2
        self.steps += 1; self.decoded_tokens += n
        finished = []
        _tnow = time.time()
        for j, s in enumerate(slots):
            req = self.active[s]
            o = out[j]
            req.out.append(o); req.pos += 1; req.next_tok = o; req.t_tok = _tnow
            req.progress.set()
            terminal = self._terminal(req, o)
            if terminal is not None:
                finished.append((s, terminal))
        for s, terminal in finished:
            self._detach(s, terminal)

    def _loop(self):
        while True:
            # The condition protects only queue admission and idle wakeups.
            # Never hold it across a prefill/decode/verify wave: request-handler
            # threads must be able to enqueue while the GPU is busy, otherwise a
            # nominally batched server silently becomes single-request serial.
            with self.cv:
                while self.running and not self.q and not self.active and not self.pref:
                    self.cv.wait(timeout=0.5)
                if not self.running and not self.active and not self.q and not self.pref:
                    return
                try:
                    self._admit()
                except EngineFatal as e:
                    self._fatal(f"during admission: {e}")
                except Exception as e:
                    # _admit handles failures after taking ownership. An
                    # exception here still belongs to the current queue head.
                    self.admission_failures += 1
                    self._metric("admission_failures")
                    print(f"[engine] admit error: {e}", flush=True)
                    if self.q:
                        r = self.q.pop(0)
                        self._finish(r, "failed", f"admission failed: {e}", 500,
                                     "server_error", "admission_failed")
                have = bool(self.active) or bool(self.pref)
                if not have and self.q:
                    # No live work can release capacity: waiting is a deadlock,
                    # not backpressure. Drop optional cache state and retry once.
                    self.admission_recoveries += 1
                    self._metric("admission_recoveries")
                    fb = self._free_blocks()
                    print(f"[engine] admission no-progress: queued={len(self.q)} "
                          f"free={fb}/{self.num_blocks}; flushing prefix cache", flush=True)
                    try:
                        self._ck_flush("admission-no-progress")
                        self._admit()
                    except EngineFatal as e:
                        self._fatal(f"during admission recovery: {e}")
                    except Exception as e:
                        self.admission_failures += 1
                        self._metric("admission_failures")
                        print(f"[engine] admission recovery error: {e}", flush=True)
                    have = bool(self.active) or bool(self.pref)
                    if not have and self.q:
                        r = self.q.pop(0)
                        self.admission_failures += 1
                        self._metric("admission_failures")
                        msg = (f"admission capacity invariant failed: "
                               f"prompt={r.n_prompt} free={self._cached_free_blocks} "
                               f"max_blocks={self.max_blocks_per_seq}")
                        self._finish(r, "failed", msg, 503,
                                     "server_overloaded", "capacity_unavailable")
                        print(f"[engine] {msg}", flush=True)
            _t = time.perf_counter()
            try:
                self._work()
                self.err_streak = 0
            except EngineFatal as e:
                # Native ownership or CUDA state is suspect. Continuing can
                # neither complete requests nor repair the context; let the
                # restart wrapper replace the process immediately.
                self._fatal(f"during execution: {e}")
            except Exception as e:
                # Pure scheduler faults are recoverable only if every owned
                # request can be failed and every slot can be reset. A cleanup
                # failure means native ownership is unknown.
                print(f"[engine] scheduler error: {e}", flush=True)
                self.last_engine_error = repr(e)
                try:
                    for s in list(self.active):
                        self._detach(
                            s, ("failed", f"execution failed: {e}", 500,
                                "server_error", "execution_failed"))
                    for s in list(self.pref):
                        self._abort_prefill(
                            s, ("failed", f"prefill failed: {e}", 500,
                                "server_error", "prefill_failed"))
                except BaseException as cleanup_exc:
                    self._fatal(
                        f"scheduler recovery after {e!r}: {cleanup_exc!r}")
            with self.cv:
                if not self.q and not self.active and not self.pref:
                    self.busy_since_mono = None
            self.t_loop += time.perf_counter() - _t

    def shutdown(self):
        # Close admission immediately, then wait for any in-flight admission to
        # leave the condition lock. A checkpoint promote/demote is synchronous
        # and may legitimately take more than a second; the shutdown watchdog
        # and production reaper bound a genuine native wedge.
        self.running = False
        with self.cv:
            # q/pref/active transitions may run outside cv while CUDA executes.
            # No request leaves _owned until _finish publishes its terminal
            # outcome, so the registry is the stable shutdown snapshot.
            with self._owned_lock:
                owned = list(self._owned.values())
            self.cv.notify_all()
        for req in owned:
            self.cancel(req, "server shutting down", 503, "failed",
                        "server_error", "server_shutdown")
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            self._fatal("engine thread did not stop within 5 seconds")
        self._native("shutdown_pool", self.L.qwn_paged_free)
        self._native("shutdown_model", self.L.qwn_free)


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
        _response_started = False
        _request_slots = threading.BoundedSemaphore(args.max_http_concurrency)


        def log_message(self, *a):
            pass

        def _json(self, code, obj, headers=None, close=False):
            body = json.dumps(obj).encode()
            self._response_started = True
            if close:
                self.close_connection = True
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            if close:
                self.send_header("Connection", "close")
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def _json_error(self, status, message, error_type="server_error",
                        code=None, param=None, headers=None, close=False):
            self._json(status, _error_body(message, error_type, code, param),
                       headers=headers, close=close)
        def _text(self, code, body, content_type):
            payload = body.encode()
            self._response_started = True
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _prometheus(self):
            m = eng.metrics_snapshot()
            terminal = (m.get("requests_completed", 0) +
                        m.get("requests_cancelled", 0) +
                        m.get("requests_failed", 0))
            with eng.phase_lock:
                phase_now = eng.phase
            phase = phase_now.replace("\\", "\\\\").replace('"', '\\"')
            rows = [
                "# HELP knivesysl_engine_up Whether the inference engine thread is alive.",
                "# TYPE knivesysl_engine_up gauge",
                f"knivesysl_engine_up {1 if eng.thread.is_alive() else 0}",
                "# HELP knivesysl_engine_busy Whether any request is queued or executing.",
                "# TYPE knivesysl_engine_busy gauge",
                f"knivesysl_engine_busy {1 if (eng.q or eng.active or eng.pref or phase_now != 'idle') else 0}",
                "# HELP knivesysl_engine_phase Current engine phase.",
                "# TYPE knivesysl_engine_phase gauge",
                f'knivesysl_engine_phase{{phase="{phase}"}} 1',
                "# TYPE knivesysl_process_start_time_seconds gauge",
                f"knivesysl_process_start_time_seconds {eng.started_unix:.6f}",
                "# HELP knivesysl_supervisor_restarts_total Process restarts by the production supervisor.",
                "# TYPE knivesysl_supervisor_restarts_total counter",
                f"knivesysl_supervisor_restarts_total {eng.supervisor_restarts}",
                "# TYPE knivesysl_supervisor_last_exit_code gauge",
                f"knivesysl_supervisor_last_exit_code {eng.supervisor_last_exit_code}",
                "# TYPE knivesysl_supervisor_last_exit_time_seconds gauge",
                f"knivesysl_supervisor_last_exit_time_seconds {eng.supervisor_last_exit_unix}",
                "# TYPE knivesysl_requests_total counter",
                f'knivesysl_requests_total{{state="submitted"}} {m.get("requests_total", 0)}',
                f'knivesysl_requests_total{{state="completed"}} {m.get("requests_completed", 0)}',
                f'knivesysl_requests_total{{state="cancelled"}} {m.get("requests_cancelled", 0)}',
                f'knivesysl_requests_total{{state="failed"}} {m.get("requests_failed", 0)}',
                f'knivesysl_requests_total{{state="rejected"}} {m.get("requests_rejected", 0)}',
                "# TYPE knivesysl_request_duration_seconds summary",
                f'knivesysl_request_duration_seconds_sum {m.get("request_duration_seconds", 0.0):.9g}',
                f"knivesysl_request_duration_seconds_count {terminal}",
                "# TYPE knivesysl_queue_duration_seconds summary",
                f'knivesysl_queue_duration_seconds_sum {m.get("queue_duration_seconds", 0.0):.9g}',
                f'knivesysl_queue_duration_seconds_count {m.get("queue_duration_count", 0)}',
                "# TYPE knivesysl_time_to_first_token_seconds summary",
                f'knivesysl_time_to_first_token_seconds_sum {m.get("time_to_first_token_seconds", 0.0):.9g}',
                f'knivesysl_time_to_first_token_seconds_count {m.get("time_to_first_token_count", 0)}',
                "# TYPE knivesysl_requests gauge",
                f'knivesysl_requests{{state="queued"}} {len(eng.q)}',
                f'knivesysl_requests{{state="prefilling"}} {len(eng.pref)}',
                f'knivesysl_requests{{state="decoding"}} {len(eng.active)}',
                "# TYPE knivesysl_kv_blocks gauge",
                f'knivesysl_kv_blocks{{state="free"}} {eng._cached_free_blocks}',
                f'knivesysl_kv_blocks{{state="total"}} {eng._cached_total_blocks}',
                "# TYPE knivesysl_tokens_total counter",
                f'knivesysl_tokens_total{{kind="prompt"}} {m.get("prompt_tokens", 0)}',
                f'knivesysl_tokens_total{{kind="generated"}} {m.get("generation_tokens", 0)}',
                "# TYPE knivesysl_scheduler_events_total counter",
                f'knivesysl_scheduler_events_total{{kind="queue_timeout"}} {m.get("queue_timeouts", 0)}',
                f'knivesysl_scheduler_events_total{{kind="request_timeout"}} {m.get("request_timeouts", 0)}',
                f'knivesysl_scheduler_events_total{{kind="admission_recovery"}} {m.get("admission_recoveries", 0)}',
                f'knivesysl_scheduler_events_total{{kind="admission_failure"}} {m.get("admission_failures", 0)}',
                f'knivesysl_scheduler_events_total{{kind="native_failure"}} {m.get("native_failures", 0)}',
                "# TYPE knivesysl_prefix_cache_events_total counter",
                f'knivesysl_prefix_cache_events_total{{kind="hit"}} {eng.pc_hits}',
                f'knivesysl_prefix_cache_events_total{{kind="miss"}} {eng.pc_misses}',
                f'knivesysl_prefix_cache_events_total{{kind="build"}} {eng.pc_builds}',
                f'knivesysl_prefix_cache_tokens_saved_total {eng.pc_saved}',
                "# TYPE knivesysl_speculative_tokens_total counter",
                f'knivesysl_speculative_tokens_total{{kind="drafted"}} {eng.spec_drafted}',
                f'knivesysl_speculative_tokens_total{{kind="committed"}} {eng.spec_committed}',
                f"knivesysl_speculative_rounds_total {eng.spec_rounds}",
            ]
            self._text(200, "\n".join(rows) + "\n",
                       "text/plain; version=0.0.4; charset=utf-8")


        def _peer_closed(self):
            """Non-consuming disconnect probe while no socket write is pending."""
            try:
                readable, _, _ = select.select([self.connection], [], [], 0)
                if not readable:
                    return False
                flags = socket.MSG_PEEK | getattr(socket, "MSG_DONTWAIT", 0)
                return self.connection.recv(1, flags) == b""
            except BlockingIOError:
                return False
            except (OSError, ValueError):
                return True

        def _require_api_key(self):
            if not args.api_key:
                return
            supplied = self.headers.get("Authorization", "")
            parts = supplied.split()
            valid = (len(parts) == 2 and parts[0].lower() == "bearer" and
                     hmac.compare_digest(parts[1].encode("utf-8"),
                                         args.api_key.encode("utf-8")))
            if not valid:
                raise RequestError(401, "invalid API key",
                                   "authentication_error", "invalid_api_key")

        def do_GET(self):
            path = self.path.partition("?")[0]
            if path.startswith("/v1/"):
                try:
                    self._require_api_key()
                except RequestError as e:
                    self._json_error(
                        e.status, e.message, e.error_type, e.code, e.param,
                        headers={"WWW-Authenticate": "Bearer"}
                        if e.status == 401 else None, close=True)
                    return
            if path == "/v1/models":
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
            elif path in ("/health", "/healthz", "/v1/healthz", "/livez", "/readyz"):
                # Do not call into CUDA from the HTTP watchdog. A wedged engine
                # stream must still yield an immediate 503 and a useful reason.
                fb, tb = eng._cached_free_blocks, eng._cached_total_blocks
                now = time.monotonic()
                alive = eng.thread.is_alive()
                with eng.phase_lock:
                    phase = eng.phase
                    phase_age_s = max(0.0, now - eng.phase_since_mono)
                    current_request_id = eng.current_request_id
                busy = bool(eng.q or eng.active or eng.pref or phase != "idle")
                checkpoints = list(eng.cks)
                spec_rounds_by_n = dict(eng.spec_rounds_by_n)
                bases = [x for x in (eng.last_progress_mono, eng.busy_since_mono)
                         if x is not None]
                progress_base = max(bases) if bases else eng.started_mono
                last_wave_age_s = max(0.0, now - progress_base)
                stalled = (not alive) or (
                    busy and last_wave_age_s > args.health_stall_seconds)
                stalled_reason = ("engine-thread-dead" if not alive else
                                  "no-forward-progress" if stalled else None)
                ready = alive and eng.running and not stalled
                unhealthy = (not alive) if path == "/livez" else (
                    not ready if path == "/readyz" else stalled)
                status = ("dead" if not alive else "live") if path == "/livez" else (
                    "ready" if ready else "not_ready") if path == "/readyz" else (
                    "stalled" if stalled else "ok")
                self._json(503 if unhealthy else 200,
                           {"status": status,
                            "ready": ready, "live": alive,
                            "stalled_reason": stalled_reason,
                            "free_blocks": fb, "total_blocks": tb,
                            "free_blocks_sample_age_s":
                                max(0.0, now - eng._cached_stats_mono),
                            "active": len(eng.active), "prefilling": len(eng.pref),
                            "queued": len(eng.q), "steps": eng.steps,
                            "decoded_tokens": eng.decoded_tokens,
                            "prefilled_tokens": eng.prefilled_tokens,
                            "engine_thread_alive": alive,
                            "last_wave_age_s": last_wave_age_s,
                            "last_engine_error": eng.last_engine_error,
                            "engine_phase": phase,
                            "engine_phase_age_s": phase_age_s,
                            "current_request_id": current_request_id,
                            "admission_recoveries": eng.admission_recoveries,
                            "admission_failures": eng.admission_failures,
                            "supervisor": {
                                "restarts": eng.supervisor_restarts,
                                "last_exit_code": eng.supervisor_last_exit_code,
                                "last_exit_time_seconds": eng.supervisor_last_exit_unix},
                            "prefix_cache": {"enabled": eng.pc_enabled,
                                             "prefix_tokens": sum(c["pos"] for c in checkpoints),
                                             "checkpoints": len(checkpoints),
                                             "hits": eng.pc_hits, "misses": eng.pc_misses,
                                             "builds": eng.pc_builds,
                                             "tokens_saved": eng.pc_saved},
                            "spec": {"enabled": eng.spec_on,
                                     "rounds": eng.spec_rounds,
                                     "committed": eng.spec_committed,
                                     "drafted": eng.spec_drafted,
                                     "rounds_by_n": spec_rounds_by_n,
                                     "tokens_per_round": (eng.spec_committed / eng.spec_rounds)
                                                         if eng.spec_rounds else 0.0}})
            elif path == "/metrics":
                self._prometheus()
            elif path == "/waveprof":
                # Never enter the native library from an HTTP thread. The
                # engine-thread ctypes interval is the authoritative call wall
                # time; kernel-only profiling belongs in nsys, off production.
                w = max(eng.prefill_waves, 1)
                other = max(eng.t_loop - eng.t_marshal - eng.t_engine, 0.0)
                self._json(200, {"waves": eng.prefill_waves,
                                 "engine_waves": eng.prefill_waves,
                                 "ms_per_wave": {
                                     "marshal": 1000.0 * eng.t_marshal / w,
                                     "engine_seen_by_python": 1000.0 * eng.t_engine / w,
                                     "engine_inside": None,
                                     "scheduler_other": 1000.0 * other / w,
                                     "loop_total": 1000.0 * eng.t_loop / w}})
            elif path == "/v1/wavelog":
                # Raw wave timeline. Host gap between consecutive engine calls is
                # rec[i].t_end - rec[i-1].t_end - rec[i].engine_ms.
                self._json(200, {"log": [list(r) for r in eng.wavelog]})
            elif path == "/v1/wavereset":
                eng.wavelog = []
                self._json(200, {"ok": True})
            else:
                self._json(404, {"error": "not found"}, close=True)

        def do_POST(self):
            self._response_started = False
            self._body_consumed = False
            self._current_req = None
            acquired = self._request_slots.acquire(blocking=False)
            if not acquired:
                self._json_error(429, "too many concurrent HTTP requests",
                                 "server_overloaded", "http_concurrency_limit",
                                 close=True)
                return
            try:
                self._require_api_key()
                self._do_POST()
            except RequestError as e:
                if not self._response_started:
                    self._json_error(
                        e.status, e.message, e.error_type, e.code, e.param,
                        headers={"WWW-Authenticate": "Bearer"}
                        if e.status == 401 else None,
                        close=not self._body_consumed)
            except (json.JSONDecodeError, UnicodeDecodeError,
                    ValueError, TypeError) as e:
                if not self._response_started:
                    self._json_error(400, f"invalid request: {e}",
                                     "invalid_request_error",
                                     "invalid_request",
                                     close=not self._body_consumed)
            except socket.timeout:
                if self._current_req is not None:
                    eng.cancel(self._current_req, "HTTP connection timed out",
                               408, "failed", "timeout_error",
                               "http_io_timeout")
                if not self._response_started:
                    self._json_error(408, "HTTP connection timed out",
                                     "timeout_error", "http_io_timeout",
                                     close=True)
            except (BrokenPipeError, ConnectionResetError):
                if self._current_req is not None:
                    eng.cancel(self._current_req, "client disconnected")
            except Exception:
                if self._current_req is not None:
                    eng.cancel(self._current_req, "request handler failed",
                               500, "failed", "server_error",
                               "handler_failed")
                traceback.print_exc()
                if not self._response_started:
                    self._json_error(500, "internal server error",
                                     "server_error", "internal_error",
                                     close=not self._body_consumed)
            finally:
                self._current_req = None
                self._request_slots.release()

        def _do_POST(self):
            path = self.path.partition("?")[0]
            if path not in ("/v1/chat/completions", "/v1/completions"):
                self._json(404, {"error": "not found"}, close=True); return
            transfer_encodings = self.headers.get_all("Transfer-Encoding") or []
            if transfer_encodings:
                raise RequestError(400, "Transfer-Encoding is not supported",
                                   "invalid_request_error",
                                   "unsupported_transfer_encoding",
                                   "Transfer-Encoding")
            content_lengths = self.headers.get_all("Content-Length") or []
            if not content_lengths:
                raise RequestError(411, "Content-Length is required",
                                   "invalid_request_error",
                                   "content_length_required")
            if len(content_lengths) != 1:
                raise RequestError(400, "exactly one Content-Length is required",
                                   "invalid_request_error",
                                   "ambiguous_content_length",
                                   "Content-Length")
            raw_length = content_lengths[0].strip()
            if not raw_length.isascii() or not raw_length.isdecimal():
                raise RequestError(400, "invalid Content-Length",
                                   "invalid_request_error",
                                   "invalid_content_length",
                                   "Content-Length")
            n = int(raw_length)
            if n > args.max_request_bytes:
                raise RequestError(413, "request body is too large",
                                      "invalid_request_error",
                                      "request_too_large")
            raw_body = self.rfile.read(n)
            self._body_consumed = len(raw_body) == n
            if not self._body_consumed:
                raise RequestError(400, "incomplete request body",
                                   "invalid_request_error",
                                   "incomplete_request_body")
            body = json.loads(raw_body or b"{}")
            if not isinstance(body, dict):
                raise RequestError(400, "request body must be a JSON object",
                                      "invalid_request_error", "invalid_json")
            model = body.get("model")
            aliases = {args.model_name, "ksl"}
            aliases.update(x.strip() for x in
                           os.environ.get("KSL_MODEL_ALIASES", "").split(",")
                           if x.strip())
            if not isinstance(model, str) or not model:
                raise RequestError(400, "model is required",
                                      "invalid_request_error",
                                      "model_required", "model")
            if model not in aliases:
                raise RequestError(404, f"model {model!r} does not exist",
                                      "invalid_request_error",
                                      "model_not_found", "model")
            stream_raw = body.get("stream", False)
            if not isinstance(stream_raw, bool):
                raise RequestError(400, "stream must be a boolean",
                                      "invalid_request_error",
                                      "invalid_stream", "stream")
            count = body.get("n", 1)
            if not isinstance(count, int) or isinstance(count, bool) or count != 1:
                raise RequestError(400, "only n=1 is supported",
                                      "invalid_request_error",
                                      "unsupported_parameter", "n")
            for key in ("logprobs", "prompt_logprobs", "top_logprobs"):
                if body.get(key) not in (None, False, 0):
                    raise RequestError(
                        400, f"{key} is not supported by this server",
                        "invalid_request_error", "unsupported_parameter", key)
            response_format = body.get("response_format")
            if response_format not in (None, {"type": "text"}):
                raise RequestError(
                    400, "structured response_format is not supported",
                    "invalid_request_error", "unsupported_parameter",
                    "response_format")
            for key in ("guided_json", "guided_regex", "guided_choice",
                        "guided_grammar", "guided_decoding_backend",
                        "structured_outputs"):
                if body.get(key) is not None:
                    raise RequestError(
                        400, f"{key} is not supported by this server",
                        "invalid_request_error", "unsupported_parameter", key)
            top_p = body.get("top_p", 1.0)
            if (not isinstance(top_p, (int, float)) or
                    isinstance(top_p, bool) or not math.isfinite(top_p) or
                    float(top_p) != 1.0):
                raise RequestError(
                    400, "top_p sampling is not supported; use top_p=1",
                    "invalid_request_error", "unsupported_parameter", "top_p")
            top_k = body.get("top_k", -1)
            if (not isinstance(top_k, int) or isinstance(top_k, bool) or
                    top_k not in (-1, 0)):
                raise RequestError(
                    400, "top_k sampling is not supported; use top_k=-1",
                    "invalid_request_error", "unsupported_parameter", "top_k")
            for key, neutral in (("presence_penalty", 0),
                                 ("frequency_penalty", 0),
                                 ("repetition_penalty", 1)):
                value = body.get(key, neutral)
                if (not isinstance(value, (int, float)) or
                        isinstance(value, bool) or not math.isfinite(value) or
                        float(value) != neutral):
                    raise RequestError(
                        400, f"{key} is not supported",
                        "invalid_request_error", "unsupported_parameter", key)
            if body.get("logit_bias"):
                raise RequestError(
                    400, "logit_bias is not supported",
                    "invalid_request_error", "unsupported_parameter",
                    "logit_bias")
            # OpenAI chat sends max_completion_tokens (guidellm's chat handler does);
            # /v1/completions sends max_tokens; Responses-style clients send
            # max_output_tokens. Reading only one silently halves the requested length.
            mt_raw = next((body[k] for k in (
                "max_tokens", "max_completion_tokens", "max_output_tokens")
                if body.get(k) is not None), None)
            stream = stream_raw
            is_chat = self.path.startswith("/v1/chat")
            tools = body.get("tools")
            if tools is not None and not isinstance(tools, list):
                raise RequestError(400, "tools must be an array",
                                      "invalid_request_error", "invalid_tools",
                                      "tools")
            if is_chat:
                msgs = body.get("messages", [])
                if not isinstance(msgs, list) or not msgs:
                    raise RequestError(400, "messages must be a non-empty array",
                                          "invalid_request_error",
                                          "invalid_messages", "messages")
                if any(not isinstance(m, dict) for m in msgs):
                    raise RequestError(400, "each message must be an object",
                                          "invalid_request_error",
                                          "invalid_messages", "messages")
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
                ctk = body.get("chat_template_kwargs")
                if ctk is not None and not isinstance(ctk, dict):
                    raise RequestError(400,
                                          "chat_template_kwargs must be an object",
                                          "invalid_request_error",
                                          "invalid_chat_template_kwargs",
                                          "chat_template_kwargs")
                if ctk is None:
                    ctk = {}
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
                prompt = body.get("prompt", "")
                if isinstance(prompt, str):
                    ids = tok(prompt, add_special_tokens=False).input_ids
                elif isinstance(prompt, list) and prompt:
                    vocab_size = len(tok)
                    if not all(isinstance(x, int) and not isinstance(x, bool)
                               and 0 <= x < vocab_size for x in prompt):
                        raise RequestError(
                            400, "prompt must be a string or a non-empty array of token IDs",
                            "invalid_request_error", "invalid_prompt", "prompt")
                    ids = list(prompt)
                else:
                    raise RequestError(
                        400, "prompt must be a string or a non-empty array of token IDs",
                        "invalid_request_error", "invalid_prompt", "prompt")
            ctx_max = int(os.environ.get("TQ_CTX", "262144"))
            win = ctx_max - len(ids) - 8
            # An explicit OpenAI generation limit is a hard contract. When the
            # field is absent, allow a useful agent response while staying
            # inside the model context window.
            ignore = body.get("ignore_eos", False)
            if not isinstance(ignore, bool):
                raise RequestError(400, "ignore_eos must be a boolean",
                                      "invalid_request_error",
                                      "invalid_ignore_eos", "ignore_eos")
            if mt_raw is not None:
                if not isinstance(mt_raw, int) or isinstance(mt_raw, bool):
                    raise RequestError(400, "max_tokens must be an integer",
                                          "invalid_request_error",
                                          "invalid_max_tokens", "max_tokens")
                max_new = mt_raw
                if max_new < 1:
                    raise RequestError(400, "max_tokens must be at least 1",
                                          "invalid_request_error",
                                          "invalid_max_tokens", "max_tokens")
                # Preserve the caller's exact cap; benchmarks and interactive
                # clients must observe the same API semantics.
            else:
                max_new = int(os.environ.get("TQ_DEF_OUT", "16384"))
            if win < 1:
                raise RequestError(
                    400, f"prompt is too long: {len(ids)} tokens for context {ctx_max}",
                    "invalid_request_error", "context_length_exceeded", "prompt")
            max_new = max(1, min(max_new, win))
            # stop strings, vLLM semantics: applied to the RAW generation (thinking
            # included), earliest match truncates and finishes with "stop".
            stop_raw = body.get("stop")
            if stop_raw is not None and not (
                    isinstance(stop_raw, str) or
                    (isinstance(stop_raw, list) and
                     all(isinstance(s0, str) for s0 in stop_raw))):
                raise RequestError(400, "stop must be a string or an array of strings",
                                      "invalid_request_error", "invalid_stop", "stop")
            stops = ([stop_raw] if isinstance(stop_raw, str)
                     else [s0 for s0 in (stop_raw or []) if s0])
            # ignore_eos: benchmark/eval harnesses (guidellm) pin the output length by
            # sending max_tokens + ignore_eos, so every request does equal work.
            req_eos = [] if ignore else eos
            stream_options = body.get("stream_options")
            if stream_options is not None and not isinstance(stream_options, dict):
                raise RequestError(400, "stream_options must be an object",
                                      "invalid_request_error",
                                      "invalid_stream_options", "stream_options")
            include_usage = (stream_options or {}).get("include_usage")
            if include_usage is not None and not isinstance(include_usage, bool):
                raise RequestError(400, "stream_options.include_usage must be a boolean",
                                   "invalid_request_error",
                                   "invalid_stream_options",
                                   "stream_options.include_usage")
            want_usage = bool(include_usage)
            # Sampling: an omitted temperature keeps the engine's greedy default
            # (agentic clients here want determinism + APC-friendly replays);
            # explicit temperature>0 samples engine-side with the spec-sampler
            # semantics (temp-scaled + TQ_MIN_P tail floor + replayable seed).
            # The native sampler implements temperature plus a server-wide min-p
            # floor. Unsupported per-request filters are rejected above rather
            # than silently producing different distributions.
            temp_raw = body.get("temperature", 0.0)
            if (not isinstance(temp_raw, (int, float)) or
                    isinstance(temp_raw, bool)):
                raise RequestError(400, "temperature must be a number",
                                      "invalid_request_error",
                                      "invalid_temperature", "temperature")
            temp = float(temp_raw)
            if not math.isfinite(temp) or temp < 0.0:
                raise RequestError(400, "temperature must be a finite non-negative number",
                                      "invalid_request_error",
                                      "invalid_temperature", "temperature")
            seed_raw = body.get("seed")
            if seed_raw is not None and (
                    not isinstance(seed_raw, int) or isinstance(seed_raw, bool)):
                raise RequestError(400, "seed must be an integer",
                                      "invalid_request_error",
                                      "invalid_seed", "seed")
            seed = (seed_raw if seed_raw is not None else
                    int.from_bytes(os.urandom(8), "little"))
            priority_raw = body.get("priority", 0)
            if (not isinstance(priority_raw, int) or
                    isinstance(priority_raw, bool)):
                raise RequestError(400, "priority must be an integer",
                                      "invalid_request_error",
                                      "invalid_priority", "priority")
            priority = priority_raw
            if abs(priority) > 1_000_000_000:
                raise RequestError(400, "priority is outside the supported range",
                                      "invalid_request_error",
                                      "invalid_priority", "priority")
            rid = self.headers.get("X-Request-ID")
            if rid is not None and (not rid.strip() or len(rid) > 256):
                raise RequestError(400, "X-Request-ID must contain 1 to 256 characters",
                                      "invalid_request_error",
                                      "invalid_request_id")
            request_timeout = (args.timeout if args.request_timeout is None
                               else args.request_timeout)
            req = eng.submit(list(ids), max_new, req_eos, temp, seed,
                             priority=priority, request_id=rid,
                             request_timeout=request_timeout,
                             queue_timeout=args.queue_timeout)
            self._current_req = req
            cid = req.id

            if stream:
                # TRUE token streaming: the engine sets req.progress as each token
                # commits, so deltas leave as they are produced (TTFT/ITL are real).
                self._response_started = True
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
                    sent_txt, sent_tok = "", 0
                    while True:
                        done = req.done.is_set()
                        if done and req.err:
                            _sse(_error_body(req.err, req.err_type,
                                            req.err_code))
                            self.wfile.write(b"data: [DONE]\n\n")
                            self.wfile.flush()
                            return
                        n_out = len(req.out)
                        # Process on new tokens AND on done-with-no-new-tokens: the
                        # last progress wake can land between the final token append
                        # and done.set(), leaving that pass's holdback (up to
                        # len(TOOL_OPEN)-1 chars for tool-sending clients) unflushed.
                        # The old `elif done: break` skipped the tail flush entirely
                        # and cut replies mid-word.
                        if n_out > sent_tok or done:
                            full = tok.decode(req.out[:n_out], skip_special_tokens=True)
                            for st0 in stops:
                                i2 = full.find(st0)
                                if i2 >= 0:
                                    full = full[:i2]
                                    if not stopped:
                                        stopped = True
                                        eng.cancel(req, None, 200, "completed",
                                                   None, None)
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
                            if done or stopped:
                                break
                        else:
                            if self._peer_closed():
                                eng.cancel(req, "client disconnected")
                                return
                            remaining = req.deadline_mono - time.monotonic()
                            if remaining <= 0:
                                if eng.cancel(req, "request deadline exceeded", 504,
                                              "failed", "timeout_error",
                                              "request_timeout"):
                                    _sse(_error_body(
                                        "request deadline exceeded", "timeout_error",
                                        "request_timeout"))
                                    self.wfile.write(b"data: [DONE]\n\n")
                                    self.wfile.flush()
                                    return
                                # Completion or another cancellation won the
                                # terminal lock. Re-read its outcome rather
                                # than publishing a contradictory timeout.
                                req.progress.wait(0.05)
                                req.progress.clear()
                                continue
                            req.progress.wait(min(0.05, remaining))
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
                    # Client walked away: stop burning a slot on a dead socket.
                    eng.cancel(req, "client disconnected")
                return

            if self._peer_closed():
                eng.cancel(req, "client disconnected")
                return

            while not req.done.is_set():
                if self._peer_closed():
                    eng.cancel(req, "client disconnected")
                    return
                remaining = req.deadline_mono - time.monotonic()
                if remaining <= 0:
                    eng.cancel(req, "request deadline exceeded", 504, "failed",
                               "timeout_error", "request_timeout")
                    req.done.wait(timeout=1.0)
                    break
                if stops and req.out:
                    partial = tok.decode(req.out, skip_special_tokens=True)
                    if any(st0 in partial for st0 in stops):
                        eng.cancel(req, None, 200, "completed", None, None)
                        req.done.wait(timeout=1.0)
                        break
                req.progress.wait(timeout=min(0.05, remaining))
                req.progress.clear()
            if req.err:
                self._json_error(req.err_status, req.err, req.err_type,
                                 req.err_code)
                return
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
    try:
        faulthandler.register(signal.SIGUSR1, all_threads=True)
    except Exception:
        pass
    startup_done = threading.Event()

    def _watch_startup():
        if startup_done.wait(args.startup_watchdog_seconds):
            return
        print(f"[engine] FATAL: startup made no progress for "
              f"{args.startup_watchdog_seconds:.1f}s", flush=True)
        faulthandler.dump_traceback(all_threads=True)
        os._exit(70)

    threading.Thread(target=_watch_startup, name="startup-watchdog",
                     daemon=True).start()
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
    try:
        eng = BatchedEngine(L, args.tqf, args.max_slots, args.num_blocks, args.page,
                            wave_cols=args.wave_cols, max_prefill=args.max_prefill,
                            fuse=args.fuse, fuse_ratio=args.fuse_ratio,
                            fuse_idle_ms=args.fuse_idle_ms,
                            decode_every=args.decode_every,
                            prefix_cache=args.prefix_cache,
                            prefix_cache_min=args.prefix_cache_min,
                            decode_min_rows=args.decode_min_rows,
                            decode_max_idle_ms=args.decode_max_idle_ms,
                            prefill_budget=args.prefill_budget,
                            max_queue=args.max_queue,
                            queue_timeout=args.queue_timeout)
    finally:
        startup_done.set()
    def _watch_engine():
        # ctypes releases the GIL around native calls. This thread therefore
        # remains able to kill a process whose engine loop is blocked in CUDA,
        # an allocator, or a Python deadlock while HTTP health still responds.
        interval = max(0.1, min(1.0, args.engine_watchdog_seconds / 4.0))
        while eng.running:
            time.sleep(interval)
            if not eng.running:
                return
            with eng.phase_lock:
                phase = eng.phase
                phase_age = time.monotonic() - eng.phase_since_mono
                current_request_id = eng.current_request_id
            if not (eng.q or eng.active or eng.pref or phase != "idle"):
                continue
            now = time.monotonic()
            bases = [x for x in (eng.last_progress_mono, eng.busy_since_mono)
                     if x is not None]
            progress_base = max(bases) if bases else eng.started_mono
            age = now - progress_base
            if age <= args.engine_watchdog_seconds:
                continue
            print(f"[engine] FATAL: no forward progress for {age:.1f}s; "
                  f"phase={phase} phase_age={phase_age:.1f}s "
                  f"request={current_request_id!r} queued={len(eng.q)} "
                  f"active={len(eng.active)} prefilling={len(eng.pref)}",
                  flush=True)
            faulthandler.dump_traceback(all_threads=True)
            os._exit(70)

    threading.Thread(target=_watch_engine, name="engine-watchdog", daemon=True).start()
    fb, tb, pg = (eng._cached_free_blocks, eng._cached_total_blocks, eng.page)
    print(f"batched server: pool blocks={tb} page={pg} max_slots={eng.max_slots} free={fb}", flush=True)
    # BaseHTTPServer's default backlog is 5: with N clients connecting at once the
    # kernel resets the surplus SYNs (ConnectionReset at the client) long before the
    # scheduler is the limit. This is a many-client server -- size it to the slots.
    class _Server(ThreadingHTTPServer):
        daemon_threads = True
        request_queue_size = max(128, 4 * args.max_slots)

        def __init__(self, *server_args, **server_kwargs):
            self._connection_slots = threading.BoundedSemaphore(
                args.max_http_concurrency)
            super().__init__(*server_args, **server_kwargs)

        def get_request(self):
            request, client_address = super().get_request()
            request.settimeout(args.http_io_timeout)
            return request, client_address

        def process_request(self, request, client_address):
            if not self._connection_slots.acquire(blocking=False):
                try:
                    body = json.dumps(_error_body(
                        "too many concurrent HTTP connections",
                        "server_overloaded", "http_concurrency_limit")).encode()
                    request.sendall(
                        b"HTTP/1.1 429 Too Many Requests\r\n"
                        b"Content-Type: application/json\r\n"
                        b"Connection: close\r\n"
                        + f"Content-Length: {len(body)}\r\n\r\n".encode()
                        + body)
                except OSError:
                    pass
                finally:
                    self.shutdown_request(request)
                eng._metric("requests_rejected")
                return
            try:
                super().process_request(request, client_address)
            except BaseException:
                self._connection_slots.release()
                raise

        def process_request_thread(self, request, client_address):
            try:
                super().process_request_thread(request, client_address)
            finally:
                self._connection_slots.release()
    httpd = _Server((args.host, args.port), make_handler(eng, tok, args))
    print(f"listening on {args.host}:{args.port}  (model={args.model_name})", flush=True)
    def _graceful_signal(_signum, _frame):
        raise KeyboardInterrupt

    old_term = signal.signal(signal.SIGTERM, _graceful_signal)
    old_hup = signal.signal(signal.SIGHUP, _graceful_signal)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGHUP, old_hup)
        shutdown_done = threading.Event()
        def _watch_shutdown():
            if shutdown_done.wait(args.engine_watchdog_seconds):
                return
            print("[Engine] FATAL: shutdown made no progress", flush=True)
            faulthandler.dump_traceback(all_threads=True)
            os._exit(70)
        threading.Thread(target=_watch_shutdown, name="shutdown-watchdog",
                         daemon=True).start()
        try:
            httpd.server_close()
            eng.shutdown()
        finally:
            shutdown_done.set()


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
    ap.add_argument("--request-timeout", type=float, default=None,
                    help="generation deadline from enqueue; defaults to --timeout")
    ap.add_argument("--queue-timeout", type=float, default=300.0,
                    help="maximum seconds a request may wait before admission")
    ap.add_argument("--max-queue", type=int, default=128,
                    help="maximum queued requests before HTTP 429")
    ap.add_argument("--max-request-bytes", type=int, default=16 * 1024 * 1024,
                    help="maximum JSON request body bytes before HTTP 413")
    ap.add_argument("--max-http-concurrency", type=int, default=256,
                    help="maximum simultaneous HTTP request handlers")
    ap.add_argument("--http-io-timeout", type=float, default=30.0,
                    help="socket read/write timeout, including request headers and bodies")
    ap.add_argument("--api-key", default=os.environ.get("KSL_API_KEY"),
                    help="optional bearer token; defaults to KSL_API_KEY")
    ap.add_argument("--health-stall-seconds", type=float,
                    default=float(os.environ.get("TQ_HEALTH_STALL_S", "60")),
                    help="return 503 when queued/active work makes no wave progress for "
                         "this many seconds; the check never enters CUDA")
    ap.add_argument("--startup-watchdog-seconds", type=float,
                    default=float(os.environ.get("TQ_STARTUP_WATCHDOG_S", "120")),
                    help="exit for supervisor restart if model/tokenizer initialization does "
                         "not finish within this many seconds")
    ap.add_argument("--engine-watchdog-seconds", type=float,
                    default=float(os.environ.get("TQ_ENGINE_WATCHDOG_S", "120")),
                    help="exit for supervisor restart after this many seconds of queued/active "
                         "work without a completed engine wave")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--bench-ns", default="1,2,4,8,16,32")
    ap.add_argument("--bench-p", type=int, default=48)
    ap.add_argument("--bench-iters", type=int, default=64)
    ap.add_argument("--clients", type=int, default=8)
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--stagger", type=float, default=0.15)
    args = ap.parse_args()
    positive_floats = {
        "--timeout": args.timeout,
        "--queue-timeout": args.queue_timeout,
        "--http-io-timeout": args.http_io_timeout,
        "--health-stall-seconds": args.health_stall_seconds,
        "--startup-watchdog-seconds": args.startup_watchdog_seconds,
        "--engine-watchdog-seconds": args.engine_watchdog_seconds,
    }
    if args.request_timeout is not None:
        positive_floats["--request-timeout"] = args.request_timeout
    for option, value in positive_floats.items():
        if not math.isfinite(value) or value <= 0:
            ap.error(f"{option} must be a finite number greater than 0")
    nonnegative_floats = {
        "--fuse-ratio": args.fuse_ratio,
        "--fuse-idle-ms": args.fuse_idle_ms,
        "--decode-max-idle-ms": args.decode_max_idle_ms,
    }
    for option, value in nonnegative_floats.items():
        if not math.isfinite(value) or value < 0:
            ap.error(f"{option} must be a finite non-negative number")
    if args.page < 1 or args.page & (args.page - 1):
        ap.error("--page must be a positive power of two")
    for option, value in {
            "--max-slots": args.max_slots,
            "--num-blocks": args.num_blocks,
            "--max-prefill": args.max_prefill,
            "--decode-min-rows": args.decode_min_rows,
            "--prefix-cache-min": args.prefix_cache_min,
    }.items():
        if value < 1:
            ap.error(f"{option} must be at least 1")
    if args.wave_cols is not None and args.wave_cols < 1:
        ap.error("--wave-cols must be at least 1")
    if args.prefill_budget is not None and args.prefill_budget < 1:
        ap.error("--prefill-budget must be at least 1")
    if args.decode_every < 0:
        ap.error("--decode-every must be non-negative")
    if not 1 <= args.port <= 65535:
        ap.error("--port must be between 1 and 65535")
    if args.max_queue < 1:
        ap.error("--max-queue must be at least 1")
    if args.max_http_concurrency < 1:
        ap.error("--max-http-concurrency must be at least 1")
    if args.max_request_bytes < 2:
        ap.error("--max-request-bytes must be at least 2")
    if args.bench:
        bench(args)
    elif args.selftest:
        selftest(args)
    else:
        serve(args)


if __name__ == "__main__":
    main()
