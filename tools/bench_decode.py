#!/usr/bin/env python3
"""Paged continuous-batch decode benchmark: ms/step and tok/s across concurrency
and context depth. Prefills N slots to depth P with ragged batched prefill waves,
then times `steps` paged decode steps after three untimed warmup steps.

Pool sizing matters: per-slot DeltaNet state is ~145 MiB (48 linear layers x a
[48,128,128] fp32 recurrent matrix + conv) and one page=128 block is ~3.44 MiB
(16 full-attention layers of Q4 K + E4M3 V), so both are sized to the case.

Run:
    CUDA_VISIBLE_DEVICES=0 TQ_KV_Q4=1 TQ_EMBED_FP8=2 TQ_CTX=262144 \\
        TQ_W_NVFP4=all python3 tools/bench_decode.py --corpus /tmp/corpus.txt \\
        --cases 1:2048,4:2048 --repeats 3 --teacher-force --json-out /tmp/decode.json

Emits:  DEC N=<n> P=<p> ms_step=<f> tok_s=<f> pf_tok_s=<f>
Native-call latency is synchronous host wall time (including ctypes/GIL), not
CUDA-event kernel time. JSON output IDs are step-major, with slots in ascending
order; seed and warmup outputs are recorded separately from timed outputs.
"""
from __future__ import annotations
import argparse, ctypes, glob, hashlib, json, math, os, time
import nvtx

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TQ_WIDE_PREFILL", "1")
os.environ.setdefault("TQ_WIDE_ATTN_MMA", "1")
from transformers import AutoTokenizer  # noqa: E402

WARMUP_STEPS = 3
INT_MAX = (1 << 31) - 1


def _first(*globs):
    for g in globs:
        hits = sorted(glob.glob(os.path.expanduser(g)))
        if hits:
            return hits[0]
    return ""


def ck(r, what):
    if r != 0:
        raise RuntimeError(f"{what} failed: {r}")
    return r


def latency_ms(samples_ns):
    ordered = sorted(v / 1e6 for v in samples_ns)

    def percentile(q):
        at = (len(ordered) - 1) * q
        lo, hi = math.floor(at), math.ceil(at)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (at - lo)

    return {"median": percentile(0.5), "p95": percentile(0.95),
            "p99": percentile(0.99), "max": ordered[-1]}


def parse_case(value):
    n, p = value.split(":")
    n, p = int(n), int(p)
    if not 1 <= n <= INT_MAX or not 1 <= p <= INT_MAX:
        raise ValueError("concurrency and context must be positive C integers")
    return n, p


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tqf", default=os.environ.get("TQ_MODEL_TQF") or
                    _first(HERE + "/*.tqf", "~/models/knivesysl/*.tqf"))
    ap.add_argument("--model-dir", default=os.environ.get("TQ_MODEL_DIR") or
                    _first("~/models/knivesysl"))
    ap.add_argument("--lib", default=os.environ.get("TQ_LIB") or
                    HERE + "/build-qwen/libforward_qwen.so")
    ap.add_argument("--page", type=int, default=128)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cases", default="1:2048,4:2048,8:2048,16:2048,32:2048",
                    help="comma-separated <concurrency>:<context> cases")
    ap.add_argument("--wave", type=int, default=256, help="prefill columns per wave")
    ap.add_argument("--corpus", default=HERE + "/src/forward_qwen.cu",
                    help="UTF-8 corpus (default: src/forward_qwen.cu, as before)")
    ap.add_argument("--json-out", help="write machine-readable results, including failures")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--teacher-force", action="store_true",
                    help="after the first generated seed, feed corpus tokens at each absolute position")
    ap.add_argument("--strict-teacher-force", action="store_true",
                    help="feed corpus at EVERY decode input, including the first seed position; overrides --teacher-force")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--nvtx", action="store_true", help="bracket selected timed decode as NVTX range 'decode'")
    ap.add_argument("--capture-case", help="N:P case to capture with --profile/--nvtx (default: first case)")
    ap.add_argument("--capture-repeat", type=int, default=1, help="one-based repeat to capture (default: 1)")
    ap.add_argument("--slots", type=int, default=0, help="force pool slots (0 = size to the case)")
    ap.add_argument("--blocks", type=int, default=0, help="force pool blocks (0 = size to the case)")
    args = ap.parse_args()
    try:
        cases = [parse_case(c) for c in args.cases.split(",")]
        capture_case = parse_case(args.capture_case) if args.capture_case else cases[0]
    except ValueError as exc:
        ap.error(f"invalid case; expected positive N:P pairs: {exc}")
    if len(set(cases)) != len(cases):
        ap.error("duplicate cases are not allowed; use --repeats")
    for name in ("page", "steps", "wave", "repeats"):
        if not 1 <= getattr(args, name) <= INT_MAX:
            ap.error(f"--{name} must be a positive C integer")
    if args.page & (args.page - 1):
        ap.error("--page must be a power of two")
    if not 0 <= args.slots <= INT_MAX or not 0 <= args.blocks <= INT_MAX:
        ap.error("--slots and --blocks must be nonnegative C integers")
    if capture_case not in cases or not 1 <= args.capture_repeat <= args.repeats:
        ap.error("capture case/repeat must belong to the requested matrix")
    for n, p in cases:
        required = n * ((p + WARMUP_STEPS + args.steps + args.page - 1) // args.page)
        if args.slots and args.slots < n:
            ap.error(f"case {n}:{p} requires at least {n} slots")
        if args.blocks and args.blocks < required:
            ap.error(f"case {n}:{p} requires at least {required} blocks including warmup and decode")
    return args, cases, capture_case


def bind_library(path):
    lib = ctypes.CDLL(path)
    lib.qwn_init.argtypes = [ctypes.c_char_p]; lib.qwn_init.restype = ctypes.c_int
    lib.qwn_paged_init.argtypes = [ctypes.c_int] * 3; lib.qwn_paged_init.restype = ctypes.c_int
    lib.qwn_paged_free.argtypes = []; lib.qwn_paged_free.restype = ctypes.c_int
    lib.qwn_paged_reset_slot.argtypes = [ctypes.c_int]; lib.qwn_paged_reset_slot.restype = ctypes.c_int
    lib.qwn_paged_decode_step.argtypes = [ctypes.POINTER(ctypes.c_int)] * 3 + [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    lib.qwn_paged_decode_step.restype = ctypes.c_int
    lib.qwn_paged_prefill_batch.argtypes = [ctypes.POINTER(ctypes.c_int)] * 7 + [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    lib.qwn_paged_prefill_batch.restype = ctypes.c_int
    lib.qwn_paged_stats.argtypes = [ctypes.POINTER(ctypes.c_int)] * 4; lib.qwn_paged_stats.restype = ctypes.c_int
    for name in ("qwn_max_seq", "qwn_wave_cap", "qwn_vocab_size"):
        fn = getattr(lib, name)
        fn.argtypes = []; fn.restype = ctypes.c_int
    lib.qwn_free.argtypes = []; lib.qwn_free.restype = None
    return lib


def pool_stats(lib):
    fb, tb, pg, mb = (ctypes.c_int() for _ in range(4))
    ck(lib.qwn_paged_stats(ctypes.byref(fb), ctypes.byref(tb), ctypes.byref(pg), ctypes.byref(mb)), "paged_stats")
    return {"free_blocks": fb.value, "total_blocks": tb.value, "page": pg.value,
            "max_blocks_per_sequence": mb.value, "used_blocks": tb.value - fb.value}


def memory_snapshot(rt, snapshots, stage, n=None, p=None, repeat=None):
    free, total = ctypes.c_size_t(), ctypes.c_size_t()
    ck(rt.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)), f"cudaMemGetInfo@{stage}")
    used = total.value - free.value
    baseline = snapshots[0]["device_used_bytes"] if snapshots else used
    snapshot = {"stage": stage, "N": n, "P": p, "repeat": repeat,
                "free_bytes": free.value, "total_bytes": total.value,
                "device_used_bytes": used, "delta_from_before_model_init_bytes": used - baseline}
    snapshots.append(snapshot)
    return snapshot


def check_tokens(rows, vocab, what):
    if any(token < 0 or token >= vocab for row in rows for token in row):
        raise RuntimeError(f"{what}: generated token outside model vocabulary")


def run_repeat(lib, args, ids, n, p, vocab, rt, capture, snapshots, repeat):
    # Reset and all host buffer allocation happen outside the timed decode region.
    for slot in range(n):
        ck(lib.qwn_paged_reset_slot(slot), f"reset_slot@{slot}")
    per = max(1, args.wave // n)
    columns = n * min(per, p)
    cT = lambda values: (ctypes.c_int * len(values))(*values)
    toks, cslot, cpos = ((ctypes.c_int * columns)() for _ in range(3))
    sslot = cT(range(n))
    soff, slen, sfin = ((ctypes.c_int * n)() for _ in range(3))
    seed = (ctypes.c_int * n)()
    # Preserve the original corpus cycle (which excludes the final token).
    cycle = len(ids) - 1
    pf_native_ns = []
    pf_start = time.perf_counter_ns()
    for w in range(0, p, per):
        cn = min(per, p - w)
        last = w + cn == p
        for slot in range(n):
            off = slot * cn
            soff[slot], slen[slot], sfin[slot] = off, cn, int(last)
            for column in range(cn):
                i = off + column
                toks[i] = ids[(slot * 7919 + w + column) % cycle]
                cslot[i], cpos[i] = slot, w + column
        start = time.perf_counter_ns()
        rc = lib.qwn_paged_prefill_batch(toks, cslot, cpos, sslot, soff, slen, sfin, n, n * cn, seed)
        pf_native_ns.append(time.perf_counter_ns() - start)
        ck(rc, f"prefill_batch@{w}")
    pf_s = (time.perf_counter_ns() - pf_start) / 1e9
    check_tokens([seed], vocab, "prefill")
    pool_prefill = pool_stats(lib)
    memory_snapshot(rt, snapshots, "after_prefill", n, p, repeat)
    total_steps = WARMUP_STEPS + args.steps
    outputs = [(ctypes.c_int * n)() for _ in range(total_steps)]
    # Native calls write directly to their final output buffers. Autoregression
    # reuses the preceding buffer; teacher forcing uses precomputed input buffers.
    if args.strict_teacher_force:
        inputs = [cT([ids[(slot * 7919 + p + step) % cycle] for slot in range(n)])
                  for step in range(total_steps)]
    elif args.teacher_force:
        inputs = [seed] + [cT([ids[(slot * 7919 + p + step) % cycle] for slot in range(n)])
                           for step in range(1, total_steps)]
    else:
        inputs = [seed] + outputs[:-1]
    positions = cT([p] * n)
    native_ns = [0] * args.steps
    step_ns = [0] * args.steps
    warm_native_ns = [0] * WARMUP_STEPS
    decode = lib.qwn_paged_decode_step
    clock = time.perf_counter_ns
    for step in range(WARMUP_STEPS):
        start = clock()
        rc = decode(inputs[step], sslot, positions, n, outputs[step])
        warm_native_ns[step] = clock() - start
        ck(rc, "warm")
        for slot in range(n):
            positions[slot] += 1
    check_tokens(outputs[:WARMUP_STEPS], vocab, "warmup")
    memory_snapshot(rt, snapshots, "after_warmup", n, p, repeat)
    profiling = capture and args.profile
    marking = capture and args.nvtx
    if profiling:
        ck(rt.cudaProfilerStart(), "cudaProfilerStart")
    try:
        if marking:
            nvtx.push_range("decode")
        try:
            start_all = clock()
            for i in range(args.steps):
                step = WARMUP_STEPS + i
                start = clock()
                rc = decode(inputs[step], sslot, positions, n, outputs[step])
                native_ns[i] = clock() - start
                ck(rc, "step")
                for slot in range(n):
                    positions[slot] += 1
                step_ns[i] = clock() - start
            decode_s = (clock() - start_all) / 1e9
        finally:
            if marking:
                nvtx.pop_range()
    finally:
        if profiling:
            ck(rt.cudaProfilerStop(), "cudaProfilerStop")
    check_tokens(outputs, vocab, "decode")
    memory_snapshot(rt, snapshots, "after_decode", n, p, repeat)
    return {"prefill_s": pf_s, "prefill_tok_s": n * p / pf_s,
            "prefill_native_call_ms": latency_ms(pf_native_ns),
            "warmup_native_call_ms": latency_ms(warm_native_ns),
            "decode_s": decode_s, "ms_step": decode_s * 1e3 / args.steps,
            "step_latency_ms": latency_ms(step_ns),
            "native_call_latency_ms": latency_ms(native_ns),
            "step_ms": [v / 1e6 for v in step_ns],
            "native_call_ms": [v / 1e6 for v in native_ns],
            "tok_s": n * args.steps / decode_s,
            "per_request_tok_s": args.steps / decode_s,
            "seed_token_ids": list(seed),
            "first_decode_input_token_ids": list(inputs[0]),
            "warmup_output_token_ids": [list(row) for row in outputs[:WARMUP_STEPS]],
            "output_token_ids": [list(row) for row in outputs[WARMUP_STEPS:]],
            "pool_after_prefill": pool_prefill, "pool_after_decode": pool_stats(lib),
            "captured": bool(profiling or marking)}


def main():
    args, cases, capture_case = parse_args()
    result = {"schema_version": 1, "status": "running", "arguments": vars(args),
              "environment": {k: v for k, v in sorted(os.environ.items())
                              if k.startswith("TQ_") or k in ("CUDA_VISIBLE_DEVICES", "CUDA_MODULE_LOADING")},
              "warmup_steps": WARMUP_STEPS, "corpus_slot_stride": 7919,
              "corpus_cycle": "token_count - 1 (legacy)",
              "decode_input_mode": ("strict_teacher_force" if args.strict_teacher_force else
                                    "teacher_force_after_seed" if args.teacher_force else "autoregressive"),
              "latency_clock": "perf_counter_ns; synchronous host call including ctypes/GIL",
              "percentile_method": "linear interpolation at (count - 1) * quantile",
              "output_layout": "step-major, ascending slot IDs",
              "cases": []}
    snapshots = []
    result["cuda_memory"] = {
        "scope": "cudaMemGetInfo device-wide usage, not process-exclusive allocation accounting",
        "sampling": "stage boundaries outside timing; maxima are sampled, NOT continuous peaks",
        "baseline": "before qwn_init, after CUDA runtime context initialization; deltas include other device users",
        "snapshots": snapshots}
    lib = None
    initialized = False
    try:
        with open(args.corpus, "rb") as f:
            corpus = f.read()
        tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
        ids = tok(corpus.decode("utf-8"), add_special_tokens=False).input_ids
        if len(ids) < 2:
            raise ValueError("corpus must tokenize to at least two tokens")
        result["corpus"] = {"path": os.path.abspath(args.corpus),
                            "sha256": hashlib.sha256(corpus).hexdigest(), "token_count": len(ids),
                            "token_ids_sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()}
        del corpus
        lib = bind_library(args.lib)
        rt = ctypes.CDLL("libcudart.so")
        rt.cudaMemGetInfo.argtypes = [ctypes.POINTER(ctypes.c_size_t)] * 2
        rt.cudaMemGetInfo.restype = ctypes.c_int
        for name in ("cudaProfilerStart", "cudaProfilerStop"):
            fn = getattr(rt, name)
            fn.argtypes = []; fn.restype = ctypes.c_int
        memory_snapshot(rt, snapshots, "before_model_init")
        ck(lib.qwn_init(args.tqf.encode()), "init")
        initialized = True
        memory_snapshot(rt, snapshots, "after_model_init")
        max_seq, wave_cap, vocab = lib.qwn_max_seq(), lib.qwn_wave_cap(), lib.qwn_vocab_size()
        result["model_limits"] = {"max_seq": max_seq, "wave_cap": wave_cap, "vocab_size": vocab}
        check_tokens([ids], vocab, "corpus")
        # Validate the complete matrix before allocating a pool or executing a case.
        geometries = []
        for n, p in cases:
            required = n * ((p + WARMUP_STEPS + args.steps + args.page - 1) // args.page)
            slots = args.slots or n
            blocks = args.blocks or max(required, n * ((p + args.page - 1) // args.page + 2))
            if p + WARMUP_STEPS + args.steps > max_seq:
                raise ValueError(f"case {n}:{p} exceeds model context {max_seq}, including warmup/decode")
            actual_wave = n * min(max(1, args.wave // n), p)
            if actual_wave > wave_cap:
                raise ValueError(f"case {n}:{p} wave has {actual_wave} columns, engine cap is {wave_cap}")
            if blocks > INT_MAX:
                raise ValueError(f"case {n}:{p} block count exceeds C integer capacity")
            geometries.append((slots, blocks, required, actual_wave))
        print(f"{'N':>4s} {'P':>7s} {'prefill s':>10s} {'pf tok/s':>9s} {'ms/step':>9s} {'tok/s':>9s} {'blocks':>11s}")
        for (n, p), (slots, blocks, required, actual_wave) in zip(cases, geometries):
            case = {"N": n, "P": p, "status": "running", "pool_slots": slots,
                    "pool_blocks": blocks, "page": args.page, "required_blocks": required,
                    "prefill_wave_columns": actual_wave, "repeats": []}
            result["cases"].append(case)
            try:
                for repeat in range(1, args.repeats + 1):
                    # Recreate the same pool (including its block free-list order and
                    # graph caches) so repeat geometry does not depend on prior use.
                    snapshot_start = len(snapshots)
                    ck(lib.qwn_paged_init(slots, blocks, args.page), f"paged_init {n}:{p} repeat={repeat}")
                    try:
                        memory_snapshot(rt, snapshots, "after_pool_init", n, p, repeat)
                        record = run_repeat(lib, args, ids, n, p, vocab, rt,
                                            (n, p) == capture_case and repeat == args.capture_repeat,
                                            snapshots, repeat)
                        record["repeat"] = repeat
                        record["cuda_memory_snapshots"] = snapshots[snapshot_start:]
                        record["max_sampled_device_used_bytes"] = max(
                            s["device_used_bytes"] for s in record["cuda_memory_snapshots"])
                        record["max_sampled_delta_from_before_model_init_bytes"] = max(
                            s["delta_from_before_model_init_bytes"] for s in record["cuda_memory_snapshots"])
                        case["repeats"].append(record)
                    finally:
                        ck(lib.qwn_paged_free(), "paged_free")
                    pool = record["pool_after_decode"]
                    print(f"{n:4d} {p:7d} {record['prefill_s']:10.2f} {record['prefill_tok_s']:9.0f} "
                          f"{record['ms_step']:9.2f} {record['tok_s']:9.1f} "
                          f"{pool['used_blocks']:5d}/{pool['total_blocks']:5d}", flush=True)
                    print(f"DEC N={n} P={p} ms_step={record['ms_step']:.4f} tok_s={record['tok_s']:.2f} "
                          f"pf_tok_s={record['prefill_tok_s']:.1f}", flush=True)
                seconds = sum(r["decode_s"] for r in case["repeats"])
                case["aggregate"] = {"decode_s": seconds, "tokens": n * args.steps * args.repeats,
                                     "tok_s": n * args.steps * args.repeats / seconds,
                                     "per_request_tok_s": args.steps * args.repeats / seconds}
                case["status"] = "ok"
            except Exception as exc:
                case["status"], case["error"] = "error", str(exc)
                raise
        result["status"] = "ok"
    except Exception as exc:
        result["status"], result["error"] = "error", str(exc)
        raise
    finally:
        if snapshots:
            result["cuda_memory"]["max_sampled_device_used_bytes"] = max(s["device_used_bytes"] for s in snapshots)
            result["cuda_memory"]["max_sampled_delta_from_before_model_init_bytes"] = max(
                s["delta_from_before_model_init_bytes"] for s in snapshots)
        if initialized:
            lib.qwn_free()
        if args.json_out:
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, allow_nan=False)
                f.write("\n")


if __name__ == "__main__":
    main()
