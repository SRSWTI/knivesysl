#!/usr/bin/env python3
"""Fixed-corpus likelihood and raw-logit comparison, not a task-quality benchmark.

Uses the real paged model via libbench_paged_logits.so. Every decode input,
including position P, is a corpus token; generated seeds never feed back.
Save a reference with --save-logits reference.npy, then compare another model
or environment with --reference-logits reference.npy. The adjacent .npy.json
sidecar binds the payload to corpus tokens, positions, shape, and baseline env.
CPU NumPy performs streamed float64 statistics; no PyTorch GPU allocations.
"""
from __future__ import annotations
import argparse, ctypes, hashlib, json, math, os, time
import numpy as np
import bench_decode as bd

SLOT_STRIDE = 7919


def json_write(path, value):
    temporary = str(path) + ".tmp"
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
    os.replace(temporary, path)


def hash_json(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lib", default=os.environ.get("TQ_LIB") or bd.HERE + "/build-qwen/libbench_paged_logits.so")
    ap.add_argument("--tqf", default=os.environ.get("TQ_MODEL_TQF") or
                    bd._first(bd.HERE + "/*.tqf", "~/models/knivesysl/*.tqf"))
    ap.add_argument("--model-dir", default=os.environ.get("TQ_MODEL_DIR") or bd._first("~/models/knivesysl"))
    ap.add_argument("--corpus", required=True, help="fixed UTF-8 corpus")
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--slots", type=int, default=0)
    ap.add_argument("--blocks", type=int, default=0)
    ap.add_argument("--page", type=int, default=128)
    ap.add_argument("--wave", type=int, default=256)
    ap.add_argument("--steps", type=int, default=30)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--save-logits", help="create float32 .npy reference and .npy.json sidecar")
    mode.add_argument("--reference-logits", help="compare against validated .npy reference and sidecar")
    ap.add_argument("--json-out", help="write metrics and failures as JSON")
    args = ap.parse_args()
    for name in ("context", "concurrency", "page", "wave", "steps"):
        if not 1 <= getattr(args, name) <= bd.INT_MAX:
            ap.error(f"--{name} must be a positive C integer")
    if args.page & (args.page - 1):
        ap.error("--page must be a power of two")
    if not 0 <= args.slots <= bd.INT_MAX or not 0 <= args.blocks <= bd.INT_MAX:
        ap.error("--slots and --blocks must be nonnegative C integers")
    if args.slots and args.slots < args.concurrency:
        ap.error("--slots cannot be smaller than --concurrency")
    required = args.concurrency * ((args.context + args.steps + args.page - 1) // args.page)
    if args.blocks and args.blocks < required:
        ap.error(f"--blocks must be at least {required}")
    path = args.save_logits or args.reference_logits
    if not path.endswith(".npy"):
        ap.error("logits path must end in .npy")
    if args.save_logits and (os.path.exists(path) or os.path.exists(path + ".json")):
        ap.error("reference output already exists; choose a new path")
    if args.json_out:
        outputs = {os.path.realpath(path), os.path.realpath(path + ".json"), os.path.realpath(args.corpus)}
        if os.path.realpath(args.json_out) in outputs:
            ap.error("--json-out must not overwrite logits, sidecar, or corpus")
    if os.path.realpath(path) == os.path.realpath(args.corpus):
        ap.error("logits path must not be the corpus")
    return args


def summarize(rows, compared):
    count = len(rows)
    result = {"requests": count, "candidate_mean_nll": math.fsum(r["candidate_nll"] for r in rows) / count,
              "candidate_max_nll": max(r["candidate_nll"] for r in rows)}
    if compared:
        elements = sum(r["elements"] for r in rows)
        squared_error = math.fsum(r["squared_error_sum"] for r in rows)
        reference_squared = math.fsum(r["reference_squared_sum"] for r in rows)
        if reference_squared == 0 and squared_error != 0:
            raise ValueError("relative L2 is undefined: reference logit norm is zero")
        result.update({
            "reference_mean_nll": math.fsum(r["reference_nll"] for r in rows) / count,
            "mean_nll_delta": math.fsum(r["nll_delta"] for r in rows) / count,
            "mean_kl_ref_to_candidate": math.fsum(r["kl_ref_to_candidate"] for r in rows) / count,
            "max_kl_ref_to_candidate": max(r["kl_ref_to_candidate"] for r in rows),
            "mean_abs_logit_error": math.fsum(r["absolute_error_sum"] for r in rows) / elements,
            "max_abs_logit_error": max(r["max_abs_logit_error"] for r in rows),
            "rms_logit_error": math.sqrt(squared_error / elements),
            "reference_rms_logits": math.sqrt(reference_squared / elements),
            "relative_l2_logit_error": math.sqrt(squared_error / reference_squared) if reference_squared else 0.0,
            "argmax_agreement": sum(r["argmax_agrees"] for r in rows) / count})
    if not all(math.isfinite(v) for v in result.values()):
        raise ValueError("nonfinite aggregate metric")
    return result


def row_metrics(candidate, target, reference=None):
    # Shift first, then normalize in FP64. NLL uses shifted logits, avoiding
    # subtraction of two very large unshifted logsumexp/logit values.
    if not np.isfinite(candidate).all():
        raise ValueError("candidate logits contain NaN or infinity")
    c = candidate.astype(np.float64)
    cmax = float(c.max())
    clp = c - cmax
    clogden = math.log(float(np.exp(clp).sum(dtype=np.float64)))
    clp -= clogden
    cargmax = int(np.argmax(c))
    result = {"target_token_id": int(target), "candidate_argmax": cargmax,
              "candidate_nll": -float(clp[target]), "candidate_logsumexp": cmax + clogden}
    if reference is not None:
        if not np.isfinite(reference).all():
            raise ValueError("reference logits contain NaN or infinity")
        r = reference.astype(np.float64)
        rmax = float(r.max())
        rlp = r - rmax
        rp = np.exp(rlp)
        rden = float(rp.sum(dtype=np.float64))
        rlogden = math.log(rden)
        rlp -= rlogden
        rp /= rden
        delta = c - r
        abs_delta = np.abs(delta)
        kl = float(np.dot(rp, rlp - clp))
        if kl < -1e-10:
            raise ValueError(f"invalid negative KL(ref||candidate): {kl}")
        rargmax = int(np.argmax(r))
        result.update({"reference_argmax": rargmax, "argmax_agrees": cargmax == rargmax,
                       "reference_nll": -float(rlp[target]), "reference_logsumexp": rmax + rlogden,
                       "nll_delta": float(rlp[target] - clp[target]), "kl_ref_to_candidate": kl,
                       "elements": int(c.size), "absolute_error_sum": float(abs_delta.sum(dtype=np.float64)),
                       "max_abs_logit_error": float(abs_delta.max()),
                       "squared_error_sum": float(np.dot(delta, delta)),
                       "reference_squared_sum": float(np.dot(r, r))})
    if not all(math.isfinite(v) for v in result.values()):
        raise ValueError("nonfinite per-request metric")
    return result


def load_reference(path, contract):
    with open(path + ".json", encoding="utf-8") as f:
        metadata = json.load(f)
    if metadata.get("schema_version") != 1 or metadata.get("status") != "complete":
        raise ValueError("reference sidecar is unsupported or incomplete")
    if metadata.get("input_contract") != contract:
        raise ValueError("reference corpus/token hashes, positions, or input layout differ")
    if not isinstance(metadata.get("baseline_environment"), dict) or not metadata.get("baseline_arguments"):
        raise ValueError("reference sidecar lacks baseline environment/arguments")
    reference = np.load(path, mmap_mode="r", allow_pickle=False)
    shape = metadata.get("shape")
    if list(reference.shape) != shape or reference.ndim != 3 or shape[:2] != contract["shape_prefix"]:
        raise ValueError("reference tensor/sidecar shapes differ")
    if reference.dtype != np.dtype("<f4") or metadata.get("dtype") != "<f4" or not reference.flags.c_contiguous:
        raise ValueError("reference must be C-contiguous little-endian float32")
    # Bind sidecar to actual payload before model initialization. Streaming through
    # mmap does not copy the full [steps,N,V] tensor into a Python allocation.
    digest = hashlib.sha256()
    for step in reference:
        if not np.isfinite(step).all():
            raise ValueError("reference payload contains NaN or infinity")
        digest.update(memoryview(step).cast("B"))
    if digest.hexdigest() != metadata.get("logits_payload_sha256"):
        raise ValueError("reference payload checksum differs from sidecar")
    return reference, metadata


def prefill(lib, args, ids, slots):
    n, p = args.concurrency, args.context
    for slot in range(n):
        bd.ck(lib.qwn_paged_reset_slot(slot), f"reset_slot@{slot}")
    per = max(1, args.wave // n)
    columns = n * min(per, p)
    tokens, col_slot, col_pos = ((ctypes.c_int * columns)() for _ in range(3))
    seg_off, seg_len, seg_final = ((ctypes.c_int * n)() for _ in range(3))
    seed = (ctypes.c_int * n)()
    start = time.perf_counter_ns()
    for w in range(0, p, per):
        cn = min(per, p - w)
        for slot in range(n):
            off = slot * cn
            seg_off[slot], seg_len[slot], seg_final[slot] = off, cn, int(w + cn == p)
            for column in range(cn):
                i = off + column
                tokens[i] = ids[(slot * SLOT_STRIDE + w + column) % (len(ids) - 1)]
                col_slot[i], col_pos[i] = slot, w + column
        bd.ck(lib.qwn_paged_prefill_batch(tokens, col_slot, col_pos, slots, seg_off, seg_len,
                                         seg_final, n, n * cn, seed), f"prefill@{w}")
    return seed, (time.perf_counter_ns() - start) / 1e9


def main():
    args = parse_args()
    environment = {k: v for k, v in sorted(os.environ.items())
                   if k.startswith("TQ_") or k in ("CUDA_VISIBLE_DEVICES", "CUDA_MODULE_LOADING")}
    snapshots = []
    result = {"schema_version": 1, "status": "running", "arguments": vars(args), "environment": environment,
              "scope": "fixed-corpus next-token likelihood and raw-logit reference comparison, not task quality",
              "numerics": "FP64 shifted logsumexp; natural-log NLL and KL(ref||candidate); raw uncentered logit errors",
              "kl_roundoff": "negative KL below -1e-10 fails; smaller negative values are reported without clipping",
              "timing_scope": "diagnostic synchronous host native calls including float32 logits D2H; not serving throughput",
              "cuda_memory": {"scope": "cudaMemGetInfo device-wide, not process-exclusive",
                              "sampling": "stage snapshots, not continuous peak; baseline excludes CUDA context initialization",
                              "snapshots": snapshots}, "steps": []}
    lib = None
    initialized = pool_initialized = False
    saved = None
    sidecar = None
    try:
        with open(args.corpus, "rb") as f:
            corpus = f.read()
        tokenizer = bd.AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
        ids = tokenizer(corpus.decode("utf-8"), add_special_tokens=False).input_ids
        if len(ids) < 2:
            raise ValueError("corpus must tokenize to at least two tokens")
        n, p = args.concurrency, args.context
        cycle = len(ids) - 1
        inputs = [[ids[(slot * SLOT_STRIDE + p + step) % cycle] for slot in range(n)] for step in range(args.steps)]
        targets = [[ids[(slot * SLOT_STRIDE + p + step + 1) % cycle] for slot in range(n)] for step in range(args.steps)]
        contract = {"corpus_sha256": hashlib.sha256(corpus).hexdigest(), "token_ids_sha256": hash_json(ids),
                    "corpus_token_count": len(ids), "corpus_cycle": cycle, "slot_stride": SLOT_STRIDE,
                    "context": p, "positions": list(range(p, p + args.steps)), "shape_prefix": [args.steps, n],
                    "decode_input_mode": "strict_teacher_force", "input_token_ids_sha256": hash_json(inputs),
                    "target_token_ids_sha256": hash_json(targets), "target_rule": "corpus at absolute input position + 1",
                    "slot_order": list(range(n))}
        del corpus
        result["input_contract"] = contract
        reference = reference_metadata = None
        if args.reference_logits:
            reference, reference_metadata = load_reference(args.reference_logits, contract)
            result["reference"] = reference_metadata
        lib = bd.bind_library(args.lib)
        step_logits = lib.qwn_bench_paged_step_logits
        step_logits.argtypes = [ctypes.POINTER(ctypes.c_int)] * 3 + [ctypes.c_int, ctypes.POINTER(ctypes.c_int),
                              ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
        step_logits.restype = ctypes.c_int
        rt = ctypes.CDLL("libcudart.so")
        rt.cudaMemGetInfo.argtypes = [ctypes.POINTER(ctypes.c_size_t)] * 2
        rt.cudaMemGetInfo.restype = ctypes.c_int
        bd.memory_snapshot(rt, snapshots, "before_model_init")
        bd.ck(lib.qwn_init(args.tqf.encode()), "init")
        initialized = True
        bd.memory_snapshot(rt, snapshots, "after_model_init")
        vocab, max_seq, wave_cap = lib.qwn_vocab_size(), lib.qwn_max_seq(), lib.qwn_wave_cap()
        bd.check_tokens([ids], vocab, "corpus")
        if p + args.steps > max_seq:
            raise ValueError(f"context plus steps exceeds model limit {max_seq}")
        actual_wave = n * min(max(1, args.wave // n), p)
        if actual_wave > wave_cap:
            raise ValueError(f"prefill wave {actual_wave} exceeds engine cap {wave_cap}")
        slots_count = args.slots or n
        required_blocks = n * ((p + args.steps + args.page - 1) // args.page)
        blocks = args.blocks or max(required_blocks, n * ((p + args.page - 1) // args.page + 2))
        if blocks > bd.INT_MAX:
            raise ValueError("pool block count exceeds C integer capacity")
        shape = (args.steps, n, vocab)
        if reference is not None and reference.shape != shape:
            raise ValueError(f"reference shape {reference.shape} differs from candidate {shape}")
        result["model_limits"] = {"vocab_size": vocab, "max_seq": max_seq, "wave_cap": wave_cap}
        result["pool"] = {"slots": slots_count, "blocks": blocks, "page": args.page,
                          "required_blocks": required_blocks, "actual_prefill_wave": actual_wave}
        payload_hash = hashlib.sha256()
        if args.save_logits:
            sidecar = {"schema_version": 1, "status": "incomplete", "input_contract": contract,
                       "shape": list(shape), "dtype": "<f4", "baseline_environment": environment,
                       "baseline_arguments": vars(args), "model_limits": result["model_limits"]}
            json_write(args.save_logits + ".json", sidecar)
            saved = np.lib.format.open_memmap(args.save_logits, mode="w+", dtype="<f4", shape=shape)
        bd.ck(lib.qwn_paged_init(slots_count, blocks, args.page), "paged_init")
        pool_initialized = True
        bd.memory_snapshot(rt, snapshots, "after_pool_init", n, p)
        slots = (ctypes.c_int * n)(*range(n))
        seed, result["prefill_s"] = prefill(lib, args, ids, slots)
        bd.check_tokens([seed], vocab, "prefill")
        result["seed_token_ids"] = list(seed)
        bd.memory_snapshot(rt, snapshots, "after_prefill", n, p)
        result["pool_after_prefill"] = bd.pool_stats(lib)
        input_buffers = [(ctypes.c_int * n)(*row) for row in inputs]
        positions = (ctypes.c_int * n)(*([p] * n))
        out = (ctypes.c_int * n)()
        logits = np.empty((n, vocab), dtype="<f4")
        logits_ptr = logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        all_rows = []
        for step in range(args.steps):
            start = time.perf_counter_ns()
            rc = step_logits(input_buffers[step], slots, positions, n, out, logits_ptr, logits.size)
            elapsed_ms = (time.perf_counter_ns() - start) / 1e6
            bd.ck(rc, f"paged_step_logits@{p + step}")
            bd.check_tokens([out], vocab, "decode")
            snapshot = bd.memory_snapshot(rt, snapshots, f"after_decode_step_{step}", n, p)
            rows = [row_metrics(logits[slot], targets[step][slot],
                                reference[step, slot] if reference is not None else None) for slot in range(n)]
            # Native greedy output may differ on exact ties, but must select a
            # maximum in its own exported raw logits. A mismatch is an invalid measurement.
            for slot, row in enumerate(rows):
                if logits[slot, out[slot]] != logits[slot, row["candidate_argmax"]]:
                    raise ValueError(f"native argmax and exported logits disagree at step={step}, slot={slot}")
            if saved is not None:
                saved[step] = logits
                payload_hash.update(memoryview(logits).cast("B"))
            record = {"step": step, "position": p + step, "input_token_ids": inputs[step],
                      "target_token_ids": targets[step], "output_token_ids": list(out),
                      "native_call_ms": elapsed_ms, "requests": rows,
                      "summary": summarize(rows, reference is not None), "cuda_memory_snapshot": snapshot}
            result["steps"].append(record)
            all_rows.extend(rows)
            for slot in range(n):
                positions[slot] += 1
        result["summary"] = summarize(all_rows, reference is not None)
        result["pool_after_decode"] = bd.pool_stats(lib)
        if saved is not None:
            saved.flush()
            sidecar["status"] = "complete"
            sidecar["logits_payload_sha256"] = payload_hash.hexdigest()
            sidecar["summary"] = result["summary"]
            json_write(args.save_logits + ".json", sidecar)
            result["saved_reference"] = sidecar
        result["status"] = "ok"
        print("PREC " + json.dumps(result["summary"], sort_keys=True, allow_nan=False), flush=True)
    except Exception as exc:
        result["status"], result["error"] = "error", str(exc)
        raise
    finally:
        if snapshots:
            result["cuda_memory"]["max_sampled_device_used_bytes"] = max(s["device_used_bytes"] for s in snapshots)
            result["cuda_memory"]["max_sampled_delta_from_before_model_init_bytes"] = max(
                s["delta_from_before_model_init_bytes"] for s in snapshots)
        try:
            if pool_initialized:
                bd.ck(lib.qwn_paged_free(), "paged_free")
        finally:
            if initialized:
                lib.qwn_free()
            if args.json_out:
                json_write(args.json_out, result)


if __name__ == "__main__":
    main()
