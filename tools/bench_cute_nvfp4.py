#!/usr/bin/env python3
"""Correctness-first SM120 CuTe DSL NVFP4 tile sweep (no production changes).

Dependencies: CUDA-enabled torch with float4_e2m1fn_x2, numpy, cuda-python,
nvidia-cutlass-dsl and its matching library wheels, CUDA toolkit/driver for
SM120. The vendored CuTeDSL requirements pin 4.8.0.dev0; installed 4.6.2 has
the inspected APIs but requires explicit --allow-version-mismatch to try it.
Compilation/launch failures are errors, never successful skipped candidates.

Run from the repo root, with the production server stopped:
  .venv/bin/python tools/bench_cute_nvfp4.py --list-candidates
  .venv/bin/python tools/bench_cute_nvfp4.py --allow-version-mismatch \
      --shapes gate_up --tokens 1,32 --candidates cooperative-k128-e64x32 \
      --warmup 5 --repeats 7 --graph-launches 32 --json-out /tmp/cute-smoke.json
  .venv/bin/python tools/bench_cute_nvfp4.py --allow-version-mismatch \
      --shapes gate_up,down --tokens 1,2,4,8,32,128,512,2048 \
      --json-out /tmp/cute-sweep.json
  .venv/bin/python tools/bench_cute_nvfp4.py --allow-version-mismatch \
      --shapes down --tokens 128 --candidates pingpong-k256-e64x32 \
      --artifacts-dir /tmp/cute-ir --json-out /tmp/cute-ir.json

Mapping: model Y[M,N] = W[M,K] @ X[K,N]. Example computes C[m,n,l]
= A[m,k,l] @ B[n,k,l]^T, so m=N tokens, n=M output channels, k=K,
l=1; A=X^T and B=W, both K-major. C=Y^T is N-major (channels contiguous).
No token padding is added to A/C; the example masks partial 128-wide tiles.
Only scale storage is padded to its required 128-row, four-scale atom.

Six bounded candidates: cooperative/pingpong, CTA (128,128,128|256), epilogue
(128,128)|(64,32), excluding K256 with epi128x128: FP32's 64 KiB epilogue
leaves insufficient shared memory for an AB stage. Both use 8 MMA warps +
1 TMA warp, cluster (1,1,1),
occupancy 1. Cooperative uses MMA warp layout (4,2,1); pingpong uses two
(2,2,1) groups. Stage counts are the example's automatic shared-memory
heuristic, recorded after compilation; arbitrary warp/stage overrides are
not supported. K must be divisible by 32 for FP4's 16-byte alignment and
by 16 for scales; output channels must be divisible by 4 for FP32 stores.
Both output and accumulation are FP32, matching native projection precision.
The imported examples' stmatrix layout-only accumulator seed is adapted to
Float16, matching native sm120_builder.inl CopyAtomC. This preserves four
values/thread; using Float32 there incorrectly halves the layout footprint.
The actual output, accumulator and universal register-to-shared store remain
Float32. The adapter is scoped to each imported example's CuTe namespace;
it does not modify vendor files or the installed CuTe module.

Inputs are seeded raw FP4 nibbles, including signs/zeros, with independently
rounded E4M3 block scales (16 elements). Reference decodes the EXACT packed
bytes and scale layout, not pre-rounding floats. FP32 GEMM has TF32 disabled;
every candidate must pass finite/elementwise checks before timing. This is
arithmetic/layout coverage, NOT model quantization-quality, TQF format or
production-kernel parity: no production weights, activation quantizer,
per-128-output-row global factors used by production, fused gate/SiLU,
residuals, KV or end-to-end inference. Global alpha is effectively one;
both operands are already quantized.

Timing captures only repeated compiled GEMM launches, with fixed inputs and
output, then uses CUDA events around graph replays. No reference, packing,
allocation, compilation or host launch loop is timed. This is warm-cache,
single-workspace throughput, not cold-cache latency or complete layer time.
Each matrix can fit the RTX 5090's measured 96 MiB L2; these results MUST NOT
be compared to full-model DRAM bandwidth as though weights were cold.
--memory-limit-mib caps the PyTorch allocator (not CUDA context/compiler
allocations); peak allocated/reserved counts include reference preparation.
Device free-memory snapshots are also reported, not claimed as a device peak.
--artifacts-dir sets supported CUTE_DSL_KEEP=ir,ir-debug,ptx and
CUTE_DSL_DUMP_DIR before CuTe import; returned artifact paths are recorded.
These env flags force recompilation for PTX retention. stdout is one JSON
object (compiler diagnostics go to stderr); --json-out checkpoints after each
candidate. Any prerequisite, compile, numerical or memory failure exits 1.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "vendor/cutlass/examples/python/CuTeDSL/cute/blackwell_geforce/kernel/blockscaled_gemm"
SHAPES = {"gate_up": (17408, 5120), "down": (5120, 17408)}
TOKENS = (1, 2, 4, 8, 32, 128, 512, 2048)
CANDIDATES = {
    f"{schedule}-k{k}-e{em}x{en}": {
        "schedule": schedule, "tile_mnk": (128, 128, k),
        "epilogue_tile": (em, en), "mma_warps": 8, "tma_warps": 1,
        "mma_warp_layout": (4, 2, 1) if schedule == "cooperative" else (2, 2, 1),
        "cluster_mnk": (1, 1, 1), "occupancy": 1, "stages_policy": "example_auto",
    }
    for schedule in ("cooperative", "pingpong")
    for k in (128, 256)
    for em, en in ((128, 128), (64, 32))
    if not (k == 256 and (em, en) == (128, 128))
}


def selection(text, allowed, label):
    selected = text.split(",")
    if not selected or any(item not in allowed for item in selected):
        raise ValueError(f"Unsupported {label}: {text}; allowed: {','.join(map(str, allowed))}")
    if len(set(selected)) != len(selected):
        raise ValueError(f"Duplicate {label}: {text}")
    return selected


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shapes", default=",".join(SHAPES))
    parser.add_argument("--tokens", default=",".join(map(str, TOKENS)))
    parser.add_argument("--candidates", default=",".join(CANDIDATES))
    parser.add_argument("--list-candidates", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--graph-launches", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--atol", type=float, default=0.125)
    parser.add_argument("--rtol", type=float, default=0.01)
    parser.add_argument("--memory-limit-mib", type=int, default=4096)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--artifacts-dir", type=Path)
    parser.add_argument("--allow-version-mismatch", action="store_true")
    args = parser.parse_args()
    try:
        args.shapes = selection(args.shapes, SHAPES, "shape")
        args.tokens = [int(n) for n in selection(args.tokens, tuple(map(str, TOKENS)), "token count")]
        args.candidates = selection(args.candidates, CANDIDATES, "candidate")
        if args.warmup < 1 or args.repeats < 1 or not 1 <= args.graph_launches <= 1024:
            raise ValueError("warmup/repeats must be positive; graph-launches must be in [1,1024]")
        if args.memory_limit_mib < 1 or not 0 <= args.seed < 2**32:
            raise ValueError("memory-limit-mib must be positive; seed must be a uint32")
        if any(not math.isfinite(v) or v < 0 for v in (args.atol, args.rtol)):
            raise ValueError("atol/rtol must be finite and nonnegative")
    except ValueError as exc:
        parser.error(str(exc))
    return args


def checkpoint(report, path):
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)


def persist_artifacts(compiled, directory, candidate, logical_mnk):
    """JitFunctionArtifacts fields contain source text, not filenames."""
    artifacts = getattr(compiled, "artifacts", None)
    retained = {}
    for key in ("MLIR", "PTX", "SASS"):
        value = getattr(artifacts, key, None)
        if value is None:
            continue
        if not isinstance(value, str):
            raise TypeError(f"Unexpected {key} artifact type: {type(value).__name__}")
        payload = value.encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        entry = {"sha256": digest, "bytes": len(payload), "path": None}
        if directory is not None:
            stem = candidate + "-" + "x".join(map(str, logical_mnk)) + "-" + digest[:16]
            path = directory / (stem + "." + key.lower())
            path.write_bytes(payload)
            entry["path"] = str(path)
        retained[key] = entry
    return retained


class Fp32EpilogueCuteNamespace:
    """Use native SM120's fixed half_t layout seed, not a half output store.

    Python examples type their layout-only stmatrix x2 seed as c_dtype.
    Native include/cutlass/epilogue/collective/builders/sm120_builder.inl
    uses Copy_Atom<SM90_U32x2_STSM_N, half_t> independently of ElementD.
    Actual FP32 R2S is chosen in blackwell_helpers, outside this namespace.
    """

    def __init__(self, cute, cutlass):
        self._cute = cute
        self._cutlass = cutlass
        self.layout_seed_rewrites = 0

    def __getattr__(self, name):
        return getattr(self._cute, name)

    def make_copy_atom(self, op, dtype, *args, **kwargs):
        if isinstance(op, self._cute.nvgpu.warp.StMatrix8x8x16bOp) and dtype is self._cutlass.Float32:
            if op.num_matrices != 2 or op.transpose:
                raise ValueError("Unexpected FP32 stmatrix seed: only the inspected n-major x2 layout is supported")
            dtype = self._cutlass.Float16
            self.layout_seed_rewrites += 1
        return self._cute.make_copy_atom(op, dtype, *args, **kwargs)


def prerequisites(args, report):
    versions = {name: importlib.metadata.version(name)
                for name in ("torch", "numpy", "cuda-python", "nvidia-cutlass-dsl")}
    requirement = ROOT / "vendor/cutlass/python/CuTeDSL/requirements.txt"
    pins = [line.split("==", 1)[1].strip() for line in requirement.read_text().splitlines()
            if line.startswith("nvidia-cutlass-dsl==")]
    if len(pins) != 1:
        raise RuntimeError(f"Cannot identify exact DSL prerequisite in {requirement}")
    report["versions"] = versions
    report["vendored_dsl_requirement"] = pins[0]
    mismatch = versions["nvidia-cutlass-dsl"] != pins[0]
    report["version_mismatch"] = mismatch
    if mismatch and not args.allow_version_mismatch:
        raise RuntimeError(f"Vendored example requires DSL {pins[0]}, installed {versions['nvidia-cutlass-dsl']}; "
                           "install matching wheels or explicitly try --allow-version-mismatch")
    # The vendor source text is unchanged by our namespace adapter. Never load
    # a cached pre-adapter kernel keyed by that same source/signature.
    os.environ["CUTE_DSL_NO_CACHE"] = "1"
    os.environ["CUTE_DSL_DISABLE_FILE_CACHING"] = "1"
    report["compiler_cache"] = "disabled_for_example_local_epilogue_adapter"
    if args.artifacts_dir:
        args.artifacts_dir = args.artifacts_dir.resolve()
        args.artifacts_dir.mkdir(parents=True, exist_ok=True)
        os.environ["CUTE_DSL_KEEP"] = "ir,ir-debug,ptx"
        os.environ["CUTE_DSL_DUMP_DIR"] = str(args.artifacts_dir)
    report["artifact_environment"] = {key: os.environ.get(key) for key in
                                      ("CUTE_DSL_KEEP", "CUTE_DSL_DUMP_DIR")}
    sys.path.insert(0, str(EXAMPLES))
    import numpy as np
    import torch
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    import cutlass.torch as cutlass_torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("This driver requires an SM120 CUDA GPU")
    if not hasattr(torch, "float4_e2m1fn_x2"):
        raise RuntimeError("Torch must expose float4_e2m1fn_x2 for packed DLPack tensors")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    props = torch.cuda.get_device_properties(0)
    limit = args.memory_limit_mib * 1024**2
    if limit >= props.total_memory:
        raise ValueError("memory-limit-mib must be smaller than total device memory")
    torch.cuda.set_per_process_memory_fraction(limit / props.total_memory, 0)
    report["device"] = {"name": props.name, "capability": [props.major, props.minor],
                        "total_memory_bytes": props.total_memory, "sm_count": props.multi_processor_count}
    report["memory_limit_bytes"] = limit
    modules = {}
    report["example_sha256"] = {}
    for schedule in {CANDIDATES[name]["schedule"] for name in args.candidates}:
        name = f"dense_blockscaled_gemm_persistent_{schedule}"
        modules[schedule] = importlib.import_module(name)
        modules[schedule].cute = Fp32EpilogueCuteNamespace(cute, cutlass)
        report["example_sha256"][name] = hashlib.sha256((EXAMPLES / (name + ".py")).read_bytes()).hexdigest()
    report["example_sha256"]["blockscaled_gemm_dispatch"] = hashlib.sha256(
        (EXAMPLES / "blockscaled_gemm_dispatch.py").read_bytes()).hexdigest()
    report["driver_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return np, torch, cutlass, cute, from_dlpack, cutlass_torch, modules


def make_operand(rows, k, rng, api):
    """Raw FP4 storage plus CUTLASS scale atom, with an independent exact decoder."""
    np, torch, _, _, from_dlpack, _, _ = api
    packed_cpu = rng.integers(0, 256, (1, rows, k // 2), dtype=np.uint8)
    packed = torch.from_numpy(packed_cpu).cuda()
    fp4 = packed.view(torch.float4_e2m1fn_x2).permute(1, 2, 0)
    tensor = from_dlpack(fp4, assumed_align=16)
    # Match cute_tensor_like(is_dynamic_layout=True) before compact marking:
    # otherwise singleton token dimensions specialize away in TMA descriptors.
    tensor.mark_layout_dynamic(leading_dim=1)
    tensor.mark_compact_shape_dynamic(mode=1, stride_order=(2, 0, 1), divisibility=2)
    if tuple(tensor.shape) != (rows, k, 1):
        raise RuntimeError(f"Packed DLPack API returned unexpected logical shape {tensor.shape}")

    padded_rows = (rows + 127) // 128 * 128
    sf_k = k // 16
    # m = m0 + 32*m1 + 128*rest_m, sf_k = k0 + 4*rest_k.
    # Physical atom order matches the example's mma_shape/mma_permute_order.
    logical = torch.from_numpy(rng.uniform(0.0625, 0.5, (padded_rows, sf_k)).astype(np.float32))
    scale_cpu = logical.to(torch.float8_e4m3fn).reshape(padded_rows // 128, 4, 32, sf_k // 4, 4)
    scale_cpu = scale_cpu.permute(0, 3, 2, 1, 4).contiguous().unsqueeze(0)
    scales = scale_cpu.cuda()
    scale_view = scales.permute(3, 4, 1, 5, 2, 0)
    scale_tensor = from_dlpack(scale_view, assumed_align=16).mark_layout_dynamic(leading_dim=3)
    # Decode actual stored E4M3 values and undo the physical atom permutation.
    scale_ref = scales.float().permute(0, 1, 4, 3, 2, 5).reshape(padded_rows, sf_k)[:rows]
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6],
                       dtype=torch.float32, device="cuda")
    dequant = torch.empty((rows, k), dtype=torch.float32, device="cuda")
    # Bound decoder temporaries independently of model shape.
    for start in range(0, rows, 128):
        end = min(rows, start + 128)
        byte = packed[0, start:end]
        chunk = dequant[start:end]
        chunk[:, 0::2] = lut[(byte & 15).long()]
        chunk[:, 1::2] = lut[(byte >> 4).long()]
        chunk.view(end - start, sf_k, 16).mul_(scale_ref[start:end, :, None])
    digest = hashlib.sha256(packed_cpu.tobytes())
    digest.update(scale_cpu.view(torch.uint8).numpy().tobytes())
    return tensor, scale_tensor, (packed, fp4, scales, scale_view), dequant, digest.hexdigest()


def error_metrics(actual, reference, torch, args):
    observed = actual.float()
    if not bool(torch.isfinite(observed).all()) or not bool(torch.isfinite(reference).all()):
        return {"passed": False, "failure": "nonfinite output or reference"}
    delta = (observed - reference).abs()
    bound = args.atol + args.rtol * reference.abs()
    failed = int((delta > bound).sum().item())
    rms_ref = reference.square().mean().sqrt()
    rms_error = delta.square().mean().sqrt()
    return {"passed": failed == 0, "failed_elements": failed, "elements": reference.numel(),
            "atol": args.atol, "rtol": args.rtol, "max_abs": delta.max().item(),
            "mean_abs": delta.mean().item(), "rms_error": rms_error.item(),
            "relative_l2": (rms_error / rms_ref.clamp_min(1e-12)).item(),
            "max_relative_floor_1e_6": (delta / reference.abs().clamp_min(1e-6)).max().item()}


def measure_candidate(name, tensors, output, reference, api, args, result):
    _, torch, cutlass, cute, _, cutlass_torch, modules = api
    config = CANDIDATES[name]
    result["configuration"] = dict(config)
    result["configuration"].update({"epilogue_layout_seed": "Float16_stmatrix_x2_native_SM120",
                                     "epilogue_actual_store": "Float32_CopyUniversalOp",
                                     "epilogue_adapter": "example_local_CuTe_namespace"})
    logical_mnk = [int(tensors[0].shape[0]), int(tensors[1].shape[0]), int(tensors[0].shape[1])]
    covered_mnk = [(size + tile - 1) // tile * tile
                   for size, tile in zip(logical_mnk, config["tile_mnk"])]
    result["configuration"].update({"logical_mnk": logical_mnk,
                                     "tensor_storage_mnk": logical_mnk,
                                     "cta_covered_mnk": covered_mnk,
                                     "scale_padded_rows_ab": [(size + 127) // 128 * 128 for size in logical_mnk[:2]]})
    result["useful_flops"] = 2 * math.prod(logical_mnk)
    result["cta_covered_flops"] = 2 * math.prod(covered_mnk)
    kernel = modules[config["schedule"]].Sm120BlockScaledGemmKernel(
        cutlass.Float32, 16, config["tile_mnk"], config["epilogue_tile"])
    active = cutlass.utils.HardwareInfo().get_max_active_clusters(1)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        cu_stream = cutlass_torch.current_stream()
    namespace = modules[config["schedule"]].cute
    rewrites_before = namespace.layout_seed_rewrites
    compile_start = time.perf_counter()
    compiled = cute.compile(kernel, *tensors, active, cu_stream)
    rewrites = namespace.layout_seed_rewrites - rewrites_before
    result["configuration"]["epilogue_layout_seed_rewrites"] = rewrites
    if rewrites < 1:
        raise RuntimeError("CuTe did not apply the required native-SM120 FP32 epilogue layout adapter")
    result["compile_seconds"] = time.perf_counter() - compile_start
    result["configuration"].update({"ab_stages": int(kernel.ab_stage), "epilogue_stages": int(kernel.epi_stage),
                                     "threads_per_cta": kernel.threads_per_cta,
                                     "max_active_clusters": int(active), "smem_capacity_bytes": kernel.smem_capacity})
    if kernel.ab_stage < 1 or kernel.epi_stage < 1:
        raise RuntimeError("Example selected an unsupported nonpositive pipeline stage count")
    result["artifacts"] = persist_artifacts(compiled, args.artifacts_dir, name, logical_mnk)
    output.fill_(float("nan"))
    torch.cuda.synchronize()
    compiled(*tensors, cu_stream)
    torch.cuda.synchronize()
    result["reference_errors"] = error_metrics(output, reference, torch, args)
    if not result["reference_errors"]["passed"]:
        raise RuntimeError("Numerical validation failed; candidate was not timed")
    for _ in range(args.warmup):
        compiled(*tensors, cu_stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(args.graph_launches):
            compiled(*tensors, cu_stream)
    # Instantiate graph and CUDA events before collecting samples.
    graph.replay()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    end.record()
    torch.cuda.synchronize()
    samples = []
    for _ in range(args.repeats):
        begin.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) * 1000 / args.graph_launches)
    if any(not math.isfinite(t) or t <= 0 for t in samples):
        raise RuntimeError(f"Invalid CUDA event measurements: {samples}")
    result["kernel_us"] = {"median": statistics.median(samples), "min": min(samples),
                            "max": max(samples), "samples": samples}
    result["useful_tflops"] = result["useful_flops"] / statistics.median(samples) / 1e6
    result["timing"] = {"method": "cuda_event_graph_replay", "warmup_launches": args.warmup,
                        "launches_per_graph": args.graph_launches, "repeats": args.repeats,
                        "workspace_count": 1, "cache_policy": "warm_reused_inputs"}
    graph.reset()
    result["status"] = "ok"


def run_shape(shape, tokens, api, args, report):
    np, torch, cutlass, _, from_dlpack, _, modules = api
    channels, k = SHAPES[shape]
    case = {"shape": shape, "model_mnk": [channels, tokens, k],
            "example_mnkl": [tokens, channels, k, 1], "status": "error", "results": []}
    report["cases"].append(case)
    # Conservative bound includes FP32 decoded operands, reference/error temporaries,
    # packed data/scales, decoder scratch and a fixed graph/library allowance.
    estimated = 12 * (tokens + channels) * k + 40 * tokens * channels + 256 * 1024**2
    case["estimated_workspace_bound_bytes"] = estimated
    if estimated > args.memory_limit_mib * 1024**2:
        raise RuntimeError("Case rejected by conservative workspace bound; raise --memory-limit-mib explicitly")
    klass = next(iter(modules.values())).Sm120BlockScaledGemmKernel
    if not klass.is_valid_tensor_alignment(tokens, channels, k, 1, cutlass.Float4E2M1FN,
                                            cutlass.Float32, "k", "k", "n"):
        raise RuntimeError("Case violates the vendored example's tensor alignment constraints")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    case["device_free_before_bytes"] = torch.cuda.mem_get_info()[0]
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, channels, k, tokens]))
    a, sfa, a_owners, a_ref, a_hash = make_operand(tokens, k, rng, api)
    b, sfb, b_owners, b_ref, b_hash = make_operand(channels, k, rng, api)
    reference = a_ref @ b_ref.T
    del a_ref, b_ref
    output = torch.empty((tokens, channels), dtype=torch.float32, device="cuda")
    c = from_dlpack(output.unsqueeze(0).permute(1, 2, 0), assumed_align=16)
    c.mark_layout_dynamic(leading_dim=1)
    c.mark_compact_shape_dynamic(mode=1, stride_order=(2, 0, 1))
    tensors = (a, b, sfa, sfb, c)
    case["input_sha256"] = {"activation_packed_and_scales": a_hash, "weight_packed_and_scales": b_hash}
    case["input_format"] = {"operands": "Float4E2M1FN", "scales": "Float8E4M3FN", "scale_vector": 16,
                             "accumulator": "Float32", "output": "Float32", "global_alpha": 1.0}
    torch.cuda.synchronize()
    preparation_peak = torch.cuda.max_memory_allocated()
    for name in args.candidates:
        result = {"candidate": name, "status": "error"}
        case["results"].append(result)
        torch.cuda.reset_peak_memory_stats()
        try:
            measure_candidate(name, tensors, output, reference, api, args, result)
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            result["memory"] = {"preparation_peak_allocated_bytes": preparation_peak,
                                "candidate_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                "peak_allocated_bytes": max(preparation_peak, torch.cuda.max_memory_allocated()),
                                "candidate_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                                "device_free_after_bytes": torch.cuda.mem_get_info()[0],
                                "scope": "torch allocator; device free is a snapshot, not peak"}
            checkpoint(report, args.json_out)
        # Owners intentionally stay live through every launch and graph replay.
        assert a_owners and b_owners
    case["status"] = "ok" if all(r["status"] == "ok" for r in case["results"]) else "error"


def main():
    args = parse_args()
    if args.list_candidates:
        print(json.dumps(CANDIDATES, indent=2))
        return 0
    # Required by deterministic torch CUDA GEMM; set before CUDA initialization.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    report = {"schema_version": 1, "benchmark": "sm120_cute_nvfp4", "status": "error",
              "seed": args.seed, "cases": [], "errors": [],
              "mapping": "model Y[M,N]=W[M,K]X[K,N]; example A=X^T,B=W,C=Y^T; (m,n,k,l)=(N,M,K,1)",
              "production_defaults_changed": False}
    try:
        with contextlib.redirect_stdout(sys.stderr):
            api = prerequisites(args, report)
            for shape in args.shapes:
                for tokens in args.tokens:
                    try:
                        run_shape(shape, tokens, api, args, report)
                    except Exception as exc:
                        report["errors"].append({"shape": shape, "tokens": tokens,
                                                 "error": f"{type(exc).__name__}: {exc}"})
                    gc.collect()
                    checkpoint(report, args.json_out)
        if not report["errors"] and all(case["status"] == "ok" for case in report["cases"]):
            report["status"] = "ok"
    except Exception as exc:
        report["errors"].append({"error": f"{type(exc).__name__}: {exc}"})
    checkpoint(report, args.json_out)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
