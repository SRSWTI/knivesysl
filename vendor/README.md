# vendor/

Third-party upstreams, each kept as its own git clone so we can pull, diff and
send patches without vendoring copies into this repo's history. Nothing in here
is tracked by the parent repo except this file and `vendor_pull_report.sh`
(see the `vendor/*` rules in the root `.gitignore`).

## Re-cloning a fresh tree

`vendor/` is not tracked, so a fresh checkout starts empty. Clone what you need:

```sh
git clone https://github.com/intel/compute-runtime.git          vendor/compute-runtime
git clone https://github.com/oneapi-src/level-zero.git          vendor/level-zero
git clone https://github.com/uxlfoundation/oneDNN.git           vendor/oneDNN
git clone https://github.com/intel/llvm.git                     vendor/llvm
git clone https://github.com/intel/sycl-tla.git                 vendor/intel-xpu/sycl-tla
git clone https://github.com/vllm-project/vllm-xpu-kernels.git \
    vendor/intel-xpu/vllm-xpu/vllm-xpu-kernels
```

`vendor_pull_report.sh` walks the tree, fetches each clone and reports what
moved upstream. Run it from the repo root.

## Why each one is here

| path | upstream | what we use it for |
|---|---|---|
| `oneDNN` | `uxlfoundation/oneDNN` | grouped NVFP4 matmul; the `SB_GROUPED_BACKEND=onednn` path |
| `intel-xpu/vllm-xpu/vllm-xpu-kernels` | `vllm-project/vllm-xpu-kernels` | origin of the Xe2 grouped MoE GEMM our NVFP4 kernel forked from |
| `intel-xpu/sycl-tla` | `intel/sycl-tla` | cute/cutlass-sycl headers and the MoE GEMM reference example |
| `compute-runtime` | `intel/compute-runtime` | NEO / Level Zero driver source; `zex` wait-on-memory used by the doorbell |
| `level-zero` | `oneapi-src/level-zero` | Level Zero loader and API headers |
| `llvm` | `intel/llvm` | SYCL runtime and the Unified Runtime Level Zero adapters |

Other clones under `vendor/` are reference material and are not on any build or
run path.

## Upstream contributions

Work sent back upstream from this tree. Kept here so the local clones and the
filed artifacts stay associated; the full bodies live in `experiments/repro/`.

### Pull requests

| PR | What |
|---|---|
| [oneDNN#5911](https://github.com/uxlfoundation/oneDNN/pull/5911) | Reject non-canonical grouped quantization layouts — turns a silent wrong answer into `dnnl_invalid_arguments` |
| [vllm-xpu-kernels#553](https://github.com/vllm-project/vllm-xpu-kernels/pull/553) | Add NVFP4 (E2M1 + E4M3 group-16) B-dtype to the Xe2 grouped MoE GEMM, with E=85 tests |
| [sycl-tla#856](https://github.com/intel/sycl-tla/pull/856) | `--experts` / `--rows_per_expert` for example 12, so the MoE fill sweep is a first-class knob |
| [oneDNN#5909](https://github.com/uxlfoundation/oneDNN/pull/5909) | Grouped NVFP4 example — **closed**, correctly: every concept was already covered elsewhere |

### Issues

| Issue | What |
|---|---|
| [compute-runtime#981](https://github.com/intel/compute-runtime/issues/981) | Device USM on BMG consumes host `MemAvailable` ~1:1 with allocation size |
| [compute-runtime#982](https://github.com/intel/compute-runtime/issues/982) | `zexCommandListAppendWaitOnMemory` on an unregistered host pointer silently waits on an internal copy |
| [compute-runtime#983](https://github.com/intel/compute-runtime/issues/983) | Feature request: timeout or abort token for the same entry point |
| [oneDNN#5908](https://github.com/uxlfoundation/oneDNN/issues/5908) | Grouped GEMM silently wrong when weight-scale memory is not canonical dense `abc` |
| [oneDNN#5912](https://github.com/uxlfoundation/oneDNN/issues/5912) | Grouped GEMM wrong with a 16-bit `binary_mul` operand at mask 0 |
| [sycl-tla#855](https://github.com/intel/sycl-tla/issues/855) | Example 12's reported GFLOP/s is dominated by per-expert fill |
| [intel/llvm#23034](https://github.com/intel/llvm/issues/23034) | SIGSEGV in `urEnqueueUSMFill` on the second device-USM memset with a default out-of-order queue |

Reproducers for all of the above are in `experiments/repro/`, each standalone
and dependent only on SYCL or benchdnn.

## Local patches

We carry no silent forks of these trees. The one local edit that existed —
an in-source `MOE_ROWS` hook in sycl-tla example 12 — was reverted and replaced
by the upstream PR above; the original is preserved as
`experiments/repro/sycl_tla_moe_rows_local.patch` for reference.

Two SpecForge edits remain unsent and are **not** in this directory's scope:
see `experiments/drafter_train/` if that work is picked back up.
