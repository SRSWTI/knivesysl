# knivesysl

**a bare-cuda inference engine for qwen3.8-27b on one rtx 5090. fp6 weights, 4-bit kv,
256k context, hand-written sm120 tensor-core kernels. no pytorch, no cublas, no cutlass
on the run path — one cuda translation unit behind a c ctypes abi.**

knivesysl is two things:

- **the engine** — 22k lines of cuda in a single tu, loaded by python over ctypes.
  the python side tokenizes and speaks http; every flop happens in `libforward_qwen.so`.
- **the format** — knivesysl fp6 (e2m3, 128-wide block scales, qmma fragment layout).
  6 bits per weight on the tensor cores, which is what makes 27b + 256k fit in 32 gb
  while staying above nvfp4 on quality.

right now it runs **exactly one model**: the qwen3.8-27b text tower (64 layers, 5120
hidden, 16 full-attention + 48 gated-deltanet). same layout as qwen3.6-27b, which also
converts. it is sm120-only and not portable — that is the trade being made.

```
qwen3.8-27b (hf, bf16)
   --> convert_qwen_tqf.py --> model.tqf   (fp6 e2m3 + block scales + mtp head, 22.6 gb)
   --> libforward_qwen.so                  (one cuda tu, compute_120f)
   --> serve_openai.py                     (single stream + mtp spec-decode)
    or serve_batched.py                    (paged kv + continuous batching)
   --> /v1/chat/completions
```

---

## why fp6

nvfp4 is 4 bits with a scale per 16 values. knivesysl fp6 is 6 bits with a pow2 scale per
128. more mantissa, coarser scaling --> better reconstruction of the weight distribution
at 1.4x the bytes, and still small enough that the whole tower plus a 256k kv cache lives
on one consumer card.

| format | bits/param | tf-top1 vs bf16 |
|---|--:|--:|
| fp8 | 8 | 95.94 |
| **knivesysl fp6 (e2m3)** | **6** | **91.30** |
| e2m1 (opt-in 4-bit tier) | 4 | 86.46 |
| nvfp4 | 4 | 85.78 |

quality figures carry over from the fork this began as and are pending our own
re-measurement; every performance number below we measured ourselves on this card.

---

## numbers

all measured 2026-08 on one rtx 5090 (gb202, sm120, 170 sm, 32 gb, 128 mb l2), cuda 13.3,
driver 595. the vllm column is vllm 0.27.1 serving `unsloth/qwen3.8-27b-nvfp4` (mixed
w8a8 + w4a4, 22.5 gb — footprint-matched to our 22.6 gb) at `--max-model-len 16384
--gpu-memory-utilization 0.90`. both engines driven by the same client, greedy, thinking
off.

### decode — we win

![decode ceiling](assets/decode-ceiling.svg)

a decode step is one pass over ~20 gib of weights whatever the batch size, so decode is
memory-bound and fewer weight bytes wins. the lead widens with concurrency because the
4-bit kv and the o(1) deltanet state keep 52 sequences resident in 30.8 gib.

single stream with mtp spec-decode: **86.8 tok/s** end-to-end vs vllm's 58.7 (+48%),
accept-length 2.7-3.3.

### decode holds its shape with depth

![decode scaling](assets/decode-scaling.svg)

after retuning the split-k gate for the paged attention: 16384/n=8 went **129.5 --> 45.9
ms/step**, 4096/n=8 **46.0 --> 25.1**. split-k is a different reduction order, so it is
eps-equivalent rather than bit-exact — 98.32% teacher-forced top-1 agreement (13 flips in
776 positions). `TQ_PAGED_SPLIT=1` forces the single-kernel path.

### prefix reuse

the single-stream path snapshots kv **and** the deltanet recurrent state at a 128-aligned
anchor, so a follow-up turn re-prefills only the suffix:

```
159k-token conversation --> cold 126.9 s --> follow-up 0.52 s   (158 848 / 158 967 reused)
  8k-token conversation --> cold   3.6 s --> follow-up 0.23 s   (  7 808 /   8 203 reused)
```

the batched server has the same primitive for a shared system prompt: **340 --> 784 tok/s**
at n=32, p50 latency 14.5 --> 6.2 s.

### prefill — we lose, and we know exactly why

![gemm headroom](assets/gemm-headroom.svg)

our hand-written fp6 gemm runs at **124 tflops**. cutlass 4.7's sm120 block-scaled
collective, on the identical numerics (e4m3 x e2m3) and the identical shapes on this gpu,
runs at **255 tflops at m=128 and 572 at m=512** — verify-passing. we are at a fifth of
the tensor cores.

that is the whole prefill gap. under a continuous-arrival benchmark (guidellm, n streams
held in flight, 256 output tokens, 20 s window) we lose every cell: 462 vs 774 tok/s at
128-token prompts / n=32, and at 4096-token prompts the window fills with prefill before
anything finishes. prefill never stops under arrivals, so a slow gemm compounds.

![prefill profile](assets/prefill-profile.svg)

and the gemm is only 55% of it. the 48 gated-deltanet layers' chunkwise scan is 22.5% —
amdahl says a cutlass port alone takes prefill 2500 --> ~4400 tok/s (parity with vllm's
2800-4400 band), and it needs the deltanet kernel too to clearly pass them.

**the honest summary: fixed batch, we hold the decode ceiling. sustained arrivals with
long prompts, vllm wins on prefill throughput. the fix is identified and measured, not
speculative.**

---

## how it works

- **weights** — fp6 e2m3, 128-wide pow2 block scales, stored in the qmma fragment layout
  the tensor cores want. ~20 gib for 27b.
- **kv cache** — 4-bit k (rotated int4 + hadamard) + e4m3 v. this is what buys 256k.
- **attention** — 16 full-attention layers on hand-written `mma.sync ...
  kind::mxf8f6f4.block_scale` with `ldmatrix ... b6x16` fp6 unpack; 48 gated-deltanet
  layers on a fused chunkwise scan.
- **spec decode** — an mtp covering tree --> batched k-split fp6 verify (weights read once
  for the whole tree) --> dense-argmax descent --> single-path commit.
- **prefill** — one wide fp6 gemm per projection --> chunk-parallel deltanet --> tensor-core
  wide attention against the q4 cache at any length.
- **batching** — paged kv pool, continuous batching, decode rows fused into prompt waves
  so both ride one weight read.

everything is compiled by stock `nvcc`/`ptxas` from cuda 13. no external assembler, no
precompiled cubins.

---

## build

```bash
export PATH=/usr/local/cuda/bin:$PATH
cmake -B build-qwen -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build-qwen --target knivesysl-forward-qwen -j
# -> build-qwen/libforward_qwen.so
```

requires an sm120 blackwell part (rtx 5090 / rtx pro 6000), cuda 12.8+, and python 3.10+
with `torch`, `transformers`, `numpy`, `safetensors` for the converter.

## get the weights

either pull our conversion:

```bash
huggingface-cli download srswti/axe-strada-knivesysl-27b --local-dir ./model
# 22.6 gb .tqf + tokenizer + chat template; --model-dir points at this same dir
```

or convert an hf checkpoint yourself:

```bash
TQ_EMIT_MTP=1 TQ_GPU_PACK=1 python3 tools/convert_qwen_tqf.py /path/to/qwen3.8-27b \
    -o model.tqf --block-scaled always --block-layout qmma-e2m3 --block-scale-policy pow2
```

`--block-scaled always` and `--block-layout qmma-e2m3` are both required; without them the
converter silently falls back to global-scale fp8. without `TQ_EMIT_MTP=1` there is no
spec-decode head. `TQ_GPU_PACK=1` quantizes on the gpu — 43 s instead of ~13 min.

verify with `python3 tools/inspect_tqf.py model.tqf`: a good file reports
`block_scaled_e2m3: True`, `has_mtp_section: True`, flags `0x53d`, ~22.6 gb.

## serve

```bash
tools/serve.sh                 # foreground, auto-sizes TQ_CTX from free vram
tools/serve.sh daemon          # detached (setsid), waits for readiness
tools/serve.sh status | logs | stop
```

or explicitly:

```bash
CUDA_VISIBLE_DEVICES=0 TQ_CTX=262144 TQ_KV_Q4=1 TQ_EMBED_FP8=2 \
    python3 tools/serve_openai.py --port 8000 \
    --tqf model.tqf --model-dir /path/to/qwen3.8-27b \
    --lib build-qwen/libforward_qwen.so
```

`TQ_EMBED_FP8=2` (6-bit embed table) is what makes the full 256k window fit in 32 gb. the
kv pool is sized by `TQ_CTX` and allocated on the first request, so an oversized context
dies mid-request rather than at load — `serve.sh` sizes it from free vram for you.

many-client mode:

```bash
CUDA_VISIBLE_DEVICES=0 TQ_CTX=32768 TQ_KV_Q4=1 python3 tools/serve_batched.py \
    --port 8100 --max-slots 32 --num-blocks 1024 --tqf model.tqf --model-dir ...
```

### reasoning effort

qwen3.8's chat template owns it: `reasoning_effort` in `low | medium | xhigh` (default
`xhigh`) selects a system-level instruction. we pass the tier through and fold `high`/`max`
onto `xhigh` so an openai-vocabulary client gets an answer instead of a 400. accepted as
`reasoning_effort`, `reasoningEffort`, or `reasoning.effort` on both endpoints.
`preserve_thinking` rides along, and preserved thinking keeps the prefix cache live across
turns.

### key flags

| flag | meaning |
|---|---|
| `TQ_CTX` | max context (engine cap 262144) |
| `TQ_KV_Q4=1` | 4-bit k + e4m3 v cache (needed for 256k) |
| `TQ_EMBED_FP8=2` | 6-bit embed table, -1.5 gib |
| `TQ_PAGED_SPLIT` | force split-k on/off for paged decode attention |
| `TQ_WAVE_MAX` | prefill wave column cap (default 128) |
| `TQ_W_E2M1=1` | opt-in 4-bit weight tier |
| `--prefill-budget` | prompt columns per wave, on top of the decode rows |
| `--prefix-cache` | materialize a shared prompt prefix once (batched server) |

## verify

```bash
python3 tools/mtp_spec_smoke.py --prompt-tokens 1024 --gen 128 --tqf model.tqf ...
python3 tools/bench_rounds.py  --prompt-tokens 65536 --rounds 200 --tqf model.tqf ...
python3 tools/needle_check.py                      # long-context retrieval gate
```

## layout

```
src/forward_qwen.cu        the engine: every kernel + the c abi, one cuda tu
tools/serve_openai.py      single-stream server (prefix cache, tools, reasoning split)
tools/serve_batched.py     multi-client server (paged kv + continuous batching)
tools/serve.sh             launcher: env defaults, vram-sized TQ_CTX, daemon/stop/status
tools/convert_qwen_tqf.py  hf qwen --> .tqf converter
tools/inspect_tqf.py       inspect a .tqf
tools/bench_coding.py      cold/follow-up coding-workload benchmark
```

see `CHANGELOG.md` for the measurement log behind every number here.

## where this is going

1. port the wide-prefill gemm to the cutlass sm120 block-scaled collective --> prefill
   parity (~4400 tok/s), which flips the sustained-load column
2. profile and rework the deltanet chunkwise scan (22.5% of prefill)
3. re-measure quality against nvfp4 ourselves instead of inheriting the number
4. more models — the converter and the format are architecture-agnostic; the kernels
   are not, yet
