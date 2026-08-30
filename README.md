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

### prefill — the gemm deficit is closed

![gemm headroom](assets/gemm-headroom.svg)

the hand-written fp6 gemm used to run at **124 tflops** against cutlass 4.8's sm120
block-scaled collective at 255 (m=128) / 572 (m=512) on the identical numerics and
shapes. it ran one warp per cta with every operand arriving from global through `__ldg`.
it is now a proper tiled kernel: 128-row x 256-column block tile, 8 warps, k staged 128
at a time into a `cp.async` circular buffer, split-k on `grid.y`. flop-weighted over this
model's real projection mix:

| columns/wave | before | after | cutlass sm120 collective |
|---|--:|--:|--:|
| 128 | 88.9 tflops | **392.8** | 255 (its m=128) |
| 256 | 88.1 tflops | **477.7** | 572 (its m=512) |

at `k_splits == 1` it is **bit-exact** vs the kernel it replaces — verified with
exactly-representable operands so summation order cannot mask an indexing bug. the two
largest projections now run at 1.29-1.40 tb/s, i.e. against the dram roofline rather
than the tensor cores.

![prefill profile](assets/prefill-profile.svg)

the gemm was 55% of prefill and the 48 gated-deltanet layers' chunkwise scan 22.5%, so
the scan was rewritten too — recurrent state held in registers across the whole chunk
instead of re-read and re-written every 8 tokens, the per-token prep hoisted out of the
sub-chunk loop and parallelised, and the idle-lane substitution phase removed. 1.56-1.59x
at every chunk width.

```
prefill, single stream, 4096-token prompt:  2579 --> 4630 tok/s   (1.80x)
prefill, paged, 32 clients x 2048 tokens:   1198 --> 2216 tok/s   (1.85x)
```

with the nvfp4 w4a4 tier, tma staging and a measured split-k config on top, the same
4096-token single-stream prefill is now **5970 tok/s** and the paged path 5973 at n=1.

**and the server-level comparison has now been run against vllm's real production
config** -- same client, both engines over http. the earlier run used `--enforce-eager`,
which i wrongly believed this checkpoint forced; `--language-model-only` skips the vision
tower and vllm runs its full config, cuda graphs included, which is worth **2.3x** to its
decode:

`vllm serve unsloth/Qwen3.8-27B-NVFP4 --max-model-len 140000 --max-num-seqs 32
--gpu-memory-utilization 0.92 --kv-cache-dtype fp8 --max-num-batched-tokens 8192
--enable-prefix-caching --language-model-only`

| | vllm | knivesysl | |
|---|--:|--:|---|
| prefill 4096, unique, cold | 8095 | 5625 | 1.44x theirs |
| prefill 4096, unique, warm | 10723 | 5985 | 1.79x theirs |
| prefill 4096, shared prefix, warm | 37074 | **175071** | **4.72x ours** |
| decode n=1 (the only clean server figure) | 68.0 | 60.6 | 1.12x theirs |

**the honest summary: we win cached prefill by ~4.7x, and we are behind on cold prefill
(1.44x) and on single-stream decode (1.12x).** their kv is fp8 and ours is int4, so this is
not like-for-like on quality. the n>1 server decode cells are prefill-contaminated in both
engines and are not quoted; engine-level our paged decode is 445 tok/s at n=8 and 1198 at
n=32, so the decode *kernels* are competitive and the *scheduler* gives much of it back
under mixed load. what remains on prefill is the kernels: 54% of this card's measured 2051
tflop/s fp4 issue roof against cutlass's 68%, and a deltanet scan pinned at a structural
1536-warp ceiling.

### decode at depth — gqa-shared paged attention

the paged decode attention ran one cta per *query* head, and this model is 24 q heads
over 4 kv heads, so six ctas independently streamed the same kv rows. at 131k context
that was ~21 gib of kv traffic per step for a 3.5 gib working set — 27 of the 46 ms. one
cta now carries all six query heads that share a kv head:

| case | before | after | |
|---|--:|--:|--:|
| n=32, ctx 2048 | 41.98 ms | **29.36** | 1.43x |
| n=1, ctx 65536 | 31.31 | **21.93** | 1.43x |
| n=1, ctx 131072 | 46.53 | **27.83** | 1.67x |
| n=1, ctx 147456 | 50.06 | **29.24** | 1.71x |

6/6 argmax match vs single-stream q4 decode. the projection gemm inside a decode step
now sustains 1.56 tb/s = 87% of this card's dram peak, so decode's remaining headroom is
attention and the deltanet recurrence, not the weight read.

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

### terminal coding client

`tools/axe_vllm.py` runs the agentic terminal client against any live
OpenAI-compatible vLLM server. It defaults to `http://127.0.0.1:8000/v1`, discovers the
served model from `/v1/models`, streams thinking separately from the final response, and
preserves assistant/tool history so server-side prefix caching can reuse completed turns.

```bash
# use a Python environment containing openai, python-dotenv, and tiktoken
# (plus ddgs and trafilatura for the web_search tool)
/path/to/vllm/.venv/bin/python tools/axe_vllm.py /path/to/project

# optional endpoint/model overrides
VLLM_BASE_URL=http://127.0.0.1:8000/v1 VLLM_MODEL=your/model \
    /path/to/vllm/.venv/bin/python tools/axe_vllm.py /path/to/project
```

Thinking is enabled and printed under `thinking>` by default; `--no-thinking` explicitly
disables it. Interactive commands are `/model`, `/clear`, and `/quit`. `--prompt` runs one
non-interactive agent turn, including tool calls, for automation and smoke tests.

`web_search` searches DuckDuckGo via `ddgs`, scrapes the top result pages with
`trafilatura`, and sends the digest to the same model the agent is running on; the
tool output is that model's summary of the links and what it found (a non-streaming
completion on the same client, `enable_thinking` off). if the summarisation call fails,
the raw scraped digest is returned instead.

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
| `TQ_PAGED_GQA=0` | revert paged decode attention to one cta per query head |
| `TQ_WIDE_GEMM=0` | revert the wide fp6 projection gemm to the 1-warp/cta kernel |
| `TQ_GEMM_STAGES` | wide-gemm `cp.async` pipeline depth (2..4, default 2) |
| `TQ_WAVE_MAX` | prefill wave column cap (default 256 = the gemm's column tile) |
| `TQ_W_E2M1=1` | opt-in 4-bit weight tier (k32 mma: memory win only, no compute win) |
| `TQ_W_NVFP4=all` | nvfp4 w4a4 tier, every projection (k64 mma: 2.05x instruction roof) |
| `TQ_W_NVFP4=mlp` | nvfp4 for the mlp only; attention + deltanet stay fp6 |
| `TQ_NVFP4_STAGES` | nvfp4 gemm pipeline depth fallback (autotuned per shape at load) |
| `TQ_NVFP4_TMA=0` | revert the nvfp4 gemm from `cp.async.bulk.tensor` to `cp.async` |
| `TQ_WIDE_ATTN_MMA=0` | revert prefill attention to the scalar/split-k decode kernel |
| `--prefill-budget` | prompt columns per wave, on top of the decode rows |
| `--prefix-cache` | materialize a shared prompt prefix once (batched server) |

## verify

```bash
python3 tools/mtp_spec_smoke.py --prompt-tokens 1024 --gen 128 --tqf model.tqf ...
python3 tools/bench_rounds.py  --prompt-tokens 65536 --rounds 200 --tqf model.tqf ...
python3 tools/needle_check.py                      # long-context retrieval gate
python3 tools/paged_smoke.py                       # paged decode == single-stream argmax
python3 tools/bench_decode.py --cases 1:2048,32:2048,1:131072
python3 tools/tf_agreement.py --chunk 256          # then diff two runs position by position
TQ_W_NVFP4=mlp python3 tools/nvfp4_check.py       # nvfp4 vs fp64 decode of the same bytes
TQ_W_NVFP4=mlp python3 tools/nvfp4_quality.py     # then diff ARGMAX against a TQ_W_NVFP4=0 run
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
tools/bench_decode.py      paged decode sweep (concurrency x context depth)
tools/tf_agreement.py      teacher-forced top-1 agreement between two libs/configs
tools/bench_endpoint.py    http endpoint sweep (any engine, unique vs shared prefix)
tools/quant_study.py       quantizer error on W@x: repack vs direct vs rotations
tools/dump_activations.py  real activations for quant_study (synthetic data lies)
tools/nvfp4_check.py       nvfp4 numeric gate (fp64 decode of the packed bytes)
tools/nvfp4_quality.py     nvfp4 wide-path argmax agreement (tf_agreement can't reach it)
```

see `CHANGELOG.md` for the measurement log behind every number here.

## where this is going

1. re-run the guidellm sustained-arrival comparison against vllm. the gemm deficit that
   lost every cell of it is closed (124 --> 392-478 tflops, against the 255-572 cutlass
   reference) and prefill is 1.8x, but the comparison itself is not re-measured
2. tensorize the deltanet chunk transforms. the scan is 1.6x faster but still pure
   scalar fp32 fma, on regular 8x128 / 128x64 shapes, and it fills only 96 of 170 sms
   (48 value heads x 2 stripes) because the per-stripe prep is duplicated. hoisting the
   prep into its own kernel unlocks nstripe=4/8
3. fold the two-pass activation quantizer into one pass by holding the k-block in
   registers instead of shared memory (it is 5.8% of prefill and reads `x` twice), and
   fuse `silu_mul` + `add_vec` into the quantizer / norm that follow them (another ~4%)
4. re-measure quality against nvfp4 ourselves instead of inheriting the number
5. more models — the converter and the format are architecture-agnostic; the kernels
   are not, yet
