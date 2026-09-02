# knivesysl

**a bare-cuda inference engine for qwen3.8-27b on one rtx 5090: knivesysl fp6 or
nvfp4 weights, q4 kv, 256k context, and hand-written sm120 tensor-core kernels.
no pytorch, cublas, cutlass, or flashinfer on the steady-state run path — one cuda
translation unit behind a c ctypes abi.**

knivesysl is two things:

- **the engine** — the cuda implementation is one translation unit loaded by python
  over ctypes. python tokenizes, schedules, and speaks http; model execution stays in
  `libforward_qwen.so`.
- **the format family** — three supported service layouts: knivesysl fp6/e2m3,
  nvfp4 w4a4 for every projection, and nvfp4-mlp with the fp6 path retained outside
  the mlp. q4 k plus e4m3 v provides the long-context kv tier.

the current target is the qwen3.8-27b text tower: 64 layers, hidden size 5120,
16 full-attention layers, and 48 gated-deltanet layers. the kernels are deliberately
sm120-specific. portability is not the goal; owning the blackwell hot path is.

```
qwen3.8-27b (hf)
   --> convert_qwen_tqf.py --> model.tqf
   --> libforward_qwen.so   --> one cuda tu, compute_120f
   --> serve_openai.py      --> single-stream service
    or serve_batched.py     --> paged kv, continuous batching, apc, optional n-gram
   --> /v1/chat/completions
```

---

## formats

the supported service formats are intentionally narrow:

| format | role | quality status |
|---|---|---:|
| **knivesysl fp6 (e2m3)** | quality tier; e4m3 activations | **91.30 tf-top1 vs bf16** |
| **nvfp4-all** | default performance/capacity tier; w4a4 on every projection | **85.78 tf-top1 vs bf16** |
| **nvfp4-mlp** | hybrid tier; nvfp4 mlp with fp6 outside it | separate end-to-end quality gate pending |

fp6 uses six-bit e2m3 values with a power-of-two scale per 128 weights. nvfp4 uses
four-bit values with a scale per 16. fp6 spends about 1.4x the weight bytes for more
mantissa; nvfp4 reaches the sm120 k64 fp4 instruction and leaves more room for kv,
slots, and checkpoints. the quality figures above carry over from the project fork
and remain labelled until our independent quality campaign is complete.

---
## current benchmark

this is the canonical performance snapshot. older decode, prefill, and speculative
charts were removed because they mixed superseded kernels, cache policy, generation
lengths, or event counts. raw schema-3 artifacts retain every repetition.

**matched contract**

- one rtx 5090: gb202, sm120, 170 sms, 32 gb, 128 mb l2; cuda 13.3, driver 595;
- exact tokenizer-constructed prompts; repetitive code-continuation workload;
- 512 generated tokens per request, temperature 0, three repeats; table values are medians;
- prefix caching disabled on every engine;
- sglang is the pinned clean reference with radix cache disabled;
- vllm serves `qwen38-27b-nvfp4-radixark` with default compiled/graph execution,
  fp8 kv, `--no-enable-prefix-caching`, and a harness warmup before measurement;
- vllm's prefix-cache query, hit, and external-hit counters remained exactly zero
  after the full campaign;
- knivesysl plain is nvfp4-all with the staged v2 paged-attention kernel;
- knivesysl boosted is the same target path plus a 16-node n-gram verify archive.

`128k*` is the only non-identical prompt length: sglang and knivesysl use 131,072
prompt tokens; vllm uses 130,496 so its 512-token completion stays inside
`--max-model-len 131072`.

### throughput

each cell is `aggregate decode / per-request decode / end-to-end aggregate`, in tok/s.

| context | n | sglang plain | vllm plain | knivesysl plain | knivesysl boosted (n-gram) | aggregate winner |
|---:|---:|---:|---:|---:|---:|---|
| 2k | 1 | 71.7 / 71.7 / 70.2 | **71.7 / 71.7 / 70.2** | 65.8 / 65.8 / 64.1 | 68.6 / 68.6 / 66.5 | sglang ~= vllm |
| 2k | 2 | 120.5 / 60.8 / 118.1 | 130.3 / 65.5 / 126.4 | 125.9 / 63.7 / 122.9 | **130.8 / 125.9 / 127.5** | **knivesysl boosted** |
| 2k | 4 | 229.2 / 58.9 / 224.6 | 246.0 / 62.9 / 234.6 | 222.5 / 57.9 / 217.7 | **247.5 / 88.9 / 241.6** | **knivesysl boosted** |
| 8k | 1 | 69.2 / 69.2 / 63.2 | 72.1 / 72.1 / 65.7 | 64.7 / 64.7 / 57.7 | **146.4 / 146.4 / 116.1** | **knivesysl boosted, 2.03x vllm** |
| 8k | 2 | 111.0 / 57.7 / 102.6 | **123.2 / 63.1 / 109.6** | 110.1 / 58.3 / 100.6 | 120.9 / 98.2 / 109.2 | vllm; boosted -1.9% |
| 8k | 4 | 112.3 / 58.2 / 107.9 | **204.8 / 57.6 / 185.6** | 177.9 / 51.4 / 165.6 | 168.4 / 57.5 / 156.3 | vllm |
| 32k | 1 | 66.9 / 66.9 / 44.0 | 69.8 / 69.8 / 46.6 | 61.1 / 61.1 / 39.7 | **103.2 / 103.2 / 53.1** | **knivesysl boosted, 1.48x vllm** |
| 32k | 2 | 77.8 / 47.5 / 59.2 | **87.3 / 52.3 / 65.0** | 68.6 / 43.0 / 51.7 | 61.3 / 43.6 / 48.0 | vllm |
| 32k | 4 | 70.8 / 47.2 / 62.1 | **103.2 / 39.7 / 85.9** | 80.2 / 30.1 / 68.3 | 72.5 / 29.8 / 61.8 | vllm |
| 64k | 1 | 63.0 / 63.0 / 26.6 | 65.9 / 65.9 / **29.0** | 58.7 / 58.7 / 25.0 | **66.0 / 66.0 / 25.8** | boosted ~= vllm decode |
| 64k | 2 | 49.1 / 38.7 / 32.1 | **50.8 / 39.1 / 32.8** | 43.3 / 33.2 / 28.8 | 39.4 / 29.6 / 27.0 | vllm |
| 64k | 4 | — | **41.0 / 32.7 / 33.6** | 38.9 / 18.3 / 31.7 | — | vllm; plain -5.1% |
| 128k* | 1 | **58.0 / 58.0 / 11.6** | 57.8 / 57.8 / **12.6** | 50.1 / 50.1 / 11.1 | 46.3 / 46.3 / 11.0 | sglang decode; vllm e2e |
| 128k* | 2 | — | **20.8 / 57.9 / 12.7** | 20.1 / 22.0 / 11.9 | — | vllm; plain -3.4% |

### aggregate decode versus vllm

| context | n | sglang / vllm | knivesysl plain / vllm | knivesysl boosted / vllm |
|---:|---:|---:|---:|---:|
| 2k | 1 | 1.00x | 0.92x | 0.96x |
| 2k | 2 | 0.92x | 0.97x | **1.00x** |
| 2k | 4 | 0.93x | 0.90x | **1.01x** |
| 8k | 1 | 0.96x | 0.90x | **2.03x** |
| 8k | 2 | 0.90x | 0.89x | **0.98x** |
| 8k | 4 | 0.55x | **0.87x** | 0.82x |
| 32k | 1 | 0.96x | 0.88x | **1.48x** |
| 32k | 2 | 0.89x | 0.79x | 0.70x |
| 32k | 4 | 0.69x | **0.78x** | 0.70x |
| 64k | 1 | 0.96x | 0.89x | **1.00x** |
| 64k | 2 | 0.97x | 0.85x | 0.78x |
| 64k | 4 | — | **0.95x** | — |
| 128k* | 1 | **1.00x** | 0.87x | 0.80x |
| 128k* | 2 | — | **0.97x** | — |

### time to first token

each cell is `maximum / median` client ttft in seconds.

| context | n | sglang | vllm | knivesysl plain | knivesysl boosted |
|---:|---:|---:|---:|---:|---:|
| 2k | 1 | 0.16 / 0.16 | **0.15 / 0.15** | 0.20 / 0.20 | 0.20 / 0.20 |
| 2k | 2 | 0.32 / 0.25 | 0.32 / 0.28 | 0.40 / 0.30 | 0.40 / 0.30 |
| 2k | 4 | **0.64 / 0.41** | 0.65 / 0.63 | 0.80 / 0.60 | 0.80 / 0.60 |
| 8k | 1 | 0.72 / 0.72 | **0.70 / 0.70** | 0.93 / 0.93 | 0.91 / 0.91 |
| 8k | 2 | 1.45 / 1.10 | **1.38 / 1.21** | 1.85 / 1.38 | 1.80 / 1.33 |
| 8k | 4 | 11.61 / 1.81 | **2.74 / 2.22** | 3.58 / 2.43 | 3.76 / 2.51 |
| 32k | 1 | 3.97 / 3.97 | **3.67 / 3.67** | 4.60 / 4.60 | 4.68 / 4.68 |
| 32k | 2 | 8.10 / 6.12 | **7.33 / 5.68** | 9.84 / 7.36 | 9.39 / 7.00 |
| 32k | 4 | 25.34 / 9.94 | **14.63 / 10.27** | 18.10 / 11.51 | 18.75 / 11.79 |
| 64k | 1 | 11.14 / 11.14 | **9.87 / 9.87** | 11.70 / 11.70 | 12.05 / 12.05 |
| 64k | 2 | 22.21 / 16.64 | **21.40 / 16.25** | 23.91 / 17.89 | 23.88 / 17.87 |
| 64k | 4 | — | 51.06 / 30.75 | **48.54 / 30.56** | — |
| 128k* | 1 | 35.14 / 35.14 | **32.11 / 32.11** | 35.77 / 35.77 | 35.55 / 35.55 |
| 128k* | 2 | — | 71.56 / **51.52** | **70.58** / 52.80 | — |

### inter-token latency

each cell is p50 / p99 milliseconds. large batched p99 values include inter-client
scheduling gaps; they are not one kernel's execution time.

| context | n | sglang | vllm | knivesysl plain | knivesysl boosted |
|---:|---:|---:|---:|---:|---:|
| 2k | 1 | 13.9 / 14.7 | **13.7 / 15.2** | 15.2 / 15.7 | 16.1 / 24.8 |
| 2k | 2 | 16.2 / 17.8 | **15.0 / 17.0** | 15.5 / 31.2 | 21.1 / 74.2 |
| 2k | 4 | 16.5 / 17.6 | **15.6 / 17.4** | 16.5 / 33.0 | 20.9 / 82.5 |
| 8k | 1 | 14.1 / 15.8 | **13.7 / 15.3** | 16.6 / 51.4 | 22.4 / 64.0 |
| 8k | 2 | 16.6 / 35.7 | 16.2 / 33.8 | **16.3 / 18.3** | 30.7 / 398.3 |
| 8k | 4 | 16.9 / 33.4 | **16.0 / 17.9** | 17.0 / 394.1 | 27.2 / 475.4 |
| 32k | 1 | 14.8 / 16.5 | **14.2 / 15.8** | 16.1 / 18.5 | 17.8 / 41.4 |
| 32k | 2 | 17.8 / 19.3 | **16.3 / 18.1** | 20.1 / 536.5 | 42.7 / 684.5 |
| 32k | 4 | 18.4 / 19.9 | **17.9 / 904.3** | 23.6 / 660.7 | 40.1 / 709.3 |
| 64k | 1 | 15.5 / 33.1 | **14.9 / 16.8** | 17.0 / 18.6 | 18.7 / 53.2 |
| 64k | 2 | 18.9 / 20.5 | **18.4 / 985.0** | 22.9 / 834.6 | 63.1 / 965.0 |
| 64k | 4 | — | **19.4 / 1042.1** | 31.9 / 1040.3 | — |
| 128k* | 1 | 17.2 / 18.9 | **16.9 / 19.7** | 19.9 / 22.5 | 21.5 / 83.3 |
| 128k* | 2 | — | **17.5 / 20.0** | 31.2 / 1545.0 | — |

### prefill throughput and total wall time

each cell is `estimated prefill tok/s / total wall seconds`.

| context | n | sglang | vllm | knivesysl plain | knivesysl boosted |
|---:|---:|---:|---:|---:|---:|
| 2k | 1 | 13,136 / 7.30 | **13,251 / 7.29** | 10,120 / 7.99 | 10,102 / 7.70 |
| 2k | 2 | 12,870 / 8.67 | **12,893 / 8.10** | 10,341 / 8.33 | 10,159 / 8.03 |
| 2k | 4 | **12,820 / 9.12** | 12,660 / 8.73 | 10,210 / 9.41 | 10,192 / 8.48 |
| 8k | 1 | 11,326 / 8.10 | **11,780 / 7.79** | 8,838 / 8.88 | 9,034 / **4.41** |
| 8k | 2 | 11,330 / 9.98 | **11,874 / 9.34** | 8,856 / 10.18 | 9,111 / 9.38 |
| 8k | 4 | 2,822 / 18.98 | **11,975 / 11.03** | 9,145 / 12.37 | 8,720 / 13.10 |
| 32k | 1 | 8,245 / 11.65 | **8,930 / 10.99** | 7,129 / 12.89 | 7,009 / **9.64** |
| 32k | 2 | 8,093 / 17.31 | **8,942 / 15.75** | 6,663 / 19.79 | 6,982 / 21.31 |
| 32k | 4 | 5,173 / 32.97 | **8,957 / 23.85** | 7,243 / 30.01 | 6,990 / 33.13 |
| 64k | 1 | 5,882 / 19.26 | **6,638 / 17.64** | 5,599 / 20.51 | 5,438 / 19.82 |
| 64k | 2 | 5,900 / 31.91 | **6,126 / 31.20** | 5,482 / 35.50 | 5,490 / 37.96 |
| 64k | 4 | — | 5,134 / **60.87** | **5,400** / 64.54 | — |
| 128k* | 1 | 3,730 / 43.97 | **4,064 / 40.67** | 3,665 / 45.95 | 3,687 / 46.47 |
| 128k* | 2 | — | 3,647 / **80.44** | **3,714** / 85.73 | — |

### n-gram behavior

| context | n | committed tokens | verify rounds | tokens/round | aggregate speedup vs plain |
|---:|---:|---:|---:|---:|---:|
| 2k | 1 | 137 | 65 | 2.11 | 1.04x |
| 2k | 2 | 668 | 129 | 5.18 | 1.04x |
| 2k | 4 | 1,747 | 255 | 6.88 | 1.11x |
| 8k | 1 | 475 | 110 | 4.32 | **2.26x** |
| 8k | 2 | 783 | 121 | 6.47 | 1.10x |
| 8k | 4 | 1,801 | 270 | 6.69 | 0.95x |
| 32k | 1 | 428 | 86 | 4.98 | **1.69x** |
| 32k | 2 | 649 | 152 | 4.09 | 0.89x |
| 32k | 4 | 1,698 | 273 | 6.22 | 0.90x |
| 64k | 1 | 322 | 93 | 3.46 | 1.12x |
| 64k | 2 | 844 | 170 | 4.70 | 0.91x |
| 128k* | 1 | 337 | 102 | 3.30 | 0.92x |

the plain engine is close, not yet ahead: vllm leads single-request decode by about
8-13%, unique prefill by about 10-25%, and the difficult 32k multi-request decode
cells by 21-22%. the gap narrows to 3-5% at the measured 64kx4 and 128kx2 capacity
edge. the n-gram path is a workload-specific accelerator, not a blanket claim:
it wins the repetitive 8k/32k single-agent cells, reaches shallow batch parity,
and loses where verification traffic or archive pressure outweighs acceptance.

unrounded data and every schema metric:
`results/comparisons/core-v2-sglang-vllm-knivesysl-plain-ngram-all-metrics.json`.

### apc — checkpointed prefix reuse

the hybrid cannot reuse kv alone at arbitrary boundaries because deltanet state is
not rewindable. an apc entry therefore owns refcounted full kv blocks, a private
partial tail, and a copy of the recurrent and convolution state. checkpoints are
saved at the lcp junction with the previous prompt and near the completed prompt;
admission adopts the deepest match and prefills only the suffix. same-prefix requests
arriving during a donor's prefill can wait briefly and adopt its checkpoint rather
than racing another full prefill.

the resident state slabs are allocated as one pool (`TQ_CKPT_POOL`, default six
151.5 mb slabs). every allocation, save, evict, and adopt failure degrades to a plain
full prefill.

measured 2026-08-31 on the production build, with `--no-prefix-cache` as the control:

| scenario | apc off | apc on | result |
|---|--:|--:|--:|
| turn append, 36k -> 50.4k context (+2.9k/turn) | 5.97-8.22 s | 0.61-0.78 s flat | 9-11x |
| session resend, all six depths | 5.6-8.2 s | **0.071-0.085 s** | **75-98x** |
| six-way fan-out, 33.2k shared, cold | wall 24.6 s | wall 5.26 s | 4.7x |
| six-way fan-out, checkpoint pre-exists | wall 24.6 s | **wall 0.67 s** | **37x** |
| decode on adopted state | 58.0 tok/s cold | 58.1 tok/s | no penalty |

adopted output is bit-identical at temperature 0.

![ttft vs depth](docs/apc/ttft_vs_depth.svg)
![speedups](docs/apc/speedups.svg)
![fan-out ladders](docs/apc/fanout_ladder.svg)
![prefill throughput](docs/apc/prefill_curve.svg)
![decode at depth](docs/apc/decode_depth.svg)

compared with dense block-aligned reuse, apc retains more sessions because its durable
state is the hybrid checkpoint rather than a dense kv image. the honest loss is a
previously unseen mid-context divergence: until that junction has been observed as an
admission boundary, the request pays a full prefill. apc prevents repeated prefill;
it does not move the weight-read roofline.

![vs vllm](docs/apc/vs_vllm.svg)
![the honest loss](docs/apc/divergence_loss.svg)

production uses `tools/serve_prod.sh`, a restart wrapper with core dumps enabled.
the engine exits after consecutive step failures instead of serving through a poisoned
cuda context.

---
## how it works

- **weights** — knivesysl fp6/e2m3 or nvfp4 w4a4, packed directly in the fragment
  layout consumed by the sm120 tensor-core kernels.
- **kv cache** — rotated asymmetric int4 k with hadamard preprocessing, e4m3 v,
  and per-row scales in a refcounted physical block pool.
- **attention** — 16 full-attention layers use owned wide-prefill and three-stage
  paged-decode kernels; gqa heads share each kv read.
- **deltanet** — 48 gated-deltanet layers use a chunk-64 wy/ut transform and an
  fp32/tf32 recurrent scan.
- **activation pipeline** — rms and silu application fuse into nvfp4 quantization
  where consumer contracts permit it.
- **speculation** — optional server-side n-gram chains are verified by one fused
  target-model wave; shallow exactness and deep cost gates select the plain path.
- **batching** — continuous scheduling over paged kv, with decode rows optionally
  riding prompt waves and apc checkpoints sharing immutable full blocks.

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
| `TQ_CTX` | max context (engine cap 262144; 131072 serves in 26.6 GB on the nvfp4 tier) |
| `TQ_KV_Q4=1` | 4-bit k + e4m3 v cache (needed for 256k) |
| `TQ_EMBED_FP8=2` | 6-bit embed table, -1.5 gib |
| `TQ_PAGED_SPLIT` | force split-k on/off for paged decode attention |
| `TQ_PAGED_GQA=0` | revert paged decode attention to one cta per query head |
| `TQ_WIDE_GEMM=0` | revert the wide fp6 projection gemm to the 1-warp/cta kernel |
| `TQ_GEMM_STAGES` | wide-gemm `cp.async` pipeline depth (2..4, default 2) |
| `TQ_WAVE_MAX` | max wave columns the engine accepts (default 2048; builders pick 512 shallow / 2048 past 16k depth) |
| `TQ_W_E2M1=1` | opt-in 4-bit weight tier (k32 mma: memory win only, no compute win) |
| `TQ_W_NVFP4=all` | nvfp4 w4a4 tier, every projection (k64 mma: 2.05x instruction roof) |
| `TQ_W_NVFP4=mlp` | nvfp4 for the mlp only; attention + deltanet stay fp6 |
| `TQ_NVFP4_STAGES` | nvfp4 gemm pipeline depth fallback (autotuned per shape at load) |
| `TQ_NVFP4_TMA=0` | revert the nvfp4 gemm from `cp.async.bulk.tensor` to `cp.async` |
| `TQ_WIDE_ATTN_MMA=0` | revert prefill attention to the scalar/split-k decode kernel |
| `TQ_WIDE_ATTN_QROWS` | force 16/32/64-row per-head attention tiles (only when gqa is off) |
| `TQ_WIDE_ATTN_GQA=0` | revert to per-query-head attention ctas (default on: one cta per kv head, 96 packed rows) |
| `TQ_WIDE_ATTN_SPLIT` | key-split chunk for the prefill attention (default 8192, S capped at 8) |
| `TQ_WIDE_ATTN_PROBE` | timing scaffolds for the gqa kernel (WRONG numerics; cost attribution only) |
| `TQ_DN_MM` | deltanet chunk-64 matmul scan: 3 = tf32 wmma (default, gates at the eps band), 1 = fp32 scan (conservative), 0 = ck8 head-split scan |
| `--fuse-idle-ms` | ride a decode row on a prompt wave only after this idle time (default 125; riding costs 1.2 ms/row/wave) |
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

the immediate goal is plain-engine parity and then a durable overtake. the working
campaign is maintained in [`docs/level-up.md`](docs/level-up.md).

the core path is deliberately profile-driven: remove global nvfp4 split-k partial
round trips, fuse projection epilogues into residual/rms/quant publication, integrate
current-row attention preparation, and remeasure the complete matrix after every
accepted stage. paged mtp and durable apc are optional increments after the plain
path meets its own gates; neither is counted as proof that plain execution is fast.
