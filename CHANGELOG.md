# Changelog

All performance numbers are measured on this machine: RTX 5090 (GB202, SM120, 170 SM,
32 GB, 128 MB L2), CUDA 13.3, driver 595, Qwen3.8-27B FP6 (E2M3) + Q4 KV, greedy,
thinking off unless stated. vLLM comparisons are vLLM 0.27.1 serving
`unsloth/Qwen3.8-27B-NVFP4` (mixed W8A8 + W4A4, 22.5 GB) at `--max-model-len 16384
--gpu-memory-utilization 0.90`, driven by the same client.

## Unreleased

### Engine (`src/forward_qwen.cu`)

- **Two-pass wide activation quantizer** (`k_tq_bfrag_absmax_wide` /
  `k_tq_bfrag_quant_wide`). The fused quantizer cached the whole `[nvar x 128]` K-block
  in shared memory (`128 * nvar * 4` bytes) to read `x` once; that hit 64 KB at
  nvar=128 and past the SM120 per-block limit beyond it. This was the true origin of
  the 128-column prefill-wave cap — the launch failure surfaced later as a sticky
  `-94` from the attention step, which sent earlier investigation down the wrong path.
  The split version reduces the block absmax in pass 1 and quantizes in pass 2, so
  shared memory is O(1) in the column count. `x` is read twice, but the second read is
  an L2 hit (10 MB at nvar=512/K=5120 against 128 MB of L2).
  **Bit-exact**: `max` is order-independent, the pow2 scale formula is unchanged, and
  the same `tq_float_to_e4m3(x/s)` is applied to the same inputs. Verified with
  `qwn_quant_bfrag_check` at K in {5120, 17408} x N in {8, 64, 128, 129, 256, 384, 512,
  1024}: byte diff 0, scale diff 0 at every point. nvar <= 128 still takes the original
  fused kernel, so the established path is untouched.
- **`TQ_WAVE_MAX`** makes the paged prefill wave column cap runtime-tunable
  (`qwn_paged_prefill_batch`, default 128). Waves of 256/384/512 columns now run.
- **Batched decode rows inside fused waves.** A decode row is a 1-column final segment
  of `qwn_paged_prefill_batch` (token-identical to `qwn_paged_decode_step`, verified at
  every N). Routing each row through the per-segment MMA attention cost one
  `k_tq_wide_attn_mma` launch per row per attention layer — 32 rows x 16 layers = 512
  tiny launches per wave, which is what made fusion a measured net loss. Leading
  1-column segments now batch into one decode-attention launch.
  Effect: rows per decode step 9.8 -> 14.8, steps per window 386 -> 296.
- **Retuned `paged_split_S`** (split-K for paged decode attention). The old gate
  disabled split-K entirely from N>=15 (`blocks0 >= 2*sm`) and capped S at 32.

  | ctx / N | before | after |
  |---|--:|--:|
  | 16384 / 8 | 129.5 ms | **45.9 ms** |
  | 4096 / 8 | 46.0 ms | **25.1 ms** |
  | 4096 / 16 | 49.5 ms | **36.1 ms** |
  | 2048 / 16 | 37.3 ms | **29.1 ms** |
  | 2048 / 32 | 41.7 ms | 41.7 ms (already SM-saturated, S=1 correct) |
  | 512 / any | unchanged | unchanged (short-ctx single kernel preserved) |

  Split-K is a different reduction order, so it is eps-equivalent rather than
  bit-exact: **98.32% teacher-forced top-1 agreement** (13 flips / 776 positions) against
  the single-kernel path on real text. `TQ_PAGED_SPLIT=1` forces the single kernel.

### Batched server (`tools/serve_batched.py`)

- **Shared-prefix cache.** `qwn_paged_load_client` copies prefix KV rows *and* the
  DeltaNet recurrent/conv state into a slot, so a fleet sharing a system prompt prefills
  it once. **340 -> 784 tok/s** at N=32 (55 hits / 2 misses, 34k prompt tokens skipped),
  p50 latency 14.5 -> 6.2 s. The recurrent state is O(1) and post-prefix, so reuse
  requires an *exact* prefix match — there is no partial rewind for the 48 DeltaNet
  layers, hence one materialized prefix rather than an N-way hash-addressed cache. It
  dedups prefix *compute*, not memory. `--prefix-cache` / `--prefix-cache-min`.
- **Amortization-aware decode gate.** A decode step is one pass over ~20 GiB of weights
  whatever the row count (17.7 ms at N=1, 29.2 ms at N=32). Stepping between every
  prefill wave measured 285 vs 340 tok/s; never stepping starves decode outright under
  continuous arrival (**9 of 41 requests finished** in a 20 s guidellm window). Now gated
  on `--decode-min-rows` (8), `--decode-max-idle-ms` (250), or a drained queue.
- **True token streaming.** SSE was previously framed *after* generation completed, so
  TTFT/ITL from any streaming client were fiction. Now emits per committed token:
  TTFT 50 ms, ITL 16.9 ms measured.
- **`ignore_eos`**, **`stream_options.include_usage`**, and **`max_completion_tokens`**.
  The OpenAI *chat* endpoint sends `max_completion_tokens`; reading only `max_tokens`
  silently halved every requested output length (guidellm asked for 256, got our 128
  default).
- **Client-disconnect cancellation.** An abandoned streaming request now sets
  `Request.cancel` and the engine detaches the slot at the next step instead of decoding
  into a dead socket.
- **Listen backlog.** Python's `HTTPServer` defaults to `request_queue_size = 5`, so >=32
  simultaneous connects were reset by the kernel long before the scheduler was the
  limit. Now sized to the slot count.
- `--max-prefill` (2), `--fuse` / `--fuse-ratio`, `--wave-cols`, `--decode-every` knobs;
  `/health` reports prefix-cache hits/misses/tokens-saved and engine token counters.

### Single-stream server (`tools/serve_openai.py`)

- **Native reasoning-effort tiers.** `reasoning_effort` is now passed through to the
  chat template (`low` / `medium` / `xhigh`), which is where Qwen3.8 implements it —
  previously the tier was parsed and dropped, so every request silently ran the
  template default (`xhigh`). `high`/`max` fold onto `xhigh` and `med` onto `medium` so
  an OpenAI-vocabulary client gets an answer instead of the template's
  `raise_exception`. Accepted as `reasoning_effort`, `reasoningEffort`, or
  `reasoning.effort` on **both** endpoints (each previously ignored the other's shape).
  Measured, same prompt: low 1474 think chars / medium 1471 / xhigh 2710.
- **`preserve_thinking`** passthrough, and `/v1/responses` now feeds `reasoning` items
  back as `reasoning_content` instead of dropping them. A preserved-thinking follow-up
  is a strict extension of the committed context, so it takes the anchor path:
  **7808 of 8203 tokens reused, 0.23 s** vs 3.6 s cold.
- **`ignore_eos`**, and `--think-effort-caps` as an optional hard ceiling per tier
  (default off — the tier's prompt instruction alone decides depth).

### Tools

- **`tools/serve.sh`** — launcher with env defaults, VRAM-sized `TQ_CTX`, and
  `daemon` / `stop` / `status` / `logs`. `daemon` runs under `setsid` so a closed
  terminal cannot kill a request mid-flight. Preflight failures print the fix (cmake
  line, converter line) rather than a stack trace.
- `tools/bench_coding.py` and `tools/serve_smoke.py`: repo-relative corpus paths
  (`TQ_REPO`) instead of a hardcoded `/root/...`; the bench pins its thinking state
  (`TQ_BENCH_THINK`) since reasoning turns route timings through the rescue phase.

### Measured: knivesysl vs vLLM 0.27.1

Two methodologies, two different answers. Both are reported because the difference
*is* the finding.

**(a) Fixed batch, no arrivals** — N requests submitted at once, wait for all to finish.
This measures the **decode ceiling**, and we hold it:

| workload | N | knivesysl | vLLM |
|---|--:|--:|--:|
| decode-heavy (40-tok prompt, 320-512 gen) | 32 | **1007-1025** | 930 |
| decode-heavy | 48 | **1151** | 927 |
| decode-heavy | 52 | **1181-1189** | 993 |
| prefill-heavy, shared prefix (645-tok prompt) | 32 | **725-784** | 675 |

**(b) guidellm, continuous arrival** — N streams held in flight, a new request starts
the moment one finishes, 256 output tokens each, 20 s window, `ignore_eos`. This is what
production traffic looks like, and **we lose every cell, 0/11**:

| prompt | N | knivesysl | tuned | vLLM | best/vLLM | q ITL | v ITL |
|---:|--:|--:|--:|--:|--:|--:|--:|
| 128 | 1 | 50.4 | 55.2 | 66.6 | 0.83x | 17.8 | 14.9 |
| 128 | 8 | 272.0 | 287.2 | 392.1 | 0.73x | 21.3 | 16.5 |
| 128 | 32 | 427.0 | 462.5 | 774.1 | 0.60x | 38.8 | 18.2 |
| 1024 | 1 | 46.4 | 45.5 | 64.7 | 0.72x | 20.1 | 15.2 |
| 1024 | 8 | 180.9 | 130.7 | 321.0 | 0.56x | 35.3 | 18.0 |
| 1024 | 32 | 26.7 | 92.9 | 666.8 | 0.14x | 56.3 | 23.1 |
| 4096 | 1 | 36.5 | 36.8 | 59.5 | 0.62x | 19.7 | 15.5 |
| 4096 | 8 | 0 | 13.3 | 202.8 | 0.07x | 75.6 | 25.1 |
| 4096 | 16 | 0 | 0 | 246.4 | 0.00x | — | 32.8 |
| 16384 | 1 | 0 | 0 | 46.0 | 0.00x | — | 16.3 |
| 16384 | 4 | 0 | 0 | 58.4 | 0.00x | — | 37.8 |

The zero cells are not crashes. `p=4096 N=8` logged **1648 input tok/s, 34.8 output
tok/s**: the whole 20 s window went to prefill and nothing reached 256 output tokens.

Both tables have the same single cause. Under (a) prefill happens once and then decode
runs unobstructed, so our FP6 weight advantage shows. Under (b) prefill never stops, and
at 124 TFLOPS against 572 achievable it consumes the window. **No scheduling policy
compensates for a 4.8x GEMM deficit** — the tuning below moved cells by 8-248% and did
not flip a single one.

### Scheduler tuning (measured, not flipping the result)

- `--prefill-budget` (default 64): prompt columns per wave **on top of** the decode rows.
  vLLM schedules running requests first and lets prefill fill the remaining token budget
  (`v1/core/sched/scheduler.py`), but its budget is 8192 so decode rows are noise; ours
  was 128, so at N=32 the rows ate a quarter of every wave. Only expressible because the
  quantizer stopped capping waves at 128 columns.
- Wave cost is **linear in columns** (0.417 / 0.418 / 0.433 ms per column at 128 / 256 /
  512), so there is no per-wave fixed cost to amortize — wave size is purely a
  decode-latency vs prefill-progress dial, and the optimum moves with prompt length:

  | config | p=128 N=32 | p=1024 N=32 |
  |---|--:|--:|
  | wave_cols 128 (old) | 427 | 26.7 |
  | wave_cols 64 | 431 | **117.5** |
  | prefill_budget 96 | **577.8** | 53.8 |
  | prefill_budget 64 (default) | 462.5 | 92.9 |

### Open: prefill is compute-bound, and our GEMM runs at ~1/5 of the tensor cores

Widening prefill waves past 128 columns now works and changes nothing:
T=128 2396 tok/s, T=256 2392, T=512 2309 (TOKENS=512). The measurement that explains it:
a 128-column wave does 6.55 TFLOP in ~53 ms = **124 TFLOPS**, while the 20 GiB weight
read accounts for only ~12 ms (23%) of the wave. There is no bandwidth left to
amortize — the hand-written FP6 wide GEMM is simply slow.

How slow, measured directly: CUTLASS 4.7's SM120 block-scaled collective was compiled
for `mx_float8_t<float_e4m3_t>` x `mx_float6_t<float_e2m3_t>` (our exact numerics, one
enum changed from example 79c) and run at our real layer shapes on this GPU. Every
shape reports `Disposition: Passed`:

| GEMM (N x K) | CUTLASS M=128 | CUTLASS M=512 |
|---|--:|--:|
| mlp gate/up (17408 x 5120) | 522 TFLOPS | **640** |
| mlp down (5120 x 17408) | 181 | **641** |
| q_proj (12288 x 5120) | 409 | **594** |
| o_proj (5120 x 6144) | 182 | **631** |
| kv_proj (1024 x 5120) | 35 | 135 |
| **FLOP-weighted mix** | **255** | **572** |

Against our 124 TFLOPS that is **4.8x at M=512**, 3.6x at M=128 — and unlike our kernel,
CUTLASS *scales with wave width* (mlp_down 181 -> 641, 3.5x from M=128 to M=512), which
is what makes the now-unlocked wide waves worth having. The quantizer fix above is the
prerequisite that lets us feed it those widths.

So the next lever is GEMM efficiency, not scheduling. Integration requires: weights
repacked from our QMMA fragment layout to CUTLASS TN (activations row-major A, weights
column-major B), and our per-128 pow2 scales replicated 4x into per-32 MX slots
(lossless for a pow2 scale, ~0.6% more scale bytes) because the SM120 builder
hard-deduces `SFVecSize == 32`. FP6 operands need 96-byte alignment
(`cutlass/detail/layout.hpp:374-390`); tile `Shape<_128,_128,_128>`, cluster `1x1x1`.
References: `examples/79_blackwell_geforce_gemm/79c_*.cu`,
`test/unit/gemm/device/sm120_blockscaled_tensorop_gemm/sm120_bs_gemm_mxf6_mxf8_f32_f32.cu`.
Benchmark harness kept at `/tmp/cbench/bench_e2m3.cu` (one `sed` from upstream 79c).
