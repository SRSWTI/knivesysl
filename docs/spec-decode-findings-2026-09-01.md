# spec decode on sm120: what the measurements say

all numbers below measured 2026-09-01 on the one rtx 5090 (gb202, sm120, 32 gb), same
client, greedy, thinking off. the reference column is sglang at pinned commit
`1cf2b8c5` serving `RadixArk/Qwen3.8-27B-NVFP4` (modelopt mixed-precision, 20.13 gb
weights, fp8 kv) -- the cookbook's *verified* rtx 5090 cell, run unmodified.

## the headline: a trained 1.86b drafter is a net loss on this card

dspark (`RadixArk/Qwen3.8-27B-DSpark`, gamma=7, verify window 8, vanilla markov rank
256) against its own engine's plain decode:

| workload | sglang plain | sglang + dspark | ratio | accept len |
|---|--:|--:|--:|--:|
| math_reasoning | 76.4 | 51.2 | **0.67x** | 3.80 |
| code_repetitive | 72.6 | 49.1 | **0.68x** | 3.30 |
| factual_qa | 76.4 | 50.2 | **0.66x** | 2.92 |
| longctx_summary | 75.0 | 50.5 | **0.67x** | 2.52 |
| chat_instruct | 75.9 | 51.2 | **0.67x** | 2.50 |
| prose_novel | 76.9 | 51.0 | **0.66x** | 2.08 |

the accepts are real and match the model card's published range (2.1-3.8 here vs their
3.43 aggregate at temp 1.0). the *speedup* is not: it is a uniform 33% slowdown, and it
is flat at 0.67x whether accept is 2.08 or 3.80 -- the round cost is fixed and dominant,
so more accepts buy nothing.

**why it inverts.** speculation assumes verify compute is nearly free. that holds on an
h200 (~990 tf/s bf16 against 4.8 tb/s) where their 2.25-3.16x was measured; it does not
hold on a 5090 (~105 tf/s against 1.79 tb/s), which is compute-poor relative to its
bandwidth. a verify wave is expensive here.

**it also costs context.** same server, same pins, draft on vs off: `max_total_num_tokens`
11,666 -> 31,219. the draft is 3.64 gb of weights plus 1.12 gb of d=8 verify
intermediates.

## the rule this establishes: draft cost must be near-zero on sm120

everything measured on this card, ranked:

| drafter | draft cost | result |
|---|---|--:|
| mtp head (1 layer, shared trunk) | ~free | **2.04x** (86.8 tok/s single-stream, fp6 tier) |
| n-gram (ours, shipped) | zero | **1.57x** draftable / parity prose |
| dspark (1.86b, 5 layers, +3.64 gb) | huge | **0.67x** |

our own n-gram design -- zero draft compute, ema + depth gated so it can never be worse
than plain -- beats a professionally-tuned trained drafter on this hardware. that is not
luck; it is the only shape that fits the card.

`tools/bench_spec_matrix.py --ns 8` on our engine confirms the concurrency half:

| cell | plain /req | ngram /req | spec rounds |
|---|--:|--:|--:|
| 2048:8 | 37.3 | 39.4 | 1 |
| 8192:8 | 30.0 | 30.2 | 5 |

at n=8 speculation self-disables (active requests exceed `TQ_PAGED_SPEC_SLOTS`=4) and
lands at parity. the fallback is clean -- no penalty for being off.

## honest deficits vs sglang

matched cells, our production nvfp4 tier vs their verified plain config:

| cell | ours /req | sglang /req | gap | ours agg | sglang agg | gap | ours ttft | sglang ttft |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 2048:1 | 58.5 | 75.2 | 1.29x | 58.5 | 75.7 | 1.29x | 0.20 | 0.15 |
| 2048:2 | 48.6 | 63.7 | 1.31x | 99.4 | 112.5 | 1.13x | 0.42 | 0.51 |
| 2048:4 | 40.7 | 56.3 | 1.38x | 173.0 | 215.9 | 1.25x | 0.79 | 0.62 |
| 8192:1 | 60.5 | 74.3 | 1.23x | 60.5 | 74.9 | 1.24x | 0.88 | 0.73 |
| 8192:2 | 55.6 | 64.3 | 1.16x | 118.9 | 129.6 | 1.09x | 1.73 | **0.71** |
| 8192:4 | 45.2 | 50.3 | 1.11x | 223.9 | 248.8 | 1.11x | 3.51 | **1.49** |

sglang wins every decode cell by 1.09-1.38x. the gap narrows with depth (8k: 1.24 ->
1.09 -> 1.11) but never inverts. the worst column is ttft under concurrency -- 2.4x
behind at 8k -- which is the known prefill deficit amplified by their chunked-prefill
scheduler interleaving better than our wave planner.

the readme's "decode -- we win" is measured against vllm (58.7 tok/s single stream) and
stands as written. it was never measured against sglang; against sglang we lose on plain
decode. the one configuration that beats them is **our mtp path at 86.8 tok/s**, which
is why porting mtp off the fp6 single-stream tier onto the paged nvfp4 server is the
decode-competitiveness item, not dspark.

not like-for-like, in both directions: their weights are modelopt mixed-precision
(20.13 gb, some layers above 4-bit) against our all-nvfp4 w4a4 (18.1 gb); their kv is
fp8 against our int4+hadamard. we hold 268,800 kv tokens against their 168,487 at n=4.
their generations here were 128 tokens against 256 in our stored baseline.

## reproducing the reference on this box

five distinct failures before it served; all avoidable:

1. `FileNotFoundError: 'ninja'` -- launching via an absolute venv path leaves
   `/tmp/sglang-env/bin` off `$PATH`, so the jit's `subprocess("ninja")` misses. export
   the venv + `/usr/local/cuda/bin` on `PATH`.
2. **host** oom, victim `cicc` -- flashinfer's jit fans ninja across all 32 threads at
   ~5.5 gb rss each, which exceeds 59 gb of system ram. cap with `MAX_JOBS=6
   NVCC_THREADS=1`. (this is a host-ram kill with no python traceback; check
   `/var/log/kern.log`, not the server log.)
3. gpu oom during cuda-graph capture at `avail_mem=0.66 GB`.
4. `ValueError: Loaded weights leave no GPU memory for the KV cache` -- lowering
   `--mem-fraction-static` is backwards: the static budget must cover weights + state +
   kv, while capture allocates from what is left *outside* it.
5. `RuntimeError: mat1 and mat2 shapes cannot be multiplied (7x5120 and 2560x248320)` --
   the draft projects through the *target's* lm_head, and an fp4-packed head needs
   `quant_method.apply`, not a matmul. that is exactly what pinned commit `1cf2b8c5`
   ("support quantized target lm_head in the dflash2 selector") adds; the pypi wheel
   0.5.18 does not have it. installing the pinned tree needs
   `SGLANG_BUILD_RUST_EXTS=none` (setup.py otherwise demands cargo for the router exts).

one sizing note worth keeping: `--mamba-full-memory-ratio 5.61` grossly over-provisions
the gdn state pool on this card. pinning `--max-mamba-cache-size` explicitly
(concurrency x s, s=4 under `extra_buffer_lazy`) moved kv from 31,219 to 168,487 tokens
at n=4. prefer the explicit pin.
