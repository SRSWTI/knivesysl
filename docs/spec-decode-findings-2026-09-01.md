# spec decode on sm120: measurements, and the instrument bug that inverted them

> **Read the first section before any number below.** An earlier revision of this file
> concluded that DSpark was a 0.67x slowdown and that "draft cost must be near-zero on
> this card." Both were artifacts of a broken measurement. They are wrong. The
> corrected numbers are the opposite sign.

## the instrument bug

`tools/bench_spec_matrix.py` counted `n_tok = len(times)` -- one entry per SSE event.
Under speculative decoding a server emits `accept_len` tokens **per event**, so the
probe was reporting *events/second* and calling it tokens/second. Every speculative
cell ever measured with it was undercounted by the accept factor (2-4x). Non-speculative
cells are unaffected (one token per event) and remain valid.

Caught by comparing against the engine's own counter on an identical request:

| | client probe | engine `x_knivesysl.gen_tok_s` | tokens returned |
|---|--:|--:|--:|
| streaming | 44.8 "tok/s" | 108.2 | 58 events |
| non-streaming | -- | 104.5 | **128 tokens** |

Independently confirmed on the sglang side: same prompts, near-identical output
lengths, plain emitted 160 deltas averaging 3.0-5.9 chars while dspark emitted 43-79
deltas averaging 9.0-13.6 chars -- and the chars/delta ratio reproduced the measured
accept length to within noise.

**Fix:** request `stream_options.include_usage` and take `completion_tokens` from the
server. ITL percentiles stay per-EVENT, which is what a client actually perceives
(speculation makes token arrival bursty, and that is a real property, not an artifact).

## corrected numbers (fixed tool, one rtx 5090, same day)

| config | 2048:1 | 8192:1 | tf-top1 |
|---|--:|--:|--:|
| ours paged nvfp4, plain | 58.5 | 60.5 | 85.78 |
| **ours paged nvfp4 + ngram** | **87.8** (1.50x) | **142.3** (2.35x) | 85.78 |
| ours single-stream fp6 + mtp | **119.5** | -- | **91.30** |
| sglang plain nvfp4 | 75.2 | 74.3 | 85.78 |
| sglang + dspark | ~162 (code prompt) | -- | 85.78 |

- **n-gram is a 1.5-2.35x win and the win GROWS with depth** (tok/round 3.45-4.17). It
  was previously shipped OFF by default on the strength of the broken numbers.
- **DSpark works.** Measured accepts 2.08 (prose) to 3.80 (math), mean 2.85; real
  throughput ~106-195 tok/s against sglang's own 72.6-76.9 plain -- roughly **1.9x
  mean**, which substantially reproduces the model card's published 2.25-3.16x at c=1.
- **fp6 + mtp single-stream is the standout combined point**: 119.5 tok/s at 91.30
  quality, versus sglang plain's 75.2 at 85.78 -- faster *and* +5.5 quality points.

## what the bug also invalidated

- the SpecMatrix phase verdict ("v2 delivers 0.33-0.73x of plain")
- shipping paged speculation OFF by default (`19f4815`)
- **the depth gate** `TQ_PAGED_SPEC_MAXPOS=65536` (`7cb005a`) -- tuned to *disable*
  deep speculation because deep cells looked like regressions. They were not. The
  deepest cell measured after the fix shows the largest win (2.35x at 8k), so the
  gate is now suppressing our best numbers.
- the derived "paged round costs 2.92 decode steps, no drafter can win"

## what still stands (measured with speculation OFF)

- sglang plain leads our paged plain by 1.09-1.38x per-request across the ctx x n grid
- TTFT under concurrency: ours 1.73s / 3.51s at 8192 n=2 / n=4 against their 0.71s /
  1.49s -- a ~2.4x prefill deficit that a `--max-prefill` sweep did **not** fix
  (4 concurrent prefills: 1.73s/3.56s, unchanged), because the wide prefill wave is
  segment-count-sensitive by design: `128 cols as 1 segment = 42.8 ms, as 32 segments
  = 73.5 ms`
- fp6 costs 0.87x of nvfp4 decode (55.9 vs 64.2 mean over six workloads) while
  achieving *higher* effective bandwidth (1258 vs 1162 GB/s) -- the nvfp4 W4A4 path
  has overhead eating into its byte savings
- dspark's VRAM bill: draft weights 3.64 GB + 1.12 GB d=8 verify intermediates, taking
  `max_total_num_tokens` from 31,219 to 11,666 on the same pins
- the quality ladder: fp8 95.94 > fp6 91.30 > e2m1 86.46 > nvfp4 85.78

## activation precision, for the record

Activations are already E4M3 (fp8) on the fp6 and e2m1 tiers; **nvfp4 is the only tier
that drops them to 4-bit**, and that is exactly what buys it the k64 `mxf4nvf4`
instruction. `TQ_W_E2M1` *is* W4A8. A16 activations cannot enter the tensor-core path
at all (`mxf8f6f4`/`mxf4nvf4` require every operand from the f8/f6/f4 family).

The lever is weak: A4->A8 is worth <=0.68 points (85.78 -> 86.46, and that conflates
scale structure), while W4->W6 is worth 4.84 (86.46 -> 91.30). **Weight precision
dominates activation precision by ~7x.**

One structural gap: `gc_weight_e2m3` is a single `__constant__` set once from a
file-level flag, so the fp6-vs-fp8 weight base is all-or-nothing. `TQ_W_NVFP4` and
`TQ_W_E2M1` are per-tensor, so tensors can be dropped 6->4 bits individually, but no
tensor can be *raised* to 8. unsloth (mixed W8A8+W4A4) and RadixArk (ModelOpt
`MIXED_PRECISION`) both ship hetero-with-8-bit; we ship hetero-down-from-6-bit.
Making 8-bit expressible per tensor -- fp8 on the SNR-worst `down_proj`/`linear_out`
only -- is the bounded engine change that would beat their mixed exports on
quality-per-byte.

## reproducing the sglang reference on this box

Five distinct failures before it served:

1. `FileNotFoundError: 'ninja'` -- launching via an absolute venv path leaves the venv
   off `$PATH`, so the jit's `subprocess("ninja")` misses. Put the venv and
   `/usr/local/cuda/bin` on `PATH`.
2. **Host** oom, victim `cicc` -- flashinfer's jit fans ninja across all 32 threads at
   ~5.5 GB rss each, exceeding 59 GB of system ram. Cap with `MAX_JOBS=6
   NVCC_THREADS=1`. No python traceback; look in `/var/log/kern.log`.
3. GPU oom during cuda-graph capture at `avail_mem=0.66 GB`.
4. `ValueError: Loaded weights leave no GPU memory for the KV cache` -- lowering
   `--mem-fraction-static` is backwards: the static budget covers weights + state + kv,
   while graph capture allocates from what is left *outside* it.
5. `RuntimeError: mat1 and mat2 shapes cannot be multiplied (7x5120 and 2560x248320)`
   -- the draft projects through the *target's* lm_head, and an fp4-packed head needs
   `quant_method.apply`, not a matmul. Pinned commit `1cf2b8c` adds exactly that; the
   pypi wheel 0.5.18 does not have it. Installing the pinned tree needs
   `SGLANG_BUILD_RUST_EXTS=none` (setup.py otherwise demands cargo for router exts).

Sizing note: `--mamba-full-memory-ratio 5.61` grossly over-provisions the gdn state
pool here. Pinning `--max-mamba-cache-size` explicitly (concurrency x S, S=4 under
`extra_buffer_lazy`) moved kv from 31,219 to 168,487 tokens at n=4. Prefer the pin.

## benching hygiene learned the hard way

`tools/serve_prod.sh` is a **restart wrapper**. `pkill -f serve_batched` alone lets it
resurrect production, and a bench then measures PRODUCTION while labelling the results
as whatever config it thought it started. `tools/bench_all_tiers.sh` now kills the
wrapper first and asserts the live server's tier and spec state against what was
requested before it will record a single cell.

Also: `$w` cannot expand into an environment assignment -- bash parses assignment
prefixes before expansion, so an expanded `VAR=VAL` becomes a command name. Use `env`.
