# re-measurement plan (post instrument fix)

## why everything is being redone

`tools/bench_spec_matrix.py` counted `n_tok = len(times)` -- one entry per SSE event.
Under speculative decoding a single event carries `accept_len` tokens, so **every
speculative cell ever measured with this tool was undercounted by the accept factor**
(2-4x). Validated against the engine's own counter: the same request read 44.8 tok/s
through the old probe and 119.5 through the fixed one, while `x_knivesysl.gen_tok_s`
reported 108-132.

Non-speculative cells are unaffected (one token per event) and stay valid.

**Void, must be regenerated:** every spec number in `README.md`, all of
`docs/spec-decode-findings-2026-09-01.md` (incl. the DSpark verdict), the SpecMatrix
phase results, and the `TQ_PAGED_SPEC_MAXPOS=65536` depth gate -- which was *tuned* to
disable deep speculation on the strength of the bad numbers.

## the axes

| axis | values | notes |
|---|---|---|
| server | knivesysl paged (`serve_batched`) · knivesysl single-stream (`serve_openai`) · sglang | paged is production; MTP only exists single-stream |
| weight tier | nvfp4 (18.1 GB, tf-top1 85.78) · fp6 (22.5 GB, 91.30) | e2m1 (86.46) exists as a third point, out of scope unless asked |
| speculation | off · ngram · mtp · dspark | see feasibility below |
| context | 2048 · 8192 · 32768 · 65536 · 94208 · 131072 | |
| concurrency | 1 · 2 · 4 · 8 | n=8 needs its own server config |
| workload | repetitive (code) · prose · math · chat · factual · longctx | accept length swings 2.08-3.80 across these |
| prefix cache | off (clean decode) · on (APC hit path) | grid runs with it OFF; APC measured separately |

## what is physically impossible (state it, don't silently skip)

- **MTP x nvfp4** -- the `-120` guard: spec-forward reads FP6 fragments that the NVFP4
  repack frees. MTP is **fp6-only**.
- **MTP x concurrency** -- MTP lives in the single-stream server, which serialises
  requests behind one lock. MTP is **n=1 only** until the paged port exists.
- **dspark x knivesysl** -- not ported; sglang-only, and we are not porting it.
- **deep x wide cells** -- pool capped. nvfp4 pool 230k tokens, fp6 pool 153.6k:
  - 131072 fits n=1 only (both tiers)
  - 94208 fits n=1,2 on nvfp4; n=1 on fp6
  - 65536 fits n=1,2 on both; never n=4
  - 32768 fits n=1,2,4 on both
- **n=8** -- needs `--max-slots 8`, whose per-slot state only fits at reduced context.
  Separate config, shallow cells only.

## the runs

### A. knivesysl paged grid (the production path) -- 4 configs
`--max-slots 4`, prefix cache off, gen 192, ctx x n = 1,2,4 with clipping.

| # | tier | spec | blocks | feasible cells |
|---|---|---|--:|--:|
| A1 | nvfp4 | off | 1800 | 14 |
| A2 | nvfp4 | ngram | 1800 | 14 |
| A3 | fp6 | off | 1200 | 13 |
| A4 | fp6 | ngram | 1200 | 13 |

Depth gate lifted (`TQ_PAGED_SPEC_MAXPOS=100000000`) so deep speculation is measured
rather than suppressed.

### B. n=8 rungs -- 4 configs
`--max-slots 8`, `TQ_CTX=32768`, 800 blocks, cells 2048:8 and 8192:8, tiers x spec.

### C. single-stream MTP ceiling -- 1 config
fp6 + MTP, n=1, all six contexts. Establishes what the paged port is worth.

### D. sglang reference -- 4 boots
nvfp4 plain at n=1 / n=2 / n=4 (`--max-running-requests` is launch-pinned, and
`--max-mamba-cache-size` must be re-pinned per rung), plus dspark at n=1 as the
external speculation reference.

### E. workload sensitivity -- 1 pass
The six-prompt mix at 2048:1 and 8192:1 for every knivesysl config. Cheap, and it is
the only thing that exposes n-gram's prose collapse vs MTP holding accept.

### F. APC confirmation -- 1 pass
Cold vs append vs exact-resend vs 6-way fan-out on the production config. Previously
9-11x / 75-98x / 37x -- all measured with speculation OFF, so believed intact, but
worth one re-confirmation now that spec will be on by default.

## metrics per cell

per-request tok/s · aggregate tok/s · TTFT · ITL p50/p99 (per *event* -- bursty under
spec, and that is what a user perceives) · tok/round · derived prefill tok/s ·
`max_total_num_tokens` for the config.

## cost

~14 server boots, ~60-75 min of GPU. Production goes down in windows and is restored
by the driver script at the end of each phase.

## open decisions

1. include the **e2m1** 4-bit tier as a third quality point (86.46)?
2. sglang at concurrency (3 boots) or n=1 only (1 boot)?
3. keep the dspark reference boot, given we are not porting it?
4. `--gen 192` enough, or longer generations for steadier decode averages?
