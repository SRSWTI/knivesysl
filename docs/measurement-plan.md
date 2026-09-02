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

## axes and feasibility

| axis | values | notes |
|---|---|---|
| server | knivesysl paged · knivesysl single-stream · SGLang | paged is production; MTP is single-stream until its paged port |
| weight tier | NVFP4-all · NVFP4-MLP · FP6 | 18.1 GB / ~20 GB / 22.5 GB; every tier is measured, not projected |
| speculation | off · n-gram · MTP · DSpark | MTP is FP6/n=1; DSpark is SGLang/n=1 |
| standard context | 2k · 8k · 16k · 32k · 65k · 94k · 131k | full service grid, pool-clipped |
| deep context | 196k · 240k · ~261k | only tiers whose actual pool admits the cell |
| concurrency | 1 · 2 · 4 · 8 · 12 · 15 · 16 | 12-16 use dedicated high-slot boots |
| workload | repetitive code · prose · math · chat · factual · long context | matrix uses code; sensitivity pass uses all six |
| prefix cache | off in clean grids · on in APC pass | prevents reuse from contaminating TTFT |
| repetitions | 3 normal · 2 deep | raw samples retained; median primary; min/max error bars |

The model position cap and aggregate capacity are different quantities:

- `TQ_MAX_SEQ=262144`; production sets `TQ_CTX=262144`.
- Production NVFP4 uses `2100 x 128 = 268800` aggregate paged KV rows.
- Each active slot also costs roughly 145-151 MB of DeltaNet state, so a pool
  serviceable at two slots may need to shrink at sixteen. The driver descends the
  requested block count until a server actually boots, then records and asserts the
  successful count. No capacity is inferred from a four-slot boot.

Impossible Cartesian cells are recorded as skipped, never sent:
`n * (context + generation + guard) > actual_pool_tokens`, `n > max_slots`, MTP
with NVFP4, MTP above n=1, or DSpark on knivesysl.

### n-gram depth at concurrency

The paged verifier has a 16-node global archive cap. With `N` active requests:

`max_draft_depth = max(0, floor(16 / N) - 1)`.

| N | 1 | 2 | 4 | 8 | 12 | 15 | 16 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| max draft depth | 15 | 7 | 3 | 1 | 0 | 0 | 0 |

n=12/15/16 n-gram-enabled cells are deliberately measured and labelled
`fallback`: they prove the enabled-mode overhead floor but are not presented as
speculative speedups. The driver uses the actual environment names
`TQ_PG_SPEC_SLOTS` and `TQ_PG_SPEC_NODES`; the former was previously misspelled
and would have silently disabled speculation above four slots.

## execution

Execution is now staged. The immediate decision matrix deliberately excludes maximum
concurrency and maximum-capacity frontiers because paged MTP and APC durability will
change per-block/per-slot memory accounting and invalidate those measurements.

Current native drivers:

```text
tools/bench_core_matrix.sh
tools/bench_core_parity.sh
```

Run the parity supplement after the base matrix. It adds only cells deliberately
omitted from the original sentinel grid; it does not overwrite or rerun completed
samples.

Current matched reference:

```text
tools/bench_sglang_core.sh
```

Persistent artifacts and live logs:

```text
results/core_matrix.log
results/sglang_core.log
results/raw/knivesysl/
results/raw/sglang/
```

The combined native core campaign has at most 53 feasible cells:

- NVFP4-all plain: 2k/8k/32k/65k/131k at n=1/2/4, pool-clipped;
- NVFP4-MLP plain: the identical context/concurrency grid, pool-clipped;
- FP6 plain: the identical context/concurrency grid, pool-clipped;
- NVFP4-all n-gram: the identical context/concurrency grid, pool-clipped;
- FP6 single-stream MTP: 2k/8k/32k/65k/131k at n=1. The current MTP server
  serializes requests and cannot validly measure n=2/4.

The base matrix's five artifacts remain authoritative. The parity supplement stores
the 18 previously omitted cells in separate artifacts so strict resume metadata
remains immutable. Consumers must union matching base and `-parity` artifacts by
weight/spec tier before comparison.

All retain tokenizer-exact prompts, 512 generated tokens, temperature zero, prefix
cache off, three repetitions, server usage token counts, and raw per-client timing.

## frozen core comparison (2026-09-02)

The authoritative SGLang reference artifacts are the `sgl-core-nocache-*` labels
(radix prefix cache disabled). The earlier `sgl-core-*` artifacts are marked
`superseded`: radix caching across the three identical repetitions contaminated
TTFT, prefill, and cohort admission (8k TTFT read 0.19s cached vs 0.72s clean;
32k read 0.36s vs 3.97s). DSpark is likewise rerun cache-disabled
(146.3/208.3 tok/s at 2k/8k, 13,276-token pool).

Frozen knivesysl-NVFP4-plain / SGLang-clean ratios (decode agg, TTFT their/ours):

```text
 2k:1 0.81x 0.77x | 8k:1 0.85x 0.82x | 32k:1 0.86x 0.89x | 65k:1 0.79x 0.96x | 131k:1 0.66x 1.01x
 8k:4 1.50x 3.22x | 32k:4 1.04x 1.36x   (we lead both concurrency cells)
```

n=1 decode ITL decomposition: flat ~3.2 ms base gap (CUDA-graph/launch, context-free)
+ context-scan gap growing to ~5.6 ms at 131k. The kernel campaign's staged GQA
decode kernel (commit 642c51e, measured after this freeze) already recovers most of
the scan gap (131k 38.9 -> 51.3 tok/s); those numbers re-enter through the
final-acceptance campaign, not by editing the frozen core artifacts.

The complete final-acceptance drivers remain:

```text
tools/bench_all_tiers.sh
tools/bench_sglang_ref.sh
```

The A-H campaign below is deferred until paged MTP and APC durability finalize the
state layout.

### A. final-acceptance primary paged grids -- 6 configs

`NVFP4-all / NVFP4-MLP / FP6 x plain / n-gram`, max-slots 4, contexts
2k-131k, concurrency 1/2/4, prefix cache off, gen 512, three repetitions.
Pools are 1800 / 1500 / 1200 blocks respectively, giving feasible cells according
to the exact `context + generation + 64` per-request guard. The obsolete depth gate is lifted.

### B. per-tier deep frontiers

- NVFP4-all plain: 196k:1, 240k:1, ~261k:1, 131k:2, starting at the production
  2100-block pool.
- NVFP4-MLP plain: ~190k:1, starting at 1500 blocks.
- FP6 plain: ~150k:1, starting at 1200 blocks.
- NVFP4 n-gram deep: a separate lower-pool boot because its eight-node DeltaNet
  verify archive costs about 1.2 GB.

Deep cells run twice. Failed requested pools are reduced in 64-block steps; the
artifact stores the successful pool and all feasibility skips.

### C. plain high-concurrency and constant-total frontiers

Each tier boots with max-slots 16. The same boot measures:

- 12/15/16 clients at 2k and 8k each;
- its actual constant-total frontier at n=4, n=8, and n=16.

For a successful pool of `B` 128-token blocks, each frontier context is computed
as `floor((128B/N - 576)/128)*128`, reserving 512 generated tokens plus a 64-token
guard per request. NVFP4 at the production pool therefore targets approximately
4x65k, 8x32k, and 16x16k; lower-capacity tiers land at their own measured frontier.

### D. high-concurrency n-gram fallback

All three tiers run n=12/15/16 at 2k and 8k with n-gram enabled and a 16-node cap.
Depth is zero by construction; artifacts must say `fallback` and show zero spec
rounds. A frontier n=16 fallback cell is included for each successful pool.

### E. active n-gram wide frontier

All tiers use an eight-slot boot and a 16-node archive. n=4 gets draft depth 3;
n=8 gets depth 1. Cells include n=8 at 2k and 8k plus the actual n=4/n=8 capacity
frontier. The archive's VRAM is paid in this configuration rather than hidden.

### F. single-stream MTP ceiling

FP6, n=1, all standard contexts through 131k, three repetitions. Q4 K + E4M3 V,
E2M3 packed embedding, wide prefill, prefix cache off, temperature 0, thinking
off, depth/k 6/3, tau 12, maxn 8, Dogs ladder
`6/3 -> 4/2 -> 2/1 -> dense`.

### G. SGLang reference

Plain NVFP4 boots separately at n=1/2/4/8/12/15/16. Each pins
`max_mamba_cache_size = 4N`, parses `max_total_num_tokens` from the live boot, and
runs exactly the same standard/deep/frontier rungs that fit. CUDA graphs are
attempted at the requested batch size; an explicitly labelled eager boot is used
only if graph capture cannot boot. DSpark remains an n=1 2k/8k external
speculation reference. All normal cells run three times.

### H. workload and APC confirmation

After the structural matrices: six-workload sensitivity at 2k:1 and 8k:1, then
cold/append/exact-resend/six-way-fan-out APC confirmation on production.

## artifact and metric contract

Every current JSON is schema 3 and is durably checkpointed after every repetition:
temporary file write, flush, file `fsync`, atomic replace, then directory `fsync`.
`--resume` accepts only an exact configuration match, skips a complete artifact,
and continues an incomplete cell from its next repetition. It contains:

- launch arguments, engine identity, pool/slot capacity, and every server-run start/end;
- all skipped cells and the exact feasibility reason;
- every raw per-client request timing for every repetition;
- median/min/max for per-request decode tok/s, aggregate decode tok/s, end-to-end
  aggregate tok/s, TTFT max/median, estimated prefill tok/s, event ITL p50/p99,
  tokens/event, spec rounds, committed tokens, and tokens/round;
- expected and observed spec behavior (`off`, `active`, `tail-active`, or `fallback`).

ITL intentionally remains per SSE event: speculative token bursts are the client
experience. Token throughput always uses the server's usage `completion_tokens`,
never event count.

## cost and resolved decisions

The immediate core pass is approximately 35 native cells plus the matched n=1/2/4
SGLang and two-cell DSpark reference: enough to freeze the kernel baseline without
spending hours on soon-to-be-invalid capacity boundaries. The exhaustive campaign
remains a roughly 4-7 GPU-hour final-acceptance gate, dominated by repeated
196k-261k prefills and high-slot boots.

Resolved: core matrix first; include exact 65,536-token cells; retain NVFP4-MLP and
FP6 as bounded sentinels; keep DSpark; gen 512; three repetitions; tokenizer-exact
input construction; server-reported prompt/completion counts; persistent `results/`
artifacts. Defer n=8/12/15/16, constant-total frontiers, and 190k-261k capacity until
paged MTP and APC finalize memory ownership.
