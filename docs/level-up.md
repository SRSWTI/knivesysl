# level up

this is the working campaign for moving knivesysl from close to the reference stacks to
plain-path parity, then a real overtake. it is an experiment plan, not a list of promised
speedups. a stage ships only when the end-to-end matrix and correctness gates agree with
the kernel-level result.

paged mtp and durable apc appear at the end as optional increments. neither is counted as
proof that the plain target-model path is fast.

## the line we are trying to cross

current nvfp4-all plain performance, no prefix cache, gen 512, three-repeat medians:

| context | vllm plain | knivesysl plain | plain gap |
|---:|---:|---:|---:|
| 2k x1 | 71.7 tok/s | 65.8 tok/s | -8.2% |
| 8k x1 | 72.1 tok/s | 64.7 tok/s | -10.3% |
| 32k x1 | 69.8 tok/s | 61.1 tok/s | -12.5% |
| 64k x1 | 65.9 tok/s | 58.7 tok/s | -10.9% |
| 128k x1 | 57.8 tok/s | 50.1 tok/s | -13.3% |
| 8k x4 | 204.8 aggregate | 177.9 aggregate | -13.1% |
| 32k x4 | 103.2 aggregate | 80.2 aggregate | -22.3% |
| 64k x4 | 41.0 aggregate | 38.9 aggregate | -5.1% |
| 128k x2 | 20.8 aggregate | 20.1 aggregate | -3.4% |

single-stream parity requires roughly 1.25-2.66 ms less work per token. prefill is the
larger gap: about 24-25% at 2k/8k, 20% at 32k, 16% at 64k, and 10% near 128k.

this is not a different performance class. it is also not small enough to fix with one
more tile-size sweep.

## rules for the campaign

1. **plain means plain.** no n-gram, mtp, apc hit, or cached prefix may enter a plain
   parity claim.
2. **same workload contract.** exact prompts, gen 512, temperature zero, three repeats,
   cache off, and the same concurrency cell on both sides.
3. **profile the current build.** old profiles guide hypotheses but never prove a new
   commit.
4. **remove work before tuning work.** prefer deleting global round trips and launches to
   moving them around.
5. **owned hot path.** no torch, cublas, cutlass, or flashinfer dependency in steady-state
   execution. source ideas may be ported and then owned.
6. **greedy gates before performance gates.** short prompt plus 2k and 32k exact output;
   then numeric/teacher-forced gates where reduction order changes.
7. **one variable per accepted rung.** every benchmark artifact names the gate and keeps
   the previous artifact immutable.
8. **honest rejection.** a microbenchmark win that moves traffic elsewhere or loses in
   the server matrix is rejected, not relabelled as groundwork.

## why the remaining gap exists

### decode: the projection mainloop, not the reducer

the current short-context profile records 496 `k_tq_nvf4_gemm_tma` launches and 352
`k_tq_nvf4_reduce` launches per token. the names made the reducer look like the obvious
target; the current-build trace disproves that:

- nvfp4 gemm mainloops consume 9.586 ms/token at 2k;
- every nvfp4 reducer combined consumes 0.259 ms/token;
- the source-derived split-partial round trip is 35.95 mb/token, whose ideal 1.56 tb/s
  floor is only 0.023 ms/token;
- forcing every projection to full-k `ks=1` removes decode-time reducers but slows the
  model from about 66.8 to 57.2 tok/s.

split-k is buying occupancy. deleting its fold without preserving that parallelism cannot
cross the plain gap, and switching globally to the vendor-style unsplit policy is worse.

### projection boundaries: apparent bytes are not sufficient

the remaining dataflow often looks like:

```text
projection -> fp32 output -> residual -> rms -> reread -> nvfp4 quant -> next projection
```

rms application and silu application already fuse into quantization where legal.
standalone experiments removed more of those apparent round trips exactly, but lost when
they serialized independent nvfp4 k64 tiles or introduced cluster rendezvous. a production
epilogue must retain the existing output-tile parallelism; launch deletion by itself has
no value under graph replay.

### attention: history dominates at depth

staged v2 fixed the original history-read disaster. on the current build, the full v2
history kernel costs 1.330 ms/token at 2k and 5.441 ms/token at 128k. the separate current
kv writer plus merge cost only 0.073 and 0.184 ms/token respectively. a standalone prepared-q
slab was exact but added traffic and never beat repeated in-cta preparation. current-row
fusion is therefore not the primary plain-parity rung.

### prefill: deltanet ownership is the remaining structural target

the fresh 8k prefill trace attributes 40.6% of gpu-busy time to nvfp4 gemms, 21.9% to
deltanet prep plus scan, 14.6% to wide attention, and 13.2% to activation quantizers.
the gemm launches are already hardware-class. producer-push dsm experiments for both the
prep stream and the output epilogue were exact but slower. the remaining credible rewrite
is a single-owner tf32/fp32 deltanet kernel that keeps the complete 128x128 recurrent state
distributed in registers; a direct fp16/bf16 donor port is numerically incompatible.

## source-mining findings

the mining pass covered the current translation unit plus the pinned cutlass, flashinfer,
flash-attention, tilegym, sglang, vllm, tensorcoreptx, and flashoverlap trees.

| area | source-grounded finding | campaign consequence |
|---|---|---|
| current nvfp4 | `k_tq_nvf4_gemm_tma` keeps fp32 accumulators in registers, publishes split-k partials, then folds them in ascending split order | preserve split occupancy and reduction order; the fold is a small tail |
| sm120 dense donors | flashinfer/vllm use unsplit `1x1x1` persistent ctas; cutlass sm120 has register mma and no tmem or stock multi-cta block-scaled builder | use their pipeline structure only; full-k ownership must first beat the live split policy |
| dsm reduction | the reusable cutlass gqa mechanism is producer-push into receiver-local mailboxes; consumer-side remote reads fault on this gb202 | producer-push is validated locally, but only inside a producer that already owns the partial |
| projection fusion | serving stacks fuse silu or add/rms/quant after materialized projections; none supplies gemm -> residual -> rms -> nvfp4 as one sm120 kernel | no library-shaped shortcut exists; a real port must own accumulator fragments and row completion |
| attention | flashattention's fused append path assumes fp16/bf16 kv; tilegym's grouped decode assumes dense kv; neither consumes q4-hadamard k plus e4m3 v | retain v2 history math and port only representation-compatible publication ideas |
| deltanet | flashinfer sm120 keeps a 128x128 fp32 state in registers across chunk-64 blocks, but converts state operands and its inverse to fp16/bf16 | retain ownership/warp-role structure; reject a direct precision port |
| scheduling | sglang launches the next forward from gpu-resident token metadata while d2h publication trails on another stream | this is the source-backed route to the measured 0.64-0.65 ms/token host-gap ceiling |

no vendor library enters the run path. provenance and license headers remain with any later
source port.

## measured campaign decisions — 2026-09-02

| thesis | decision | measured reason |
|---|---|---|
| standalone cluster split-k fold | **reject** | exact and faster in isolation, but all live reducers total only 0.259 ms/token |
| global full-k/unsplit nvfp4 | **reject** | `ks=1` removes post-load reducers and loses 14.4% end-to-end decode |
| one-cta residual/rms/nvfp4 publication | **reject** | bit-exact, 2.25x slower under graph replay; 80 k64 tiles became serial |
| four-cta deltanet norm/quant epilogue | **reject** | bit-exact, 1.90x slower under graph replay despite 6 mib removed |
| standalone prepared-q slab | **reject** | bit-exact, 0-25% slower; publication/reload replaces cheap parallel arithmetic |
| dsm-broadcast deltanet prep | **reject** | exact transport, 15.1% slower with 256 cluster barriers per 2k head |
| same-thread scheduler overlap | **accept for implementation experiment** | current traces leave 0.64-0.65 ms/token outside gpu kernels |
| single-owner tf32/fp32 deltanet prefill | **revise and retain** | only ownership pattern with a structural payoff; direct bf16/fp16 math is rejected |

rejected mechanisms remain documented and their standalone sources remain in `tools/`.

## stage 0 — refresh the profile map

### runs

- 2k x1 plain decode;
- 128k x1 plain decode;
- 8k x1 plain prefill;
- 32k x4 full request;
- 64k x4 capacity-edge request.

capture kernel totals, call counts, gpu-busy time, host gaps, memory allocation events,
and the exact benchmark artifact/config. separate prefill and decode windows.

### questions

- how much time and global traffic remains in `k_tq_nvf4_reduce`?
- how many projection shapes use split-k at n=1, n=2, and n=4?
- how much of quantization, norm, residual, and copy time can be attributed to full-width
  fp32 tensors?
- which part of the 32k multi-request gap is prefill residency, decode, or scheduling?
- how much current-row attention work remains outside the v2 history kernel?

### gate

at least 95% of gpu-busy time must have an owner before a kernel rewrite starts. empty or
unflushed profiler output is not evidence.

### current evidence — 2026-09-02

fresh current-build artifacts:

- `results/profiles/current-plain-2k-n1-full.sqlite`: 120 decode steps, 2k x1;
- `results/profiles/current-plain-128k-n1.sqlite`: 120 decode steps, 128k x1;
- `results/profiles/current-plain-prefill-8k-n1.sqlite`: one 8k prefill;
- `results/microbench/level_up_experiments.json`: standalone controls, probes, resources,
  exactness, and decisions.

| owner | 2k ms/token | 128k ms/token | depth delta |
|---|---:|---:|---:|
| `k_tq_nvf4_gemm_tma` | 9.586 | 9.660 | +0.074 |
| nvfp4 activation quantizers | 1.698 | 1.709 | +0.011 |
| paged-attention v2 history scan | 1.330 | 5.441 | +4.111 |
| deltanet decode kernels | 0.513 | 0.522 | +0.009 |
| `k_tq_nvf4_reduce` | 0.259 | 0.265 | +0.006 |
| fp6 lm head | 0.570 | 0.570 | 0.000 |
| residual/rms helpers | 0.279 | 0.298 | +0.019 |
| current kv writer plus attention merge | 0.073 | 0.184 | +0.111 |

the 2k window spans 1.800 s, contains 1.723 s of summed gpu work, and leaves about
0.64 ms/token outside kernels. the 128k window spans 2.322 s, contains 2.244 s of gpu
work, and leaves about 0.65 ms/token outside kernels. both are about 96.6-96.7% gpu-busy.

the 8k prefill window spans 840.7 ms and contains 827.5 ms of gpu work (98.4% busy).
nvfp4 gemm, wide attention, deltanet prep/scan, and the three activation quantizers alone
own 90.3%; the next residual, norm, convolution, and gated-norm kernels take attribution
well above the 95% gate. batch-server scheduling still needs its own stage-4 trace; it is
not inferred from these single-client windows.

## stage 1 — cluster-reduced nvfp4 decode projection

### hypothesis

an sm120 thread-block cluster can keep split-k partials inside cluster-visible storage,
perform a deterministic final fold, and publish the output once. this removes the global
partials buffer and the standalone reducer from the common decode path.

```text
nvfp4 weight tiles -> fp4 mma -> cluster-local reduction -> fused epilogue -> output
```

### first implementation shape

- n=1 decode only;
- one cluster owns one output tile;
- ctas divide k without duplicating the output publication;
- deterministic reduction order matching the accepted path;
- bounded distributed/shared storage with an explicit register and smem ledger;
- existing tma weight layout and scales stay unchanged;
- env gate with the current path as the control.

### research probes

1. confirm sm120 cluster dimensions and distributed shared-memory limits for the actual
   launch shape;
2. measure cluster occupancy after the accumulator and scale registers are counted;
3. compare cluster-local fold, cooperative-groups fold, and a persistent-cta work queue;
4. determine whether the final epilogue can consume accumulator fragments directly;
5. confirm cuda graph capture/replay accepts the final launch contract;
6. test n=2 and n=4 only after n=1 moves end-to-end decode.

### acceptance

- short, 2k, and 32k greedy outputs pass;
- no global partial allocation or `k_tq_nvf4_reduce` for accepted n=1 shapes;
- at least 0.7 ms/token removed at 2k x1;
- no 128k regression above measurement noise;
- no extra host synchronization;
- server throughput moves with the kernel result.

### rejection

reject the rung if cluster occupancy loss causes another kernel to recover the saved time,
if a global atomic changes greedy output, or if the win exists only before graph capture.

### measured decision — standalone rung rejected

`tools/microbench_cluster_reduce.cu` isolates the exact sm120 mechanism with deterministic
fp32 partials, eager and cuda-graph controls, actual model output widths, register/smem
reporting, and bitwise output comparison.

- ~~owner-pull dsm reduction~~ failed: peer reads faulted at remote shared offset `0x500`
  under both dynamic and static allocation;
- producer-push into rank 0's dsm mailbox passed bitwise (`max_abs=0`) and removed the
  global partial round trip plus one launch;
- `ks=2`, `n=1`: every tested width passed, 2.04-2.10x eager and 1.12-2.00x graph,
  with a 1,024-byte mailbox per cta;
- `ks=4`, `n=1`: useful only for narrower outputs; neutral by width 17,408 and
  graph-shape-dependent;
- `ks=8`, `n=1`: useful only for small outputs, loses from width 5,120 upward, and width
  17,408 is repeated-launch unstable;
- wider `n=2/4` cases become neutral or slower and include launch-unstable shapes.

weighted over the actual 352 reducer-bearing projections per token, the synthetic saving is
about 0.408 ms/token eager and 0.274 ms/token under graph replay. the independent real-kernel
measurement is consistent: reducers consume 0.259 ms/token and the ideal partial-workspace
traffic floor adds only about 0.023 ms/token.

the second control forced the real engine to one split policy across every projection:

| policy | ms/token | tok/s | versus per-weight autotune |
|---:|---:|---:|---:|
| per-weight autotune | about 14.97 | about 66.8 | reference |
| `ks=1` | 17.497 | 57.15 | -14.4% |
| `ks=2` | 15.590 | 64.14 | -3.9% |
| `ks=4` | 15.314 | 65.30 | -2.2% |
| `ks=8` | 15.074 | 66.34 | -0.7% |

`results/profiles/current-plain-2k-ks1-verified.sqlite` contains 288 reducer launches during
load-time autotuning and **zero** after load, proving the override removed the decode-time
fold. therefore ~~ship cluster reduction as a standalone decode rewrite~~ and ~~replace
split-k with a vendor-style full-k owner~~ are both **rejected**. producer-push dsm remains
a valid primitive only if a future gemm producer can use it without losing split occupancy.

## stage 2 — projection epilogue and next-consumer publication

### target chains

1. attention `o_proj -> residual -> rms statistics -> mlp nvfp4 input`;
2. mlp `down_proj -> residual -> next-layer rms statistics -> nvfp4 input`;
3. deltanet `linear_out -> residual -> mlp nvfp4 input`;
4. any final layer where the committed fp32 residual and quantized consumer view can be
   published from one pass.

### design constraints

- the committed residual remains fp32;
- consumer nvfp4 codes and scales must match the accepted quantizer when the mathematical
  order is unchanged;
- rms reductions cannot race split-k completion;
- a projection with multiple consumers does not fuse until every consumer contract is
  explicit;
- no second convention beside `wide_quant_*` memo ownership.

### experiments

- fuse one projection family at a time;
- record bytes removed per row as well as kernel time;
- compare direct epilogue quantization with an epilogue that emits only rms factors;
- measure decode n=1 and prefill widths 256/512/2048 separately;
- keep a forced-unfused control in the same binary.

### acceptance

- exact greedy gates;
- numeric worst case remains inside the current nvfp4 gate;
- at least 0.5 ms/token removed after stage 1, or at least 8% at 8k prefill;
- 32k x4 improves rather than trading throughput for a prettier n=1 number.

### measured decision — post-projection fusion rejected

`tools/microbench_resid_quant.cu` reproduces the exact 5,120-wide residual/rms factor and
native nvfp4 tile contract. both the no-add and residual-add variants produced zero
residual or packed-word mismatches. under graph replay the current two-kernel controls took
8.20 us; the one-cta fused variants took 18.44 us. one block serialized 80 independent
k64 quant tiles and allowed only one active block per sm.

`tools/microbench_dn_epilogue.cu` then tested the larger four-cta alternative over a real
64-token, 48-head, 128-wide deltanet wave. producer-push dsm preserved every fp32 result and
packed word while removing a modeled 6 mib, but took 25.50 us versus 13.45 us for the
three-kernel graph control.

therefore ~~collapse the post-projection chain in a standalone mega-epilogue~~ is rejected.
an accumulator-owned epilogue remains admissible only if the gemm already owns complete
output tiles and retains existing tile parallelism; it is not an active rung without a
full producer microkernel that clears the stage gate.

## stage 3 — current-row attention integration

### hypothesis

q normalization, rope, current k/v quantization, paged publication, and q preparation can
share an existing producer boundary. the history scan must not pay another global
round trip for a separately prepared q slab.

### measured decision — standalone preparation rejected

`tools/microbench_qprep.cu` copies the exact q norm -> rope -> hadamard arithmetic and
compares repeated in-cta work against one global prepared-q producer. every tested split
count was bit-identical. the producer/reload path was slower:

| split count | eager control / staged | graph control / staged |
|---:|---:|---:|
| 4 | 4.96 / 5.84 us | 6.14 / 7.79 us |
| 16 | 5.01 / 5.86 us | 6.15 / 8.19 us |
| 64 | 5.73 / 6.20 us | 6.15 / 8.19 us |
| 85 | 10.32 / 10.75 us | 12.28 / 12.28 us |

the current writer plus merge are only 0.073 ms/token at 2k and 0.184 ms/token at 128k.
~~publish a standalone prepared-q tensor~~ is rejected, and this stage is removed as a
plain-parity priority. q/k preparation may still be fused into the already-mandatory
current-row writer if a later attention rewrite needs it, but that work must keep kv bytes,
page/apc ownership, and short/2k/32k outputs unchanged.

## stage 4 — scheduler and graph overlap

this is now an accepted bounded implementation experiment, not cleanup. full graph replay
previously moved only about 1% because it removed dispatch without changing the synchronous
host boundary. the current traces leave 0.64-0.65 ms/token outside kernels at both 2k and
128k. that is the maximum single-stream gain, not a promise.

- keep stable input, state-index, sample, and publication buffers per batch bucket;
- launch batch t while the same host thread processes completed metadata for t-1;
- relay the next token through device-resident storage; do not wait for its d2h copy before
  launching the next forward;
- no speculative assumption in the plain scheduler;
- preserve cancellation, sampling, slot reuse, and error publication order;
- measure host gaps directly, plus staggered-client ttft and starvation.

accept only an end-to-end win with unchanged outputs, unchanged maximum ttft, and no
request-starvation regression. reject it if the overlap merely moves delay into response
publication or lets a new step overwrite buffers still owned by the copy stream.

## stage 5 — deltanet and prefill dataflow

the fresh 8k profile assigns 65.1 ms to prep and 116.3 ms to the tf32 scan: 21.9% of
gpu-busy prefill time. two tempting transport/finalization rewrites have now failed:

- `tools/microbench_dn_prep_flow.cu` models all 32 chunks and 48 heads. global publication
  plus the four stripe readers took 1.341 ms under graph replay; exact producer-push dsm
  streaming took 1.579 ms, 15.1% slower, despite removing the 226 mib prep write;
- the exact four-cta norm/quant epilogue took 25.50 us versus 13.45 us, 1.90x slower.

therefore ~~broadcast prep through dsm~~ and ~~cluster-fuse the raw-core epilogue~~ are
rejected. the flashinfer source supplies one remaining structural idea: one 384-thread cta
owns a full head, carries the 128x128 fp32 state in registers across all chunk-64 blocks,
and publishes it once. its arithmetic is not a drop-in:

- flashinfer uses fp16/bf16 mma operands and a fixed fp16 triangular inverse;
- knivesysl keeps prep, solve, and recurrent state fp32 and rounds only scan mma operands
  to tf32;
- the prior bf16 checkpoint experiment diverged at 2k/32k and emitted an empty 8k output.

the stage is **revised and retained** as a dedicated tf32/fp32 single-owner prototype. it
must compare final recurrent state, packed pre-`linear_out` codes/scales, ragged 129-token
handling, and 2k/32k greedy output. acceptance requires at least 8% faster 8k prefill, no
32k regression, no state-sized checkpoint image, and no bf16/fp16 boundary substitution.

## stage 6 — plain acceptance matrix

rerun sglang, vllm, and knivesysl plain with cache off, exact prompts, gen 512, and three
repeats. do not count n-gram, mtp, or apc hits.

### target

| area | acceptance target |
|---|---|
| n=1 decode | geometric mean at or above vllm |
| n=1 floor | no measured cell below 0.95x vllm; then tighten to parity |
| 2k/8k batch | match or beat vllm aggregate |
| 32k x2/x4 | close the current 21-22% deficit |
| 64k x4 and 128k x2 | keep the existing capacity-edge parity |
| prefill | first get every cell within 10%; then cross parity |
| ttft | no throughput win bought by worse admission latency |
| correctness | every kernel and server gate passes |

one matrix is not enough to claim robustness. repeat the winner after a clean server boot
and retain min/median/max plus raw samples.

## experiment record

use this header for every rung:

```text
hypothesis:
changed symbols:
control env:
probe env:
correctness gates:
profile artifact:
benchmark artifact:
expected bytes/launches removed:
observed kernel delta:
observed server delta:
accepted or rejected:
reason:
```

### experiment 1 — dsm split-k reduction

```text
hypothesis: remove global split-k partials and the standalone reducer
changed symbols: tools/microbench_cluster_reduce.cu only
control env: sm120 global partial kernel plus global reducer
probe env: compile-time k-cluster, producer-push dsm mailbox, owner-local fold
correctness gates: bitwise float output identity for every completed case
profile artifact: results/profiles/current-plain-2k-n1-full.sqlite
benchmark artifact: results/microbench/level_up_experiments.json
expected bytes/launches removed: (2 * ks) fp32 partial bytes per output, one launch
observed kernel delta: weighted 0.408 ms/token eager; 0.274 ms/token graph estimate
observed server delta: not run; production source was intentionally unchanged
accepted or rejected: rejected as a standalone rung; primitive retained only inside a future gemm producer
reason: measured payoff misses the gate; full-k loses 14.4%, and wide/multi-request cluster cases are neutral or unstable
```

### experiment 2 — post-projection residual/rms/nvfp4

```text
hypothesis: one row owner can commit residual, compute exact rms, and publish nvfp4 once
changed symbols: tools/microbench_resid_quant.cu only
control env: exact factor kernel plus 80 independent k64 quant blocks
probe env: one 1024-thread cta, 5120-value shared row, direct packed publication
correctness gates: zero residual and packed-word mismatches
profile artifact: results/profiles/current-plain-2k-n1-full.sqlite
benchmark artifact: results/microbench/level_up_experiments.json
expected bytes/launches removed: 40960 read bytes and one launch per row
observed kernel delta: graph 8.20 us -> 18.44 us
observed server delta: not run; standalone failed
accepted or rejected: rejected
reason: quant tile parallelism collapsed and occupancy fell to one block/sm
```

### experiment 3 — deltanet output cluster epilogue

```text
hypothesis: four scan stripes can normalize and quantize without a raw-core round trip
changed symbols: tools/microbench_dn_epilogue.cu only
control env: publish, exact gated rmsnorm, native nvfp4 quant
probe env: four-cta producer-push dsm cluster
correctness gates: zero fp32 and packed-word mismatches
profile artifact: results/profiles/current-plain-prefill-8k-n1.sqlite
benchmark artifact: results/microbench/level_up_experiments.json
expected bytes/launches removed: 6291456 bytes and two launches per 64-token wave
observed kernel delta: graph 13.45 us -> 25.50 us
observed server delta: not run; standalone failed
accepted or rejected: rejected
reason: dsm copies and cluster synchronization cost more than removed traffic
```

### experiment 4 — prepared-q staging

```text
hypothesis: compute q norm/rope/hadamard once instead of once per attention split
changed symbols: tools/microbench_qprep.cu only
control env: repeated exact per-split prologue
probe env: one producer, 24576-byte global staging slab, lightweight consumers
correctness gates: bitwise identity at split counts 4, 16, 64, and 85
profile artifact: results/profiles/current-plain-128k-n1.sqlite
benchmark artifact: results/microbench/level_up_experiments.json
expected bytes/launches removed: repeated q arithmetic; adds one launch and staging round trip
observed kernel delta: never faster; equal only at split 85 under graph replay
observed server delta: not run; standalone failed
accepted or rejected: rejected
reason: parallel q transforms are cheaper than publication and reload
```

### experiment 5 — deltanet prep transport

```text
hypothesis: stream prep through producer-push dsm into four state-stripe owners
changed symbols: tools/microbench_dn_prep_flow.cu only
control env: 226 mib global publication plus 755 mib stripe reads
probe env: 705 mib producer-push dsm stores, 256 cluster barriers per head
correctness gates: identical 48x4 output checksums
profile artifact: results/profiles/current-plain-prefill-8k-n1.sqlite
benchmark artifact: results/microbench/level_up_experiments.json
expected bytes/launches removed: 226492416 global write bytes and one launch
observed kernel delta: graph 1.341 ms -> 1.579 ms
observed server delta: not run; standalone failed
accepted or rejected: rejected
reason: dsm fan-out, barriers, and 72 registers outweigh global traffic removal
```

## optional increment a — paged mtp

paged mtp is the model-native overtake path after the plain target path is strong. unlike
n-gram, it can draft novel prose and code. it is still speculation, so its numbers remain
separate from plain acceptance.

### ownership invariant

```text
main kv cursor
  = deltanet cursor
  = mtp kv cursor
  = mtp root-hidden cursor
  = scheduler cursor
```

### implementation rungs

1. add a native committed cursor per slot and reject skipped, repeated, or stale positions;
2. make multi-block allocation rollback atomic on pool exhaustion;
3. extend the existing physical block namespace with the mtp attention layer's q4/e4m3
   kv rows;
4. allocate per-slot fp32 root hidden plus validity/cursor metadata;
5. advance the mtp trunk after every successfully committed paged prefill, plain decode,
   and speculative input, using verified main-model hidden rows;
6. include mtp kv, root hidden, cursor, deltanet state, and main kv in one checkpoint
   publication boundary;
7. extend resident adopt/fork, private-tail copy, host demote, and host promote;
8. add slot-aware batched mtp draft and target verify;
9. cost-gate depth and subgroup scheduling by context, concurrency, acceptance, and archive
   memory;
10. run fp6, nvfp4-all, and nvfp4-mlp grids only after equivalence and apc gates pass.

### correctness gates

- cold prefill and plain decode;
- zero, one, partial, and full draft acceptance;
- short, 2k, and 32k committed token equivalence;
- aligned and non-aligned checkpoint adopt/fork;
- host demote/promote with a private partial tail;
- pool and slab exhaustion rollback;
- cancellation and slot reuse;
- concurrency 1, 2, 4, 8, 12, 15, and 16;
- no cursor publication until the stream succeeds.

### decision rule

mtp ships only where total accepted-token savings exceed draft, verify, state, and memory
cost. low-acceptance and high-concurrency cells fall back to the improved plain path.

## optional increment b — durable apc

resident apc already gives 9-11x turn append, 75-98x exact resend, and up to 37x steady
fan-out on the measured workload. the optional durability increment makes those
checkpoints survive memory pressure and process restarts.

### host tier validation

- exercise `qwn_paged_ckpt_demote`, `promote`, and `tier` end to end;
- cover aligned and partial-tail checkpoints;
- prove every failed allocation or transfer leaves the original checkpoint usable;
- measure pinned-host bytes, demote latency, promote latency, and saved re-prefill time;
- validate multiple resident and demoted entries under lru pressure.

### restart persistence

- serialize checkpoint registry metadata, block-row images, deltanet/conv state, sampling
  state where required, committed cursors, and optional mtp state;
- use versioned, checksummed, atomic files;
- publish a registry entry only after every payload reaches durable storage;
- reject model, tokenizer, format, geometry, and engine-version mismatches;
- restore into newly allocated blocks and slabs without assuming old physical ids;
- make an interrupted restore degrade to a cold prefill rather than poison the registry.

### template-aware boundaries

- record stable chat-template message boundaries rather than only incidental token offsets;
- keep the lcp junction and near-end resend target;
- add tool-call/result and assistant-turn boundaries that recur in agentic traffic;
- cap checkpoint count by measured reuse value, not by every possible boundary;
- keep previously unseen divergence as an explicit honest-loss case.

### durability measurements

- restart-to-ready time with zero, one, and many checkpoints;
- restore latency versus cold re-prefill at 8k, 32k, 64k, and 128k;
- resident gpu capacity, pinned-host capacity, and disk capacity;
- six-way fan-out after restart;
- corrupt/truncated registry recovery;
- model-upgrade invalidation;
- concurrent restore and new admission behavior.

### decision rule

durable apc ships only when restore is faster than cold prefill, failure is atomic, and the
storage budget does not crowd out the live kv capacity the service needs.

## optional increments are additive

plain parity is evaluated without n-gram, mtp, or apc hits. after that gate:

- n-gram remains a cost-gated accelerator for repetitive histories;
- paged mtp targets novel agentic continuations;
- durable apc removes repeated prefill across turns and restarts.

these mechanisms solve different work. they can stack, but none may hide a regression in
another layer.
