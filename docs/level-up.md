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

### decode: global split-k reduction traffic

the current short-context profile records roughly 351 `k_tq_nvf4_reduce` launches per
decode step. the nvfp4 projection path divides k across ctas, writes fp32 partials to
global memory, launches a reducer, rereads the partials, and writes the final output.
graph replay removes dispatch cost but not that traffic.

large nvfp4 weight launches already stream around 1.56 tb/s, roughly 87% of the practical
dram ceiling. asking the same algorithm to stream weights a little faster cannot close
the whole gap. the partial and epilogue traffic has to disappear.

### projection boundaries: fp32 materialization

the remaining dataflow often looks like:

```text
projection -> fp32 output -> residual -> rms -> reread -> nvfp4 quant -> next projection
```

rms application and silu application already fuse into quantization where legal. the
projection output and residual pipeline still cross global memory more often than the
consumer contract requires.

### attention: current-row preparation is separate

decode q/k/v projection, normalization, rope preparation, q4/e4m3 kv publication, history
scan, and gated merge remain separate stages. staged v2 fixed the history-read disaster;
the current row still pays preparation and publication traffic before the scan can use it.

### prefill: peak gemm is not enough

the measured 8k attribution before the final matrix was approximately 42% nvfp4 gemm,
21% deltanet scan and prep, 13% wide attention, and 13% activation quantization/fusion
kernels. the gemm's large launches are already hardware-class. prefill parity therefore
needs a better activation and state dataflow around the gemm, not a benchmark-only gemm
number.

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

## stage 3 — current-row attention integration

### hypothesis

q normalization, rope, current k/v quantization, paged publication, and q preparation can
share a prologue or at least one owned preparation kernel. the history scan then consumes
the prepared row without another full global round trip.

### work

- map every current-row read/write in the 16 full-attention layers;
- fuse q/k normalization and rope where the exact reduction order can be kept;
- publish q4 k, scales, e4m3 v, and v scale once;
- make the v2 attention kernel consume the prepared q representation directly;
- preserve page-table and apc tail ownership;
- keep the spec chain and plain decode contracts separate where their q grouping differs.

### acceptance

- kv bytes are identical to the existing writer;
- paged parity and short/2k/32k gates pass;
- at least 0.3 ms/token saved at 2k or a measured context-scan improvement at 128k;
- no new allocation in the decode loop.

## stage 4 — scheduler and graph overlap

this is a cleanup rung, not the main bet. full graph replay measured about 1% because the
work inside the graph stayed the same. after stages 1-3 remove kernels and buffers, revisit
its value.

- stable input and state-index buffers per batch bucket;
- launch batch t while the host processes completed metadata for t-1;
- no speculative assumption in the plain scheduler;
- preserve cancellation, sampling, and error publication order;
- measure host gaps directly rather than inferring them from wall time.

accept only an end-to-end win with unchanged ttft and no request-starvation regression.

## stage 5 — deltanet and prefill dataflow

our chunk-64 wy/ut transform is already structurally close to the reference algorithm.
the failed serial/parallel split proved that fp32 state checkpoint traffic can erase a
faster serial kernel. do not retry bf16 checkpoints: they produced early divergence and
an empty 8k output in the measured stack.

research directions:

- fuse the third reconstruction phase into the following projection consumer;
- retain incoming state in a persistent cta without emitting the full s-pre image;
- checkpoint only a compact algebraic intermediate if the consumer can reconstruct from
  it without another state-sized read;
- co-schedule independent value-head stripes without reducing live-state occupancy;
- compare tf32 and fp32 at the existing teacher-forced gate, not a live source corpus.

acceptance is an 8k and 32k prefill win with the existing quality band and no state-sized
scratch explosion.

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
