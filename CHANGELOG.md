# Changelog

All performance numbers are measured on this machine: RTX 5090 (GB202, SM120, 170 SM,
32 GB, 128 MB L2), CUDA 13.3, driver 595, Qwen3.8-27B FP6 (E2M3) + Q4 KV, greedy,
thinking off unless stated. vLLM comparisons are vLLM 0.27.1 serving
`unsloth/Qwen3.8-27B-NVFP4` (mixed W8A8 + W4A4, 22.5 GB), this session at
`--max-model-len 116032 --gpu-memory-utilization 0.92 --kv-cache-dtype fp8
--max-num-batched-tokens 8192 --enable-prefix-caching --language-model-only`
(140000 does not fit at util 0.92 right now: 4.59 GiB KV needed, 3.87 free),
driven by the same client (`tools/bench_endpoint.py`).

## Unreleased

### GQA-shared prefill attention: the 16k-96k campaign

**`k_tq_wide_attn_mma6`** -- one CTA per KV HEAD instead of per query head: the
six query heads of a GQA group are packed into the MMA's M dimension as six
16-row tiles (QROWS=96), so ONE K dequant and ONE V read feed all of them.
K4 rows stage through smem via cp.async at a 144 B conflict-free stride, with
the NEXT super-tile's copy issued right after the wmax barrier (it lands under
the score+V phases); Q prep runs 8 rows per pass with shared barriers. Default
ON (`TQ_WIDE_ATTN_GQA=0` reverts); wired on both the engine wide path and the
paged server path. Split default moves 16384 -> 8192 (measured on this kernel).

Bit-exact by construction and by gate: same fragment layout, same key walk,
same reduction order per row -- 65/65 argmax vs the per-head qr64 kernel on
the same build, TF identical on FP6, paged parity 11/11, needle 2/3 at a 128k
haystack (identical to baseline, including the documented 120k think-preamble
miss).

**Probe-driven development.** `TQ_WIDE_ATTN_PROBE` template scaffolds (default
off, WRONG numerics, timing only) decompose the 64k attention cost:
K loads 1.57 s | MMA issue 1.35 s | Q prep 0.58 s | expf 0.17 s | K-dequant
ALU 0.14 s | V decode 0.12 s. Two hypotheses died honestly: dequant ALU is 1%
(not the tax), and a first V-staging attempt lost at 64k because the pair-round
refactor tripled the e4m3 decode count. MMA issue is the sm120 floor
(m16n8k16 is the widest bf16 shape; no wgmma on consumer Blackwell).

**Engine (NVFP4-all, same chunk both sides):** 2k 8624->8737 | 4k 8316->8559 |
8k 7681->8122 | 16k 6828->7334 | 32k 5966->6381 | 64k 4806->5189 |
96k 3975->4306 | 128k 3383->3659. Gains grow with depth (+1.3% -> +8.3%).

**Server, cold, same client, both servers fresh, vLLM prefix cache OFF**
(their cache-ON runs emitted 84.8k/125.5k "tok/s" cells -- prefix hits,
discarded):

| P | vllm | knivesysl | ratio |
|---|--:|--:|--:|
| 16384 | 9173 | 6881 | 1.33x |
| 32768 | 7978 | 5971 | 1.34x |
| 65536 | 6090 | 4843 | 1.26x |
| 98304 | 4908 | 4054 | 1.21x |
| 114688 | 4487 | -- | their max-len |
| 131072 | HTTP 400 | 3424 | ours alone |

The band moves from 1.28-1.60x behind to **1.21-1.34x**, still narrowing with
depth. Remaining deficit by probe arithmetic: the MMA-issue floor plus GEMM
(down-projection) and DeltaNet shares -- i.e. the megakernel/TMA-GEMM lever,
not further attention scheduling.

**Harness lesson, paid twice now:** `nvfp4_quality` defaulted its prompt to
the LIVE SOURCE FILE; the first in-window edit of the session (a helper at
line 656) shifted every gate from token ~9.8k and burned four kernel bisects
on phantom flips (the flips started at sample 39/65 = the file offset of the
edit). The tool now defaults to the pinned `tf_corpus_e3cdb42.txt`.

### 128k: memory truth, V staging, and the capability line

**The "6.5 GB NVFP4 accounting bug" decomposed into a leak and a mirage.** The
autotuner's shape cache sat AFTER its synthetic-activation malloc, so every cache
hit leaked `d_x` -- 489 of 496 weights at 5-18 MB each. `TQ_MEM_TRACE=1` (new)
shows the conversion itself was always clean (+4320 MiB actual vs +4359 claimed)
and the remaining ~3.5 GB appears at the FIRST kernel launch: the driver's
local-memory pool, a context-lifetime cost every tier pays. The old comparison
measured FP6 before any launch, which is how a leak plus a driver pool read as
"NVFP4 costs 2.9 GB". Post-launch truth on this build: FP6 leaves 9760 MiB free,
NVFP4-`all` leaves **10578** -- the tier finally delivers its memory win, and
262k contexts now allocate on every tier (NVFP4 single-stream decode still
returns its documented -120).

**V8 super-tile staged via cp.async** in the 64-row attention kernel: the copy
issues before the QK phase and hides under K-dequant + mma; the V phase reads
smem instead of scattered per-byte gathers. Bit-identical (64/64 argmax vs the
unstaged 32-row kernel); 64k 4813 -> 4866, 128k 3385 -> 3422.

**First 128k measurements** (`all` tier, unlocked by the memory fix):

| scenario | vllm | knivesysl | |
|---|--:|--:|---|
| prefill 98304, server | 4834 | 3790 | 1.28x theirs |
| prefill 114688 (their max-len 116032) | 4491 | -- | their ceiling |
| prefill 131072, server | **HTTP 400** | **3259** | ours alone |
| prefill 131072, engine | cannot | 3422 | -- |
| decode @131072 ctx | cannot | 25.7 ms/step | -- |

vLLM's first 114688 response on a fresh server is 4491 tok/s (an earlier 8413
reading was prefix-cache contamination -- 13.6 s wall for MORE tokens than a
20.3 s 96k run; discarded). At 128k this checkpoint+config cannot answer at all:
`--max-model-len 140000` does not fit at util 0.92 and 116032 is its ceiling,
while this engine serves 131072 in 26.6 GB with 4.4 GB to spare. Note the
long-context deficit narrows as depth grows: 1.64x at 16k, 1.60x at 64k,
1.28x at 96k.

Quality spot check (temp 0, NVFP4-`all` + tf32 scan): the linked-list merge
prompt produces correct, idiomatic, documented code -- indistinguishable in
correctness from vLLM's output on the same prompt. Needle at a 128k haystack
(FP6, the tier whose single-stream decode the harness needs): 2/3 -- depths
4000 and 64000 RECOVERED, depth 120000 flips into a think-preamble and
exhausts the 32-token answer window (the depth-8000 near-tie class; this
depth was never measured before, so it is a data point, not a regression). GEMM stages=4 re-confirmed dead (110 KB smem vs the 101 KB
opt-in cap, already screened by the tuner); the producer/consumer rewrite
remains the recorded next step for the warm 2-4k cells.

### Long-context attention: depth-adaptive tiles, wide deep waves

The 64k profile put the wide-prefill attention at **54.1%** of the whole prefill;
everything else is a footnote at depth. Root cause is arithmetic, not mystery: KV
traffic scales as `total_query_rows / M_tile x depth`, and the kernel ran M=32
(wide-shared) / M=16 (paged) tiles. FA2's hd256 config runs M=64 but is smem-capped
there by its bf16 KV tiles (96 KB CTA); our int4-K/e4m3-V tiles are 4-8x smaller, so
M=64 fits in 55 KB of dynamic shared memory with the identical per-row fragment
plan. Three changes:

- **`k_tq_wide_attn_mma4`**: a 64-query-row generalization of the QT=1/2 kernel
  (same fragment floor plan, loops over four 16-row tiles). Per-row numerics are
  IDENTICAL: 16k wide-path argmax 64/64 vs the 32-row kernel, needle 3/3 with the
  path forced. Registers pin it at 1 CTA/SM (a forced 2-CTA build spilled and lost),
  so it only wins where KV streaming dominates: auto-routed at prefix depth >= 24k
  (wide-shared) / segment depth thresholds 4k/24k for 16/32/64 rows (paged -- which
  had been running 16-row tiles at EVERY depth and reading KV 4x more than the
  engine path at 64k).
- **Wave width by depth**: the engine cap rises to 2048 (a MAXIMUM; wave builders
  choose), and `serve_batched` widens a wave to the cap once a prefilling client's
  cursor passes 16k -- four fewer weight passes where attention dominates anyway.
  Shallow prompts keep 512 (8055 vs 7903 tok/s at P=2048 single-wave).
- Measured: engine 64k prefill 3966 -> **4567** (+15%), 16k 6252 -> 6553 at
  2048-col waves; server 64k **2677 -> 4267 (+59%)**, 16k 5288 -> 5853. The 64k
  deficit vs vLLM narrows 2.64x -> **1.68x**. Paged parity 11/11 shallow + 6/6
  deep, teacher-forced and FP6 regression gates re-run on the final build below
  (FP6 single-stream 4265 prefill / 135.8 decode-only, in band).

GEMM close-out, honestly: the z-batched column tiles (+2.6%) were this session's
GEMM win. Gate/up fusion was evaluated and skipped -- the two projections share an
L2-hot B, so a merged launch saves only launch overhead. What remains between our
54% of the 2051 TFLOP/s issue roof and CUTLASS's 68% is the producer/consumer
pipeline depth that TMA+mbarrier alone did not buy; that is a multi-week rewrite,
recorded here rather than started half-finished.

### Prefill campaign: scheduler truth, DeltaNet matmul scan, 512 waves, z-batched GEMM

Engine-level NVFP4 (`all`) prefill 6408 -> 7752 tok/s at 4096 tokens (+21%); 8042 at
2048. Server-level: cold-unique prefill 6264 -> 7357 (vs vLLM's 5218, **1.41x ahead**),
hot-unique 6264 -> 7432 (vs 9758, deficit 1.56x -> 1.31x), shared-prefix hot 96976
(2.88x ahead), TTFT p50 at N=8 1.68 -> 1.404 s (now under vLLM's 1.425). Four changes,
each gated:

**Scheduler: the "2x scheduler loss" claim was wrong -- withdrawn.** A per-wave
timeline (`/v1/wavelog`, new) showed 14 ms of host gaps in a 4.4 s N=8 run: the
Python scheduler was innocent all along. The real decode-cell gap decomposes ~85%
prefill residency in the measurement window, ~10% ride policy, ~5% step speed.
The measured ride cost is 1.2 ms/row/wave while a tail step is row-invariant, so
eager riding bought nothing under synchronized load: rides are now starve-gated
(`--fuse-idle-ms`, default 125) and the legacy min-rows/every-N step triggers only
apply with fuse off. N=8 212 -> 218, N=16 252 -> 262 at identical kernels.

**DeltaNet chunk-64 matmul scan (`TQ_DN_MM`, default on).** The head-split scan was
structurally capped at 96 CTAs walking 32 serial sub-chunks per 256 tokens; column
striping was exhausted. The per-sub-chunk map is affine in the state, so every
state-independent factor hoists into a fully parallel prep (4x4-register-tiled Gram,
DP=129 anti-bank-conflict stride -- the stride-128 version burned 60% of the kernel
in 32-way conflicts -- and a 16-row blocked forward substitution), and the scan
collapses to N/64 serial steps of register-tiled fp32 matmuls. 269 -> 188
us/layer-wave (1.42x). Numerics: maxdiff 2.1e-5 vs the per-token reference (ship
tier was 8.8e-6; N=512 both land ~2.5e-5), needle 4/4 at 262k, paged parity 11/11,
teacher-forced drift 16/257 vs the 7/257 unmodified-engine eps band. `TQ_DN_MM=0`
reverts. Side effect: FP6 spec decode 133.6 -> 141.4 tok/s (>=128-token chunk
advances route through it).

**tf32 wmma scan tier (`TQ_DN_MM=3`) -- first rejected, then SHIPPED as default
after the rejection turned out to be a harness bug.** The teacher-forced gate
corpus was the LIVE engine source file; the `#include <mma.h>` this very tier
added near the top shifted the prompt window, so the "3.1% agreement" compared
different prompts between builds. With the corpus pinned (initial-commit blob,
`TQ_BENCH_TEXT` overrides), the true same-prompt numbers on the final build:
fp32 scan 96.11% vs the old scan, **tf32 tier 97.28%** -- exactly the value the
unmodified-engine wave-width control measured, i.e. inside the engine's own eps
band -- and tf32-vs-fp32 98.05%, 16k wide argmax 60/64, needle 3/4 (the same
depth-8000 near-tie the fp32 path flips). The kernel-level distance is real and
stays documented: 1.3e-2 core error from tf32 rounding under the delta rule's
~20x cancellation, vs 2.1e-5 fp32 -- and vLLM's GDN scan runs these same
contractions in bf16, 8x coarser. +5-7% prefill on top of the fp32 scan
(4k 7773 -> 8349, 16k 6569 -> 7044, 64k 4586 -> 4813). `TQ_DN_MM=1` is the
conservative fp32 tier.

**Wave cap 256 -> 512 (default, `TQ_WAVE_MAX` overrides).** The second 256-col GEMM
pass re-reads each projection weight from L2; after the scan rewrite the weight
re-stream stopped being hidden: 7043 -> 7556 at 4096 tokens. 768+ declines
(attention's deepest-position charge grows with the wave). The old width-insensitive
result predated TMA + autotune + the scan rewrite.

**NVFP4 GEMM column tiles batched in one launch (`blockIdx.z`).** A 512-col wave ran
two host-serialized 256-col passes; tile 1's CTAs now fill tile 0's grid tail.
7556 -> 7752. Numeric gate 75 cells at ns up to 512, worst 9.8e-08.

Diagnostic footnote: two real-vs-synthetic "mysteries" during this work were my own
tooling -- a CSV parse that split kernel template names at commas (making the
harness look 30x faster than production), and an ftz/denormal theory the data then
refuted. The bank-conflict and cancellation findings above are what survived.

Final head-to-head (server-level, same client, both engines fully warm; ksl =
NVFP4 `all`, 20 slots / 340 blocks at 2-4k, 4 slots / 560 blocks at 64k):

| scenario | vllm | knivesysl | |
|---|--:|--:|---|
| prefill 2048 unique, fresh server | 5218 | 7357 | 1.41x ours |
| prefill 2048 unique, warm | 10304 | 7432 | 1.39x theirs |
| prefill 4096 unique, warm | 16080 | 7099 | 2.27x theirs |
| prefill 4096 shared prefix | 40282 | 174093 | 4.32x ours |
| prefill 16384 (conc 2) | 10104 | 5279 | 1.91x theirs |
| prefill 65536 | 7162 | 2710 | 2.64x theirs |
| ttft p50 8x2048 | 1.425 s | 1.404 s | ours |
| decode n=1 paged | 69.2 | 61.1 | 1.13x theirs |
| decode n=1 fp6 spec | 69.2 | 141.4 | 2.04x ours |
| decode n=8 / n=16 unique | 468 / 675 | 244 / 295 | prefill-residency-bound |
| decode n=8 / n=16 shared | 506 / 950 | 438 / 765 | 1.15x / 1.24x theirs |

Engine-level (chunk 512, tok/s): FP6 4869/4821/4539/3190 and NVFP4-all
8042/7769/6317/3966 at P=2048/4096/16384/65536. Paged decode (`all`):
16.7/18.0/20.0 ms/step at N=1/8/16 @2k ctx; 16.6 @16k; 20.0 @64k. vLLM's
lead grows with context: its flashattention prefill scales better than the
wide-attention kernel, which is now the largest single deficit.

Final gate sweep on the committed build: paged parity 11/11, NVFP4 numeric gate
75 cells worst 9.8e-08, FP6 single-stream 4246 prefill / 135.9 decode-only
(inside the 133.6-141.4 session band). Needle at 262k: 3/4 -- depth 8000 flips
to a `<think>` preamble and exhausts the 32-token answer window. It flips
IDENTICALLY with `TQ_DN_MM=0` (the untouched ck8 scan), so it is the documented
near-tie trajectory class at 512-column waves, not a scan regression; depths
1000/4000/11000 recover on both paths.

### Three levers closed by measurement, one small win

Working the remaining prefill list. Two of the four turned out to be worth less than
estimated, and I would rather record the numbers than the plan.

**Residual add folded into the post-norm (kept, +0.3-0.6%).** `k_tq_add_rmsnorm_b` existed
for exactly this and was dead code: all seven `add_vec(resid, h, o)` -> `wide_mlp(...,
resid, ...)` sites ran the add as its own kernel and then re-read `resid` in the post-norm.
One launch and one full read of `resid` per layer-wave, gone. Wide prefill 6386 -> 6395,
paged N=1 6300 -> 6322, N=8 4665 -> 4682, FP6 single-stream 3989 -> 4011. That is at the
harness noise floor; kept because it is strictly less work with no extra compute and no
change to any reduction order, not because the number is convincing.

**Fusing silu*mul into the NVFP4 activation quantizer -- REVERTED, measured loss twice.**
The idea was to drop the `[n x I]` fp32 intermediate (17.8 MB per layer at n=256). Two
independent reasons it does not work:

| attempt | prefill | vs baseline |
|---|--:|--:|
| baseline (materialised intermediate) | 6386 | — |
| naive fusion | 6140 | **-3.4%** |
| + stage values in smem | 5684 | -11.0% |
| + pad smem to kill bank conflicts | 5766 | -9.7% |

The quantizer reads its source **twice** (absmax pass, then encode), so fusing computes
`expf` twice per element -- the transcendental, not the read, was the price. Staging the
silu'd values in shared memory to compute it once moved that work into the kernel's
*already serialized* 8-lane absmax phase (only lanes 0..7 own a column), while 24 lanes
idle -- strictly worse. Padding the row stride to 65 floats recovered part of the
bank-conflict loss but nowhere near baseline. Fixing this properly means restructuring the
absmax so all 32 lanes participate, which changes the reduction order, for a kernel that
is 6.9% of prefill with `silu_mul` at 2.1%. Not worth it.

**Warp-specialized producer group over TMA -- not built, bounded under 0.6% of prefill.**
I had estimated ~1.1x for this, reasoning that TMA removes the `cp.async` bandwidth wall
that made the earlier attempt lose 3.7x. The wall is gone, but so is the prize: with TMA
the producer's entire job is ~5 instructions per stage (4 box loads + an `expect_tx`), so
there is almost nothing to dedicate a warp to. A producer warp removes issue *delay*;
deeper staging adds issue *depth*, which is strictly more. Measured stage sensitivity of
the shipped TMA kernel:

| shape | st=2 | st=3 | st=4 | st=5 | st=3 -> 5 |
|---|--:|--:|--:|--:|--:|
| `mlp_down` @nt=128 | 883.0 | 900.2 | 910.3 | 913.9 | **+1.5%** |
| `mlp_gate/up` @nt=128 | 806.0 | 835.8 | 850.0 | 850.0 | **+1.7%** |
| `mlp_down` @nt=256 | 1103.1 | **1119.0** | smem | smem | — |

Depth saturates at st=4 for +1.5-1.7%, and at the real nt=256 tile the shared-memory
budget caps staging at 3 anyway. So the producer-warp ceiling is under 1.7% of the GEMM =
**under 0.6% of prefill**. Warp specialization is the wrong lever on this chip for this
kernel, in both its `cp.async` and TMA forms; the 46-54%-of-issue-roof gap is elsewhere.

**Activation quantizer double-read of `x` -- moot.** That lever was the FP6 two-pass
`bfrag_absmax_wide`/`quant_wide` pair. On the NVFP4 path it does not exist:
`k_tq_nvf4_quant_x` is already single-pass. Dropped.

#### NVFP4 spec decode: limitation made explicit, not fixed

`qwn_decode`/`qwn_decode_graph` returned a documented `-120` under `TQ_W_NVFP4`, but the
spec-decode surface -- `qwn_spec_round`, `qwn_spec_forward_test`/`_graph`, `qwn_mtp_step`,
`qwn_mtp_advance`, `qwn_mtp_advance_wave`, `qwn_mtp_tree_build` -- did not. That is the
path `serve_openai.py` drives for the flagship 133.7 tok/s. It failed rather than faulted,
but with `-77`: an incidental format-gate rejection two levels down that would silently
change meaning with any edit to the wide-path format checks. All seven now return `-120`
with the reason inline. Actually supporting it needs the fused persistent layer kernel to
read NVFP4 fragments -- real work, not done, and stated as such.

#### Current prefill breakdown, for whoever picks this up

At 6392 tok/s (4096 tokens, 256-column waves, `TQ_W_NVFP4=all`):

| kernel | % |
|---|--:|
| `k_tq_nvf4_gemm_tma` | **34.8** |
| `k_tq_deltanet_chunk_hs` | **29.7** |
| `k_tq_wide_attn_mma` | 10.7 |
| `k_tq_nvf4_quant_x` | 6.9 |
| `k_tq_nvf4_reduce` | 2.9 |
| `k_tq_deltanet_prep_hs` | 2.7 |
| rmsnorm / silu_mul / add_vec | 2.2 / 2.1 / 2.1 |

The two big ones are both at documented structural limits: the GEMM at parity with CUTLASS
for the operating tile (0.99x at N=256) and 54% of the issue roof, and the scan at a
1536-warp parallelism ceiling. Neither moves without a decomposition change -- a two-level
associative scan for DeltaNet, and for the GEMM something other than warp specialization.

### DeltaNet: the state-independent prep hoisted out of the scan

The scan's dominant phase built `Lmat`/`Amat` -- `L*L` inner-product pairs, each a serial
128-step walk over D. At CK=8 that is 64 pairs against 512 threads, so **448 threads idled
through the most expensive step**, and because it is a pure function of q,k within the
sub-chunk, every stripe of a head recomputed it identically.

Split into `k_tq_deltanet_prep_hs`, grid `(nsub, value_heads)` = 1536 blocks at N=256
against the scan's 96. **Bit-exact, not eps-equivalent**: one thread per `(j,m)` pair
walking d in the same order, one per token for the `qfac`/`kfac` L2 sums, one per sub-chunk
for the serial `cum` decay. The chunk-check maxdiff against the per-token reference is
*identical* in every config (8.792e-06, 7.689e-06, 7.302e-06, 8.732e-06).

| | before | after |
|---|--:|--:|
| scan @N=256 (ck=8,ns=2,g=8) | 0.3173 ms | **0.2685** (1.18x) |
| wide prefill 4096 tok | 6008 | **6365 tok/s** |
| paged prefill N=1 | 5946 | **6309** |
| paged prefill N=8 | 4575 | **4662** |
| FP6 single-stream prefill | 3853 | **3989** |

Prep is 21.9 us of the resulting 263. Gated on `nsub>=4`: ragged multi-client waves give
`seg_len=8` per launch (32 clients over a 256-column wave), so `nsub=1`, 48 blocks, nothing
to spread -- that measured **4.3% slower** at N=32 while 5.9% faster at N=1. Above the
threshold N=32 is neutral. `TQ_DN_PREP=0` reverts.

#### What bounds this, measured

The scan's thread count is `value_heads * D * G` -- **independent of `nstripe`**. Striping
over D redistributes the same 49152 threads into more, smaller blocks and duplicates
per-block work; it buys no parallelism at all:

| ck | g | ns=1 | ns=2 | ns=4 | warps (all) |
|--:|--:|--:|--:|--:|--:|
| 16 | 8 | 0.6291 | **0.3020** | 0.4610 | 1536 |
| 8 | 8 | — | **0.2677** | 0.4205 | 1536 |
| 8 | 16 | — | 0.2939 | 0.3593 | 3072 |

Doubling warps via `g=16` does not help either. So the scan sits at a structural ceiling of
~1536 warps = 2.3 per scheduler slot on 170 SMs, and passing it needs a different
decomposition -- a two-level associative scan over sub-chunks with prefix composition --
not tuning. My earlier ~1.2-1.3x estimate for this lever assumed tensorising the state
contractions; **I am deliberately not doing that in bf16.** The state is recurrent over
150k tokens and bf16 operands would push the scan's maxdiff from 8.8e-06 to ~1e-3.

### Corrected vLLM baseline: CUDA graphs were worth 2.3x to them

Every earlier vLLM comparison in this file used `--enforce-eager`, which I had believed was
forced by this checkpoint OOMing during cudagraph memory profiling at 32 GB. It was not:
`--language-model-only` skips the vision tower, and with it vLLM runs its full production
config on this card. Re-measured, same client, both engines over HTTP:

`vllm serve unsloth/Qwen3.8-27B-NVFP4 --max-model-len 140000 --max-num-seqs 32
--gpu-memory-utilization 0.92 --kv-cache-dtype fp8 --max-num-batched-tokens 8192
--enable-prefix-caching --language-model-only`

| | vLLM eager (old) | vLLM production | gain to them |
|---|--:|--:|--:|
| prefill 4096 unique, cold | 7985 | 8095 | 1.01x |
| prefill 4096 unique, warm | 9206 | 10723 | 1.16x |
| decode N=1 | 29.2 | **68.0** | **2.33x** |
| decode N=8 | 170.8 | **461.7** | **2.70x** |

So **the claim "we win decode at every concurrency" was an artifact of a crippled
baseline** and is withdrawn. Against the real config:

| | vLLM | knivesysl | |
|---|--:|--:|---|
| prefill 4096 unique, cold | 8095 | 5625 | 1.44x theirs |
| prefill 4096 unique, warm | 10723 | 5985 | 1.79x theirs |
| prefill 4096 shared prefix, warm | 37074 | **175071** | **4.72x ours** |
| decode N=1 (clean) | 68.0 | 60.6 | 1.12x theirs |

Two things I will not paper over. **Their KV is fp8, ours is int4** -- ours is smaller and
lossier, so this is not a like-for-like quality comparison. And the **N>1 server decode
numbers are prefill-contaminated in both engines**: with `--max-prefill 2` and 256-column
waves, 8x4096 tokens of prefill overlaps the decode phase, while their
`--max-num-batched-tokens 8192` retires each prompt in one pass. Engine-level our paged
decode is 445 tok/s at N=8 (17.96 ms/step) and 1198 at N=32; the honest reading is that our
decode *kernels* are competitive and our *scheduler* gives much of it back under mixed
load. N=1 is the only uncontaminated server figure, and there we are 1.12x behind.

### The multi-client server was never running the prefill attention kernel

`TQ_WIDE_ATTN_MMA` was opt-in while the tensor-core prefill attention was being brought
up, and the flag outlived the bring-up. `serve_openai.py` set it. `bench_decode.py` set
it. **`serve_batched.py` never did.** So every number in this repo came from the MMA
path, while the multi-client paged server -- the one that actually serves the paged
prefill wave -- silently ran the memory-bound paged *decode* kernel across prompt
columns. It is now on by default; `TQ_WIDE_ATTN_MMA=0` reverts.

Found by profiling both processes and diffing `cuda_gpu_kern_sum`. Same prefill work,
different attention kernel (load-time repack/autotune are identical in both -- 496
launches, 475 vs 474 ms -- so they cancel):

| process | attention kernel | ms | launches | ms/launch |
|---|---|--:|--:|--:|
| `serve_batched.py` | `k_tq_paged_attn_q4_split_gqa<6>` | 725.2 | 384 | **1.89** |
| `bench_decode.py` | `k_tq_wide_attn_mma<4>` | 77.5 | 256 | **0.30** |

| | before | after |
|---|--:|--:|
| server-side prefill 4096 tok (HTTP) | 3336 | **5672 tok/s** |
| per-wave engine time, `T=256` | 82.1 ms | **48.3 ms** |
| single-stream wide prefill 2048 tok | 2606 | **3860 tok/s** |

**This closes the 1.79x server-vs-engine prefill gap.** That gap had been bounded to
"inside the engine call" and left open: the scheduler (0.026 ms/wave), pool geometry,
threading, streaming, tokenization (2.9 ms) and CPU contention had all been excluded by
measurement, correctly -- none of them was ever the cause. 48.3 ms/wave now agrees with
`bench_decode`'s 50.6 ms for identical `T=256`, so there is no residual to explain. The
lesson is that "I have excluded everything host-side" is not the same as "it must be
intrinsic": both processes were compared on the assumption they ran the same kernels,
and that assumption was never checked until it was profiled.

#### Third instance of one bug class

A Python-side copy of an engine default going stale, now three times in this work:

| mirror | assumed | truth | symptom |
|---|---|---|---|
| `serve_batched.py` `WAVE_MAX` | 128 columns | 256 (GEMM tile) | every wave half width, `--prefill-budget` ignored |
| `mtp_spec_smoke.py` `mma_on` | env unset == off | on | wide cap pinned at 16384; 140k prompts fell to the chunked ABI and failed `-97` |
| `serve_batched.py` attention | — | MMA available | 6.3x slower attention |

Both mirrors now ask the library: `qwn_wide_attn_mma()` joins `qwn_wave_cap()`. The
general rule this earns: **an engine default that a harness needs must be published over
the ABI, never duplicated.** A duplicated default is not wrong when written -- it is
wrong later, silently, and it corrupts the measurement rather than crashing.

#### Corrected head-to-head

Every server-level vLLM comparison in this file was measured against the slow path and
is superseded. vLLM 0.27.1 NVFP4, `--enforce-eager` (its cudagraph profiling OOMs on
this checkpoint at 32 GB), `--max-model-len 8192`, same client, both engines over HTTP:

| metric | vLLM | knivesysl | ratio |
|---|--:|--:|--:|
| prefill 4096, unique, cold | 7985 | 5648 | 0.71x |
| prefill 4096, unique, warm | 9206 | 5668 | 0.62x |
| prefill 4096, shared prefix, warm | 35324 | **171435** | **4.85x** |
| decode N=1 | 29.2 | **61.2** | **2.10x** |
| decode N=8, unique | 170.8 | **205.4** | **1.20x** |
| decode N=8, shared prefix | 220.4 | **435.1** | **1.97x** |

The cold unique-prompt prefill deficit went from **2.98x to 1.41x**. That remainder is
now genuinely the kernels -- 54% of the 2051 TFLOP/s issue roof against CUTLASS's 68%,
plus a DeltaNet chunk scan that is still scalar FP32 with no `mma.sync` in it at all.
Everything else on this card, we win.

### TMA staging + a measured launch config for the NVFP4 GEMM

The prefill GEMM was at 49% of the measured 2051 TFLOP/s `mxf4nvf4` issue roof against
CUTLASS's 68%. Everything except one term had already been excluded by measurement:
shared-memory bandwidth (warp-tile reshape 2mx16n -> 4mx8n, 1-2%), copy-issue parallelism
(warp specialisation scaled LINEARLY with producer warps and never caught all-8-warps
issue), and one of the two per-stage barriers (taken, ~2%). What remained was that all 8
warps converge on a **CTA-wide barrier every stage** -- unavoidable with `cp.async`,
because the warps that consume the data are the ones that issued it, so "bytes landed" is
only observable collectively.

`cp.async.bulk.tensor` breaks that coupling: one elected thread issues 4 box loads per
stage, the DMA engine transfers, and completion lands on an **mbarrier** each warp waits
on independently. A second barrier set (`empty[s]`, one arrival per warp) replaces the
CTA fence for buffer reuse, so a warp that finishes early goes straight to the next
stage instead of blocking on the slowest.

| N (FLOP-weighted, 64 layers) | CUTLASS | cp.async | TMA | gain | vs CUTLASS |
|--:|--:|--:|--:|--:|--:|
| 64 | 293.5 | 565.2 | 566.2 | 1.00x | **1.93x** |
| 128 | 599.2 | 712.1 | 760.8 | 1.07x | **1.27x** |
| 256 | 935.8 | 848.1 | 922.0 | 1.09x | 0.99x |
| 512 | 1185.4 | 846.5 | 930.1 | 1.10x | 0.78x |

`mlp_down` @256 is **1104 TF = 1.33x CUTLASS**; `kv_proj` 1.80x. N=256 (the prefill wave)
moves from 0.91x to **0.99x** of CUTLASS. Deterministic over 3 runs, all 28 harness cells
and all 60 in-engine cells exact (9.8e-08).

Two hardware constraints shaped it, both found by failing:

1. **`cuTensorMapEncodeTiled` caps every box dimension at 256 ELEMENTS.** A stage is 2
   k64 tiles = 288 u32 along the contiguous axis, so a stage is *two* box loads, not one.
   That also splits the shared layout: TMA writes a box contiguously, so the two k-halves
   are separate regions rather than interleaved at stride 288.
2. **The shared destination must be 128-BYTE aligned.** B's k64 half is `NG*72` u32 --
   aligned only for NG>=4. At NG=1/2 it landed at 288/576 B and the copy failed. The
   standalone harness never caught it because its smallest N is 64 (NG=8); the in-engine
   gate, which runs down to N=1, caught it immediately. Fixed by padding the half stride
   to 32 u32, with `expect_tx` still counting the *unpadded* transferred bytes (the
   barrier completes on byte count, so padding must not be counted).

#### The bigger win the hunt exposed: split-K was mispriced

End-to-end, TMA alone moved prefill ~2% -- far less than the kernel gain. The reason was
not TMA: the engine's `(ks, stages)` heuristic disagreed with the harness optimum, so the
kernel was measured at one operating point and shipped at another.

| shape | mblocks | harness best | old heuristic |
|---|--:|---|---|
| `mlp_gate/up` | 136 | k1 | k2 |
| `q_proj` | 96 | k1 | k2 |
| `mlp_down` | 40 | k4 | k8 |
| `lin_in_z` | 48 | k2 | k4 |

The heuristic priced **SM occupancy only**. Split-K also adds a full reduce pass over
`nvar*M` floats -- for `q_proj` at 256 columns that is ~25 MB of traffic against a 39 us
GEMM, ~35% overhead -- so it over-split every large-M shape. Rather than fit a better
guess, `nvf4_autotune` now **measures** `(ks, stages)` once at load, cached per `(M,K)`
(the model has ~7 distinct projection shapes, so this is a few dozen timed launches). It
reproduces the swept optimum exactly: `gate=k1s3 down=k4s3 lqkv=k2s3`.

The tuner needed one guard that is worth stating, because the first version got it wrong:
**a config that exceeds the opt-in shared limit fails to launch and therefore returns
instantly, which a timing loop reports as the fastest config.** It duly picked stages=4 at
NG=32 (110 KB against a 101 KB limit) and produced garbage. It now screens shared memory
up front and verifies the warm launch actually ran.

#### Measured

| | FP6 | NVFP4 before | + autotune | + TMA |
|---|--:|--:|--:|--:|
| wide prefill 4096 tok | 4470 | 5138 | 5847 | **5970** |
| paged prefill 2048, N=1 | 4441 | — | — | **5973** |
| paged prefill, N=8 | 3609 | — | — | **4580** |
| paged prefill, N=32 | 2204 | — | — | **2532** |
| paged decode N=1 ms/step | 18.74 | — | — | **16.68** |
| paged decode N=8 | 20.43 | — | — | **17.96** |
| paged decode N=32 | 29.51 | — | — | **26.73** |
| server-side prefill (HTTP) | — | 2678 | 3115 | **3336** |

1.34x over FP6 on wide prefill, 1.35x on paged prefill at N=1, 1.10-1.14x on paged decode.
`TQ_NVFP4_TMA=0` reverts to cp.async.

One measurement caveat worth recording: `bench_rounds.py`, `tf_agreement.py` and
`nvfp4_quality.py` all use `src/forward_qwen.cu` as their prompt corpus, so their absolute
numbers drift as that file is edited. A 139.7 -> 133.7 decode-only tok/s "regression" was
chased through three reference builds before this was the answer: all three, including the
one that had measured 139.7, gave an identical 134.1 on the current corpus. Comparisons
must be same-corpus, back to back.

### NVFP4 W4A4 tier (`TQ_W_NVFP4`)

A second weight format alongside FP6, in NVIDIA's NVFP4 numerics: E2M1 codes + per-16
`ue4m3` block scales + one fp32 global per 128-row block. **4.502 bits/weight** vs FP6's
6.002. FP6 is unchanged and remains the default; NVFP4 is a per-weight flag, not a fork.

The reason this tier exists at all is the instruction, not the bit width. The existing
`TQ_W_E2M1` tier embeds 4-bit codes in the k32 `mxf8f6f4` MMA, so it saves memory and
wins **zero** compute. NVFP4 reaches `mma.sync.kind::mxf4nvf4`, which is **k64** — double
the FLOPs per instruction at the same register and shared-memory cost. Measured on this
card with a pure issue-rate microbenchmark (`/tmp/gembench/mma_peak.cu`, operands resident
in registers, 8 independent accumulator chains, zero memory traffic):

| instruction | TFLOP/s | note |
|---|--:|---|
| `mxf8f6f4` k32 `e2m3`x`e4m3` | 998.7 | FP6, what we shipped |
| `mxf4nvf4` k64 `e2m1`x`e2m1` 4X `ue4m3` | **2051.3** | NVFP4, 2.05x |
| `mxf8f6f4` k32 `e2m1`x`e2m1` | 515.4 | the E2M1 tier — *half* rate, wrong instruction |
| plain k16 `bf16` | 253.8 | reference |

#### Format

Every layout constant was decoded on hardware with one-hot MMA probes
(`/tmp/nvf4/probe{,2,3}.cu`), not read off a spec. Per m16k64 weight tile, 576 B:

```
words [0,128)    E2M1 codes, lane L owns words [4L,4L+4)      -> one LDS.128
words [128,144)  per-16 ue4m3 scales, word 128+r = row r's 4 k-group bytes
                 lane L reads word 128 + ((L>>2) + 8*(L&1))   -> one LDS.32
A: row = (L>>2) + 8*(reg&1),  k = 32*(reg>>1) + 16*((L>>1)&1) + 8*(L&1) + nib
B: col = L>>2,                k = 32*reg      + 16*((L>>1)&1) + 8*(L&1) + nib
ue4m3: 0x38 == 1.0, 0x40 == 2.0
```

This is exactly what the MMA wants, so global->shared is a flat 16 B `cp.async`: no
swizzle, no `ldmatrix`, no software unpack. FP6 needs 3x `LDS.32` plus a LOP3 unpack per
fragment. The fp32 global is **constant over K** by construction — a per-K-block global
cannot be factored out of the K sum — so it folds once in the epilogue and the per-16
`ue4m3` absorbs all K variation.

**One probe was worth the whole exercise.** My first k-labelling was a *permutation* of
the hardware's, so each scale byte appeared to own a scattered set of 16 k-values instead
of a contiguous group. Every shape then failed the numeric gate at a suspiciously uniform
~11% relative error — which reads like a numerics problem and is actually a layout bug.
`probe3.cu` (which scale byte doubles C for which k?) settled it in one run.

#### Pipeline: one barrier per stage, and the trap in it

`k_tq_nvf4_gemm` uses **one** full-CTA barrier per stage, not two: the barrier at the top
of iteration `stg` already proves every warp finished reading `buf_{stg-1}`, so that
buffer is refilled right there. Worth ~2% FLOP-weighted, and every cell stayed exact.

The bookkeeping is load-bearing. Stage `s` must map to `cp.async` group `s` or the wait
retires the wrong copy. The prologue issues `STAGES-1` stages and iteration `stg` issues
stage `stg+STAGES-1`, giving group index == stage index. Past the last real stage an
**empty** group is still committed per iteration to keep the retire count advancing — and
those empties land *after* every real group, so they can't displace one. My first attempt
committed the empty at `stg=0`, i.e. *before* the real groups; that permanently offset the
mapping and silently read a stage whose bytes had not landed. It failed exactly the
`STAGES=2` configs. **Order of the empty commit is not cosmetic.**

#### Warp specialization: built, measured, rejected

CUTLASS's producer warpgroup does not transfer to `cp.async`. Full producer/consumer
kernel (8 consumer warps + PW producer warps, two named `bar.sync`/`bar.arrive` barriers
per stage so consumers block only on data-ready), `gate/up` at N=256:

| producer warps | TFLOP/s |
|--:|--:|
| 1 | 356.9 |
| 2 | 539.7 |
| 4 | 702.5 |
| **all 8 warps issue (no specialization)** | **959.4** |

Linear in producer count. **With `cp.async` a producer's memory-level parallelism IS its
warp count**, so dedicating warps to copying only relocates the bottleneck; pw=8 would
need 512 threads, which the 128-accumulator budget forbids. This is why CUTLASS needs
**TMA**: one elected thread issues a bulk-tensor copy and the hardware DMA supplies the
parallelism for free. Kept as `k_nvf4_gemm_ws` in `/tmp/nvf4/bench_nvf4.cu` with the
numbers in-comment so it is not re-derived.

#### Head-to-head vs CUTLASS 4.8 sm120 NVFP4

FLOP-weighted over all 64 layers, our kernel vs a standalone CUTLASS NVFP4 GEMM
(`vendor/cutlass/examples/79_blackwell_geforce_gemm`) at the same shapes:

| N | CUTLASS | ours | | regime |
|--:|--:|--:|--:|---|
| 64 | 293.5 | **565.2** | **1.93x** | decode / low concurrency |
| 128 | 599.2 | **712.1** | **1.19x** | decode at high concurrency |
| 256 | 935.8 | 848.1 | 0.91x | prefill wave |
| 512 | 1185.4 | 846.5 | 0.71x | prefill wave |

Best single cell `mlp_down` @256 = **999 TFLOP/s**. `kv_proj` is **3.81x** and `mlp_down`
**3.36x** at N=64 because CUTLASS has an **~18.3 us per-launch floor** — its `kv_proj`
time is identical (0.0183 ms) from M=64 through M=512 — where ours is 4.8 us. At ~500 GEMM
launches per decode step that floor is ~9 ms/step of pure overhead.

**We do not win prefill.** We sit at 49% of the 2051 TFLOP/s instruction roof, CUTLASS at
68%. Excluded by measurement: shared-memory bandwidth (reshaping the warp tile 2mx16n ->
4mx8n cut loads-per-MMA by a third for 1-2%), copy-issue parallelism (above), and one of
the two barriers (taken, ~2%). What remains is that all 8 warps still converge on a
CTA-wide barrier per stage. That is a **TMA gap, not a scheduling gap**.

#### Two tiers

- `TQ_W_NVFP4=all` — full W4A4, every projection.
- `TQ_W_NVFP4=mlp` — MLP-only FP4; attention and DeltaNet projections stay FP6. The MLP
  is ~70% of both FLOPs and weight bytes, so this keeps most of the win at a fraction of
  the quality cost — structurally what unsloth's mixed W8A8+W4A4 checkpoint does.
- Also `delta`, `attn`, and fine-grained `gate,up,down`, matching `TQ_W_E2M1`.

Produced by a **load-time repack** from the dense E2M3 payload already on device, read
through the logical accessor `tq_e2m3_code_at(payload, Kt, row, col)` so the packer never
has to know the FP6 fragment layout. Same precedent as `TQ_W_E2M1`. This is a
re-quantization (E2M3 -> E2M1) and therefore strictly lossier than converting from the
original bf16; emitting an NVFP4 section from `tools/convert_qwen_tqf.py` is the quality
follow-up and is **not** done. The repack needs no re-conversion of the 22.6 GB `.tqf`.

#### Measured, end to end

| | FP6 | NVFP4 `mlp` | NVFP4 `all` |
|---|--:|--:|--:|
| weights on device | 22482 MB | 19418 MB (-3063) | **18122 MB (-4359)** |
| prefill 4096 tok, 256-col waves | 4472 tok/s | 4918 (1.10x) | **5305 (1.19x)** |
| paged decode N=1 @2048 | 18.60 ms/step | - | **16.26 (1.14x)** |
| paged decode N=8 @2048 | 20.39 ms/step | - | **17.62 (1.16x)** |
| paged decode N=32 @2048 | 29.38 ms/step | - | **26.45 (1.11x)** |

Decode is faster at every concurrency level tested, and 4.36 GB freed is ~29 more paged
KV blocks' worth of headroom.

#### Verification

- `tools/nvfp4_check.py` — in-situ numeric gate on real model weights: runs the shipping
  path, compares a subsample against an fp64 reference that decodes the **same packed
  bytes**. Worst **1.11e-07** across 48 (layer, kind, N) cells for both tiers — pure fp32
  rounding. This validates fragment layout, scale mapping and MMA together, and it is what
  caught the k-group permutation and a `WM`/`NG` pairing bug (at N=8, `NG=1` with `WM=2`
  gives 4 warp columns, so warps 1-3 read past the end of the B stage buffer).
- `tools/nvfp4_quality.py` — wide-path greedy argmax over 64 independent 64-layer
  forwards (16384-token prompt, 256-column waves). Agreement vs FP6: **96.88%** for
  `mlp` (2 flips), **92.19%** for `all` (5 flips). For scale, this engine's own
  float-eps sensitivity — same weights, only the wave width changed from 256 to 128
  columns — is 97.28% (7 flips in 257). So MLP-only sits inside the engine's existing
  noise band; full W4A4 is measurably worse, which is the expected cost of 4-bit
  activations and exactly why both tiers exist.

#### Known limitation, stated plainly

**Single-stream decode (`qwn_decode` / `qwn_decode_graph`) does not support NVFP4.** It is
the fused persistent/GEMV route and reads `w->d_A`, which the repack frees. Both entries
now return **-120** with a one-line explanation instead of dereferencing a freed payload
(that surfaced as an illegal memory access three kernels downstream). NVFP4 serves prefill
and the batched/paged decode loops, which both go through `wide_proj`. For the
single-stream MTP spec-decode path, use `TQ_W_NVFP4=0`. Verified unaffected: with the
tier off, single-stream spec decode is 20.77 ms/round and 139.2 decode-only tok/s, inside
the 20.82 ms / 140.8 tok/s band from before this work.

Two existing gates drive that path and therefore **cannot** measure this tier:
`tools/tf_agreement.py` (teacher-forced top-1) and the `PARITY` test in
`tools/paged_smoke.py` (which compares paged decode *against* single-stream). Coverage for
the paged path is instead: the per-projection numeric gate above (1.11e-07 on real
weights, same GEMM the paged loop calls), the full-forward argmax agreement below, and
clean 20-step runs at N=1/8/32. No paged-specific code was touched by this work.

#### vLLM head-to-head, run

vLLM 0.27.1 serving `unsloth/Qwen3.8-27B-NVFP4` vs this engine at `TQ_W_NVFP4=all`,
**driven by the same HTTP client** (`/tmp/gembench/vll.py`), prefix caching ON in both.
TTFT with `max_tokens=1` is the prefill number; decode is N concurrent streams measured
after the last TTFT. Two prompt regimes, because they answer different questions and
conflating them is how engines get mis-marketed: `unique` = a distinct random 4096-token
window per request so nothing can be reused (raw kernel speed); `shared` = one prompt
reused (multi-turn / system-prompt / retry traffic, i.e. most real load).

|  | unique: vLLM | ksl | | shared: vLLM | ksl | |
|---|--:|--:|--:|--:|--:|--:|
| prefill cold tok/s | 7985 | 2678 | 0.34x | 8557 | 2679 | 0.31x |
| prefill warm tok/s | 9206 | 2788 | 0.30x | 35324 | **176590** | **5.00x** |
| decode N=1 tok/s | 29.2 | **63.3** | **2.17x** | 29.1 | **63.2** | **2.17x** |
| decode N=8 tok/s | 171 | 151 | 0.89x | 220 | **447** | **2.03x** |
| decode N=32 tok/s | 309 | 179 | 0.58x | 409 | **1079** | **2.64x** |

**We win decode and cached prefill; we lose cold prefill by ~3x.** On shared-prefix
traffic decode is 2.0-2.6x ahead and cached prefill 5x ahead (our prefix cache replays a
whole prefix; vLLM's Mamba `align` mode is experimental and only reached a 13.6% block
hit rate). On cold unique prefill vLLM is 3.2x faster, which is the same story the
kernel-level table tells -- we trail CUTLASS at N=256/512, the prefill regime.

Two separate losses, worth not conflating:

1. **Kernel**: we are at 49% of the 2051 TFLOP/s instruction roof vs CUTLASS's 68%. The
   remaining term is the CTA-wide barrier per stage; the fix is TMA (see above).
2. **Scheduler**: engine-level prefill is 5281 tok/s but only 2679 through our own
   server, and engine-level paged decode at N=32 is 1216 tok/s but 179 through the
   server on unique prompts (1079 on shared). So on a cold mixed load our scheduler
   gives back about half the prefill and most of the decode by starving decode while it
   prefills. That is a `serve_batched.py` scheduling problem, not a kernel problem, and
   it is the larger of the two levers on this workload.

Caveats, all in vLLM's disfavour and none of them fabricated away: it needed
`--enforce-eager` on this box (its cudagraph memory profiling OOMs -- the checkpoint
loads as multimodal + hybrid-mamba, so vLLM picks a 1568-token attention block to match
the mamba page size and then cannot fit even a minimal profiling KV cache in 32 GB), so
its decode runs without CUDA graphs; and it ran at `--max-model-len 8192` where we ran
`TQ_CTX=16384`. Its prefill numbers are unaffected by both.

#### Two quantizer changes measured and NOT built

`tools/quant_study.py` reproduces all three quantizers exactly (E2M3, NVFP4 weight,
NVFP4 activation) in fp64 and reports error on the **product** `W@x`, not on the weight
-- a rotation changes what "weight error" means, and the product is what the model
consumes. Activations come from `tools/dump_activations.py`, which pulls the engine's
**real** residual stream, because this decision cannot be made on synthetic data (see
below). Both take repo-relative / env-overridable paths.

**1. Emitting NVFP4 from the converter: real but small, so deferred.** The shipped tier
repacks weights that are already E2M3, paying two rounding steps; a converter emitting
NVFP4 would pay one. Measured product-error ratio: **1.03-1.09x**. The 4-bit E2M1 grid
dominates the error, not the intermediate E2M3 step -- so a format change on both sides
plus a 22.6 GB re-conversion buys under 10%. Not worth it yet; recorded so the next
person does not re-derive it.

**2. Hadamard rotation: actively HARMFUL here. Do not add it.** This is the interesting
one, because it contradicts the standard recipe (QuaRot, and QuTLASS/vLLM's own
"NVFP4 + Hadamard" curves). `W@x == (W@H) @ (H.T@x)` for orthogonal H, and rotating
along K is supposed to flatten outlier channels. The engine even already ships a 256-wide
FWHT for the K cache, so it would have been cheap to wire up.

First, the outliers are real and severe -- per-channel peak/median of the residual
stream, measured on this model:

| depth | peak / median |
|---|--:|
| layer 0 (embedding) | 1.9x |
| layer 8 | 81x |
| layer 32 | **245x** |

And yet rotation makes it worse, monotonically in the rotation block size (product error,
real layer-32 activations):

| tensor | shipped | no rotation | had16 | had32 | had64 | had256 |
|---|--:|--:|--:|--:|--:|--:|
| `mlp.gate_proj` | 0.1167 | **0.1068** | 0.1242 | 0.1306 | 0.1276 | 0.1547 |
| `mlp.down_proj` | 0.1052 | **0.0997** | 0.1112 | 0.1143 | 0.1222 | 0.1407 |
| `linear_attn.in_proj_qkv` | 0.1055 | **0.1014** | 0.1115 | 0.1126 | 0.1183 | 0.1391 |

**Why: NVFP4's scale group is only 16 elements.** An outlier damages exactly its own
group of 16 and none of the other 319; a rotation SPREADS it across every group in the
block, raising all of their scales. Localizing an outlier beats diluting it once the
scale granularity is at or below the outlier's own footprint. The monotonic rise with
block size is the mechanism showing itself -- and even `had16`, matched exactly to the
scale group, loses, because mixing turns one large-plus-fifteen-small group (which E2M1's
8 magnitudes represent well) into sixteen mid-magnitude values (which it does not).

Rotation is the right tool for per-tensor or per-channel scaling. At per-16 it is a
pessimization, and the ~1-2% activation-FWHT cost would have been paid for negative
quality. This is why the study ran before the implementation.

**A caution on how this was almost gotten wrong:** the first pass used Gaussian synthetic
activations and reported Hadamard as a flat 1.03-1.05x -- i.e. "no effect". That was not a
result, it was a broken experiment: a Gaussian has no outlier channels, so nothing exists
to flatten and any rotation must look neutral. Real activations were required to see that
the effect is not merely absent but negative.

### Hot kernels: prefill GEMM, DeltaNet scan, paged decode attention

The three kernels that owned prefill and long-context decode were rewritten. Everything
below is measured on this card against the immediately preceding commit, built as a
second `.so` from `git show HEAD:src/forward_qwen.cu` and driven by the same harness.

#### `k_tq_fp6_gemm_mma` — tiled, `cp.async`-pipelined wide FP6 GEMM

`k_tq_fp6_wide_gemm` ran **one warp per CTA** with every operand arriving straight from
global through `__ldg`, so each `mma.sync` sat behind its own load-use chain, occupancy
was register-capped at ~12 warps/SM, and the per-128-K-block weight and activation
scales were re-fetched on every k32 tile. The replacement changes only the schedule:

- block tile 128 rows x `NG*8` columns, 8 warps as 4(M) x 2(N) — warp tile 32 x `NG*4`,
  so `NG` MMAs per k32 tile off `NG*2` accumulator registers;
- K staged 128 at a time = **exactly one block scale**, so `sfa`/`sfb` are loaded once
  per stage instead of once per MMA group per warp;
- A and B `cp.async`'d into a `STAGES`-deep circular smem buffer. The `.tqf` QMMA
  fragment layout *is* the layout the MMA wants, so the copy is a flat 16 B
  `cp.async.cg` with no swizzle and no transform, and operand A comes back as three
  bank-conflict-free `LDS.32` per lane (word stride 3 is coprime with 32). CUTLASS needs
  `Swizzle<3,4,3>` + `ldmatrix.b6x16_p32` only because its global layout is canonical
  TN; ours is not, and here that is an advantage;
- split-K rides `grid.y` into `[split][nvar*M]` partials, chosen by the largest split
  that still lands the whole grid in one wave of the 170 SMs (the kernel is 1 CTA/SM at
  >48 KB smem, so a second wave costs a full extra pass);
- column tile raised 128 -> 256 (`NG=32`, 128 accumulator registers, 240 registers
  total, **0 bytes spilled**) because the operands now come from smem rather than 32
  live `__ldg` results. That is what makes a wide wave amortize the weight read.

`tk` is still accumulated in ascending order into the same C register, so at
`k_splits == 1` this is **bit-exact** vs the kernel it replaces. Verified with
order-independent operands — E2M3 codes drawn from {0.5, 1, 1.5, 2}, E4M3 activations
from {0.5, 1, 2, 4} and pow2 block scales from {1, 2}, so every partial product is a
multiple of 0.25 and the K-sum stays far below 2^24. The fp32 result is then *exact* and
therefore independent of summation order, which means a mis-indexed operand or block
scale shows up as a real difference while pure reassociation cannot hide one. Zero
differing bits at every (shape, N, stages, split) in the sweep. With split-K the
reduction order does change: max relative difference 3.8e-6 under the realistic pow2
scale distribution.

FLOP-weighted over this model's real projection mix (`mlp_gate/up` 17408x5120,
`mlp_down` 5120x17408, `q_proj` 12288x5120, `kv_proj` 1024x5120, `o_proj`/`linear_out`
5120x6144, `linear_in_qkv` 10240x5120, `linear_in_z` 6144x5120, `linear_in_b`/`_a`
48x5120, weighted by the 16 full-attention + 48 DeltaNet layer counts):

| columns/wave | before | after | CUTLASS 4.8 sm120 collective |
|---|--:|--:|--:|
| 128 | 88.9 TFLOPS | **392.8** | 255 (its M=128) |
| 256 | 88.1 TFLOPS | **477.7** | 572 (its M=512) |

Per shape at 256 columns: `mlp_down` 63.7 -> **579.8** TFLOPS (9.1x), `mlp_gate/up`
206.4 -> **538.0** (2.6x), `lin_in_qkv` 116.5 -> **499.9**, `o_proj` 63.9 -> **451.2**,
`q_proj` 148.5 -> **391.0**, `kv_proj` 12.8 -> **166.2**. `mlp_down` and `mlp_gate/up`
at 128 columns run at 1.40 and 1.29 TB/s — i.e. the GEMM is now against the DRAM
roofline, not the tensor cores. In the decode regime it sustains **1.56 TB/s, 87% of
the 1.79 TB/s peak**, so there is nothing left to take there.

**Pipeline bug worth recording.** The first version dropped the `cp.async.commit_group`
when the tail had nothing left to issue. `cp.async.wait_group N` means *leave at most N
groups pending*, so once the outstanding count falls below `STAGES` the wait stops
covering the stage about to be read: the last `STAGES-1` stages are consumed before
their bytes land. It only reproduced on the largest weight, where the A stream misses
L2, and it presented as ~2% of outputs differing non-deterministically between two runs
of the identical config. The tail now commits an empty group.

#### `k_tq_deltanet_chunk_hs` — chunkwise gated-DeltaNet scan

Once the GEMM was fixed the 48 DeltaNet layers became the co-bottleneck (22.5% -> 38.5%
of prefill). Three changes, all order-preserving by construction:

- **the recurrent state lives in registers across the whole chunk.** Each thread owns
  `D/G = 16` rows of its value column, so the `[D x SV]` stripe is read from global once
  at entry and written back once at exit. Before, step 3 re-read and step 8 re-read *and
  wrote* the full 32 KB stripe every CK=8 tokens: at N=256 that is 32 sub-chunks x 96 KB
  = 3 MB per CTA, 288 MB per layer per wave, for a 3 MB working set.
- **the per-token prep is hoisted and parallelised.** The q/k inverse-L2 factors ran on
  8 threads of 512, each looping `d = 0..127` over uncoalesced global reads, and the
  beta / cumulative-log-decay loop ran on thread 0 alone — both *inside* the sub-chunk
  loop, so 512-thread barriers serialised behind ~8 and ~1 active threads, `2 * N/CK`
  times per launch. They are now a pre-pass over a super-chunk of up to `NMAX=512`
  tokens: one thread per token for the factors, one per sub-chunk for beta/la. Each item
  keeps its own serial accumulation order.
- **no idle-lane phase.** The `(I+L)Delta=R` substitution ran under `klane == 0`, 1 of
  G lanes, and Delta was then shuffle-broadcast. Those G lanes are in the same warp, so
  masking them off saves nothing — the warp issues the same instructions either way.
  Reducing `kS0`/`qS0` with an XOR butterfly instead of a `shfl_down` tree leaves every
  lane holding the value the old lane 0 held (same operands, same tree depth, sibling
  operands only commuted, and fp add is commutative), so all lanes run the substitution
  and the broadcast disappears.

Shipping config `ck=8 nstripe=2 g=8`: **0.2525 -> 0.1616 ms** at N=128, **0.5017 ->
0.3156** at N=256, **1.0005 -> 0.6398** at N=512 (1.56-1.59x), against the per-token
reference at 18.4x. Agreement with that reference is unchanged in magnitude (5.3e-6 vs
4.9e-6 at N=128, 8.8e-6 vs 7.6e-6 at N=256, 2.54e-5 vs 2.52e-5 at N=512): the same
float-eps band, not bit-identical — FMA contraction shifts under the new register
envelope. `__launch_bounds__` added so the wide-G configs stay launchable.

#### `k_tq_paged_attn_q4_split_gqa` — GQA-shared paged decode attention

`k_tq_paged_attn_q4_split` runs one CTA per **query** head, and this model is nh=24 /
nkv=4, so six CTAs independently streamed the same KV rows. At 1 client x 131072
positions the 16 full-attention layers hold 3.52 GiB of Q4 K + E4M3 V and a decode step
was moving ~21 GiB of it — 27 of the 46 ms. One CTA now owns a (kv head, column, split)
and carries the `nh/nkv` query heads that share it, so each K row and each V row is
fetched once and fed to all six. Preserved element-for-element: the Q norm/RoPE/FWHT
arithmetic and its 256-thread reduction, the `e = 0..7` order of the K dot (with the
dequantized `(code - zp) * scale` hoisted out of the head loop — same expression), the
256-thread max/denominator tree (walked once for all six heads instead of six times),
and the V fold written `s * vcode * vscl` rather than `s * (vcode * vscl)` so the
association matches the scalar kernel. The output layout is unchanged, so
`k_tq_paged_attn_q4_merge` is reused verbatim and S=1 also routes through split+merge
(at S=1 the merge is `exp(0)=1 * acc/l * sigmoid(gate)`, exactly what the single kernel
writes). `paged_split_S_gqa` compensates for having 6x fewer CTAs per column and lifts
the split cap 32 -> 64. `TQ_PAGED_GQA=0` reverts.

Gate: **6/6 argmax match** vs single-stream Q4 decode at 1200 context
(`tools/paged_smoke.py` PARITY).

#### Smaller fixes

- **`tq_norm_stripes`.** `k_tq_qwen_rmsnorm_b` was launched `dim3(8, rows)` everywhere.
  The 8-way `.x` split exists so a *single* row still engages many SMs, but every block
  re-reads the whole row to build `sum_sq`, so past ~170/rows stripes it is pure read
  amplification — a 256-row prefill wave was reading 8 x 5.24 MB per norm, and there are
  three norms per layer. The stripe count is now `clamp(170/rows, 1, 8)` at the
  wide/batched/paged call sites; bit-identical either way, since the reduction order
  depends on `blockDim.x` and `N` and never on `gridDim.x`.
- **Per-wave host sync removed.** `run_wide_chunk_forward` built the position array with
  `malloc` + H2D + `cudaStreamSynchronize` — draining the entire pipeline once per wave
  (8 full stalls in a 2048-token prefill) purely to know when the host buffer could be
  freed. Positions are an iota, so `k_tq_fill_iota` writes them on the device; the
  capture-mode guard is gone because the device fill is graph-capturable.
- **`TQ_WAVE_MAX` default 128 -> 256**, matching the GEMM's native column tile so a wave
  reads the ~20 GiB weight stripe once for twice the prompt columns. The old 128 was
  inherited from the single-stream wide path (whose batched attention really does cap at
  128, `-57`).
- `qwn_wide_gemm_check` / `_bench` size their partial slab for whichever split heuristic
  runs, and `launch_fp6_gemm_mma_tile` clamps its split to the caller's slab so a
  heuristic change can never overrun it.

#### Measured: end to end

Prefill, single stream (`qwn_prefill_wide`, `TQ_WIDE_ATTN_MMA=1`):

| case | HEAD | new | |
|---|--:|--:|--:|
| 2048 tok, 256-col waves | 2647 tok/s | **4428** | 1.67x |
| 4096 tok, 256-col waves | 2579 | **4501** | 1.75x |
| 4096 tok, 512-col waves | 2579 | **4630** | 1.80x |

Prefill, paged multi-client ragged waves (N clients x 2048 tokens):

| N | HEAD | new | |
|--:|--:|--:|--:|
| 1 | 2644 tok/s | **4453** | 1.68x |
| 4 | 2386 | **4071** | 1.71x |
| 8 | 2090 | **3632** | 1.74x |
| 16 | 1723 | **2954** | 1.71x |
| 32 | 1198 | **2216** | 1.85x |

Decode, paged continuous batch:

| case | HEAD | new | | HEAD tok/s | new tok/s |
|---|--:|--:|--:|--:|--:|
| N=1, ctx 2048 | 19.70 ms | **18.65** | 1.06x | 50.8 | **53.6** |
| N=8, ctx 2048 | 21.67 | **20.36** | 1.06x | 369 | **393** |
| N=16, ctx 2048 | 29.14 | **26.39** | 1.10x | 549 | **606** |
| N=32, ctx 2048 | 41.98 | **29.36** | 1.43x | 762 | **1090** |
| N=1, ctx 65536 | 31.31 | **21.93** | 1.43x | 31.9 | **45.6** |
| N=1, ctx 131072 | 46.53 | **27.83** | 1.67x | 21.5 | **35.9** |
| N=1, ctx 147456 | 50.06 | **29.24** | 1.71x | 20.0 | **34.2** |

(the 147456 row is `TQ_PAGED_GQA=0` vs `=1` on the same build — the reference lib is not
needed there because attention is the only thing that moved at that depth)

Prefill kernel time moved from GEMM 55% / DeltaNet 22.5% to GEMM 47.5% / DeltaNet 26.9%
(nsys, 2048-token prefill, 256-column waves), i.e. both shrank and the mix is still
balanced. A decode step at ctx 2048 / N=8 is now 66.6% projection GEMM (at 87% of the
DRAM roofline), 17.6% -> ~3% attention, 8.1% DeltaNet recurrence.

#### Divergence, honestly

Split-K and the DeltaNet FMA shift are reassociations, so greedy trajectories drift. The
control that matters: the **unmodified** engine, changing only the prefill wave width
from 256 to 128 columns, already gives 97.28% teacher-forced top-1 agreement (7 flips in
257 positions on a 2048-token prompt). The new engine against `HEAD` at the same wave
width gives **96.50%** (9 flips) — inside the engine's own float-eps sensitivity. The
DeltaNet rewrite alone accounts for 8 of those 9. Long-context retrieval is unaffected:
`tools/needle_check.py` recovers **4/4** needles at depths 1000/4000/8000/11000 in an
11.5k haystack, and **2/2** at depths 4000/11000 in a **140 000-token** haystack — the
regime the GQA attention rewrite targets.

#### Not regressed

The single-stream MTP spec-decode path decodes through the persistent-layer kernels, not
the wide GEMM, so it is expected to be flat and is: **20.82 -> 20.84 ms/round**, 135.7 ->
140.8 decode-only tok/s at a 2048-token context (`tools/bench_rounds.py`, 120 rounds;
the tok/s delta is accept length 2.83 -> 2.93, i.e. near-tie noise, not a speedup). Its
*prefill* does go through the wide path: 2630 -> 3891 tok/s.

#### Tools

- **`tools/bench_decode.py`** — paged continuous-batch decode sweep: ragged batched
  prefill to depth P across N slots, then timed decode steps. Sizes the pool per case
  (per-slot DeltaNet state is ~145 MiB and a page=128 block ~3.44 MiB, so a fixed
  `--max-slots`/`--num-blocks` either fails to allocate or starves the KV pool). This is
  what reproduces the decode table above.
- **`tools/tf_agreement.py`** — teacher-forced top-1 agreement: prefill, then feed the
  *true* continuation token at every step and record the argmax, so two libs driven
  identically diverge only through numerics. Run the same lib at two wave widths first —
  that control is the engine's own float-eps sensitivity, and a real regression has to be
  worse than it.
- Both take repo-relative defaults and `TQ_MODEL_TQF` / `TQ_MODEL_DIR` / `TQ_LIB`.

#### Observed, pre-existing, not fixed here

`tools/needle_check.py` without `TQ_WIDE_PREFILL=1` falls through to `qwn_prefill_chunk`
and fails with `-97` on this model at `TQ_CTX=262144`. Reproduced identically on the
pre-change lib, so it is not a regression from this work — but the non-wide chunked
prefill path is evidently untested at full context. Both servers set
`TQ_WIDE_PREFILL=1`, which is why it has not surfaced.

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
- **`tools/axe_vllm.py`** — local agentic terminal client for OpenAI-compatible vLLM.
  It discovers the live model/context, streams thinking separately from answer text,
  preserves assistant/tool turns for prefix-cache reuse, executes bounded tool loops,
  and supports both interactive sessions and `--prompt` automation.
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
