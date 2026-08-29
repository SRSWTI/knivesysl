# Changelog

All performance numbers are measured on this machine: RTX 5090 (GB202, SM120, 170 SM,
32 GB, 128 MB L2), CUDA 13.3, driver 595, Qwen3.8-27B FP6 (E2M3) + Q4 KV, greedy,
thinking off unless stated. vLLM comparisons are vLLM 0.27.1 serving
`unsloth/Qwen3.8-27B-NVFP4` (mixed W8A8 + W4A4, 22.5 GB) at `--max-model-len 16384
--gpu-memory-utilization 0.90`, driven by the same client.

## Unreleased

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
