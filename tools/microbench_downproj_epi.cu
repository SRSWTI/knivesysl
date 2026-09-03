// Standalone sm120 proof for docs/level-up.md stage 1b: persistent tile-parallel
// down/o-projection epilogue. Does NOT modify or link the production engine.
//
// Boundary under test (per projection, n=1 decode):
//   control: producer(split-K partials) -> reduce -> add_rms_fac_b -> quant_x_nw   (4 launches)
//   fused:   producer(+DSM fold +residual add +exact-order RMS +register quant)    (1 launch)
//
// Bit-exactness contract (fused vs control, same synthetic inputs):
//   - residual bytes identical
//   - rms factor bytes identical (owner-0 emulates the exact 1024-lane shfl tree)
//   - packed NVFP4 codes + scale words identical (production encode math cloned)
//
// The producer is shared source between arms so partials are identical by
// construction; only the boundary organization differs. Producer uses scalar
// NVFP4 dequant FMA (slower than the production TMA GEMM in absolute terms;
// both arms pay it equally, so the arm delta isolates the boundary cost).
//
// Build: nvcc -std=c++17 -O3 -lineinfo -generate-code=arch=compute_120,code=sm_120 \
//        -Xptxas=-v -o results/microbench/downproj_epi tools/microbench_downproj_epi.cu
// Run:   results/microbench/downproj_epi <iters> <warmup> [K] [ks]

#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>

namespace cg = cooperative_groups;

#define CUDA_OK(expr) do { \
    cudaError_t status_ = (expr); \
    if (status_ != cudaSuccess) { \
        std::fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__, #expr, \
                     cudaGetErrorString(status_)); \
        std::exit(1); \
    } } while (0)

// ---------------------------------------------------------------------------
// Production math clones (src/forward_qwen.cu). Values must stay bit-identical.
// ---------------------------------------------------------------------------

static __host__ __device__ __forceinline__ float bf16_to_float(uint16_t h) {
    uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

// tq_ue4m3_to_f (forward_qwen.cu:11351)
static __host__ __device__ __forceinline__ float ue4m3_to_f(uint8_t b) {
    int e = (b >> 3) & 0xF, m = b & 0x7;
    if (e == 0) return (float)m * 0.125f * 0.015625f;
    return ldexpf(1.0f + (float)m * 0.125f, e - 7);
}

// tq_f2ue4m3 (forward_qwen.cu:11356)
static __host__ __device__ __forceinline__ uint8_t f2ue4m3(float v) {
    if (!(v > 0.0f)) return 0;
    int e; float m = frexpf(v, &e);
    int E = e + 6, f = (int)lrintf((2.0f * m - 1.0f) * 8.0f);
    if (f > 7) { f = 0; E++; }
    if (E > 15) { E = 15; f = 7; }
    if (E < 1) return 0;
    return (uint8_t)((E << 3) | f);
}

// tq_quant_e2m1_code (forward_qwen.cu:11238)
static __device__ __forceinline__ uint32_t quant_e2m1_code(float v) {
    float a = fminf(fabsf(v), 6.0f);
    uint32_t code;
    if (a < 0.75f) {
        code = (a < 0.25f) ? 0u : 1u;
    } else {
        int e = (a < 2.0f) ? 1 : (a < 4.0f ? 2 : 3);
        float sc = (float)(1 << (e - 1));
        int m = (int)((a / sc - 1.0f) * 2.0f + 0.5f);
        if (m >= 2) { m = 0; e += 1; if (e > 3) { e = 3; m = 1; } }
        code = ((uint32_t)e << 1) | (uint32_t)m;
    }
    if (v < 0.0f) code |= 0x8u;
    return code;
}

// tq_nvf4_b_k (forward_qwen.cu:11370 comment): activation-side slot->k map.
static __host__ __device__ __forceinline__ int nvf4_b_k(int lane, int reg, int nib) {
    return 32 * reg + 16 * ((lane >> 1) & 1) + 8 * (lane & 1) + nib;
}

// E2M1 magnitude grid for the synthetic weight dequant (sign in bit 3).
__device__ __constant__ float c_e2m1_lut[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};

// ---------------------------------------------------------------------------
// Problem shape. M fixed at 5120 (down/o projection output width).
//   M tiles: 40 x 128 rows; each tile owner also owns 2 k64 quant groups of the
//   next activation (Kt64 = 80). TQ_NVF4_BW = 72 u32 per n8k64 activation tile.
// ---------------------------------------------------------------------------

#define M_DIM      5120
#define TILE_ROWS  128
#define N_TILES    (M_DIM / TILE_ROWS)          // 40
#define KT64_OUT   (M_DIM / 64)                 // 80 activation tiles downstream
#define NVF4_BW    72
#define RMS_EPS    1e-6f

// Synthetic weight layout per (tile, kt): 1024 u32 packed codes (128 rows x 64 k
// at 4 bits) + 128x4 scale bytes as 128 u32. Row-major within the tile.
struct WeightBuf {
    const uint32_t *codes;   // [tile][kt][row][8]  (8 u32 = 64 codes)
    const uint32_t *scales;  // [tile][kt][row]     (4 e4m3 bytes = 4 k16 groups)
};

// ---------------------------------------------------------------------------
// Shared producer: one CTA computes a split-K partial for one 128-row tile.
//   grid.x = N_TILES * KS (control) or cluster-organized identically (fused).
//   256 threads: row = tid & 127, half = tid >> 7 splits the k-range in two;
//   halves combine through smem in a fixed order -> identical in both arms.
// Returns the partial in p_out[row] (smem), and the k-range sum association is
// fixed: serial kt ascending, g ascending, t ascending, halves folded lo+hi.
// ---------------------------------------------------------------------------

template <int KS>
static __device__ __forceinline__ void producer_partial(
    const WeightBuf w, const float *__restrict__ x, int K, int tile, int split,
    float *p_half, float *p_out) {
    const int tid  = threadIdx.x;
    const int row  = tid & (TILE_ROWS - 1);
    const int half = tid >> 7;
    const int Kt   = K / 64;
    const int kt_per_split = Kt / KS;
    const int kt_lo = split * kt_per_split + half * (kt_per_split / 2);
    const int kt_hi = kt_lo + kt_per_split / 2;
    float acc = 0.0f;
    for (int kt = kt_lo; kt < kt_hi; kt++) {
        const size_t base = ((size_t)tile * Kt + kt);
        const uint32_t sw = w.scales[base * TILE_ROWS + row];
        const uint32_t *cw = w.codes + (base * TILE_ROWS + row) * 8;
        for (int g = 0; g < 4; g++) {
            const float s = ue4m3_to_f((uint8_t)((sw >> (8 * g)) & 0xFF));
            uint32_t w0 = cw[g * 2 + 0], w1 = cw[g * 2 + 1];
            for (int t = 0; t < 8; t++) {
                uint32_t c0 = (w0 >> (4 * t)) & 0xF;
                uint32_t c1 = (w1 >> (4 * t)) & 0xF;
                float v0 = c_e2m1_lut[c0 & 7] * ((c0 & 8) ? -s : s);
                float v1 = c_e2m1_lut[c1 & 7] * ((c1 & 8) ? -s : s);
                int k = kt * 64 + g * 16;
                acc = fmaf(v0, x[k + t], acc);
                acc = fmaf(v1, x[k + 8 + t], acc);
            }
        }
    }
    p_half[half * TILE_ROWS + row] = acc;
    __syncthreads();
    if (half == 0) p_out[row] = p_half[row] + p_half[TILE_ROWS + row];
    __syncthreads();
}

// ---------------------------------------------------------------------------
// Control arm kernels (production-shaped clones).
// ---------------------------------------------------------------------------

template <int KS>
__global__ void __launch_bounds__(256, 2)
k_ctl_producer(const uint32_t *codes, const uint32_t *scales,
               const float *__restrict__ x, int K, float *partials) {
    __shared__ float p_half[2 * TILE_ROWS];
    __shared__ float p_out[TILE_ROWS];
    const int tile = blockIdx.x / KS, split = blockIdx.x % KS;
    WeightBuf w{codes, scales};
    producer_partial<KS>(w, x, K, tile, split, p_half, p_out);
    const int tid = threadIdx.x;
    if (tid < TILE_ROWS)
        partials[(size_t)split * M_DIM + tile * TILE_ROWS + tid] = p_out[tid];
}

// k_tq_nvf4_reduce shape: fold split partials serially in split order.
template <int KS>
__global__ void k_ctl_reduce(const float *partials, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M_DIM) return;
    float acc = partials[i];
    for (int s = 1; s < KS; s++) acc += partials[(size_t)s * M_DIM + i];
    y[i] = acc;
}

// k_tq_add_rms_fac_b clone (forward_qwen.cu:4729), N rows = 1.
__global__ void __launch_bounds__(1024, 1)
k_ctl_add_rms_fac(float *fac, float *resid, const float *a, const float *b,
                  int N, float eps) {
    __shared__ float smem[32];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;
    float sum_sq = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float v = a[i] + b[i];
        resid[i] = v;
        sum_sq += v * v;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    if (lane == 0) smem[warp_id] = sum_sq;
    __syncthreads();
    if (warp_id == 0) {
        sum_sq = (lane < (blockDim.x + 31) / 32) ? smem[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        if (lane == 0) fac[0] = rsqrtf(sum_sq / (float)N + eps);
    }
}

// k_tq_nvf4_quant_x_nw clone (forward_qwen.cu:11888), nvar=1.
__global__ void k_ctl_quant_x_nw(uint32_t *b, const float *X, const float *fac,
                                 const uint16_t *nw, int nvar, int K,
                                 int NGgroups, int Kt64) {
    long tile = (long)blockIdx.x;
    if (tile >= (long)NGgroups * Kt64) return;
    int g8 = (int)(tile / Kt64), kt = (int)(tile % Kt64);
    int lane = threadIdx.x;
    uint32_t *dst = b + tile * NVF4_BW;
    __shared__ float sc[8][4];
    if (lane < 8) {
        int col = g8 * 8 + lane;
        float f = (col < nvar) ? fac[col] : 0.f;
        uint32_t w = 0;
        for (int g = 0; g < 4; g++) {
            float mx = 0.f;
            for (int t = 0; t < 16; t++) {
                int k = kt * 64 + g * 16 + t;
                float v = (col < nvar && k < K)
                    ? X[(size_t)col * K + k] * f * (1.0f + bf16_to_float(nw[k])) : 0.f;
                mx = fmaxf(mx, fabsf(v));
            }
            uint8_t sb = f2ue4m3(mx / 6.0f);
            if (sb == 0) sb = 1;
            w |= (uint32_t)sb << (8 * g);
            sc[lane][g] = ue4m3_to_f(sb);
        }
        dst[64 + lane] = w;
    }
    __syncthreads();
    uint32_t words[2] = {0, 0};
    int col = g8 * 8 + (lane >> 2);
    float f = (col < nvar) ? fac[col] : 0.f;
    for (int reg = 0; reg < 2; reg++) {
        for (int j = 0; j < 8; j++) {
            int k = nvf4_b_k(lane, reg, j);
            int gk = kt * 64 + k;
            float v = (col < nvar && gk < K)
                ? X[(size_t)col * K + gk] * f * (1.0f + bf16_to_float(nw[gk])) : 0.f;
            float s = sc[lane >> 2][k >> 4];
            words[reg] |= quant_e2m1_code(s > 0.f ? v / s : 0.f) << (4 * j);
        }
    }
    dst[lane * 2 + 0] = words[0];
    dst[lane * 2 + 1] = words[1];
}

// ---------------------------------------------------------------------------
// Fused arm: one launch. Cluster = KS CTAs over one tile. Rank 0 owns the tile:
//   DSM fold (split order) -> residual add -> global resid publish ->
//   sense-barrier -> owner-0 exact-order RMS -> fac publish -> all owners
//   quantize their 2 k64 groups from registers/smem with production encode.
// ---------------------------------------------------------------------------

struct FusedCtl {
    float *resid;        // [M]
    float *fac;          // [1]
    uint32_t *qout;      // [KT64_OUT * NVF4_BW]
    float *y_dbg;        // optional fold output for debugging (may be null)
    unsigned int *arrive;   // sense-barrier counter
    volatile unsigned int *epoch;  // barrier epoch flag
    volatile unsigned int *fac_ready;
};

template <int KS>
__global__ void __launch_bounds__(256, 2)
k_fused_epi(const uint32_t *codes, const uint32_t *scales,
            const float *__restrict__ x, int K,
            const float *__restrict__ resid_in, const uint16_t *__restrict__ nw,
            FusedCtl ctl, unsigned int pass) {
    __shared__ float p_half[2 * TILE_ROWS];
    __shared__ float p_out[TILE_ROWS];
    __shared__ float mailbox[KS > 1 ? (KS - 1) * TILE_ROWS : 1];
    __shared__ float tile_vals[TILE_ROWS];       // finalized residual rows
    __shared__ float rms_stage[1024];            // owner-0 exact-tree staging
    __shared__ float s_fac;

    cg::cluster_group cluster = cg::this_cluster();
    const int rank = (int)cluster.block_rank();
    const int tile = blockIdx.x / KS;
    const int tid  = threadIdx.x;

    WeightBuf w{codes, scales};
    producer_partial<KS>(w, x, K, tile, rank, p_half, p_out);

    if (KS > 1) {
        if (rank != 0) {
            // Push this split's partial into the owner's mailbox slot rank-1.
            float *dst = cluster.map_shared_rank(mailbox, 0);
            if (tid < TILE_ROWS) dst[(rank - 1) * TILE_ROWS + tid] = p_out[tid];
        }
        cluster.sync();
        if (rank != 0) return;   // non-owners done (cluster stays resident until all exit)
    }

    // Owner: fold in split order (matches k_ctl_reduce association), add residual.
    if (tid < TILE_ROWS) {
        float acc = p_out[tid];
        for (int s = 1; s < KS; s++) acc += mailbox[(s - 1) * TILE_ROWS + tid];
        int gi = tile * TILE_ROWS + tid;
        float v = resid_in[gi] + acc;
        ctl.resid[gi] = v;
        tile_vals[tid] = v;
        if (ctl.y_dbg) ctl.y_dbg[gi] = acc;
    }
    __syncthreads();
    __threadfence();

    // Sense barrier arrival (one per owner CTA).
    if (tid == 0) atomicAdd(ctl.arrive, 1u);

    if (tile == 0) {
        // Owner-0 emulates k_tq_add_rms_fac_b's exact 1024-lane reduction over the
        // globally published residual. Wait for all owners first.
        if (tid == 0) {
            while (atomicAdd(ctl.arrive, 0u) < pass * N_TILES) { }
        }
        __syncthreads();
        // Virtual lanes: vt in [0,1024); serial strided sums exactly as tid-strided
        // loop in the 1024-thread kernel.
        for (int vt = tid; vt < 1024; vt += blockDim.x) {
            float sum_sq = 0.0f;
            for (int i = vt; i < M_DIM; i += 1024) {
                float v = ctl.resid[i];
                sum_sq += v * v;
            }
            rms_stage[vt] = sum_sq;
        }
        __syncthreads();
        // Exact shfl-down tree per virtual warp, then warp-0 tree over warp sums.
        if (tid < 32) {
            float lanes[32];
            for (int l = 0; l < 32; l++) lanes[l] = rms_stage[tid * 32 + l];
            for (int offset = 16; offset > 0; offset >>= 1)
                for (int l = 0; l < offset; l++) lanes[l] += lanes[l + offset];
            rms_stage[tid] = lanes[0];
        }
        __syncthreads();
        if (tid == 0) {
            float lanes[32];
            for (int l = 0; l < 32; l++) lanes[l] = rms_stage[l];
            for (int offset = 16; offset > 0; offset >>= 1)
                for (int l = 0; l < offset; l++) lanes[l] += lanes[l + offset];
            float f = rsqrtf(lanes[0] / (float)M_DIM + RMS_EPS);
            ctl.fac[0] = f;
            __threadfence();
            *ctl.fac_ready = pass;
        }
    }

    // All owners: wait for the factor, then quantize their 2 k64 groups with the
    // production layout (only lanes 0..3 carry data at nvar=1; others zero/sb=1).
    if (tid == 0) {
        while (*ctl.fac_ready < pass) { }
        s_fac = ctl.fac[0];
    }
    __syncthreads();
    const float f = s_fac;

    // 64 active threads: kt_local in {0,1}, lane in [0,32).
    if (tid < 64) {
        int kt_local = tid >> 5;
        int lane = tid & 31;
        int kt = tile * 2 + kt_local;
        uint32_t *dst = ctl.qout + (size_t)kt * NVF4_BW;
        __shared__ float sc_sh[2][8][4];
        if (lane < 8) {
            uint32_t wword = 0;
            for (int g = 0; g < 4; g++) {
                float mx = 0.f;
                if (lane == 0) {
                    for (int t = 0; t < 16; t++) {
                        int k = kt * 64 + g * 16 + t;      // k inside [tile*128, ...)
                        int local = k - tile * TILE_ROWS;
                        float v = tile_vals[local] * f * (1.0f + bf16_to_float(nw[k]));
                        mx = fmaxf(mx, fabsf(v));
                    }
                }
                uint8_t sb = f2ue4m3(mx / 6.0f);
                if (sb == 0) sb = 1;
                wword |= (uint32_t)sb << (8 * g);
                sc_sh[kt_local][lane][g] = ue4m3_to_f(sb);
            }
            dst[64 + lane] = wword;
        }
        __syncwarp();
        uint32_t words[2] = {0, 0};
        int col = lane >> 2;                      // nvar=1: only col 0 real
        for (int reg = 0; reg < 2; reg++) {
            for (int j = 0; j < 8; j++) {
                int k = nvf4_b_k(lane, reg, j);
                int gk = kt * 64 + k;
                float v = 0.f;
                if (col == 0) {
                    int local = gk - tile * TILE_ROWS;
                    v = tile_vals[local] * f * (1.0f + bf16_to_float(nw[gk]));
                }
                float s = sc_sh[kt_local][lane >> 2][k >> 4];
                words[reg] |= quant_e2m1_code(s > 0.f ? v / s : 0.f) << (4 * j);
            }
        }
        dst[lane * 2 + 0] = words[0];
        dst[lane * 2 + 1] = words[1];
    }
}

// Stage-7 e2 bit-1 payoff shape: with the rms scalar deferred (quant needs only the
// static (1+nw)), the boundary fuses with NO synchronization at all: DSM fold +
// residual add + register-resident quant + a per-tile sumsq partial. A trailing
// one-thread kernel folds the 40 partials into fac for the NEXT gemm's scalar.
template <int KS>
__global__ void __launch_bounds__(256, 2)
k_fused_nobar(const uint32_t *codes, const uint32_t *scales,
              const float *__restrict__ x, int K,
              const float *__restrict__ resid_in, const uint16_t *__restrict__ nw,
              float *__restrict__ resid_out, uint32_t *__restrict__ qout,
              float *__restrict__ sumsq_part) {
    __shared__ float p_half[2 * TILE_ROWS];
    __shared__ float p_out[TILE_ROWS];
    __shared__ float mailbox[KS > 1 ? (KS - 1) * TILE_ROWS : 1];
    __shared__ float tile_vals[TILE_ROWS];
    cg::cluster_group cluster = cg::this_cluster();
    const int rank = (int)cluster.block_rank();
    const int tile = blockIdx.x / KS;
    const int tid  = threadIdx.x;
    WeightBuf w{codes, scales};
    producer_partial<KS>(w, x, K, tile, rank, p_half, p_out);
    if (KS > 1) {
        if (rank != 0) {
            float *dst = cluster.map_shared_rank(mailbox, 0);
            if (tid < TILE_ROWS) dst[(rank - 1) * TILE_ROWS + tid] = p_out[tid];
        }
        cluster.sync();
        if (rank != 0) return;
    }
    if (tid < TILE_ROWS) {
        float acc = p_out[tid];
        for (int s = 1; s < KS; s++) acc += mailbox[(s - 1) * TILE_ROWS + tid];
        int gi = tile * TILE_ROWS + tid;
        float v = resid_in[gi] + acc;
        resid_out[gi] = v;
        tile_vals[tid] = v;
    }
    __syncthreads();
    // per-tile sumsq partial (order-free for the deferred scalar's tolerance class)
    float ss = (tid < TILE_ROWS) ? tile_vals[tid] * tile_vals[tid] : 0.f;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, off);
    if ((tid & 31) == 0) p_half[tid >> 5] = ss;
    __syncthreads();
    if (tid == 0) {
        float t = 0.f;
        for (int i = 0; i < 8; i++) t += p_half[i];
        sumsq_part[tile] = t;
    }
    // quantize with (1+nw) only: no factor wait, values already in smem
    if (tid < 64) {
        int kt_local = tid >> 5;
        int lane = tid & 31;
        int kt = tile * 2 + kt_local;
        uint32_t *dst = qout + (size_t)kt * NVF4_BW;
        __shared__ float sc_sh[2][8][4];
        if (lane < 8) {
            uint32_t wword = 0;
            for (int g = 0; g < 4; g++) {
                float mx = 0.f;
                if (lane == 0) {
                    for (int t = 0; t < 16; t++) {
                        int k = kt * 64 + g * 16 + t;
                        int local = k - tile * TILE_ROWS;
                        float v = tile_vals[local] * (1.0f + bf16_to_float(nw[k]));
                        mx = fmaxf(mx, fabsf(v));
                    }
                }
                uint8_t sb = f2ue4m3(mx / 6.0f);
                if (sb == 0) sb = 1;
                wword |= (uint32_t)sb << (8 * g);
                sc_sh[kt_local][lane][g] = ue4m3_to_f(sb);
            }
            dst[64 + lane] = wword;
        }
        __syncwarp();
        uint32_t words[2] = {0, 0};
        int col = lane >> 2;
        for (int reg = 0; reg < 2; reg++) {
            for (int j = 0; j < 8; j++) {
                int k = nvf4_b_k(lane, reg, j);
                int gk = kt * 64 + k;
                float v = 0.f;
                if (col == 0) {
                    int local = gk - tile * TILE_ROWS;
                    v = tile_vals[local] * (1.0f + bf16_to_float(nw[gk]));
                }
                float s = sc_sh[kt_local][lane >> 2][k >> 4];
                words[reg] |= quant_e2m1_code(s > 0.f ? v / s : 0.f) << (4 * j);
            }
        }
        dst[lane * 2 + 0] = words[0];
        dst[lane * 2 + 1] = words[1];
    }
}

__global__ void k_fac_combine(const float *__restrict__ sumsq_part, float *__restrict__ fac,
                              int ntiles, int N) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float t = 0.f;
        for (int i = 0; i < ntiles; i++) t += sumsq_part[i];
        fac[0] = rsqrtf(t / (float)N + RMS_EPS);
    }
}

// ---------------------------------------------------------------------------
// Host harness.
// ---------------------------------------------------------------------------

static float elapsed_ms(cudaEvent_t a, cudaEvent_t b, int iters) {
    float ms = 0.f;
    CUDA_OK(cudaEventElapsedTime(&ms, a, b));
    return ms / (float)iters;
}

template <int KS>
static void run_case(int K, int iters, int warmup) {
    const int Kt = K / 64;
    if (Kt % KS || (Kt / KS) % 2) {
        std::printf("{\"K\":%d,\"ks\":%d,\"skip\":\"k-range not divisible\"}\n", K, KS);
        return;
    }
    cudaStream_t stream;
    CUDA_OK(cudaStreamCreate(&stream));

    // Synthetic inputs (host-seeded, deterministic).
    const size_t n_code_w = (size_t)N_TILES * Kt * TILE_ROWS * 8;
    const size_t n_scale_w = (size_t)N_TILES * Kt * TILE_ROWS;
    std::vector<uint32_t> h_codes(n_code_w);
    std::vector<uint32_t> h_scales(n_scale_w);
    std::vector<float> h_x(K);
    std::vector<float> h_resid(M_DIM);
    std::vector<uint16_t> h_nw(M_DIM);
    uint64_t rng = 0x9e3779b97f4a7c15ull;
    auto next = [&]() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; };
    for (auto &v : h_codes) v = (uint32_t)next();
    for (auto &v : h_scales) {
        uint32_t w = 0;
        for (int g = 0; g < 4; g++) w |= (uint32_t)(0x30 + (next() & 0xF)) << (8 * g);
        v = w;
    }
    for (auto &v : h_x) v = ((float)(next() & 0xFFFF) / 65536.0f - 0.5f) * 0.25f;
    for (auto &v : h_resid) v = ((float)(next() & 0xFFFF) / 65536.0f - 0.5f) * 2.0f;
    for (auto &v : h_nw) {
        float wv = ((float)(next() & 0xFFFF) / 65536.0f - 0.5f) * 0.125f;
        uint32_t u; memcpy(&u, &wv, 4); v = (uint16_t)(u >> 16);
    }

    uint32_t *d_codes, *d_scales, *d_q_ctl, *d_q_fused;
    float *d_x, *d_resid_in, *d_resid_ctl, *d_resid_fused, *d_partials, *d_y;
    float *d_fac_ctl, *d_fac_fused;
    uint16_t *d_nw;
    unsigned int *d_arrive, *d_epoch, *d_facready;
    CUDA_OK(cudaMalloc(&d_codes, n_code_w * 4));
    CUDA_OK(cudaMalloc(&d_scales, n_scale_w * 4));
    CUDA_OK(cudaMalloc(&d_x, K * 4));
    CUDA_OK(cudaMalloc(&d_resid_in, M_DIM * 4));
    CUDA_OK(cudaMalloc(&d_resid_ctl, M_DIM * 4));
    CUDA_OK(cudaMalloc(&d_resid_fused, M_DIM * 4));
    CUDA_OK(cudaMalloc(&d_partials, (size_t)KS * M_DIM * 4));
    CUDA_OK(cudaMalloc(&d_y, M_DIM * 4));
    CUDA_OK(cudaMalloc(&d_fac_ctl, 4));
    CUDA_OK(cudaMalloc(&d_fac_fused, 4));
    CUDA_OK(cudaMalloc(&d_q_ctl, (size_t)KT64_OUT * NVF4_BW * 4));
    CUDA_OK(cudaMalloc(&d_q_fused, (size_t)KT64_OUT * NVF4_BW * 4));
    CUDA_OK(cudaMalloc(&d_nw, M_DIM * 2));
    CUDA_OK(cudaMalloc(&d_arrive, 4));
    CUDA_OK(cudaMalloc(&d_epoch, 4));
    CUDA_OK(cudaMalloc(&d_facready, 4));
    CUDA_OK(cudaMemcpy(d_codes, h_codes.data(), n_code_w * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_scales, h_scales.data(), n_scale_w * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_x, h_x.data(), K * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_resid_in, h_resid.data(), M_DIM * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_nw, h_nw.data(), M_DIM * 2, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemset(d_arrive, 0, 4));
    CUDA_OK(cudaMemset(d_epoch, 0, 4));
    CUDA_OK(cudaMemset(d_facready, 0, 4));
    CUDA_OK(cudaMemset(d_q_ctl, 0, (size_t)KT64_OUT * NVF4_BW * 4));
    CUDA_OK(cudaMemset(d_q_fused, 0, (size_t)KT64_OUT * NVF4_BW * 4));

    // Occupancy guard for the resident sense barrier.
    cudaDeviceProp prop{};
    CUDA_OK(cudaGetDeviceProperties(&prop, 0));
    int per_sm = 0;
    CUDA_OK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, k_fused_epi<KS>, 256, 0));
    const int grid_ctas = N_TILES * KS;
    const bool resident = (long)per_sm * prop.multiProcessorCount >= grid_ctas;

    auto launch_control = [&](cudaStream_t st) {
        k_ctl_producer<KS><<<N_TILES * KS, 256, 0, st>>>(d_codes, d_scales, d_x, K, d_partials);
        k_ctl_reduce<KS><<<(M_DIM + 255) / 256, 256, 0, st>>>(d_partials, d_y);
        k_ctl_add_rms_fac<<<dim3(1, 1), 1024, 0, st>>>(d_fac_ctl, d_resid_ctl, d_resid_in, d_y, M_DIM, RMS_EPS);
        k_ctl_quant_x_nw<<<KT64_OUT, 32, 0, st>>>(d_q_ctl, d_resid_ctl, d_fac_ctl, d_nw, 1, M_DIM, 1, KT64_OUT);
    };

    unsigned int pass_no = 0;
    auto launch_fused = [&](cudaStream_t st, unsigned int pass) {
        FusedCtl ctl{d_resid_fused, d_fac_fused, d_q_fused, nullptr,
                     d_arrive, d_epoch, d_facready};
        cudaLaunchConfig_t cfg{};
        cfg.gridDim = dim3(N_TILES * KS, 1, 1);
        cfg.blockDim = dim3(256, 1, 1);
        cfg.stream = st;
        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeClusterDimension;
        attr[0].val.clusterDim.x = KS; attr[0].val.clusterDim.y = 1; attr[0].val.clusterDim.z = 1;
        cfg.attrs = attr; cfg.numAttrs = 1;
        CUDA_OK(cudaLaunchKernelEx(&cfg, k_fused_epi<KS>, d_codes, d_scales, d_x, K,
                                   d_resid_in, d_nw, ctl, pass));
    };

    float *d_resid_nb = NULL, *d_fac_nb = NULL, *d_sumsq = NULL;
    uint32_t *d_q_nb = NULL;
    CUDA_OK(cudaMalloc(&d_resid_nb, M_DIM * 4));
    CUDA_OK(cudaMalloc(&d_fac_nb, 4));
    CUDA_OK(cudaMalloc(&d_sumsq, N_TILES * 4));
    CUDA_OK(cudaMalloc(&d_q_nb, (size_t)KT64_OUT * NVF4_BW * 4));
    CUDA_OK(cudaMemset(d_q_nb, 0, (size_t)KT64_OUT * NVF4_BW * 4));
    auto launch_nobar = [&](cudaStream_t st) {
        cudaLaunchConfig_t cfg{};
        cfg.gridDim = dim3(N_TILES * KS, 1, 1);
        cfg.blockDim = dim3(256, 1, 1);
        cfg.stream = st;
        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeClusterDimension;
        attr[0].val.clusterDim.x = KS; attr[0].val.clusterDim.y = 1; attr[0].val.clusterDim.z = 1;
        cfg.attrs = attr; cfg.numAttrs = 1;
        CUDA_OK(cudaLaunchKernelEx(&cfg, k_fused_nobar<KS>, d_codes, d_scales, d_x, K,
                                   d_resid_in, d_nw, d_resid_nb, d_q_nb, d_sumsq));
        k_fac_combine<<<1, 32, 0, st>>>(d_sumsq, d_fac_nb, N_TILES, M_DIM);
    };

    // Correctness pass.
    launch_control(stream);
    pass_no += 1;
    launch_fused(stream, pass_no);
    CUDA_OK(cudaStreamSynchronize(stream));

    std::vector<float> r_ctl(M_DIM), r_fused(M_DIM);
    float fac_ctl_h = 0.f, fac_fused_h = 0.f;
    std::vector<uint32_t> q_ctl(KT64_OUT * NVF4_BW), q_fused(KT64_OUT * NVF4_BW);
    CUDA_OK(cudaMemcpy(r_ctl.data(), d_resid_ctl, M_DIM * 4, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(r_fused.data(), d_resid_fused, M_DIM * 4, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(&fac_ctl_h, d_fac_ctl, 4, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(&fac_fused_h, d_fac_fused, 4, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(q_ctl.data(), d_q_ctl, q_ctl.size() * 4, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(q_fused.data(), d_q_fused, q_fused.size() * 4, cudaMemcpyDeviceToHost));

    long resid_mm = 0, code_mm = 0;
    for (int i = 0; i < M_DIM; i++)
        if (memcmp(&r_ctl[i], &r_fused[i], 4)) resid_mm++;
    for (size_t i = 0; i < q_ctl.size(); i++)
        if (q_ctl[i] != q_fused[i]) code_mm++;
    const bool fac_exact = !memcmp(&fac_ctl_h, &fac_fused_h, 4);

    // Timing: eager.
    cudaEvent_t ev_a, ev_b;
    CUDA_OK(cudaEventCreate(&ev_a));
    CUDA_OK(cudaEventCreate(&ev_b));
    for (int i = 0; i < warmup; i++) launch_control(stream);
    CUDA_OK(cudaEventRecord(ev_a, stream));
    for (int i = 0; i < iters; i++) launch_control(stream);
    CUDA_OK(cudaEventRecord(ev_b, stream));
    CUDA_OK(cudaStreamSynchronize(stream));
    const float ctl_eager_us = elapsed_ms(ev_a, ev_b, iters) * 1000.f;

    float fused_eager_us = -1.f, ctl_graph_us = -1.f, fused_graph_us = -1.f;
    if (resident) {
        for (int i = 0; i < warmup; i++) { pass_no += 1; launch_fused(stream, pass_no); }
        CUDA_OK(cudaEventRecord(ev_a, stream));
        for (int i = 0; i < iters; i++) { pass_no += 1; launch_fused(stream, pass_no); }
        CUDA_OK(cudaEventRecord(ev_b, stream));
        CUDA_OK(cudaStreamSynchronize(stream));
        fused_eager_us = elapsed_ms(ev_a, ev_b, iters) * 1000.f;
    }

    // Timing: graph replay (control chain as one graph; fused cannot capture the
    // monotonically increasing pass counter, so replay a 2-pass graph whose
    // counter advance is baked in by resetting arrive/fac_ready with memsets).
    {
        cudaGraph_t g; cudaGraphExec_t ge;
        CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        launch_control(stream);
        CUDA_OK(cudaStreamEndCapture(stream, &g));
        CUDA_OK(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0));
        for (int i = 0; i < warmup; i++) CUDA_OK(cudaGraphLaunch(ge, stream));
        CUDA_OK(cudaEventRecord(ev_a, stream));
        for (int i = 0; i < iters; i++) CUDA_OK(cudaGraphLaunch(ge, stream));
        CUDA_OK(cudaEventRecord(ev_b, stream));
        CUDA_OK(cudaStreamSynchronize(stream));
        ctl_graph_us = elapsed_ms(ev_a, ev_b, iters) * 1000.f;
        CUDA_OK(cudaGraphExecDestroy(ge));
        CUDA_OK(cudaGraphDestroy(g));
    }
    if (resident) {
        cudaGraph_t g; cudaGraphExec_t ge;
        CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        CUDA_OK(cudaMemsetAsync(d_arrive, 0, 4, stream));
        CUDA_OK(cudaMemsetAsync(d_facready, 0, 4, stream));
        launch_fused(stream, 1);
        CUDA_OK(cudaStreamEndCapture(stream, &g));
        CUDA_OK(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0));
        for (int i = 0; i < warmup; i++) CUDA_OK(cudaGraphLaunch(ge, stream));
        CUDA_OK(cudaEventRecord(ev_a, stream));
        for (int i = 0; i < iters; i++) CUDA_OK(cudaGraphLaunch(ge, stream));
        CUDA_OK(cudaEventRecord(ev_b, stream));
        CUDA_OK(cudaStreamSynchronize(stream));
        fused_graph_us = elapsed_ms(ev_a, ev_b, iters) * 1000.f;
        CUDA_OK(cudaGraphExecDestroy(ge));
        CUDA_OK(cudaGraphDestroy(g));
    }
    float nobar_eager_us = -1.f, nobar_graph_us = -1.f;
    long resid_nb_mm = -1;
    float fac_nb_h = 0.f;
    {
        launch_nobar(stream);
        CUDA_OK(cudaStreamSynchronize(stream));
        std::vector<float> r_nb(M_DIM);
        CUDA_OK(cudaMemcpy(r_nb.data(), d_resid_nb, M_DIM * 4, cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(&fac_nb_h, d_fac_nb, 4, cudaMemcpyDeviceToHost));
        resid_nb_mm = 0;
        for (int i = 0; i < M_DIM; i++)
            if (memcmp(&r_ctl[i], &r_nb[i], 4)) resid_nb_mm++;
        for (int i = 0; i < warmup; i++) launch_nobar(stream);
        CUDA_OK(cudaEventRecord(ev_a, stream));
        for (int i = 0; i < iters; i++) launch_nobar(stream);
        CUDA_OK(cudaEventRecord(ev_b, stream));
        CUDA_OK(cudaStreamSynchronize(stream));
        nobar_eager_us = elapsed_ms(ev_a, ev_b, iters) * 1000.f;
        cudaGraph_t g; cudaGraphExec_t ge;
        CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        launch_nobar(stream);
        CUDA_OK(cudaStreamEndCapture(stream, &g));
        CUDA_OK(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0));
        for (int i = 0; i < warmup; i++) CUDA_OK(cudaGraphLaunch(ge, stream));
        CUDA_OK(cudaEventRecord(ev_a, stream));
        for (int i = 0; i < iters; i++) CUDA_OK(cudaGraphLaunch(ge, stream));
        CUDA_OK(cudaEventRecord(ev_b, stream));
        CUDA_OK(cudaStreamSynchronize(stream));
        nobar_graph_us = elapsed_ms(ev_a, ev_b, iters) * 1000.f;
        CUDA_OK(cudaGraphExecDestroy(ge));
        CUDA_OK(cudaGraphDestroy(g));
    }
    const float fac_rel = fabsf(fac_nb_h - fac_ctl_h) / (fabsf(fac_ctl_h) + 1e-30f);

    std::printf("{\"K\":%d,\"ks\":%d,\"resident\":%s,\"per_sm\":%d,"
                "\"resid_mismatches\":%ld,\"fac_exact\":%s,\"code_mismatches\":%ld,"
                "\"control_eager_us\":%.3f,\"fused_eager_us\":%.3f,"
                "\"control_graph_us\":%.3f,\"fused_graph_us\":%.3f,"
                "\"nobar_resid_mismatches\":%ld,\"nobar_fac_rel\":%.2e,"
                "\"nobar_eager_us\":%.3f,\"nobar_graph_us\":%.3f}\n",
                K, KS, resident ? "true" : "false", per_sm,
                resid_mm, fac_exact ? "true" : "false", code_mm,
                ctl_eager_us, fused_eager_us, ctl_graph_us, fused_graph_us,
                resid_nb_mm, fac_rel, nobar_eager_us, nobar_graph_us);

    CUDA_OK(cudaFree(d_codes)); CUDA_OK(cudaFree(d_scales)); CUDA_OK(cudaFree(d_x));
    CUDA_OK(cudaFree(d_resid_in)); CUDA_OK(cudaFree(d_resid_ctl)); CUDA_OK(cudaFree(d_resid_fused));
    CUDA_OK(cudaFree(d_partials)); CUDA_OK(cudaFree(d_y));
    CUDA_OK(cudaFree(d_fac_ctl)); CUDA_OK(cudaFree(d_fac_fused));
    CUDA_OK(cudaFree(d_q_ctl)); CUDA_OK(cudaFree(d_q_fused)); CUDA_OK(cudaFree(d_nw));
    CUDA_OK(cudaFree(d_arrive)); CUDA_OK(cudaFree(d_epoch)); CUDA_OK(cudaFree(d_facready));
    CUDA_OK(cudaFree(d_resid_nb)); CUDA_OK(cudaFree(d_fac_nb));
    CUDA_OK(cudaFree(d_sumsq)); CUDA_OK(cudaFree(d_q_nb));
    CUDA_OK(cudaStreamDestroy(stream));
}

int main(int argc, char **argv) {
    int iters = argc > 1 ? std::atoi(argv[1]) : 2000;
    int warmup = argc > 2 ? std::atoi(argv[2]) : 200;
    int K = argc > 3 ? std::atoi(argv[3]) : 0;
    int ks = argc > 4 ? std::atoi(argv[4]) : 0;
    auto run_k = [&](int Kv) {
        if (!ks || ks == 2) run_case<2>(Kv, iters, warmup);
        if (!ks || ks == 4) run_case<4>(Kv, iters, warmup);
        if (!ks || ks == 8) run_case<8>(Kv, iters, warmup);
    };
    if (K) run_k(K);
    else { run_k(17408); run_k(6144); }
    return 0;
}
