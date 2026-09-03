// Standalone sm120 proof for docs/level-up.md stage 2.
//
// This does NOT modify or link the production engine. It isolates one n=1,
// M=5120 projection boundary. The control follows the current global split-K
// fold -> add/RMS factor -> NVFP4 quantization sequence. The probe uses the
// stage-1 producer-push DSM fold to publish the committed residual directly,
// then maps the exact 1024-thread RMS reduction over an 8-CTA cluster and
// retains the residual in DSM for NVFP4 encoding.
//
// Build:
//   nvcc -std=c++17 -O3 -lineinfo --generate-code=arch=compute_120,code=sm_120 \
//     -o results/microbench/epilogue tools/microbench_epilogue.cu

#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace cg = cooperative_groups;

#define CUDA_OK(expr) do {                                                        \
    cudaError_t status_ = (expr);                                                 \
    if (status_ != cudaSuccess) {                                                 \
        std::fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__,      \
                     #expr, cudaGetErrorString(status_));                         \
        std::exit(1);                                                             \
    }                                                                             \
} while (0)

constexpr int D = 5120;
constexpr int KS = 4;
constexpr int TILE = 128;
constexpr int GROUP = 16;
constexpr int GROUPS = D / GROUP;
constexpr int RMS_CTAS = 8;
constexpr int RMS_THREADS = 128;
constexpr int RMS_LOGICAL_THREADS = RMS_CTAS * RMS_THREADS;
constexpr int RMS_VALUES_PER_THREAD = (D + RMS_LOGICAL_THREADS - 1) / RMS_LOGICAL_THREADS;
constexpr float EPS = 1.0e-6f;

static __host__ __device__ __forceinline__ float input_partial(int i, int split) {
    return float((i & 31) + 1) * 0.03125f + float(split + 1) * 0.125f;
}

static __host__ __device__ __forceinline__ float input_residual(int i) {
    return float((i % 17) - 8) * 0.0625f;
}

static __host__ __device__ __forceinline__ float input_norm_weight(int i) {
    return float((i % 13) - 6) * 0.015625f;
}

static __device__ __forceinline__ uint8_t quant_e2m1(float v) {
    float a = fminf(fabsf(v), 6.0f);
    uint32_t code;
    if (a < 0.75f) {
        code = (a < 0.25f) ? 0u : 1u;
    } else {
        int e = (a < 2.0f) ? 1 : (a < 4.0f ? 2 : 3);
        float sc = float(1 << (e - 1));
        int m = int((a / sc - 1.0f) * 2.0f + 0.5f);
        if (m >= 2) {
            m = 0;
            ++e;
            if (e > 3) { e = 3; m = 1; }
        }
        code = (uint32_t(e) << 1) | uint32_t(m);
    }
    if (v < 0.0f) code |= 0x8u;
    return uint8_t(code);
}

static __device__ __forceinline__ float ue4m3_to_f(uint8_t b) {
    int e = (b >> 3) & 0xF, m = b & 0x7;
    if (e == 0) return float(m) * 0.125f * 0.015625f;
    return ldexpf(1.0f + float(m) * 0.125f, e - 7);
}

static __device__ __forceinline__ uint8_t f2ue4m3(float v) {
    if (!(v > 0.0f)) return 0;
    int e;
    float m = frexpf(v, &e);
    int E = e + 6;
    int f = int(lrintf((2.0f * m - 1.0f) * 8.0f));
    if (f > 7) { f = 0; ++E; }
    if (E > 15) { E = 15; f = 7; }
    if (E < 1) return 0;
    return uint8_t((E << 3) | f);
}

__global__ void fill_inputs(float *partials, float *residual, float *norm_weight) {
    int i = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (i >= D) return;
    residual[i] = input_residual(i);
    norm_weight[i] = input_norm_weight(i);
    #pragma unroll
    for (int split = 0; split < KS; ++split)
        partials[size_t(split) * D + i] = input_partial(i, split);
}

__global__ void global_reduce(float *projection, const float *partials) {
    int i = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (i >= D) return;
    float sum = 0.0f;
    #pragma unroll
    for (int split = 0; split < KS; ++split)
        sum += partials[size_t(split) * D + i];
    projection[i] = sum;
}

// Same logical reduction order as k_tq_add_rms_fac_b for N=5120.
__global__ __launch_bounds__(1024, 1)
void add_rms_factor(float *factor, float *committed, const float *residual,
                    const float *projection) {
    __shared__ float warp_sum[32];
    int tid = int(threadIdx.x);
    int lane = tid & 31;
    int warp = tid >> 5;
    float sum_sq = 0.0f;
    for (int i = tid; i < D; i += int(blockDim.x)) {
        float v = residual[i] + projection[i];
        committed[i] = v;
        sum_sq += v * v;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_sq += __shfl_down_sync(0xffffffffu, sum_sq, offset);
    if (lane == 0) warp_sum[warp] = sum_sq;
    __syncthreads();
    if (warp == 0) {
        sum_sq = lane < 32 ? warp_sum[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum_sq += __shfl_down_sync(0xffffffffu, sum_sq, offset);
        if (lane == 0) factor[0] = rsqrtf(sum_sq / float(D) + EPS);
    }
}

// N=1 specialization of the existing quantizer's work assignment: one warp per
// k64 tile; lane 0 computes four sequential 16-value scales, then all lanes
// encode two values. Codes are kept one byte each only to make comparison easy.
__global__ void quant_control(uint8_t *codes, uint8_t *scales,
                              const float *committed, const float *factor,
                              const float *norm_weight) {
    int base = int(blockIdx.x) * 64;
    int lane = int(threadIdx.x);
    __shared__ uint8_t scale_sm[4];
    if (lane == 0) {
        float f = factor[0];
        #pragma unroll
        for (int group = 0; group < 4; ++group) {
            float mx = 0.0f;
            #pragma unroll
            for (int t = 0; t < GROUP; ++t) {
                int i = base + group * GROUP + t;
                float v = committed[i] * f * (1.0f + norm_weight[i]);
                mx = fmaxf(mx, fabsf(v));
            }
            uint8_t sb = f2ue4m3(mx / 6.0f);
            if (sb == 0) sb = 1;
            scale_sm[group] = sb;
            scales[base / GROUP + group] = sb;
        }
    }
    __syncthreads();
    #pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
        int local = lane + pass * 32;
        int i = base + local;
        float v = committed[i] * factor[0] * (1.0f + norm_weight[i]);
        float scale = ue4m3_to_f(scale_sm[local / GROUP]);
        codes[i] = quant_e2m1(scale > 0.0f ? v / scale : 0.0f);
    }
}

// Producer-push DSM: each split rank writes its register-resident analogue to
// rank 0. Rank 0 folds in split order, adds the residual, and publishes only the
// committed value. The standalone probe reads synthetic partials from DRAM;
// the production form would push directly from MMA accumulator registers.
__global__ __launch_bounds__(256, 1) __cluster_dims__(KS, 1, 1)
void cluster_epilogue(float *committed, const float *partials,
                      const float *residual) {
    cg::cluster_group cluster = cg::this_cluster();
    int rank = int(cluster.block_rank());
    int tile = int(blockIdx.x) / KS;
    int base = tile * TILE;
    extern __shared__ float mailbox[];
    float *owner = rank == 0 ? mailbox : cluster.map_shared_rank(mailbox, 0);
    for (int local = int(threadIdx.x); local < TILE; local += int(blockDim.x)) {
        int i = base + local;
        owner[rank * TILE + local] = i < D ? partials[size_t(rank) * D + i] : 0.0f;
    }
    cluster.sync();
    if (rank == 0) {
        for (int local = int(threadIdx.x); local < TILE; local += int(blockDim.x)) {
            int i = base + local;
            if (i >= D) continue;
            float sum = 0.0f;
            #pragma unroll
            for (int split = 0; split < KS; ++split)
                sum += mailbox[split * TILE + local];
            committed[i] = residual[i] + sum;
        }
    }
}

// The 1024 logical RMS threads are rank*128+thread. Each computes the same
// strided sum and each logical warp reduces in the same order as add_rms_factor.
// Its up-to-five committed values stay in local DSM. After rank 0 broadcasts
// the exact factor, each rank quantizes the 40 groups already present in its
// local cache, avoiding a second global committed-value read.
__global__ __launch_bounds__(RMS_THREADS, 1) __cluster_dims__(RMS_CTAS, 1, 1)
void cluster_rms_quant(uint8_t *codes, uint8_t *scales, float *factor,
                       const float *committed, const float *norm_weight) {
    cg::cluster_group cluster = cg::this_cluster();
    int rank = int(cluster.block_rank());
    int local_tid = int(threadIdx.x);
    int logical_tid = rank * RMS_THREADS + local_tid;
    int lane = local_tid & 31;
    int warp = local_tid >> 5;
    extern __shared__ float sm[];
    float *cache = sm;
    float *mailbox = sm + RMS_VALUES_PER_THREAD * RMS_THREADS;
    float *owner = rank == 0 ? mailbox : cluster.map_shared_rank(mailbox, 0);

    float sum_sq = 0.0f;
    #pragma unroll
    for (int j = 0; j < RMS_VALUES_PER_THREAD; ++j) {
        int i = logical_tid + j * RMS_LOGICAL_THREADS;
        float v = i < D ? committed[i] : 0.0f;
        cache[j * RMS_THREADS + local_tid] = v;
        if (i < D) sum_sq += v * v;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_sq += __shfl_down_sync(0xffffffffu, sum_sq, offset);
    if (lane == 0) owner[rank * 4 + warp] = sum_sq;
    cluster.sync();

    if (rank == 0 && warp == 0) {
        float v = mailbox[lane];
        for (int offset = 16; offset > 0; offset >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, offset);
        if (lane == 0) {
            float f = rsqrtf(v / float(D) + EPS);
            mailbox[0] = f;
            factor[0] = f;
        }
    }
    cluster.sync();
    float f = owner[0];

    // Eight half-warps process eight groups at once. Five batches cover the 40
    // groups cached by this rank. Lane 0 retains the control's sequential fmax.
    int halfwarp = local_tid >> 4;
    int half_lane = local_tid & 15;
    unsigned half_mask = (lane & 16) ? 0xffff0000u : 0x0000ffffu;
    __shared__ uint8_t scale_sm[8];
    #pragma unroll
    for (int batch = 0; batch < RMS_VALUES_PER_THREAD; ++batch) {
        int group = batch * 64 + rank * 8 + halfwarp;
        int cache_base = batch * RMS_THREADS + halfwarp * GROUP;
        if (half_lane == 0) {
            float mx = 0.0f;
            #pragma unroll
            for (int t = 0; t < GROUP; ++t) {
                int i = group * GROUP + t;
                float v = cache[cache_base + t] * f * (1.0f + norm_weight[i]);
                mx = fmaxf(mx, fabsf(v));
            }
            uint8_t sb = f2ue4m3(mx / 6.0f);
            if (sb == 0) sb = 1;
            scale_sm[halfwarp] = sb;
            scales[group] = sb;
        }
        __syncwarp(half_mask);
        int i = group * GROUP + half_lane;
        float v = cache[cache_base + half_lane] * f * (1.0f + norm_weight[i]);
        float scale = ue4m3_to_f(scale_sm[halfwarp]);
        codes[i] = quant_e2m1(scale > 0.0f ? v / scale : 0.0f);
        __syncwarp(half_mask);
    }
}

// One cooperative clustered launch for the whole M=5120 boundary. Forty
// four-rank clusters exactly cover the output tiles. All clusters remain
// resident, so a grid barrier can separate committed publication, the exact
// global RMS fold, and NVFP4 encoding while each owner keeps its tile in DSM.
__global__ __launch_bounds__(256, 1) __cluster_dims__(KS, 1, 1)
void cooperative_epilogue(uint8_t *codes, uint8_t *scales, float *factor,
                          float *committed, float *rms_warp,
                          const float *partials, const float *residual,
                          const float *norm_weight) {
    cg::cluster_group cluster = cg::this_cluster();
    cg::grid_group grid = cg::this_grid();
    int rank = int(cluster.block_rank());
    int tile = int(blockIdx.x) / KS;
    int base = tile * TILE;
    int tid = int(threadIdx.x);
    extern __shared__ float mailbox[];
    float *owner = rank == 0 ? mailbox : cluster.map_shared_rank(mailbox, 0);

    for (int local = tid; local < TILE; local += int(blockDim.x)) {
        int i = base + local;
        owner[rank * TILE + local] = partials[size_t(rank) * D + i];
    }
    cluster.sync();
    if (rank == 0) {
        for (int local = tid; local < TILE; local += int(blockDim.x)) {
            int i = base + local;
            float sum = 0.0f;
            #pragma unroll
            for (int split = 0; split < KS; ++split)
                sum += mailbox[split * TILE + local];
            float h = residual[i] + sum;
            committed[i] = h;
            mailbox[local] = h;
        }
    }
    grid.sync();

    // Blocks 0..3 provide the original 1024 logical RMS threads. Their eight
    // warps each write the same 32 warp sums as the control's one 1024-thread CTA.
    if (blockIdx.x < 4) {
        int logical_tid = int(blockIdx.x) * 256 + tid;
        float sum_sq = 0.0f;
        for (int i = logical_tid; i < D; i += 1024) {
            float v = committed[i];
            sum_sq += v * v;
        }
        for (int offset = 16; offset > 0; offset >>= 1)
            sum_sq += __shfl_down_sync(0xffffffffu, sum_sq, offset);
        if ((tid & 31) == 0)
            rms_warp[int(blockIdx.x) * 8 + (tid >> 5)] = sum_sq;
    }
    grid.sync();
    if (blockIdx.x == 0 && tid < 32) {
        float v = rms_warp[tid];
        for (int offset = 16; offset > 0; offset >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, offset);
        if (tid == 0) factor[0] = rsqrtf(v / float(D) + EPS);
    }
    grid.sync();

    if (rank == 0 && tid < TILE) {
        int halfwarp = tid >> 4;
        int half_lane = tid & 15;
        unsigned half_mask = (tid & 16) ? 0xffff0000u : 0x0000ffffu;
        __shared__ uint8_t scale_sm[8];
        int group = tile * 8 + halfwarp;
        if (half_lane == 0) {
            float mx = 0.0f;
            #pragma unroll
            for (int t = 0; t < GROUP; ++t) {
                int i = group * GROUP + t;
                float v = mailbox[halfwarp * GROUP + t] * factor[0]
                        * (1.0f + norm_weight[i]);
                mx = fmaxf(mx, fabsf(v));
            }
            uint8_t sb = f2ue4m3(mx / 6.0f);
            if (sb == 0) sb = 1;
            scale_sm[halfwarp] = sb;
            scales[group] = sb;
        }
        __syncwarp(half_mask);
        int i = group * GROUP + half_lane;
        float v = mailbox[halfwarp * GROUP + half_lane] * factor[0]
                * (1.0f + norm_weight[i]);
        float scale = ue4m3_to_f(scale_sm[halfwarp]);
        codes[i] = quant_e2m1(scale > 0.0f ? v / scale : 0.0f);
    }
}

// Same fused boundary without cooperative-grid barriers. The launch is admitted
// only when every four-rank cluster can be resident. A reusable sense barrier
// across the 40 owners publishes committed tiles; four blocks then reproduce
// the control's 32 RMS warp sums and publish a completion epoch.
__global__ __launch_bounds__(256, 1) __cluster_dims__(KS, 1, 1)
void resident_epilogue(uint8_t *codes, uint8_t *scales, float *factor,
                       float *committed, float *rms_warp, unsigned *sync,
                       const float *partials, const float *residual,
                       const float *norm_weight) {
    cg::cluster_group cluster = cg::this_cluster();
    int rank = int(cluster.block_rank());
    int tile = int(blockIdx.x) / KS;
    int base = tile * TILE;
    int tid = int(threadIdx.x);
    extern __shared__ float mailbox[];
    float *owner = rank == 0 ? mailbox : cluster.map_shared_rank(mailbox, 0);
    __shared__ unsigned commit_start;
    __shared__ unsigned ready_start;
    __shared__ unsigned rms_ticket;
    if (tid == 0) {
        commit_start = atomicAdd(sync + 1, 0u);
        ready_start = atomicAdd(sync + 3, 0u);
    }
    __syncthreads();

    for (int local = tid; local < TILE; local += int(blockDim.x)) {
        int i = base + local;
        owner[rank * TILE + local] = partials[size_t(rank) * D + i];
    }
    cluster.sync();
    if (rank == 0) {
        for (int local = tid; local < TILE; local += int(blockDim.x)) {
            int i = base + local;
            float sum = 0.0f;
            #pragma unroll
            for (int split = 0; split < KS; ++split)
                sum += mailbox[split * TILE + local];
            float h = residual[i] + sum;
            committed[i] = h;
            mailbox[local] = h;
        }
        __syncthreads();
        if (tid == 0) {
            __threadfence();
            unsigned ticket = atomicAdd(sync + 0, 1u);
            if (ticket == D / TILE - 1) {
                atomicExch(sync + 0, 0u);
                __threadfence();
                atomicAdd(sync + 1, 1u);
            } else {
                while (atomicAdd(sync + 1, 0u) == commit_start) {}
            }
        }
        __syncthreads();
    }
    cluster.sync();

    if (blockIdx.x < 4) {
        int logical_tid = int(blockIdx.x) * 256 + tid;
        float sum_sq = 0.0f;
        for (int i = logical_tid; i < D; i += 1024) {
            float v = committed[i];
            sum_sq += v * v;
        }
        for (int offset = 16; offset > 0; offset >>= 1)
            sum_sq += __shfl_down_sync(0xffffffffu, sum_sq, offset);
        if ((tid & 31) == 0)
            rms_warp[int(blockIdx.x) * 8 + (tid >> 5)] = sum_sq;
        __syncthreads();
        if (tid == 0) {
            __threadfence();
            rms_ticket = atomicAdd(sync + 2, 1u);
        }
        __syncthreads();
        if (rms_ticket == 3u && tid < 32) {
            float v = rms_warp[tid];
            for (int offset = 16; offset > 0; offset >>= 1)
                v += __shfl_down_sync(0xffffffffu, v, offset);
            if (tid == 0) {
                factor[0] = rsqrtf(v / float(D) + EPS);
                atomicExch(sync + 2, 0u);
                __threadfence();
                atomicAdd(sync + 3, 1u);
            }
        }
    }

    if (rank == 0) {
        if (tid == 0)
            while (atomicAdd(sync + 3, 0u) == ready_start) {}
        __syncthreads();
        if (tid < TILE) {
            int halfwarp = tid >> 4;
            int half_lane = tid & 15;
            unsigned half_mask = (tid & 16) ? 0xffff0000u : 0x0000ffffu;
            __shared__ uint8_t scale_sm[8];
            int group = tile * 8 + halfwarp;
            if (half_lane == 0) {
                float mx = 0.0f;
                #pragma unroll
                for (int t = 0; t < GROUP; ++t) {
                    int i = group * GROUP + t;
                    float v = mailbox[halfwarp * GROUP + t] * factor[0]
                            * (1.0f + norm_weight[i]);
                    mx = fmaxf(mx, fabsf(v));
                }
                uint8_t sb = f2ue4m3(mx / 6.0f);
                if (sb == 0) sb = 1;
                scale_sm[halfwarp] = sb;
                scales[group] = sb;
            }
            __syncwarp(half_mask);
            int i = group * GROUP + half_lane;
            float v = mailbox[halfwarp * GROUP + half_lane] * factor[0]
                    * (1.0f + norm_weight[i]);
            float scale = ue4m3_to_f(scale_sm[halfwarp]);
            codes[i] = quant_e2m1(scale > 0.0f ? v / scale : 0.0f);
        }
    }
}

struct Events {
    cudaEvent_t begin{}, end{};
    Events() { CUDA_OK(cudaEventCreate(&begin)); CUDA_OK(cudaEventCreate(&end)); }
    ~Events() { cudaEventDestroy(begin); cudaEventDestroy(end); }
};

static float elapsed_ms(cudaEvent_t begin, cudaEvent_t end, int iters) {
    CUDA_OK(cudaEventSynchronize(end));
    float ms = 0.0f;
    CUDA_OK(cudaEventElapsedTime(&ms, begin, end));
    return ms / float(iters);
}

template <typename Launch>
static float measure_eager(Launch launch, int warmup, int iters, cudaStream_t stream) {
    for (int i = 0; i < warmup; ++i) launch();
    CUDA_OK(cudaStreamSynchronize(stream));
    Events events;
    CUDA_OK(cudaEventRecord(events.begin, stream));
    for (int i = 0; i < iters; ++i) launch();
    CUDA_OK(cudaEventRecord(events.end, stream));
    return elapsed_ms(events.begin, events.end, iters);
}

template <typename Launch>
static float measure_graph(Launch launch, int warmup, int iters, cudaStream_t stream) {
    cudaGraph_t graph{};
    cudaGraphExec_t exec{};
    CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    launch();
    CUDA_OK(cudaStreamEndCapture(stream, &graph));
    CUDA_OK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    for (int i = 0; i < warmup; ++i) CUDA_OK(cudaGraphLaunch(exec, stream));
    CUDA_OK(cudaStreamSynchronize(stream));
    Events events;
    CUDA_OK(cudaEventRecord(events.begin, stream));
    for (int i = 0; i < iters; ++i) CUDA_OK(cudaGraphLaunch(exec, stream));
    CUDA_OK(cudaEventRecord(events.end, stream));
    float ms = elapsed_ms(events.begin, events.end, iters);
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    return ms;
}

static cudaLaunchConfig_t epilogue_config(cudaStream_t stream, cudaLaunchAttribute *attr) {
    cudaLaunchConfig_t config{};
    config.gridDim = dim3((D / TILE) * KS, 1, 1);
    config.blockDim = dim3(256, 1, 1);
    config.dynamicSmemBytes = KS * TILE * sizeof(float);
    config.stream = stream;
    attr->id = cudaLaunchAttributeClusterDimension;
    attr->val.clusterDim.x = KS;
    attr->val.clusterDim.y = 1;
    attr->val.clusterDim.z = 1;
    config.attrs = attr;
    config.numAttrs = 1;
    return config;
}

static cudaLaunchConfig_t rms_config(cudaStream_t stream, cudaLaunchAttribute *attr) {
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(RMS_CTAS, 1, 1);
    config.blockDim = dim3(RMS_THREADS, 1, 1);
    config.dynamicSmemBytes =
        (RMS_VALUES_PER_THREAD * RMS_THREADS + 32) * sizeof(float);
    config.stream = stream;
    attr->id = cudaLaunchAttributeClusterDimension;
    attr->val.clusterDim.x = RMS_CTAS;
    attr->val.clusterDim.y = 1;
    attr->val.clusterDim.z = 1;
    config.attrs = attr;
    config.numAttrs = 1;
    return config;
}

static cudaLaunchConfig_t cooperative_config(
        cudaStream_t stream, cudaLaunchAttribute *attrs) {
    cudaLaunchConfig_t config{};
    config.gridDim = dim3((D / TILE) * KS, 1, 1);
    config.blockDim = dim3(256, 1, 1);
    config.dynamicSmemBytes = KS * TILE * sizeof(float);
    config.stream = stream;
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = KS;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    attrs[1].id = cudaLaunchAttributeCooperative;
    attrs[1].val.cooperative = 1;
    config.attrs = attrs;
    config.numAttrs = 2;
    return config;
}

int main(int argc, char **argv) {
    int iters = argc > 1 ? std::max(10, std::atoi(argv[1])) : 20000;
    int warmup = argc > 2 ? std::max(1, std::atoi(argv[2])) : 1000;
    cudaDeviceProp prop{};
    CUDA_OK(cudaGetDeviceProperties(&prop, 0));

    float *partials{}, *residual{}, *norm_weight{}, *projection{};
    float *control_h{}, *probe_h{}, *coop_h{}, *spin_h{};
    float *control_factor{}, *probe_factor{}, *coop_factor{}, *spin_factor{};
    float *coop_rms_warp{}, *spin_rms_warp{};
    unsigned *spin_sync{};
    uint8_t *control_codes{}, *probe_codes{}, *coop_codes{}, *spin_codes{};
    uint8_t *control_scales{}, *probe_scales{}, *coop_scales{}, *spin_scales{};
    CUDA_OK(cudaMalloc(&partials, size_t(KS) * D * sizeof(float)));
    CUDA_OK(cudaMalloc(&residual, D * sizeof(float)));
    CUDA_OK(cudaMalloc(&norm_weight, D * sizeof(float)));
    CUDA_OK(cudaMalloc(&projection, D * sizeof(float)));
    CUDA_OK(cudaMalloc(&control_h, D * sizeof(float)));
    CUDA_OK(cudaMalloc(&probe_h, D * sizeof(float)));
    CUDA_OK(cudaMalloc(&control_factor, sizeof(float)));
    CUDA_OK(cudaMalloc(&probe_factor, sizeof(float)));
    CUDA_OK(cudaMalloc(&control_codes, D));
    CUDA_OK(cudaMalloc(&probe_codes, D));
    CUDA_OK(cudaMalloc(&control_scales, GROUPS));
    CUDA_OK(cudaMalloc(&probe_scales, GROUPS));
    CUDA_OK(cudaMalloc(&coop_h, D * sizeof(float)));
    CUDA_OK(cudaMalloc(&coop_factor, sizeof(float)));
    CUDA_OK(cudaMalloc(&coop_rms_warp, 32 * sizeof(float)));
    CUDA_OK(cudaMalloc(&coop_codes, D));
    CUDA_OK(cudaMalloc(&coop_scales, GROUPS));
    CUDA_OK(cudaMalloc(&spin_h, D * sizeof(float)));
    CUDA_OK(cudaMalloc(&spin_factor, sizeof(float)));
    CUDA_OK(cudaMalloc(&spin_rms_warp, 32 * sizeof(float)));
    CUDA_OK(cudaMalloc(&spin_sync, 4 * sizeof(unsigned)));
    CUDA_OK(cudaMemset(spin_sync, 0, 4 * sizeof(unsigned)));
    CUDA_OK(cudaMalloc(&spin_codes, D));
    CUDA_OK(cudaMalloc(&spin_scales, GROUPS));

    cudaStream_t stream{};
    CUDA_OK(cudaStreamCreate(&stream));
    fill_inputs<<<(D + 255) / 256, 256, 0, stream>>>(partials, residual, norm_weight);
    CUDA_OK(cudaStreamSynchronize(stream));

    auto control = [&] {
        global_reduce<<<(D + 255) / 256, 256, 0, stream>>>(projection, partials);
        add_rms_factor<<<1, 1024, 0, stream>>>(control_factor, control_h, residual, projection);
        quant_control<<<D / 64, 32, 0, stream>>>(control_codes, control_scales,
                                                 control_h, control_factor, norm_weight);
    };
    auto probe = [&] {
        cudaLaunchAttribute attr_ep{};
        cudaLaunchConfig_t cfg_ep = epilogue_config(stream, &attr_ep);
        CUDA_OK(cudaLaunchKernelEx(&cfg_ep, cluster_epilogue, probe_h, partials, residual));
        cudaLaunchAttribute attr_rms{};
        cudaLaunchConfig_t cfg_rms = rms_config(stream, &attr_rms);
        CUDA_OK(cudaLaunchKernelEx(&cfg_rms, cluster_rms_quant, probe_codes, probe_scales,
                                   probe_factor, probe_h, norm_weight));
    };
    auto cooperative = [&] {
        cudaLaunchAttribute attrs[2]{};
        cudaLaunchConfig_t cfg = cooperative_config(stream, attrs);
        CUDA_OK(cudaLaunchKernelEx(&cfg, cooperative_epilogue, coop_codes, coop_scales,
                                   coop_factor, coop_h, coop_rms_warp,
                                   partials, residual, norm_weight));
    };
    auto resident = [&] {
        cudaLaunchAttribute attr{};
        cudaLaunchConfig_t cfg = epilogue_config(stream, &attr);
        CUDA_OK(cudaLaunchKernelEx(&cfg, resident_epilogue, spin_codes, spin_scales,
                                   spin_factor, spin_h, spin_rms_warp, spin_sync,
                                   partials, residual, norm_weight));
    };

    control();
    probe();
    cooperative();
    resident();
    CUDA_OK(cudaStreamSynchronize(stream));

    std::vector<float> ch(D), ph(D);
    std::vector<uint8_t> cc(D), pc(D), cs(GROUPS), ps(GROUPS);
    float cf = 0.0f, pf = 0.0f;
    std::vector<float> coh(D);
    std::vector<uint8_t> coc(D), cos(GROUPS);
    float cof = 0.0f;
    std::vector<float> sh(D);
    std::vector<uint8_t> sc(D), ss(GROUPS);
    float sf = 0.0f;
    CUDA_OK(cudaMemcpy(ch.data(), control_h, D * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(ph.data(), probe_h, D * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(cc.data(), control_codes, D, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(pc.data(), probe_codes, D, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(cs.data(), control_scales, GROUPS, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(ps.data(), probe_scales, GROUPS, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(&cf, control_factor, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(&pf, probe_factor, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(coh.data(), coop_h, D * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(coc.data(), coop_codes, D, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(cos.data(), coop_scales, GROUPS, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(&cof, coop_factor, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(sh.data(), spin_h, D * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(sc.data(), spin_codes, D, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(ss.data(), spin_scales, GROUPS, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(&sf, spin_factor, sizeof(float), cudaMemcpyDeviceToHost));

    float max_h_abs = 0.0f;
    int code_mismatches = 0, scale_mismatches = 0;
    for (int i = 0; i < D; ++i) {
        max_h_abs = std::max(max_h_abs, std::fabs(ch[i] - ph[i]));
        code_mismatches += cc[i] != pc[i];
    }
    for (int i = 0; i < GROUPS; ++i) scale_mismatches += cs[i] != ps[i];
    bool factor_exact = std::memcmp(&cf, &pf, sizeof(float)) == 0;
    float coop_max_h_abs = 0.0f;
    int coop_code_mismatches = 0, coop_scale_mismatches = 0;
    for (int i = 0; i < D; ++i) {
        coop_max_h_abs = std::max(coop_max_h_abs, std::fabs(ch[i] - coh[i]));
        coop_code_mismatches += cc[i] != coc[i];
    }
    for (int i = 0; i < GROUPS; ++i)
        coop_scale_mismatches += cs[i] != cos[i];
    bool coop_factor_exact = std::memcmp(&cf, &cof, sizeof(float)) == 0;
    float spin_max_h_abs = 0.0f;
    int spin_code_mismatches = 0, spin_scale_mismatches = 0;
    for (int i = 0; i < D; ++i) {
        spin_max_h_abs = std::max(spin_max_h_abs, std::fabs(ch[i] - sh[i]));
        spin_code_mismatches += cc[i] != sc[i];
    }
    for (int i = 0; i < GROUPS; ++i)
        spin_scale_mismatches += cs[i] != ss[i];
    bool spin_factor_exact = std::memcmp(&cf, &sf, sizeof(float)) == 0;

    float control_eager = measure_eager(control, warmup, iters, stream);
    float probe_eager = measure_eager(probe, warmup, iters, stream);
    float control_graph = measure_graph(control, warmup, iters, stream);
    float probe_graph = measure_graph(probe, warmup, iters, stream);
    float coop_eager = measure_eager(cooperative, warmup, iters, stream);
    float coop_graph = measure_graph(cooperative, warmup, iters, stream);
    float spin_eager = measure_eager(resident, warmup, iters, stream);
    float spin_graph = measure_graph(resident, warmup, iters, stream);

    cudaFuncAttributes reduce_attr{}, add_attr{}, quant_attr{}, ep_attr{}, rms_attr{};
    CUDA_OK(cudaFuncGetAttributes(&reduce_attr, global_reduce));
    CUDA_OK(cudaFuncGetAttributes(&add_attr, add_rms_factor));
    CUDA_OK(cudaFuncGetAttributes(&quant_attr, quant_control));
    CUDA_OK(cudaFuncGetAttributes(&ep_attr, cluster_epilogue));
    CUDA_OK(cudaFuncGetAttributes(&rms_attr, cluster_rms_quant));
    int ep_clusters = 0, rms_clusters = 0;
    cudaLaunchAttribute ep_occ_attr{};
    cudaLaunchConfig_t ep_occ = epilogue_config(stream, &ep_occ_attr);
    cudaError_t ep_occ_status = cudaOccupancyMaxActiveClusters(
        &ep_clusters, (const void *)cluster_epilogue, &ep_occ);
    cudaLaunchAttribute rms_occ_attr{};
    cudaLaunchConfig_t rms_occ = rms_config(stream, &rms_occ_attr);
    cudaError_t rms_occ_status = cudaOccupancyMaxActiveClusters(
        &rms_clusters, (const void *)cluster_rms_quant, &rms_occ);
    cudaFuncAttributes coop_attr{};
    CUDA_OK(cudaFuncGetAttributes(&coop_attr, cooperative_epilogue));
    int coop_clusters = 0;
    cudaLaunchAttribute coop_occ_attrs[2]{};
    cudaLaunchConfig_t coop_occ = cooperative_config(stream, coop_occ_attrs);
    cudaError_t coop_occ_status = cudaOccupancyMaxActiveClusters(
        &coop_clusters, (const void *)cooperative_epilogue, &coop_occ);
    cudaFuncAttributes spin_attr{};
    CUDA_OK(cudaFuncGetAttributes(&spin_attr, resident_epilogue));
    int spin_clusters = 0;
    cudaLaunchAttribute spin_occ_attr{};
    cudaLaunchConfig_t spin_occ = epilogue_config(stream, &spin_occ_attr);
    cudaError_t spin_occ_status = cudaOccupancyMaxActiveClusters(
        &spin_clusters, (const void *)resident_epilogue, &spin_occ);

    unsigned long long control_bytes =
        (unsigned long long)(KS * 4 + 4 + 12 + 16 + 1) * D + GROUPS + 4 + 80 * 4;
    unsigned long long probe_bytes =
        (unsigned long long)(KS * 4 + 4 + 4 + 4 + 8 + 1) * D + GROUPS + 4;

    std::printf(
        "{\"device\":\"%s\",\"cc\":\"%d.%d\",\"D\":%d,\"ks\":%d,"
        "\"max_h_abs\":%.9g,\"factor_exact\":%s,\"control_factor\":%.9g,"
        "\"probe_factor\":%.9g,\"code_mismatches\":%d,\"scale_mismatches\":%d,"
        "\"control_eager_us\":%.3f,\"probe_eager_us\":%.3f,"
        "\"control_graph_us\":%.3f,\"probe_graph_us\":%.3f,"
        "\"eager_speedup\":%.4f,\"graph_speedup\":%.4f,"
        "\"control_launches\":3,\"probe_launches\":2,"
        "\"control_dram_bytes\":%llu,\"probe_dram_bytes\":%llu,"
        "\"dram_bytes_removed\":%llu,\"reduce_regs\":%d,\"add_regs\":%d,"
        "\"quant_regs\":%d,\"epilogue_regs\":%d,\"rms_quant_regs\":%d,"
        "\"epilogue_dynamic_smem\":%zu,\"rms_quant_dynamic_smem\":%zu,"
        "\"epilogue_active_clusters\":%d,\"rms_quant_active_clusters\":%d,"
        "\"epilogue_occupancy_status\":%d,\"rms_quant_occupancy_status\":%d,"
        "\"iters\":%d,\"warmup\":%d}\n",
        prop.name, prop.major, prop.minor, D, KS, max_h_abs,
        factor_exact ? "true" : "false", cf, pf, code_mismatches, scale_mismatches,
        control_eager * 1000.0f, probe_eager * 1000.0f,
        control_graph * 1000.0f, probe_graph * 1000.0f,
        control_eager / probe_eager, control_graph / probe_graph,
        control_bytes, probe_bytes, control_bytes - probe_bytes,
        reduce_attr.numRegs, add_attr.numRegs, quant_attr.numRegs,
        ep_attr.numRegs, rms_attr.numRegs,
        ep_occ.dynamicSmemBytes, rms_occ.dynamicSmemBytes,
        ep_occ_status == cudaSuccess ? ep_clusters : -1,
        rms_occ_status == cudaSuccess ? rms_clusters : -1,
        int(ep_occ_status), int(rms_occ_status), iters, warmup);
    std::printf(
        "{\"variant\":\"cooperative\",\"max_h_abs\":%.9g,"
        "\"factor_exact\":%s,\"factor\":%.9g,\"code_mismatches\":%d,"
        "\"scale_mismatches\":%d,\"eager_us\":%.3f,\"graph_us\":%.3f,"
        "\"eager_speedup\":%.4f,\"graph_speedup\":%.4f,\"launches\":1,"
        "\"regs\":%d,\"dynamic_smem\":%zu,\"active_clusters\":%d,"
        "\"occupancy_status\":%d}\n",
        coop_max_h_abs, coop_factor_exact ? "true" : "false", cof,
        coop_code_mismatches, coop_scale_mismatches,
        coop_eager * 1000.0f, coop_graph * 1000.0f,
        control_eager / coop_eager, control_graph / coop_graph,
        coop_attr.numRegs, coop_occ.dynamicSmemBytes,
        coop_occ_status == cudaSuccess ? coop_clusters : -1,
        int(coop_occ_status));
    std::printf(
        "{\"variant\":\"resident-spin\",\"max_h_abs\":%.9g,"
        "\"factor_exact\":%s,\"factor\":%.9g,\"code_mismatches\":%d,"
        "\"scale_mismatches\":%d,\"eager_us\":%.3f,\"graph_us\":%.3f,"
        "\"eager_speedup\":%.4f,\"graph_speedup\":%.4f,\"launches\":1,"
        "\"regs\":%d,\"dynamic_smem\":%zu,\"active_clusters\":%d,"
        "\"occupancy_status\":%d}\n",
        spin_max_h_abs, spin_factor_exact ? "true" : "false", sf,
        spin_code_mismatches, spin_scale_mismatches,
        spin_eager * 1000.0f, spin_graph * 1000.0f,
        control_eager / spin_eager, control_graph / spin_graph,
        spin_attr.numRegs, spin_occ.dynamicSmemBytes,
        spin_occ_status == cudaSuccess ? spin_clusters : -1,
        int(spin_occ_status));

    cudaFree(spin_scales); cudaFree(spin_codes); cudaFree(spin_sync);
    cudaFree(spin_rms_warp); cudaFree(spin_factor); cudaFree(spin_h);
    cudaFree(coop_scales); cudaFree(coop_codes);
    cudaFree(coop_rms_warp); cudaFree(coop_factor); cudaFree(coop_h);
    cudaStreamDestroy(stream);
    cudaFree(probe_scales); cudaFree(control_scales);
    cudaFree(probe_codes); cudaFree(control_codes);
    cudaFree(probe_factor); cudaFree(control_factor);
    cudaFree(probe_h); cudaFree(control_h); cudaFree(projection);
    cudaFree(norm_weight); cudaFree(residual); cudaFree(partials);
    return (max_h_abs == 0.0f && factor_exact && code_mismatches == 0 &&
            scale_mismatches == 0 && coop_max_h_abs == 0.0f &&
            coop_factor_exact && coop_code_mismatches == 0 &&
            coop_scale_mismatches == 0 && spin_max_h_abs == 0.0f &&
            spin_factor_exact && spin_code_mismatches == 0 &&
            spin_scale_mismatches == 0) ? 0 : 2;
}
