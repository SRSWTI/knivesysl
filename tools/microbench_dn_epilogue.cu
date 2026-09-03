// Standalone sm120 proof for docs/level-up.md DeltaNet output publication.
//
// This does not link or modify the production engine. It isolates one 64-token
// chunk after k_tq_dnmm_scan_tf32 has produced four 32-value stripes per head.
// The control publishes FP32 core_raw, launches the exact gated RMSNorm tree,
// then runs the production NVFP4 activation layout. The probe clusters the four
// stripes, performs the identical reduction/gate/quantization from live shared
// values, and publishes only packed NVFP4 codes/scales.

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace cg = cooperative_groups;

#define CUDA_OK(expr) do { \
    cudaError_t status_ = (expr); \
    if (status_ != cudaSuccess) { \
        std::fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__, \
                     #expr, cudaGetErrorString(status_)); \
        std::exit(1); \
    } \
} while (0)

constexpr int T = 64;
constexpr int D = 128;
constexpr int VS = 32;
constexpr int VALUE_HEADS = 48;
constexpr int K = VALUE_HEADS * D;
constexpr int STRIPES = D / VS;
constexpr int NGROUPS = T / 8;
constexpr int KT64 = K / 64;
constexpr int NVF4_BW = 72;
constexpr float EPS = 1.0e-6f;

static __host__ __device__ __forceinline__ int b_k(int lane, int reg, int nib) {
    return 32 * reg + 16 * ((lane >> 1) & 1) + 8 * (lane & 1) + nib;
}

static __device__ __forceinline__ uint32_t quant_e2m1(float v) {
    float a = fminf(fabsf(v), 6.0f);
    uint32_t code;
    if (a < 0.75f) code = a < 0.25f ? 0u : 1u;
    else {
        int e = a < 2.0f ? 1 : (a < 4.0f ? 2 : 3);
        float sc = float(1 << (e - 1));
        int m = int(lrintf(a / sc - 1.0f));
        if (m > 1) { m = 0; ++e; }
        if (e > 3) { e = 3; m = 1; }
        code = uint32_t((e << 1) | m);
    }
    return signbit(v) ? code | 8u : code;
}

static __device__ __forceinline__ float ue4m3_to_f(uint8_t b) {
    int E = (b >> 3) & 15, f = b & 7;
    if (E == 0) return 0.0f;
    return ldexpf(1.0f + float(f) * 0.125f, E - 7);
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

__global__ void fill_inputs(float *raw, float *z, float *norm_weight) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < T * K) {
        raw[i] = sinf(float(i) * 0.00317f) * 0.8f
               + cosf(float(i) * 0.00091f) * 0.3f;
        z[i] = sinf(float(i) * 0.00173f) * 1.7f;
    }
    if (i < D)
        norm_weight[i] = 1.0f + sinf(float(i) * 0.017f) * 0.125f;
}

// Same head/stripe geometry as the scan's final global publication.
__global__ __launch_bounds__(256, 2)
void publish_control(float *core, const float *source) {
    int head = blockIdx.x / STRIPES;
    int stripe = blockIdx.x % STRIPES;
    int tid = threadIdx.x;
    for (int idx = tid; idx < T * VS; idx += blockDim.x) {
        int token = idx / VS, v = stripe * VS + idx % VS;
        size_t p = size_t(token) * K + head * D + v;
        core[p] = source[p];
    }
}

// Exact k_tq_deltanet_norm reduction tree and expression.
__global__ __launch_bounds__(128)
void norm_control(float *out, const float *core, const float *z,
                  const float *norm_weight) {
    int blk = blockIdx.x;
    int v = threadIdx.x;
    int head = blk % VALUE_HEADS;
    int token = blk / VALUE_HEADS;
    size_t base = size_t(token) * K + head * D;
    __shared__ float red[D];
    float c = core[base + v];
    red[v] = c * c;
    __syncthreads();
    for (int stride = D >> 1; stride > 0; stride >>= 1) {
        if (v < stride) red[v] += red[v + stride];
        __syncthreads();
    }
    float normed = c * rsqrtf(red[0] / float(D) + EPS) * norm_weight[v];
    float gate = z[base + v];
    out[base + v] = normed * (gate / (1.0f + expf(-gate)));
}

// Exact production tile layout: eight token rows by one k64 tile.
__global__ void quant_control(uint32_t *packed, const float *x) {
    int tile = blockIdx.x;
    int g8 = tile / KT64, kt = tile % KT64;
    int lane = threadIdx.x;
    uint32_t *dst = packed + size_t(tile) * NVF4_BW;
    __shared__ float sc[8][4];
    if (lane < 8) {
        int col = g8 * 8 + lane;
        uint32_t sw = 0;
        for (int g = 0; g < 4; ++g) {
            float mx = 0.0f;
            for (int t = 0; t < 16; ++t) {
                int k = kt * 64 + g * 16 + t;
                mx = fmaxf(mx, fabsf(x[size_t(col) * K + k]));
            }
            uint8_t sb = f2ue4m3(mx / 6.0f);
            if (sb == 0) sb = 1;
            sw |= uint32_t(sb) << (8 * g);
            sc[lane][g] = ue4m3_to_f(sb);
        }
        dst[64 + lane] = sw;
    }
    __syncthreads();
    uint32_t words[2] = {0, 0};
    int col = g8 * 8 + (lane >> 2);
    for (int reg = 0; reg < 2; ++reg) {
        for (int j = 0; j < 8; ++j) {
            int k = b_k(lane, reg, j);
            float v = x[size_t(col) * K + kt * 64 + k];
            float scale = sc[lane >> 2][k >> 4];
            words[reg] |= quant_e2m1(scale > 0.0f ? v / scale : 0.0f) << (4 * j);
        }
    }
    dst[lane * 2] = words[0];
    dst[lane * 2 + 1] = words[1];
}

__global__ __launch_bounds__(256, 1) __cluster_dims__(STRIPES, 1, 1)
void fused_probe(uint32_t *packed, float *debug_norm, const float *source,
                 const float *z, const float *norm_weight, int publish_debug) {
    cg::cluster_group cluster = cg::this_cluster();
    int rank = int(cluster.block_rank());
    int head = int(blockIdx.x) / STRIPES;
    int tid = threadIdx.x;
    int warp = tid >> 5, lane = tid & 31;
    extern __shared__ float sm[];
    float *local = sm;                         // [T][VS], this stripe
    float *mailbox = local + T * VS;           // [T][D], rank-0 gather / peer half
    float *factor = mailbox + T * D;           // [T], pushed to every rank
    float *owner_box = rank == 0 ? mailbox : cluster.map_shared_rank(mailbox, 0);

    // Every producer retains its stripe locally and pushes a second copy to
    // rank 0. No consumer performs a remote shared-memory load.
    for (int idx = tid; idx < T * VS; idx += blockDim.x) {
        int token = idx / VS, d = rank * VS + idx % VS;
        float v = source[size_t(token) * K + head * D + d];
        local[idx] = v;
        owner_box[rank * T * VS + idx] = v;
    }
    cluster.sync();

    if (rank == 0) {
        #pragma unroll
        for (int batch = 0; batch < T / 8; ++batch) {
            int token = batch * 8 + warp;
            const float *row = mailbox + token * VS;
            float a = row[lane];
            float b = row[2 * T * VS + lane];
            float c = row[T * VS + lane];
            float d = row[3 * T * VS + lane];
            float aa = __fmul_rn(a, a), bb = __fmul_rn(b, b);
            float cc = __fmul_rn(c, c), dd = __fmul_rn(d, d);
            float sum = __fadd_rn(__fadd_rn(aa, bb), __fadd_rn(cc, dd));
            for (int offset = 16; offset > 0; offset >>= 1)
                sum = __fadd_rn(
                    sum, __shfl_down_sync(0xffffffffu, sum, offset));
            if (lane == 0) factor[token] = rsqrtf(sum / float(D) + EPS);
        }
        __syncthreads();
        if (tid < T) {
            float f = factor[tid];
            #pragma unroll
            for (int dst_rank = 1; dst_rank < STRIPES; ++dst_rank) {
                float *remote_factor = cluster.map_shared_rank(factor, dst_rank);
                remote_factor[tid] = f;
            }
        }
    }
    cluster.sync();

    for (int idx = tid; idx < T * VS; idx += blockDim.x) {
        int token = idx / VS, d = rank * VS + idx % VS;
        size_t p = size_t(token) * K + head * D + d;
        float gate = z[p];
        float normed = local[idx] * factor[token] * norm_weight[d]
                     * (gate / (1.0f + expf(-gate)));
        local[idx] = normed;
        if (publish_debug) debug_norm[p] = normed;
        if (rank == 1 || rank == 3) {
            float *peer_box = cluster.map_shared_rank(mailbox, rank - 1);
            peer_box[idx] = normed;
        }
    }
    cluster.sync();

    // Ranks 0 and 2 now own both local 32-value halves without a remote read.
    // Every warp publishes one eight-token tile in production B-fragment order.
    if ((rank & 1) == 0) {
        int kt = head * 2 + rank / 2;
        int g8 = warp;
        const float *peer = mailbox;
        __shared__ float scale_sm[8][8][4];
        if (lane < 8) {
            int token = g8 * 8 + lane;
            uint32_t sw = 0;
            for (int g = 0; g < 4; ++g) {
                float mx = 0.0f;
                for (int t = 0; t < 16; ++t) {
                    int k64 = g * 16 + t;
                    const float *src = k64 < VS ? local : peer;
                    int lk = k64 < VS ? k64 : k64 - VS;
                    mx = fmaxf(mx, fabsf(src[token * VS + lk]));
                }
                uint8_t sb = f2ue4m3(mx / 6.0f);
                if (sb == 0) sb = 1;
                sw |= uint32_t(sb) << (8 * g);
                scale_sm[warp][lane][g] = ue4m3_to_f(sb);
            }
            packed[(size_t(g8) * KT64 + kt) * NVF4_BW + 64 + lane] = sw;
        }
        __syncwarp();
        uint32_t words[2] = {0, 0};
        int token = g8 * 8 + (lane >> 2);
        for (int reg = 0; reg < 2; ++reg) {
            for (int j = 0; j < 8; ++j) {
                int k64 = b_k(lane, reg, j);
                const float *src = k64 < VS ? local : peer;
                int lk = k64 < VS ? k64 : k64 - VS;
                float v = src[token * VS + lk];
                float scale = scale_sm[warp][lane >> 2][k64 >> 4];
                words[reg] |= quant_e2m1(scale > 0.0f ? v / scale : 0.0f) << (4 * j);
            }
        }
        size_t dst = (size_t(g8) * KT64 + kt) * NVF4_BW + lane * 2;
        packed[dst] = words[0];
        packed[dst + 1] = words[1];
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

static cudaLaunchConfig_t probe_config(cudaStream_t stream,
                                       cudaLaunchAttribute *attr) {
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(VALUE_HEADS * STRIPES, 1, 1);
    config.blockDim = dim3(256, 1, 1);
    config.dynamicSmemBytes = (T * VS + T * D + T) * sizeof(float);
    config.stream = stream;
    attr->id = cudaLaunchAttributeClusterDimension;
    attr->val.clusterDim.x = STRIPES;
    attr->val.clusterDim.y = 1;
    attr->val.clusterDim.z = 1;
    config.attrs = attr;
    config.numAttrs = 1;
    return config;
}

int main(int argc, char **argv) {
    int iters = argc > 1 ? std::max(10, std::atoi(argv[1])) : 5000;
    int warmup = argc > 2 ? std::max(1, std::atoi(argv[2])) : 250;
    constexpr size_t values = size_t(T) * K;
    constexpr size_t packed_words = size_t(NGROUPS) * KT64 * NVF4_BW;
    float *source{}, *z{}, *norm_weight{}, *core{}, *control_norm{}, *probe_norm{};
    uint32_t *control_packed{}, *probe_packed{};
    CUDA_OK(cudaMalloc(&source, values * sizeof(float)));
    CUDA_OK(cudaMalloc(&z, values * sizeof(float)));
    CUDA_OK(cudaMalloc(&norm_weight, D * sizeof(float)));
    CUDA_OK(cudaMalloc(&core, values * sizeof(float)));
    CUDA_OK(cudaMalloc(&control_norm, values * sizeof(float)));
    CUDA_OK(cudaMalloc(&probe_norm, values * sizeof(float)));
    CUDA_OK(cudaMalloc(&control_packed, packed_words * sizeof(uint32_t)));
    CUDA_OK(cudaMalloc(&probe_packed, packed_words * sizeof(uint32_t)));
    cudaStream_t stream{};
    CUDA_OK(cudaStreamCreate(&stream));
    fill_inputs<<<(values + 255) / 256, 256, 0, stream>>>(source, z, norm_weight);
    CUDA_OK(cudaStreamSynchronize(stream));
    auto control = [&] {
        publish_control<<<VALUE_HEADS * STRIPES, 256, 0, stream>>>(core, source);
        norm_control<<<T * VALUE_HEADS, D, 0, stream>>>(
            control_norm, core, z, norm_weight);
        quant_control<<<NGROUPS * KT64, 32, 0, stream>>>(
            control_packed, control_norm);
    };
    auto probe = [&] {
        cudaLaunchAttribute attr{};
        cudaLaunchConfig_t config = probe_config(stream, &attr);
        CUDA_OK(cudaLaunchKernelEx(&config, fused_probe, probe_packed, probe_norm,
                                   source, z, norm_weight, 0));
    };
    control();
    {
        cudaLaunchAttribute attr{};
        cudaLaunchConfig_t config = probe_config(stream, &attr);
        CUDA_OK(cudaLaunchKernelEx(&config, fused_probe, probe_packed, probe_norm,
                                   source, z, norm_weight, 1));
    }
    CUDA_OK(cudaStreamSynchronize(stream));
    std::vector<float> control_host(values), probe_host(values);
    std::vector<uint32_t> control_pack_host(packed_words), probe_pack_host(packed_words);
    CUDA_OK(cudaMemcpy(control_host.data(), control_norm, values * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(probe_host.data(), probe_norm, values * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(control_pack_host.data(), control_packed, packed_words * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(probe_pack_host.data(), probe_packed, packed_words * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    size_t norm_mismatches = 0, packed_mismatches = 0;
    float max_abs = 0.0f;
    for (size_t i = 0; i < values; ++i) {
        norm_mismatches += std::memcmp(&control_host[i], &probe_host[i], 4) != 0;
        max_abs = std::max(max_abs, std::fabs(control_host[i] - probe_host[i]));
    }
    for (size_t i = 0; i < packed_words; ++i)
        packed_mismatches += control_pack_host[i] != probe_pack_host[i];
    float control_eager = measure_eager(control, warmup, iters, stream);
    float probe_eager = measure_eager(probe, warmup, iters, stream);
    float control_graph = measure_graph(control, warmup, iters, stream);
    float probe_graph = measure_graph(probe, warmup, iters, stream);
    cudaFuncAttributes pub_attr{}, norm_attr{}, quant_attr{}, probe_attr{};
    CUDA_OK(cudaFuncGetAttributes(&pub_attr, publish_control));
    CUDA_OK(cudaFuncGetAttributes(&norm_attr, norm_control));
    CUDA_OK(cudaFuncGetAttributes(&quant_attr, quant_control));
    CUDA_OK(cudaFuncGetAttributes(&probe_attr, fused_probe));
    cudaLaunchAttribute occ_launch_attr{};
    cudaLaunchConfig_t occ_config = probe_config(stream, &occ_launch_attr);
    int active_clusters = 0;
    cudaError_t occ_status = cudaOccupancyMaxActiveClusters(
        &active_clusters, (const void *)fused_probe, &occ_config);
    cudaDeviceProp prop{};
    CUDA_OK(cudaGetDeviceProperties(&prop, 0));
    unsigned long long removed_bytes = 4ULL * values * sizeof(float);
    std::printf(
        "{\"device\":\"%s\",\"cc\":\"%d.%d\",\"tokens\":%d,"
        "\"value_heads\":%d,\"dim\":%d,\"norm_mismatches\":%zu,"
        "\"packed_mismatches\":%zu,\"max_abs\":%.9g,"
        "\"control_eager_us\":%.3f,\"probe_eager_us\":%.3f,"
        "\"control_graph_us\":%.3f,\"probe_graph_us\":%.3f,"
        "\"eager_speedup\":%.4f,\"graph_speedup\":%.4f,"
        "\"control_launches\":3,\"probe_launches\":1,"
        "\"modeled_bytes_removed\":%llu,\"publish_regs\":%d,"
        "\"norm_regs\":%d,\"quant_regs\":%d,\"probe_regs\":%d,"
        "\"probe_dynamic_smem\":%zu,\"active_clusters\":%d,"
        "\"occupancy_status\":%d,\"iters\":%d,\"warmup\":%d}\n",
        prop.name, prop.major, prop.minor, T, VALUE_HEADS, D,
        norm_mismatches, packed_mismatches, max_abs,
        control_eager * 1000.0f, probe_eager * 1000.0f,
        control_graph * 1000.0f, probe_graph * 1000.0f,
        control_eager / probe_eager, control_graph / probe_graph,
        removed_bytes, pub_attr.numRegs, norm_attr.numRegs, quant_attr.numRegs,
        probe_attr.numRegs, occ_config.dynamicSmemBytes,
        occ_status == cudaSuccess ? active_clusters : -1, int(occ_status),
        iters, warmup);
    cudaStreamDestroy(stream);
    cudaFree(probe_packed); cudaFree(control_packed);
    cudaFree(probe_norm); cudaFree(control_norm); cudaFree(core);
    cudaFree(norm_weight); cudaFree(z); cudaFree(source);
    return (norm_mismatches == 0 && packed_mismatches == 0) ? 0 : 2;
}
