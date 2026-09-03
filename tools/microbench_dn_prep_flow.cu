// Standalone sm120 dataflow proof for docs/level-up.md. Models one 2048-token
// DeltaNet prefill wave (32 chunk-64 blocks, 48 heads) without copying math from
// the engine: current global prep publication/readback versus producer-push DSM
// streaming into the four recurrent-state stripe owners. No production linkage.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace cg = cooperative_groups;

#define CUDA_OK(expr) do { cudaError_t s_ = (expr); if (s_ != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__, #expr, \
                 cudaGetErrorString(s_)); std::exit(1); } } while (0)

constexpr int HEADS = 48;
constexpr int CHUNKS = 32;
constexpr int T = 64;
constexpr int D = 128;
constexpr int STRIPES = 4;
constexpr int VS = D / STRIPES;
constexpr int MAT = T * D;
constexpr int AM = T * T;
constexpr int PREP = 4 * MAT + AM;
constexpr int KW_OFF = 0;
constexpr int D0_OFF = MAT;
constexpr int EQ_OFF = 2 * MAT;
constexpr int KC_OFF = 3 * MAT;
constexpr int AM_OFF = 4 * MAT;

__global__ void fill_source(float *source, size_t n) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        uint32_t x = uint32_t(i) * 2654435761u + 1013904223u;
        x ^= x >> 15; x *= 2246822519u; x ^= x >> 13;
        source[i] = __uint_as_float(0x3f000000u | (x & 0x007fffffu));
    }
}

__global__ void publish_control(float *prep, const float *source) {
    int item = blockIdx.x;
    size_t base = size_t(item) * PREP;
    for (int i = threadIdx.x; i < PREP; i += blockDim.x)
        prep[base + i] = source[base + i];
}

__device__ __forceinline__ uint32_t xor_block(uint32_t v) {
    for (int off = 16; off > 0; off >>= 1)
        v ^= __shfl_down_sync(0xffffffffu, v, off);
    __shared__ uint32_t warp_xor[8];
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) warp_xor[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = lane < 8 ? warp_xor[lane] : 0u;
        for (int off = 16; off > 0; off >>= 1)
            v ^= __shfl_down_sync(0xffffffffu, v, off);
    }
    __syncthreads();
    return v;
}

__global__ void consume_control(uint32_t *out, const float *prep) {
    int head = blockIdx.x / STRIPES;
    int rank = blockIdx.x % STRIPES;
    uint32_t total = 0;
    for (int chunk = 0; chunk < CHUNKS; ++chunk) {
        const float *p = prep + size_t(head * CHUNKS + chunk) * PREP;
        constexpr int full_offsets[] = {KW_OFF, EQ_OFF, KC_OFF};
        #pragma unroll
        for (int phase = 0; phase < 3; ++phase) {
            const float *m = p + full_offsets[phase];
            for (int i = threadIdx.x; i < MAT; i += blockDim.x)
                total ^= __float_as_uint(m[i]);
        }
        for (int i = threadIdx.x; i < AM; i += blockDim.x)
            total ^= __float_as_uint(p[AM_OFF + i]);
        for (int i = threadIdx.x; i < T * VS; i += blockDim.x) {
            int token = i / VS, local = i % VS;
            total ^= __float_as_uint(p[D0_OFF + token * D + rank * VS + local]);
        }
    }
    total = xor_block(total);
    if (threadIdx.x == 0) out[head * STRIPES + rank] = total;
}

__device__ __forceinline__ uint32_t stream_phase(
        cg::cluster_group cluster, float *box, const float *source,
        int count, int rank, uint32_t total) {
    int global_tid = rank * blockDim.x + threadIdx.x;
    for (int i = global_tid; i < count; i += STRIPES * blockDim.x) {
        float v = source[i];
        #pragma unroll
        for (int dst_rank = 0; dst_rank < STRIPES; ++dst_rank) {
            float *dst = dst_rank == rank ? box : cluster.map_shared_rank(box, dst_rank);
            dst[i] = v;
        }
    }
    cluster.sync();
    for (int i = threadIdx.x; i < count; i += blockDim.x)
        total ^= __float_as_uint(box[i]);
    cluster.sync();
    return total;
}

__global__ __launch_bounds__(256, 1) __cluster_dims__(STRIPES, 1, 1)
void consume_probe(uint32_t *out, const float *source) {
    cg::cluster_group cluster = cg::this_cluster();
    int rank = int(cluster.block_rank());
    int head = int(blockIdx.x) / STRIPES;
    extern __shared__ float box[];
    uint32_t total = 0;
    for (int chunk = 0; chunk < CHUNKS; ++chunk) {
        const float *p = source + size_t(head * CHUNKS + chunk) * PREP;
        total = stream_phase(cluster, box, p + KW_OFF, MAT, rank, total);
        total = stream_phase(cluster, box, p + EQ_OFF, MAT, rank, total);
        total = stream_phase(cluster, box, p + KC_OFF, MAT, rank, total);
        total = stream_phase(cluster, box, p + AM_OFF, AM, rank, total);
        for (int i = threadIdx.x; i < T * VS; i += blockDim.x) {
            int token = i / VS, local = i % VS;
            total ^= __float_as_uint(
                p[D0_OFF + token * D + rank * VS + local]);
        }
    }
    total = xor_block(total);
    if (threadIdx.x == 0) out[head * STRIPES + rank] = total;
}

static cudaLaunchConfig_t probe_config(cudaStream_t stream,
                                       cudaLaunchAttribute *attr) {
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(HEADS * STRIPES, 1, 1);
    cfg.blockDim = dim3(256, 1, 1);
    cfg.dynamicSmemBytes = MAT * sizeof(float);
    cfg.stream = stream;
    attr->id = cudaLaunchAttributeClusterDimension;
    attr->val.clusterDim.x = STRIPES;
    attr->val.clusterDim.y = 1;
    attr->val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    return cfg;
}

struct Events {
    cudaEvent_t begin{}, end{};
    Events() { CUDA_OK(cudaEventCreate(&begin)); CUDA_OK(cudaEventCreate(&end)); }
    ~Events() { cudaEventDestroy(begin); cudaEventDestroy(end); }
};

static float elapsed(cudaEvent_t a, cudaEvent_t b, int iters) {
    CUDA_OK(cudaEventSynchronize(b));
    float ms = 0.0f;
    CUDA_OK(cudaEventElapsedTime(&ms, a, b));
    return ms * 1000.0f / float(iters);
}

template <typename F>
static float eager(F fn, int warmup, int iters, cudaStream_t stream) {
    for (int i = 0; i < warmup; ++i) fn();
    CUDA_OK(cudaStreamSynchronize(stream));
    Events e;
    CUDA_OK(cudaEventRecord(e.begin, stream));
    for (int i = 0; i < iters; ++i) fn();
    CUDA_OK(cudaEventRecord(e.end, stream));
    return elapsed(e.begin, e.end, iters);
}

template <typename F>
static float graph(F fn, int warmup, int iters, cudaStream_t stream) {
    cudaGraph_t g{};
    cudaGraphExec_t exec{};
    CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    fn();
    CUDA_OK(cudaStreamEndCapture(stream, &g));
    CUDA_OK(cudaGraphInstantiate(&exec, g, nullptr, nullptr, 0));
    for (int i = 0; i < warmup; ++i) CUDA_OK(cudaGraphLaunch(exec, stream));
    CUDA_OK(cudaStreamSynchronize(stream));
    Events e;
    CUDA_OK(cudaEventRecord(e.begin, stream));
    for (int i = 0; i < iters; ++i) CUDA_OK(cudaGraphLaunch(exec, stream));
    CUDA_OK(cudaEventRecord(e.end, stream));
    float us = elapsed(e.begin, e.end, iters);
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(g);
    return us;
}

int main(int argc, char **argv) {
    int iters = argc > 1 ? std::max(10, std::atoi(argv[1])) : 500;
    int warmup = argc > 2 ? std::max(1, std::atoi(argv[2])) : 25;
    constexpr size_t values = size_t(HEADS) * CHUNKS * PREP;
    float *source{}, *prep{};
    uint32_t *control_out{}, *probe_out{};
    CUDA_OK(cudaMalloc(&source, values * sizeof(float)));
    CUDA_OK(cudaMalloc(&prep, values * sizeof(float)));
    CUDA_OK(cudaMalloc(&control_out, HEADS * STRIPES * sizeof(uint32_t)));
    CUDA_OK(cudaMalloc(&probe_out, HEADS * STRIPES * sizeof(uint32_t)));
    cudaStream_t stream{};
    CUDA_OK(cudaStreamCreate(&stream));
    fill_source<<<(values + 255) / 256, 256, 0, stream>>>(source, values);
    CUDA_OK(cudaStreamSynchronize(stream));

    auto control = [&] {
        publish_control<<<HEADS * CHUNKS, 256, 0, stream>>>(prep, source);
        consume_control<<<HEADS * STRIPES, 256, 0, stream>>>(control_out, prep);
    };
    auto probe = [&] {
        cudaLaunchAttribute attr{};
        cudaLaunchConfig_t cfg = probe_config(stream, &attr);
        CUDA_OK(cudaLaunchKernelEx(&cfg, consume_probe, probe_out, source));
    };

    control();
    probe();
    CUDA_OK(cudaStreamSynchronize(stream));
    std::vector<uint32_t> c(HEADS * STRIPES), p(HEADS * STRIPES);
    CUDA_OK(cudaMemcpy(c.data(), control_out, c.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(p.data(), probe_out, p.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    size_t mismatches = 0;
    for (size_t i = 0; i < c.size(); ++i) mismatches += c[i] != p[i];

    float ce = eager(control, warmup, iters, stream);
    float pe = eager(probe, warmup, iters, stream);
    float cg = graph(control, warmup, iters, stream);
    float pg = graph(probe, warmup, iters, stream);
    cudaFuncAttributes attr{};
    CUDA_OK(cudaFuncGetAttributes(&attr, consume_probe));
    cudaLaunchAttribute occ_attr{};
    cudaLaunchConfig_t occ_cfg = probe_config(stream, &occ_attr);
    int active_clusters = 0;
    cudaError_t occ_status = cudaOccupancyMaxActiveClusters(
        &active_clusters, (const void *)consume_probe, &occ_cfg);
    cudaDeviceProp prop{};
    CUDA_OK(cudaGetDeviceProperties(&prop, 0));
    unsigned long long prep_bytes = values * sizeof(float);
    unsigned long long control_reads =
        size_t(HEADS) * CHUNKS * (STRIPES * (3 * MAT + AM) + MAT) * sizeof(float);
    unsigned long long probe_dsm =
        size_t(HEADS) * CHUNKS * STRIPES * (3 * MAT + AM) * sizeof(float);
    std::printf(
        "{\"device\":\"%s\",\"cc\":\"%d.%d\",\"tokens\":%d,"
        "\"heads\":%d,\"chunks\":%d,\"mismatches\":%zu,"
        "\"control_eager_us\":%.3f,\"probe_eager_us\":%.3f,"
        "\"control_graph_us\":%.3f,\"probe_graph_us\":%.3f,"
        "\"eager_speedup\":%.4f,\"graph_speedup\":%.4f,"
        "\"control_launches\":2,\"probe_launches\":1,"
        "\"prep_global_write_bytes\":%llu,\"control_global_read_bytes\":%llu,"
        "\"probe_dsm_store_bytes\":%llu,\"probe_regs\":%d,"
        "\"probe_dynamic_smem\":%zu,\"active_clusters\":%d,"
        "\"occupancy_status\":%d,\"iters\":%d,\"warmup\":%d}\n",
        prop.name, prop.major, prop.minor, T * CHUNKS, HEADS, CHUNKS,
        mismatches, ce, pe, cg, pg, ce / pe, cg / pg, prep_bytes,
        control_reads, probe_dsm, attr.numRegs, occ_cfg.dynamicSmemBytes,
        occ_status == cudaSuccess ? active_clusters : -1, int(occ_status),
        iters, warmup);

    cudaStreamDestroy(stream);
    cudaFree(probe_out); cudaFree(control_out); cudaFree(prep); cudaFree(source);
    return mismatches == 0 ? 0 : 2;
}
