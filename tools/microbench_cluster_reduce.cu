// Standalone sm120 proof for docs/level-up.md stage 1.
//
// This does NOT modify or link the production engine. It isolates the finalization
// mechanism used after split-K: the control writes every partial to DRAM and launches
// a second reducer; the probe writes the same values to per-CTA DSM, synchronizes a
// cluster, and lets rank 0 fold ranks in ascending order before one global publish.
//
// Build:
//   nvcc -std=c++17 -O3 -lineinfo -gencode=arch=compute_120,code=sm_120 \
//     -o results/microbench/cluster_reduce tools/microbench_cluster_reduce.cu

#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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

static __device__ __forceinline__ float partial_value(int row, int col, int split) {
    // Exactly represented binary fractions keep the comparison focused on fold order.
    return float((row & 31) + 1) * 0.03125f
         + float(col + 1) * 0.125f
         + float(split + 1) * 0.5f;
}

__global__ __launch_bounds__(256, 1)
void global_partial(float *__restrict__ partials, int M, int nvar, int ks) {
    const int split = int(blockIdx.x) % ks;
    const int mblock = int(blockIdx.x) / ks;
    const int base_row = mblock * 128;
    const size_t plane = size_t(M) * nvar;
    for (int local = int(threadIdx.x); local < nvar * 128; local += int(blockDim.x)) {
        const int col = local / 128;
        const int row = base_row + local % 128;
        if (row < M)
            partials[size_t(split) * plane + size_t(col) * M + row] =
                partial_value(row, col, split);
    }
}

__global__ __launch_bounds__(256, 1)
void global_reduce(float *__restrict__ out, const float *__restrict__ partials,
                   int count, int ks) {
    const int i = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (i >= count) return;
    float sum = 0.0f;
    for (int split = 0; split < ks; ++split)
        sum += partials[size_t(split) * count + i];
    out[i] = sum;
}

template <int KS>
__global__ __launch_bounds__(256, 1) __cluster_dims__(KS, 1, 1)
void cluster_reduce(float *__restrict__ out, int M, int nvar) {
    cg::cluster_group cluster = cg::this_cluster();
    const int rank = int(cluster.block_rank());
    const int mblock = int(blockIdx.x) / KS;
    const int base_row = mblock * 128;
    extern __shared__ float mailbox[];
    const int stride = nvar * 128;
    float *owner = rank == 0 ? mailbox : cluster.map_shared_rank(mailbox, 0);
    for (int local = int(threadIdx.x); local < stride; local += int(blockDim.x)) {
        const int col = local / 128;
        const int row = base_row + local % 128;
        owner[rank * stride + local] =
            row < M ? partial_value(row, col, rank) : 0.0f;
    }
    cluster.sync();
    if (rank == 0) {
        for (int local = int(threadIdx.x); local < stride; local += int(blockDim.x)) {
            const int col = local / 128;
            const int row = base_row + local % 128;
            if (row >= M) continue;
            float sum = 0.0f;
            #pragma unroll
            for (int split = 0; split < KS; ++split)
                sum += mailbox[split * stride + local];
            out[size_t(col) * M + row] = sum;
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

template <int KS>
static cudaLaunchConfig_t cluster_config(int M, int nvar, cudaLaunchAttribute *attr) {
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(((M + 127) / 128) * KS, 1, 1);
    config.blockDim = dim3(256, 1, 1);
    config.dynamicSmemBytes = size_t(KS) * nvar * 128 * sizeof(float);
    attr->id = cudaLaunchAttributeClusterDimension;
    attr->val.clusterDim.x = KS;
    attr->val.clusterDim.y = 1;
    attr->val.clusterDim.z = 1;
    config.attrs = attr;
    config.numAttrs = 1;
    return config;
}

static void launch_control(float *out, float *partials, int M, int nvar, int ks,
                           cudaStream_t stream) {
    const int count = M * nvar;
    global_partial<<<((M + 127) / 128) * ks, 256, 0, stream>>>(partials, M, nvar, ks);
    global_reduce<<<(count + 255) / 256, 256, 0, stream>>>(out, partials, count, ks);
}

template <int KS>
static void launch_cluster(float *out, int M, int nvar, cudaStream_t stream) {
    cudaLaunchAttribute attr{};
    cudaLaunchConfig_t config = cluster_config<KS>(M, nvar, &attr);
    config.stream = stream;
    CUDA_OK(cudaLaunchKernelEx(&config, cluster_reduce<KS>, out, M, nvar));
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
    const float ms = elapsed_ms(events.begin, events.end, iters);
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    return ms;
}

template <int KS>
static void run_case(int M, int nvar, int warmup, int iters, cudaStream_t stream) {
    const size_t count = size_t(M) * nvar;
    float *control_out = nullptr, *cluster_out = nullptr, *partials = nullptr;
    CUDA_OK(cudaMalloc(&control_out, count * sizeof(float)));
    CUDA_OK(cudaMalloc(&cluster_out, count * sizeof(float)));
    CUDA_OK(cudaMalloc(&partials, count * KS * sizeof(float)));

    auto control = [&] { launch_control(control_out, partials, M, nvar, KS, stream); };
    auto probe = [&] { launch_cluster<KS>(cluster_out, M, nvar, stream); };

    control();
    probe();
    CUDA_OK(cudaStreamSynchronize(stream));
    std::vector<float> host_control(count), host_probe(count);
    CUDA_OK(cudaMemcpy(host_control.data(), control_out, count * sizeof(float),
                       cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(host_probe.data(), cluster_out, count * sizeof(float),
                       cudaMemcpyDeviceToHost));
    float max_abs = 0.0f;
    for (size_t i = 0; i < count; ++i)
        max_abs = std::max(max_abs, std::fabs(host_control[i] - host_probe[i]));

    const float control_eager = measure_eager(control, warmup, iters, stream);
    const float cluster_eager = measure_eager(probe, warmup, iters, stream);
    const float control_graph = measure_graph(control, warmup, iters, stream);
    const float cluster_graph = measure_graph(probe, warmup, iters, stream);

    cudaFuncAttributes partial_attr{}, reduce_attr{}, cluster_attr{};
    CUDA_OK(cudaFuncGetAttributes(&partial_attr, global_partial));
    CUDA_OK(cudaFuncGetAttributes(&reduce_attr, global_reduce));
    CUDA_OK(cudaFuncGetAttributes(&cluster_attr, cluster_reduce<KS>));
    int partial_blocks = 0, reduce_blocks = 0, active_clusters = 0;
    CUDA_OK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &partial_blocks, global_partial, 256, 0));
    CUDA_OK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &reduce_blocks, global_reduce, 256, 0));
    cudaLaunchAttribute attr{};
    cudaLaunchConfig_t config = cluster_config<KS>(M, nvar, &attr);
    const cudaError_t occ_status = cudaOccupancyMaxActiveClusters(
        &active_clusters, (const void *)cluster_reduce<KS>, &config);

    const unsigned long long payload = count * sizeof(float);
    const unsigned long long control_dram = (2ULL * KS + 1ULL) * payload;
    std::printf(
        "{\"M\":%d,\"n\":%d,\"ks\":%d,\"max_abs\":%.9g,"
        "\"control_eager_us\":%.3f,\"cluster_eager_us\":%.3f,"
        "\"control_graph_us\":%.3f,\"cluster_graph_us\":%.3f,"
        "\"eager_speedup\":%.4f,\"graph_speedup\":%.4f,"
        "\"control_dram_bytes\":%llu,\"cluster_dram_bytes\":%llu,"
        "\"dram_bytes_removed\":%llu,\"control_launches\":2,\"cluster_launches\":1,"
        "\"partial_regs\":%d,\"reduce_regs\":%d,\"cluster_regs\":%d,"
        "\"cluster_static_smem\":%zu,\"cluster_dynamic_smem\":%zu,"
        "\"partial_blocks_per_sm\":%d,\"reduce_blocks_per_sm\":%d,"
        "\"active_clusters\":%d,\"occupancy_status\":%d}\n",
        M, nvar, KS, max_abs,
        control_eager * 1000.0f, cluster_eager * 1000.0f,
        control_graph * 1000.0f, cluster_graph * 1000.0f,
        control_eager / cluster_eager, control_graph / cluster_graph,
        control_dram, payload, control_dram - payload,
        partial_attr.numRegs, reduce_attr.numRegs, cluster_attr.numRegs,
        size_t(cluster_attr.sharedSizeBytes),
        size_t(KS) * nvar * 128 * sizeof(float), partial_blocks, reduce_blocks,
        occ_status == cudaSuccess ? active_clusters : -1, int(occ_status));
    std::fflush(stdout);

    cudaFree(partials);
    cudaFree(cluster_out);
    cudaFree(control_out);
}

template <int KS>
static void run_shapes(const std::vector<int> &shapes, int warmup, int iters,
                       cudaStream_t stream) {
    for (int M : shapes)
        for (int nvar : {1, 2, 4})
            run_case<KS>(M, nvar, warmup, iters, stream);
}

int main(int argc, char **argv) {
    int warmup = 100;
    int iters = 2000;
    if (argc > 1) iters = std::max(10, std::atoi(argv[1]));
    if (argc > 2) warmup = std::max(1, std::atoi(argv[2]));

    cudaDeviceProp prop{};
    CUDA_OK(cudaGetDeviceProperties(&prop, 0));
    int cluster_supported = 0;
    CUDA_OK(cudaDeviceGetAttribute(&cluster_supported, cudaDevAttrClusterLaunch, 0));
    std::printf("{\"device\":\"%s\",\"cc\":\"%d.%d\",\"sm_count\":%d,"
                "\"cluster_launch\":%d,\"iters\":%d,\"warmup\":%d}\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount,
                cluster_supported, iters, warmup);

    cudaStream_t stream{};
    CUDA_OK(cudaStreamCreate(&stream));
    if (argc == 6) {
        const int M = std::atoi(argv[3]);
        const int nvar = std::atoi(argv[4]);
        const int ks = std::atoi(argv[5]);
        if (M < 1 || (nvar != 1 && nvar != 2 && nvar != 4) ||
            (ks != 2 && ks != 4 && ks != 8)) {
            std::fprintf(stderr, "single case requires M>0, n={1,2,4}, ks={2,4,8}\n");
            return 2;
        }
        if (ks == 2) run_case<2>(M, nvar, warmup, iters, stream);
        if (ks == 4) run_case<4>(M, nvar, warmup, iters, stream);
        if (ks == 8) run_case<8>(M, nvar, warmup, iters, stream);
        cudaStreamDestroy(stream);
        return 0;
    }
    // Exact output widths appearing in the Qwen3.8 projection families.
    const std::vector<int> shapes{256, 1024, 2560, 5120, 6144, 10240, 12288, 17408};
    run_shapes<2>(shapes, warmup, iters, stream);
    run_shapes<4>(shapes, warmup, iters, stream);
    run_shapes<8>(shapes, warmup, iters, stream);
    cudaStreamDestroy(stream);
    return 0;
}
