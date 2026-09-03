// Standalone sm120 proof for docs/level-up.md attention Q staging.
//
// This does not link or modify the production engine. It copies the exact
// norm -> RoPE -> Hadamard prologue from k_tq_paged_attn_q4_split_gqa_v2.
// The control repeats that work in every split CTA. The probe prepares each
// query head once, then each split CTA loads the staged values it would use.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CUDA_OK(expr) do { \
    cudaError_t status_ = (expr); \
    if (status_ != cudaSuccess) { \
        std::fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__, \
                     #expr, cudaGetErrorString(status_)); \
        std::exit(1); \
    } \
} while (0)

constexpr int HD = 256;
constexpr int NH = 24;
constexpr int NKV = 4;
constexpr int GQA = NH / NKV;
constexpr int Q_M = NH * HD * 2;
constexpr float EPS = 1.0e-6f;
constexpr float ROPE_THETA = 1000000.0f;

static __device__ __forceinline__ void fwht256(float *buf, int tid) {
    #pragma unroll
    for (int h = 1; h < 256; h <<= 1) {
        float a = buf[tid], b = buf[tid ^ h];
        __syncthreads();
        buf[tid] = (tid & h) ? (b - a) : (a + b);
        __syncthreads();
    }
    buf[tid] *= 0.0625f;
    __syncthreads();
}

static __device__ __forceinline__ void prepare_q(
        float *qn, float *partial, float *q_sum_shared,
        const float *q_proj, const __nv_bfloat16 *q_norm_w,
        int kv_head, int tid, int pos) {
    #pragma unroll
    for (int g = 0; g < GQA; ++g) {
        int head = kv_head * GQA + g;
        float qv = q_proj[head * (2 * HD) + tid];
        float qsum = qv * qv;
        for (int offset = 16; offset > 0; offset >>= 1)
            qsum += __shfl_down_sync(0xffffffffu, qsum, offset);
        if ((tid & 31) == 0) partial[tid >> 5] = qsum;
        __syncthreads();
        if (tid < 8) {
            float vtmp = partial[tid];
            for (int offset = 4; offset > 0; offset >>= 1)
                vtmp += __shfl_down_sync(0xffu, vtmp, offset);
            if (tid == 0) *q_sum_shared = vtmp;
        }
        __syncthreads();
        float rms = rsqrtf(*q_sum_shared / float(HD) + EPS);
        qv = qv * rms * (1.0f + __bfloat162float(q_norm_w[tid]));
        if (tid < 64) {
            int idx = tid & 31;
            float freq = powf(ROPE_THETA, -float(2 * idx) / 64.0f);
            float angle = float(pos) * freq;
            float c = cosf(angle), s = sinf(angle);
            int pi = tid < 32 ? tid + 32 : tid - 32;
            float q_pair = q_proj[head * (2 * HD) + pi];
            q_pair = q_pair * rms * (1.0f + __bfloat162float(q_norm_w[pi]));
            float q_rot = tid < 32 ? -q_pair : q_pair;
            qv = qv * c + q_rot * s;
        }
        qn[g * HD + tid] = qv;
        __syncthreads();
        fwht256(qn + g * HD, tid);
    }
}

__global__ __launch_bounds__(256, 2)
void qprep_control(float *out, const float *q_proj,
                   const __nv_bfloat16 *q_norm_w, int pos, int splits) {
    __shared__ float qn[GQA * HD];
    __shared__ float partial[8];
    __shared__ float q_sum_shared;
    int kv_head = blockIdx.x;
    int split = blockIdx.y;
    int tid = threadIdx.x;
    prepare_q(qn, partial, &q_sum_shared, q_proj, q_norm_w,
              kv_head, tid, pos);
    #pragma unroll
    for (int g = 0; g < GQA; ++g) {
        size_t dst = ((size_t(split) * NKV + kv_head) * GQA + g) * HD + tid;
        out[dst] = qn[g * HD + tid];
    }
}

__global__ __launch_bounds__(256, 2)
void qprep_stage(float *staged, const float *q_proj,
                 const __nv_bfloat16 *q_norm_w, int pos) {
    __shared__ float qn[GQA * HD];
    __shared__ float partial[8];
    __shared__ float q_sum_shared;
    int kv_head = blockIdx.x;
    int tid = threadIdx.x;
    prepare_q(qn, partial, &q_sum_shared, q_proj, q_norm_w,
              kv_head, tid, pos);
    #pragma unroll
    for (int g = 0; g < GQA; ++g)
        staged[(size_t(kv_head) * GQA + g) * HD + tid] = qn[g * HD + tid];
}

__global__ __launch_bounds__(256, 2)
void qprep_consume(float *out, const float *staged) {
    __shared__ float qn[GQA * HD];
    int kv_head = blockIdx.x;
    int split = blockIdx.y;
    int tid = threadIdx.x;
    #pragma unroll
    for (int g = 0; g < GQA; ++g)
        qn[g * HD + tid] = staged[(size_t(kv_head) * GQA + g) * HD + tid];
    __syncthreads();
    #pragma unroll
    for (int g = 0; g < GQA; ++g) {
        size_t dst = ((size_t(split) * NKV + kv_head) * GQA + g) * HD + tid;
        out[dst] = qn[g * HD + tid];
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
static float measure_eager(Launch launch, int warmup, int iters,
                           cudaStream_t stream) {
    for (int i = 0; i < warmup; ++i) launch();
    CUDA_OK(cudaStreamSynchronize(stream));
    Events events;
    CUDA_OK(cudaEventRecord(events.begin, stream));
    for (int i = 0; i < iters; ++i) launch();
    CUDA_OK(cudaEventRecord(events.end, stream));
    return elapsed_ms(events.begin, events.end, iters);
}

template <typename Launch>
static float measure_graph(Launch launch, int warmup, int iters,
                           cudaStream_t stream) {
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

static void run_case(int splits, int warmup, int iters, cudaStream_t stream,
                     const float *q_proj, const __nv_bfloat16 *q_norm_w) {
    size_t count = size_t(splits) * NKV * GQA * HD;
    float *control_out{}, *probe_out{}, *staged{};
    CUDA_OK(cudaMalloc(&control_out, count * sizeof(float)));
    CUDA_OK(cudaMalloc(&probe_out, count * sizeof(float)));
    CUDA_OK(cudaMalloc(&staged, size_t(NKV) * GQA * HD * sizeof(float)));
    int pos = splits == 4 ? 2047 : (splits == 16 ? 8191 :
              (splits == 64 ? 32767 : 131071));
    auto control = [&] {
        qprep_control<<<dim3(NKV, splits), 256, 0, stream>>>(
            control_out, q_proj, q_norm_w, pos, splits);
    };
    auto probe = [&] {
        qprep_stage<<<NKV, 256, 0, stream>>>(staged, q_proj, q_norm_w, pos);
        qprep_consume<<<dim3(NKV, splits), 256, 0, stream>>>(probe_out, staged);
    };
    control();
    probe();
    CUDA_OK(cudaStreamSynchronize(stream));
    std::vector<float> control_host(count), probe_host(count);
    CUDA_OK(cudaMemcpy(control_host.data(), control_out, count * sizeof(float),
                       cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(probe_host.data(), probe_out, count * sizeof(float),
                       cudaMemcpyDeviceToHost));
    size_t bit_mismatches = 0;
    float max_abs = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        bit_mismatches += std::memcmp(&control_host[i], &probe_host[i], 4) != 0;
        max_abs = std::max(max_abs, std::fabs(control_host[i] - probe_host[i]));
    }
    float control_eager = measure_eager(control, warmup, iters, stream);
    float probe_eager = measure_eager(probe, warmup, iters, stream);
    float control_graph = measure_graph(control, warmup, iters, stream);
    float probe_graph = measure_graph(probe, warmup, iters, stream);
    std::printf(
        "{\"splits\":%d,\"position\":%d,\"bit_mismatches\":%zu,"
        "\"max_abs\":%.9g,\"control_eager_us\":%.3f,"
        "\"probe_eager_us\":%.3f,\"control_graph_us\":%.3f,"
        "\"probe_graph_us\":%.3f,\"eager_speedup\":%.4f,"
        "\"graph_speedup\":%.4f,\"control_qprep_ctas\":%d,"
        "\"probe_qprep_ctas\":%d,\"probe_consumer_ctas\":%d,"
        "\"staged_bytes\":%zu}\n",
        splits, pos, bit_mismatches, max_abs,
        control_eager * 1000.0f, probe_eager * 1000.0f,
        control_graph * 1000.0f, probe_graph * 1000.0f,
        control_eager / probe_eager, control_graph / probe_graph,
        NKV * splits, NKV, NKV * splits,
        size_t(NKV) * GQA * HD * sizeof(float));
    cudaFree(staged);
    cudaFree(probe_out);
    cudaFree(control_out);
}

int main(int argc, char **argv) {
    int iters = argc > 1 ? std::max(10, std::atoi(argv[1])) : 10000;
    int warmup = argc > 2 ? std::max(1, std::atoi(argv[2])) : 500;
    cudaDeviceProp prop{};
    CUDA_OK(cudaGetDeviceProperties(&prop, 0));
    std::printf("{\"device\":\"%s\",\"cc\":\"%d.%d\","
                "\"iters\":%d,\"warmup\":%d}\n",
                prop.name, prop.major, prop.minor, iters, warmup);
    std::vector<float> q_host(Q_M);
    std::vector<__nv_bfloat16> nw_host(HD);
    for (int i = 0; i < Q_M; ++i)
        q_host[i] = std::sin(float(i) * 0.0137f) * 0.7f
                  + std::cos(float(i) * 0.0031f) * 0.2f;
    for (int i = 0; i < HD; ++i)
        nw_host[i] = __float2bfloat16_rn(std::sin(float(i) * 0.021f) * 0.125f);
    float *q_proj{};
    __nv_bfloat16 *q_norm_w{};
    CUDA_OK(cudaMalloc(&q_proj, Q_M * sizeof(float)));
    CUDA_OK(cudaMalloc(&q_norm_w, HD * sizeof(__nv_bfloat16)));
    CUDA_OK(cudaMemcpy(q_proj, q_host.data(), Q_M * sizeof(float),
                       cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(q_norm_w, nw_host.data(), HD * sizeof(__nv_bfloat16),
                       cudaMemcpyHostToDevice));
    cudaStream_t stream{};
    CUDA_OK(cudaStreamCreate(&stream));
    for (int splits : {4, 16, 64, 85})
        run_case(splits, warmup, iters, stream, q_proj, q_norm_w);
    cudaStreamDestroy(stream);
    cudaFree(q_norm_w);
    cudaFree(q_proj);
    return 0;
}
