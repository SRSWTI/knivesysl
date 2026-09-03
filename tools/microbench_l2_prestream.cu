// Standalone sm120 probe for docs/level-up.md stage 7 rung e3: l2 weight prestream.
// Does NOT modify or link the production engine.
//
// Question: while a compute-bound phase runs (attention / deltanet windows leave
// DRAM ~60% idle for up to ~340 us per layer at deep context), can a low-priority
// side-stream prefetcher pull the NEXT projection's weight bytes into L2 so the
// following GEMV reads them at L2 bandwidth instead of DRAM bandwidth - and does
// the warmth survive until consumption?
//
// Sequence per simulated layer, over R round-robin cold weight buffers:
//   control:  compute(phase_us)                        ; gemv(W[i])
//   probe:    compute(phase_us) || prefetch(W[i], cap) ; gemv(W[i])
//
// Build: nvcc -std=c++17 -O3 -lineinfo --generate-code=arch=compute_120,code=sm_120 \
//        -o results/microbench/l2_prestream tools/microbench_l2_prestream.cu
// Run:   results/microbench/l2_prestream <iters> [wmb] [capmb]

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

#define CUDA_OK(expr) do { \
    cudaError_t status_ = (expr); \
    if (status_ != cudaSuccess) { \
        std::fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__, #expr, \
                     cudaGetErrorString(status_)); \
        std::exit(1); \
    } } while (0)

// bandwidth-dominated GEMV-like consumer: grid-stride uint4 stream + fma sink.
__global__ void k_gemv_stream(const uint4 *__restrict__ w, size_t n4, float x,
                              float *__restrict__ y) {
    float acc = 0.f;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4;
         i += (size_t)gridDim.x * blockDim.x) {
        uint4 v = __ldg(&w[i]);
        acc = fmaf((float)(v.x & 0xFF), x, acc);
        acc = fmaf((float)(v.y & 0xFF), x, acc);
        acc = fmaf((float)(v.z & 0xFF), x, acc);
        acc = fmaf((float)(v.w & 0xFF), x, acc);
    }
    if (acc == 3.402823466e38f) y[threadIdx.x] = acc;   // never true; defeats dce
}

// compute-bound phase: pure fma chains, negligible memory traffic, fixed cycles.
__global__ void __launch_bounds__(256, 2)
k_compute_phase(long long cycles, float *__restrict__ sink) {
    long long t0 = clock64();
    float a = 1.0001f + threadIdx.x * 1e-6f, b = 0.9999f;
    while (clock64() - t0 < cycles) {
        #pragma unroll
        for (int i = 0; i < 64; i++) { a = fmaf(a, b, 1e-9f); b = fmaf(b, a, -1e-9f); }
    }
    if (a == 3.402823466e38f) sink[threadIdx.x] = a + b;
}

// low-priority prefetcher: touch up to cap4 uint4 of the target buffer.
__global__ void k_prefetch(const uint4 *__restrict__ w, size_t cap4,
                           unsigned int *__restrict__ progress) {
    unsigned int acc = 0;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < cap4;
         i += (size_t)gridDim.x * blockDim.x) {
        uint4 v = __ldg(&w[i]);
        acc += v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0xFFFFFFFFu) atomicAdd(progress, 1u);    // defeats dce
}

int main(int argc, char **argv) {
    int iters = argc > 1 ? std::atoi(argv[1]) : 42;
    int wmb = argc > 2 ? std::atoi(argv[2]) : 160;      // one projection group
    int capmb = argc > 3 ? std::atoi(argv[3]) : 112;    // prefetch cap (< L2)
    const int R = argc > 4 ? std::atoi(argv[4]) : 6;    // round-robin cold buffers
    const size_t wbytes = (size_t)wmb << 20;
    const size_t n4 = wbytes / 16;
    const size_t cap4 = ((size_t)capmb << 20) / 16;

    cudaDeviceProp prop{};
    CUDA_OK(cudaGetDeviceProperties(&prop, 0));
    int lo = 0, hi = 0;
    CUDA_OK(cudaDeviceGetStreamPriorityRange(&lo, &hi));   // lo = least urgent
    cudaStream_t main_st, pf_st;
    CUDA_OK(cudaStreamCreateWithPriority(&main_st, cudaStreamNonBlocking, hi));
    CUDA_OK(cudaStreamCreateWithPriority(&pf_st, cudaStreamNonBlocking, lo));

    std::vector<uint4 *> W(R);
    for (int r = 0; r < R; r++) {
        CUDA_OK(cudaMalloc(&W[r], wbytes));
        CUDA_OK(cudaMemset(W[r], 0x5A + r, wbytes));
    }
    float *sink; unsigned int *progress;
    CUDA_OK(cudaMalloc(&sink, 1024 * 4));
    CUDA_OK(cudaMalloc(&progress, 4));
    CUDA_OK(cudaMemset(progress, 0, 4));

    const int gemv_grid = prop.multiProcessorCount * 6, gemv_blk = 256;
    const int pf_grid = prop.multiProcessorCount, pf_blk = 256;
    int khz = 0;
    CUDA_OK(cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, 0));
    const double ghz = khz / 1e6;                       // kHz -> GHz

    cudaEvent_t a, b;
    CUDA_OK(cudaEventCreate(&a));
    CUDA_OK(cudaEventCreate(&b));

    auto run_arm = [&](double phase_us, bool prefetch, float *pf_wait_ms_out) {
        k_compute_phase<<<prop.multiProcessorCount * 2, 256, 0, main_st>>>(1000, sink);
        CUDA_OK(cudaStreamSynchronize(main_st));
        long long cyc = (long long)(phase_us * 1000.0 * ghz);
        float total_ms = 0.f, pf_wait_total = 0.f;
        for (int it = 0; it < iters; it++) {
            int r = it % R;
            k_compute_phase<<<prop.multiProcessorCount * 2, 256, 0, main_st>>>(cyc, sink);
            if (prefetch)
                k_prefetch<<<pf_grid, pf_blk, 0, pf_st>>>(W[r], cap4, progress);
            CUDA_OK(cudaStreamSynchronize(main_st));
            cudaEvent_t pa, pb;
            if (prefetch) {                              // residual prefetch tail cost
                CUDA_OK(cudaEventCreate(&pa)); CUDA_OK(cudaEventCreate(&pb));
                CUDA_OK(cudaEventRecord(pa, pf_st));
                CUDA_OK(cudaStreamSynchronize(pf_st));
                CUDA_OK(cudaEventRecord(pb, pf_st));
                CUDA_OK(cudaEventSynchronize(pb));
                float pw = 0.f; CUDA_OK(cudaEventElapsedTime(&pw, pa, pb));
                pf_wait_total += pw;
                CUDA_OK(cudaEventDestroy(pa)); CUDA_OK(cudaEventDestroy(pb));
            }
            CUDA_OK(cudaEventRecord(a, main_st));
            k_gemv_stream<<<gemv_grid, gemv_blk, 0, main_st>>>(W[r], n4, 1.0f / (it + 1), sink);
            CUDA_OK(cudaEventRecord(b, main_st));
            CUDA_OK(cudaStreamSynchronize(main_st));
            float ms = 0.f;
            CUDA_OK(cudaEventElapsedTime(&ms, a, b));
            total_ms += ms;
        }
        if (pf_wait_ms_out) *pf_wait_ms_out = pf_wait_total / iters;
        return total_ms / iters;
    };

    auto pf_solo_us = [&]() {
        CUDA_OK(cudaEventRecord(a, pf_st));
        k_prefetch<<<pf_grid, pf_blk, 0, pf_st>>>(W[0], cap4, progress);
        CUDA_OK(cudaEventRecord(b, pf_st));
        CUDA_OK(cudaStreamSynchronize(pf_st));
        float ms = 0.f;
        CUDA_OK(cudaEventElapsedTime(&ms, a, b));
        return ms * 1000.f;
    };

    std::printf("{\"device\":\"%s\",\"sms\":%d,\"l2_mb\":%d,\"w_mb\":%d,\"cap_mb\":%d,"
                "\"pf_solo_us\":%.1f}\n",
                prop.name, prop.multiProcessorCount,
                (int)(prop.l2CacheSize >> 20), wmb, capmb, pf_solo_us());

    const double phases[] = {20.0, 90.0, 340.0};        // ~2k / 32k / 131k per-layer windows
    for (double p : phases) {
        float pfw = 0.f;
        float cold = run_arm(p, false, nullptr);
        float warm = run_arm(p, true, &pfw);
        double gb = (double)wbytes / 1e9;
        std::printf("{\"phase_us\":%.0f,\"cold_gemv_ms\":%.3f,\"warm_gemv_ms\":%.3f,"
                    "\"cold_gb_s\":%.0f,\"warm_gb_s\":%.0f,\"gemv_speedup\":%.3f,"
                    "\"pf_tail_wait_ms\":%.3f}\n",
                    p, cold, warm, gb / (cold / 1e3), gb / (warm / 1e3),
                    cold / warm, pfw);
    }
    return 0;
}
