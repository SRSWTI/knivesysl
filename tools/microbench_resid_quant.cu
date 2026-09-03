// Standalone sm120 proof for docs/level-up.md: collapse one decode-row
// residual/RMS/NVFP4 publication boundary. This does not link the engine.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CUDA_OK(expr) do { cudaError_t s_ = (expr); if (s_ != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__, #expr, \
                 cudaGetErrorString(s_)); std::exit(1); } } while (0)

constexpr int H = 5120;
constexpr int KT64 = H / 64;
constexpr int BW = 72;
constexpr float EPS = 1.0e-6f;

__device__ __forceinline__ float bf16_to_float(uint16_t x) {
    return __uint_as_float(uint32_t(x) << 16);
}

__host__ __device__ __forceinline__ int b_k(int lane, int reg, int nib) {
    return 32 * reg + 16 * ((lane >> 1) & 1) + 8 * (lane & 1) + nib;
}

__host__ __device__ __forceinline__ uint8_t quant_e2m1(float v) {
    float a = fminf(fabsf(v), 6.0f);
    uint32_t code;
    if (a < 0.75f) code = (a < 0.25f) ? 0u : 1u;
    else if (a < 1.5f) code = 2u;
    else if (a < 2.5f) code = 3u;
    else if (a < 3.5f) code = 4u;
    else if (a < 5.0f) code = 5u;
    else code = 6u;
    return uint8_t(code | (v < 0.0f ? 8u : 0u));
}

__host__ __device__ __forceinline__ float ue4m3_to_f(uint8_t b) {
    int e = (b >> 3) & 0xF, m = b & 7;
    return ldexpf(1.0f + float(m) * 0.125f, e - 6);
}

__host__ __device__ __forceinline__ uint8_t f2ue4m3(float v) {
    if (!(v > 0.0f)) return 0;
    int e;
    float m = frexpf(v, &e);
    m = v / ldexpf(1.0f, e - 1);
    int E = e + 6;
    int f = int(lrintf((2.0f * m - 1.0f) * 8.0f));
    if (f == 8) { f = 0; ++E; }
    if (E < 0) return 0;
    if (E > 15) return uint8_t((15 << 3) | 7);
    return uint8_t((E << 3) | (f & 7));
}

__global__ void fill_inputs(float *a, float *b, uint16_t *nw) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < H) {
        a[i] = sinf(float(i) * 0.00317f) * 0.8f
             + cosf(float(i) * 0.00091f) * 0.3f;
        b[i] = sinf(float(i) * 0.00173f) * 0.41f;
        float w = sinf(float(i) * 0.017f) * 0.125f;
        nw[i] = uint16_t(__float_as_uint(w + copysignf(0x1p-17f, w)) >> 16);
    }
}

// Production-order factor kernel for a row that already exists.
__global__ __launch_bounds__(1024, 1)
void rms_factor(float *fac, const float *x) {
    __shared__ float smem[32];
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    float sum_sq = 0.0f;
    for (int i = tid; i < H; i += blockDim.x) {
        float v = x[i];
        sum_sq += v * v;
    }
    for (int off = 16; off > 0; off >>= 1)
        sum_sq += __shfl_down_sync(0xffffffffu, sum_sq, off);
    if (lane == 0) smem[warp] = sum_sq;
    __syncthreads();
    if (warp == 0) {
        sum_sq = smem[lane];
        for (int off = 16; off > 0; off >>= 1)
            sum_sq += __shfl_down_sync(0xffffffffu, sum_sq, off);
        if (lane == 0) fac[0] = rsqrtf(sum_sq / float(H) + EPS);
    }
}

// Production-order residual update plus factor kernel.
__global__ __launch_bounds__(1024, 1)
void add_rms_factor(float *fac, float *resid, const float *a, const float *b) {
    __shared__ float smem[32];
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    float sum_sq = 0.0f;
    for (int i = tid; i < H; i += blockDim.x) {
        float v = a[i] + b[i];
        resid[i] = v;
        sum_sq += v * v;
    }
    for (int off = 16; off > 0; off >>= 1)
        sum_sq += __shfl_down_sync(0xffffffffu, sum_sq, off);
    if (lane == 0) smem[warp] = sum_sq;
    __syncthreads();
    if (warp == 0) {
        sum_sq = smem[lane];
        for (int off = 16; off > 0; off >>= 1)
            sum_sq += __shfl_down_sync(0xffffffffu, sum_sq, off);
        if (lane == 0) fac[0] = rsqrtf(sum_sq / float(H) + EPS);
    }
}

// Exact one-row specialization of k_tq_nvf4_quant_x_nw. Seven padded rows are
// emitted too, preserving the complete [8 rows x k64] native tile contract.
__global__ void quant_control(uint32_t *packed, const float *x,
                              const float *fac, const uint16_t *nw) {
    int kt = blockIdx.x, lane = threadIdx.x;
    uint32_t *dst = packed + size_t(kt) * BW;
    __shared__ float sc[8][4];
    if (lane < 8) {
        float f = lane == 0 ? fac[0] : 0.0f;
        uint32_t sw = 0;
        for (int g = 0; g < 4; ++g) {
            float mx = 0.0f;
            for (int t = 0; t < 16; ++t) {
                int k = kt * 64 + g * 16 + t;
                float v = lane == 0
                    ? x[k] * f * (1.0f + bf16_to_float(nw[k])) : 0.0f;
                mx = fmaxf(mx, fabsf(v));
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
    int col = lane >> 2;
    for (int reg = 0; reg < 2; ++reg) {
        for (int j = 0; j < 8; ++j) {
            int k64 = b_k(lane, reg, j), k = kt * 64 + k64;
            float v = col == 0
                ? x[k] * fac[0] * (1.0f + bf16_to_float(nw[k])) : 0.0f;
            float scale = sc[col][k64 >> 4];
            words[reg] |= quant_e2m1(scale > 0.0f ? v / scale : 0.0f) << (4 * j);
        }
    }
    dst[lane * 2] = words[0];
    dst[lane * 2 + 1] = words[1];
}

template <bool ADD>
__global__ __launch_bounds__(1024, 1)
void fused_row(uint32_t *packed, float *resid, const float *a, const float *b,
               const uint16_t *nw) {
    __shared__ float row[H];
    __shared__ float smem[32];
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    float sum_sq = 0.0f;
    for (int i = tid; i < H; i += blockDim.x) {
        float v = ADD ? a[i] + b[i] : a[i];
        if (ADD) resid[i] = v;
        row[i] = v;
        sum_sq += v * v;
    }
    for (int off = 16; off > 0; off >>= 1)
        sum_sq += __shfl_down_sync(0xffffffffu, sum_sq, off);
    if (lane == 0) smem[warp] = sum_sq;
    __syncthreads();
    if (warp == 0) {
        sum_sq = smem[lane];
        for (int off = 16; off > 0; off >>= 1)
            sum_sq += __shfl_down_sync(0xffffffffu, sum_sq, off);
        if (lane == 0) smem[0] = rsqrtf(sum_sq / float(H) + EPS);
    }
    __syncthreads();
    float f = smem[0];
    for (int i = tid; i < H; i += blockDim.x)
        row[i] = row[i] * f * (1.0f + bf16_to_float(nw[i]));
    __syncthreads();

    // Thirty-two warps cooperatively cover all 80 k64 tiles. Scale bytes move
    // through warp shuffles; no per-tile block or global factor is required.
    for (int kt = warp; kt < KT64; kt += 32) {
        uint32_t sw = 0;
        if (lane < 8) {
            for (int g = 0; g < 4; ++g) {
                float mx = 0.0f;
                if (lane == 0) {
                    for (int t = 0; t < 16; ++t)
                        mx = fmaxf(mx, fabsf(row[kt * 64 + g * 16 + t]));
                }
                uint8_t sb = f2ue4m3(mx / 6.0f);
                if (sb == 0) sb = 1;
                sw |= uint32_t(sb) << (8 * g);
            }
            packed[size_t(kt) * BW + 64 + lane] = sw;
        }
        uint32_t owner_sw = __shfl_sync(0xffffffffu, sw, lane >> 2);
        uint32_t words[2] = {0, 0};
        int col = lane >> 2;
        for (int reg = 0; reg < 2; ++reg) {
            for (int j = 0; j < 8; ++j) {
                int k64 = b_k(lane, reg, j);
                float v = col == 0 ? row[kt * 64 + k64] : 0.0f;
                float scale = ue4m3_to_f(uint8_t(owner_sw >> (8 * (k64 >> 4))));
                words[reg] |= quant_e2m1(scale > 0.0f ? v / scale : 0.0f) << (4 * j);
            }
        }
        packed[size_t(kt) * BW + lane * 2] = words[0];
        packed[size_t(kt) * BW + lane * 2 + 1] = words[1];
    }
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
    int iters = argc > 1 ? std::max(10, std::atoi(argv[1])) : 20000;
    int warmup = argc > 2 ? std::max(1, std::atoi(argv[2])) : 1000;
    float *a{}, *b{}, *resid_control{}, *resid_probe{}, *fac{};
    uint16_t *nw{};
    uint32_t *pack_control{}, *pack_probe{};
    CUDA_OK(cudaMalloc(&a, H * sizeof(float)));
    CUDA_OK(cudaMalloc(&b, H * sizeof(float)));
    CUDA_OK(cudaMalloc(&resid_control, H * sizeof(float)));
    CUDA_OK(cudaMalloc(&resid_probe, H * sizeof(float)));
    CUDA_OK(cudaMalloc(&fac, sizeof(float)));
    CUDA_OK(cudaMalloc(&nw, H * sizeof(uint16_t)));
    CUDA_OK(cudaMalloc(&pack_control, KT64 * BW * sizeof(uint32_t)));
    CUDA_OK(cudaMalloc(&pack_probe, KT64 * BW * sizeof(uint32_t)));
    cudaStream_t stream{};
    CUDA_OK(cudaStreamCreate(&stream));
    fill_inputs<<<(H + 255) / 256, 256, 0, stream>>>(a, b, nw);
    CUDA_OK(cudaStreamSynchronize(stream));

    auto control_plain = [&] {
        rms_factor<<<1, 1024, 0, stream>>>(fac, a);
        quant_control<<<KT64, 32, 0, stream>>>(pack_control, a, fac, nw);
    };
    auto probe_plain = [&] {
        fused_row<false><<<1, 1024, 0, stream>>>(pack_probe, resid_probe, a, b, nw);
    };
    auto control_add = [&] {
        add_rms_factor<<<1, 1024, 0, stream>>>(fac, resid_control, a, b);
        quant_control<<<KT64, 32, 0, stream>>>(pack_control, resid_control, fac, nw);
    };
    auto probe_add = [&] {
        fused_row<true><<<1, 1024, 0, stream>>>(pack_probe, resid_probe, a, b, nw);
    };

    control_plain();
    probe_plain();
    CUDA_OK(cudaStreamSynchronize(stream));
    std::vector<uint32_t> cpack(KT64 * BW), ppack(KT64 * BW);
    CUDA_OK(cudaMemcpy(cpack.data(), pack_control, cpack.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(ppack.data(), pack_probe, ppack.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    size_t plain_mismatches = 0;
    for (size_t i = 0; i < cpack.size(); ++i) plain_mismatches += cpack[i] != ppack[i];

    control_add();
    probe_add();
    CUDA_OK(cudaStreamSynchronize(stream));
    CUDA_OK(cudaMemcpy(cpack.data(), pack_control, cpack.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(ppack.data(), pack_probe, ppack.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    std::vector<float> cres(H), pres(H);
    CUDA_OK(cudaMemcpy(cres.data(), resid_control, H * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(pres.data(), resid_probe, H * sizeof(float), cudaMemcpyDeviceToHost));
    size_t add_pack_mismatches = 0, resid_mismatches = 0;
    for (size_t i = 0; i < cpack.size(); ++i) add_pack_mismatches += cpack[i] != ppack[i];
    for (int i = 0; i < H; ++i) resid_mismatches += std::memcmp(&cres[i], &pres[i], 4) != 0;

    float cp_e = eager(control_plain, warmup, iters, stream);
    float pp_e = eager(probe_plain, warmup, iters, stream);
    float ca_e = eager(control_add, warmup, iters, stream);
    float pa_e = eager(probe_add, warmup, iters, stream);
    float cp_g = graph(control_plain, warmup, iters, stream);
    float pp_g = graph(probe_plain, warmup, iters, stream);
    float ca_g = graph(control_add, warmup, iters, stream);
    float pa_g = graph(probe_add, warmup, iters, stream);

    cudaFuncAttributes fused_attr{};
    CUDA_OK(cudaFuncGetAttributes(&fused_attr, fused_row<true>));
    int active_blocks = 0;
    CUDA_OK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks, fused_row<true>, 1024, 0));
    cudaDeviceProp prop{};
    CUDA_OK(cudaGetDeviceProperties(&prop, 0));
    std::printf(
        "{\"device\":\"%s\",\"cc\":\"%d.%d\",\"h\":%d,"
        "\"plain_pack_mismatches\":%zu,\"add_pack_mismatches\":%zu,"
        "\"resid_mismatches\":%zu,\"plain_control_eager_us\":%.3f,"
        "\"plain_probe_eager_us\":%.3f,\"plain_control_graph_us\":%.3f,"
        "\"plain_probe_graph_us\":%.3f,\"add_control_eager_us\":%.3f,"
        "\"add_probe_eager_us\":%.3f,\"add_control_graph_us\":%.3f,"
        "\"add_probe_graph_us\":%.3f,\"plain_eager_speedup\":%.4f,"
        "\"plain_graph_speedup\":%.4f,\"add_eager_speedup\":%.4f,"
        "\"add_graph_speedup\":%.4f,\"control_launches\":2,"
        "\"probe_launches\":1,\"modeled_read_bytes_removed\":%zu,"
        "\"probe_regs\":%d,\"probe_static_smem\":%zu,"
        "\"active_blocks_per_sm\":%d,\"iters\":%d,\"warmup\":%d}\n",
        prop.name, prop.major, prop.minor, H, plain_mismatches,
        add_pack_mismatches, resid_mismatches, cp_e, pp_e, cp_g, pp_g,
        ca_e, pa_e, ca_g, pa_g, cp_e / pp_e, cp_g / pp_g,
        ca_e / pa_e, ca_g / pa_g, size_t(H) * 2 * sizeof(float),
        fused_attr.numRegs, fused_attr.sharedSizeBytes, active_blocks, iters, warmup);

    cudaStreamDestroy(stream);
    cudaFree(pack_probe); cudaFree(pack_control); cudaFree(nw); cudaFree(fac);
    cudaFree(resid_probe); cudaFree(resid_control); cudaFree(b); cudaFree(a);
    return (plain_mismatches == 0 && add_pack_mismatches == 0 && resid_mismatches == 0) ? 0 : 2;
}
