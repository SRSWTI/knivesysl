// Standalone sm120 probe for docs/level-up.md "attention v3 memory path".
// Does NOT modify or link the production engine.
//
// Question: why does k_tq_paged_attn_q4_split_gqa_v2 sustain only ~640 GB/s of
// KV traffic when the NVFP4 GEMM sustains 1.56 TB/s on the same silicon?
//
// Method: replicate the kernel's exact pool layout, grid shape, smem budget,
// staging pipeline, and (optionally) its compute phases against synthetic
// pools, then toggle one mechanism at a time:
//   clone     - full replica: Q prep + 3-stage cp.async staging + scores +
//               online softmax + j-ascending V fold (time-faithful, not
//               numerics-faithful: synthetic weights/codes)
//   noprep    - clone minus the per-head Q norm/RoPE/Hadamard prologue
//   nocompute - staging + waits + syncs only; bytes consumed by checksum
//   hoistbt   - clone with block_table hoisted once per chunk (2 registers)
//   direct    - no cp.async: plain ldg of the same bytes + same compute
//   nst4      - 4-deep pipeline at CH=16 (same smem class, deeper in-flight)
//
// Per variant: kernel wall time for a 16-layer-equivalent back-to-back burst,
// derived aggregate GB/s over the touched KV bytes. Grid mirrors production:
// (nkv=4, cols=N, S from the flow-split formula), 256 threads, 2 CTAs/SM cap.
//
// Build: nvcc -std=c++17 -O3 -lineinfo --generate-code=arch=compute_120,code=sm_120 \
//        -Xptxas=-v -o results/microbench/attn_bw tools/microbench_attn_bw.cu
// Run:   results/microbench/attn_bw <iters> <rows> <N> [variant]

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_fp16.h>
#include <cmath>
#include <vector>
#include <string>

#define CUDA_OK(expr) do { \
    cudaError_t status_ = (expr); \
    if (status_ != cudaSuccess) { \
        std::fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__, #expr, \
                     cudaGetErrorString(status_)); \
        std::exit(1); \
    } } while (0)

// ---- production constants (Qwen3.8-27B full-attention layers) ----
#define NKV      4
#define HD       256
#define GQA      6
#define QM       (24 * HD * 2)        // q_proj row stride used by the kernel
#define PAGE     128
#define PAGE_LOG 7
#define CH       24
#define NST      3
#define SKB      (CH * 128)
#define SCB      (CH * 32)            // fp16 (scale,zp) path: 32 B per row
#define SVB      (CH * 256)
#define OFF_QN   0
#define OFF_SK   6144
#define OFF_SC   (OFF_SK + NST * SKB)
#define OFF_SV   (OFF_SC + NST * SCB)
#define OFF_VS   (OFF_SV + NST * SVB)
#define OFF_SCH  (OFF_VS + NST * CH * 4)
#define OFF_CTL  (OFF_SCH + GQA * CH * 4)
#define SMEM_SZ  (OFF_CTL + 256)

// nst4 variant: CH=16, 4 buffers
#define CH4      16
#define SKB4     (CH4 * 128)
#define SCB4     (CH4 * 32)
#define SVB4     (CH4 * 256)
#define OFF4_SK  6144
#define OFF4_SC  (OFF4_SK + 4 * SKB4)
#define OFF4_SV  (OFF4_SC + 4 * SCB4)
#define OFF4_VS  (OFF4_SV + 4 * SVB4)
#define OFF4_SCH (OFF4_VS + 4 * CH4 * 4)
#define OFF4_CTL (OFF4_SCH + GQA * CH4 * 4)
#define SMEM4_SZ (OFF4_CTL + 256)

static __device__ __forceinline__ void cp16(uint32_t smaddr, const void *gaddr) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smaddr), "l"(gaddr));
}
static __device__ __forceinline__ float bf16f(uint16_t h) {
    uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f;
}
__device__ __constant__ float c_e4m3_lut[256];

// 8-stage in-place FWHT over 256 floats (time-equivalent to production helper).
static __device__ __forceinline__ void fwht256(float *b, int tid) {
    #pragma unroll
    for (int h = 1; h < 256; h <<= 1) {
        __syncthreads();
        int lo = tid & (h - 1), grp = tid >> (31 - __clz(h));
        int i = (grp * 2 * h) + lo;
        if (i + h < 256 && tid < 128) {
            float a = b[i], c = b[i + h];
            b[i] = a + c; b[i + h] = a - c;
        }
    }
    __syncthreads();
}

// ---------------------------------------------------------------------------
// The probe kernel. MODE: 0=clone 1=noprep 2=nocompute 3=hoistbt(+full compute)
// ---------------------------------------------------------------------------
template <int MODE>
__global__ __launch_bounds__(256, 2)
void k_probe_v2(const float *q_proj_base, const uint16_t *q_norm_w,
                const uint8_t *k4_pool, const uint16_t *kq4s_pool,
                const uint8_t *v8_pool, const float *vscale_pool,
                const int *positions, const int *slot_ids,
                const int *block_table, int max_blocks,
                int S, float *part_acc, float *part_ml, float *sink) {
    extern __shared__ uint32_t sm_u32[];
    char *sm = (char *)sm_u32;
    float *qn = (float *)(sm + OFF_QN);
    float *sch = (float *)(sm + OFF_SCH);
    float *ctl = (float *)(sm + OFF_CTL);
    float *m_run = ctl, *l_run = ctl + GQA, *m_new_s = ctl + 2 * GQA,
          *alpha_s = ctl + 3 * GQA, *l_chunk_s = ctl + 4 * GQA;
    float *partial = ctl + 5 * GQA;
    float *q_sum_shared = partial + 8;
    const int kv_head = blockIdx.x, col = blockIdx.y, split = blockIdx.z, tid = threadIdx.x;
    const int pos = positions[col];
    const int slot = slot_ids[col];
    const float *q_proj = q_proj_base + (size_t)col * QM;

    if (MODE == 0 || MODE == 3) {                      // faithful Q prep
        #pragma unroll
        for (int g = 0; g < GQA; g++) {
            int head = kv_head * GQA + g;
            float qv = q_proj[head * (2 * HD) + tid];
            float qsum = qv * qv;
            for (int off = 16; off > 0; off >>= 1) qsum += __shfl_down_sync(0xffffffffu, qsum, off);
            if ((tid & 31) == 0) partial[tid >> 5] = qsum;
            __syncthreads();
            if (tid < 8) {
                float vt = partial[tid];
                for (int off = 4; off > 0; off >>= 1) vt += __shfl_down_sync(0xff, vt, off);
                if (tid == 0) *q_sum_shared = vt;
            }
            __syncthreads();
            float rms = rsqrtf(*q_sum_shared / (float)HD + 1e-6f);
            qv = qv * rms * (1.0f + bf16f(q_norm_w[tid]));
            if (tid < 64) {
                int idx = tid & 31;
                float freq = powf(5000000.0f, -((float)(2 * idx) / 64.0f));
                float angle = (float)pos * freq;
                float c = cosf(angle), s = sinf(angle);
                int pi = (tid < 32) ? tid + 32 : tid - 32;
                float qp = q_proj[head * (2 * HD) + pi];
                qp = qp * rms * (1.0f + bf16f(q_norm_w[pi]));
                float qr = (tid < 32) ? -qp : qp;
                qv = qv * c + qr * s;
            }
            qn[g * 256 + tid] = qv;
            __syncthreads();
            fwht256(&qn[g * 256], tid);
        }
    } else {
        for (int g = tid; g < GQA * 256; g += 256) qn[g] = 0.001f * (g & 255);
        __syncthreads();
    }
    if (tid < GQA) { m_run[tid] = -3.402823466e38f; l_run[tid] = 0.0f; }
    __syncthreads();

    const size_t bt_base = (size_t)slot * max_blocks;
    const size_t k4_pp = (size_t)(NKV * HD) >> 1, kq4s_pp = (size_t)NKV * 16, v8_pp = (size_t)NKV * HD;
    const int warp = tid >> 5, lane = tid & 31;
    const float scale = rsqrtf((float)HD);
    float acc[GQA];
    #pragma unroll
    for (int g = 0; g < GQA; g++) acc[g] = 0.0f;
    const int total = pos + 1;
    const int per = (total + S - 1) / S;
    const int lo = split * per;
    int hi = lo + per; if (hi > total) hi = total;
    if (lo >= hi) return;

    auto issue = [&](int c0i) {
        int clen = hi - c0i; if (clen > CH) clen = CH;
        int buf = ((c0i - lo) / CH) % NST;
        int pg0v = 0, pg1v = 0, pg0i = 0;
        if (MODE == 3) {                                // hoist block_table
            pg0i = c0i >> PAGE_LOG;
            pg0v = block_table[bt_base + pg0i];
            pg1v = block_table[bt_base + ((c0i + clen - 1) >> PAGE_LOG)];
        }
        for (int id = tid; id < (clen << 3); id += 256) {
            int j = id >> 3, seg = id & 7, t = c0i + j;
            int phys = (MODE == 3) ? (((t >> PAGE_LOG) == pg0i) ? pg0v : pg1v)
                                   : block_table[bt_base + (t >> PAGE_LOG)];
            size_t pr = (size_t)phys * PAGE + (t & (PAGE - 1));
            cp16((uint32_t)__cvta_generic_to_shared(sm + OFF_SK + buf * SKB + j * 128 + seg * 16),
                 k4_pool + pr * k4_pp + (size_t)kv_head * (HD >> 1) + seg * 16);
        }
        for (int id = tid; id < (clen << 1); id += 256) {
            int j = id >> 1, seg = id & 1, t = c0i + j;
            int phys = (MODE == 3) ? (((t >> PAGE_LOG) == pg0i) ? pg0v : pg1v)
                                   : block_table[bt_base + (t >> PAGE_LOG)];
            size_t pr = (size_t)phys * PAGE + (t & (PAGE - 1));
            cp16((uint32_t)__cvta_generic_to_shared(sm + OFF_SC + buf * SCB + j * 32 + seg * 16),
                 (const char *)(kq4s_pool + pr * kq4s_pp + (size_t)kv_head * 16) + seg * 16);
        }
        for (int id = tid; id < (clen << 4); id += 256) {
            int j = id >> 4, seg = id & 15, t = c0i + j;
            int phys = (MODE == 3) ? (((t >> PAGE_LOG) == pg0i) ? pg0v : pg1v)
                                   : block_table[bt_base + (t >> PAGE_LOG)];
            size_t pr = (size_t)phys * PAGE + (t & (PAGE - 1));
            cp16((uint32_t)__cvta_generic_to_shared(sm + OFF_SV + buf * SVB + j * 256 + seg * 16),
                 v8_pool + pr * v8_pp + (size_t)kv_head * HD + seg * 16);
        }
        for (int j = tid; j < clen; j += 256) {
            int t = c0i + j;
            int phys = (MODE == 3) ? (((t >> PAGE_LOG) == pg0i) ? pg0v : pg1v)
                                   : block_table[bt_base + (t >> PAGE_LOG)];
            size_t pr = (size_t)phys * PAGE + (t & (PAGE - 1));
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                :: "r"((uint32_t)__cvta_generic_to_shared(sm + OFF_VS + buf * (CH * 4) + j * 4)),
                   "l"(vscale_pool + pr * NKV + kv_head));
        }
        asm volatile("cp.async.commit_group;\n");
    };

    issue(lo);
    if (lo + CH < hi) issue(lo + CH);
    else asm volatile("cp.async.commit_group;\n");
    float sink_acc = 0.0f;
    for (int c0 = lo; c0 < hi; c0 += CH) {
        int clen = hi - c0; if (clen > CH) clen = CH;
        const int buf = ((c0 - lo) / CH) % NST;
        if (c0 + 2 * CH < hi) issue(c0 + 2 * CH);
        else asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_group 2;\n");
        __syncthreads();
        const uint8_t *skb = (const uint8_t *)(sm + OFF_SK + buf * SKB);
        const uint16_t *scb = (const uint16_t *)(sm + OFF_SC + buf * SCB);
        if (MODE == 2) {                               // consume staged bytes only
            const float *vsb = (const float *)(sm + OFF_VS + buf * (CH * 4));
            for (int j = warp; j < clen; j += 8)
                sink_acc += (float)skb[j * 128 + lane] + (float)scb[j * 16 + (lane & 15)]
                          + (float)((const uint8_t *)(sm + OFF_SV + buf * SVB))[j * 256 + lane] + vsb[j];
            __syncthreads();
            continue;
        }
        if (MODE == 5) {                               // staging + V fold only
            #pragma unroll
            for (int g = 0; g < GQA; g++) acc[g] *= 0.9999f;
            const uint8_t *vb5 = (const uint8_t *)(sm + OFF_SV + buf * SVB);
            const float *vsb5 = (const float *)(sm + OFF_VS + buf * (CH * 4));
            for (int j = 0; j < clen; j++) {
                float vcode = c_e4m3_lut[vb5[j * 256 + tid]];
                float vscl = vsb5[j];
                #pragma unroll
                for (int g = 0; g < GQA; g++) acc[g] += 0.001f * vcode * vscl;
            }
            __syncthreads();
            continue;
        }
        if (MODE == 6) {
            // two rows per iteration: independent dequant/dot/shfl chains double ILP;
            // per-row arithmetic association identical to the reference loop.
            for (int j = warp; j < clen; j += 16) {
                const int j2 = j + 8;
                const bool has2 = j2 < clen;
                const uint8_t *krA = skb + j * 128, *krB = skb + j2 * 128;
                const uint16_t *ksA = scb + j * 16, *ksB = scb + j2 * 16;
                float kdA[8], kdB[8];
                #pragma unroll
                for (int e = 0; e < 8; e++) {
                    int d = lane + 32 * e;
                    uint8_t byA = krA[d >> 1];
                    uint8_t byB = has2 ? krB[d >> 1] : 0;
                    float cA = (float)((d & 1) ? (byA >> 4) : (byA & 15));
                    float cB = (float)((d & 1) ? (byB >> 4) : (byB & 15));
                    float scA = __half2float(((const __half *)ksA)[2 * e]);
                    float zpA = __half2float(((const __half *)ksA)[2 * e + 1]);
                    float scB = has2 ? __half2float(((const __half *)ksB)[2 * e]) : 0.f;
                    float zpB = has2 ? __half2float(((const __half *)ksB)[2 * e + 1]) : 0.f;
                    kdA[e] = (cA - zpA) * scA;
                    kdB[e] = (cB - zpB) * scB;
                }
                #pragma unroll
                for (int g = 0; g < GQA; g++) {
                    float dA = 0.0f, dB = 0.0f;
                    #pragma unroll
                    for (int e = 0; e < 8; e++) {
                        float qv = qn[g * 256 + lane + 32 * e];
                        dA += qv * kdA[e];
                        dB += qv * kdB[e];
                    }
                    #pragma unroll
                    for (int off = 16; off > 0; off >>= 1) {
                        dA += __shfl_down_sync(0xffffffffu, dA, off);
                        dB += __shfl_down_sync(0xffffffffu, dB, off);
                    }
                    if (lane == 0) {
                        sch[g * CH + j] = dA * scale;
                        if (has2) sch[g * CH + j2] = dB * scale;
                    }
                }
            }
        } else
        for (int j = warp; j < clen; j += 8) {
            if (MODE == 7) {                            // no dequant: fixed kd
                float kd7[8];
                #pragma unroll
                for (int e = 0; e < 8; e++) kd7[e] = 0.001f * (float)(lane + e);
                #pragma unroll
                for (int g = 0; g < GQA; g++) {
                    float dot = 0.0f;
                    #pragma unroll
                    for (int e = 0; e < 8; e++) dot += qn[g * 256 + lane + 32 * e] * kd7[e];
                    #pragma unroll
                    for (int off = 16; off > 0; off >>= 1) dot += __shfl_down_sync(0xffffffffu, dot, off);
                    if (lane == 0) sch[g * CH + j] = dot * scale;
                }
                continue;
            }
            const uint8_t *krow = skb + j * 128;
            const uint16_t *ksz = scb + j * 16;
            float kd[8];
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                int d = lane + 32 * e;
                uint8_t byte = krow[d >> 1];
                float code = (float)((d & 1) ? (byte >> 4) : (byte & 15));
                float sc = __half2float(((const __half *)ksz)[2 * e]);
                float zp = __half2float(((const __half *)ksz)[2 * e + 1]);
                kd[e] = (code - zp) * sc;
            }
            if (MODE == 8) {                            // dequant + dot, no shfl tree
                #pragma unroll
                for (int g = 0; g < GQA; g++) {
                    float dot = 0.0f;
                    #pragma unroll
                    for (int e = 0; e < 8; e++) dot += qn[g * 256 + lane + 32 * e] * kd[e];
                    if (lane == 0) sch[g * CH + j] = dot * scale;
                    else if (lane == 1) sch[g * CH + j] += dot * 1e-9f;
                }
                continue;
            }
            #pragma unroll
            for (int g = 0; g < GQA; g++) {
                float dot = 0.0f;
                #pragma unroll
                for (int e = 0; e < 8; e++) dot += qn[g * 256 + lane + 32 * e] * kd[e];
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) dot += __shfl_down_sync(0xffffffffu, dot, off);
                if (lane == 0) sch[g * CH + j] = dot * scale;
            }
        }
        __syncthreads();
        if (MODE == 4) { sink_acc += sch[tid & (GQA * CH - 1)]; continue; }
        if (warp < GQA) {
            const int g = warp;
            float v0 = (lane < clen) ? sch[g * CH + lane] : -3.402823466e38f;
            float m = v0;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
            float m_old = m_run[g], m_new = fmaxf(m_old, m);
            float p0 = (lane < clen) ? expf(v0 - m_new) : 0.0f;
            if (lane < clen) sch[g * CH + lane] = p0;
            float ls = p0;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) ls += __shfl_xor_sync(0xffffffffu, ls, off);
            if (lane == 0) {
                float alpha = expf(m_old - m_new);
                m_new_s[g] = m_new; alpha_s[g] = alpha; l_chunk_s[g] = ls;
                l_run[g] = l_run[g] * alpha + ls; m_run[g] = m_new;
            }
        }
        __syncthreads();
        #pragma unroll
        for (int g = 0; g < GQA; g++) acc[g] *= alpha_s[g];
        const uint8_t *vb = (const uint8_t *)(sm + OFF_SV + buf * SVB);
        const float *vsb = (const float *)(sm + OFF_VS + buf * (CH * 4));
        for (int j = 0; j < clen; j++) {
            float vcode = c_e4m3_lut[vb[j * 256 + tid]];
            float vscl = vsb[j];
            #pragma unroll
            for (int g = 0; g < GQA; g++) acc[g] += sch[g * CH + j] * vcode * vscl;
        }
        __syncthreads();
    }
    if (MODE == 2 || MODE == 4) { if (sink) sink[(blockIdx.z * gridDim.x + blockIdx.x) * 256 + tid] = sink_acc; return; }
    #pragma unroll
    for (int g = 0; g < GQA; g++) {
        int head = kv_head * GQA + g;
        size_t pidx = ((size_t)(head * gridDim.y + col) * S + split);
        part_acc[pidx * HD + tid] = acc[g];
        if (tid == 0) { part_ml[pidx * 2 + 0] = m_run[g]; part_ml[pidx * 2 + 1] = l_run[g]; }
    }
}

// direct-load variant: same compute, no cp.async staging (ldg straight to regs)
__global__ __launch_bounds__(256, 2)
void k_probe_direct(const float *q_proj_base, const uint16_t *q_norm_w,
                    const uint8_t *k4_pool, const uint16_t *kq4s_pool,
                    const uint8_t *v8_pool, const float *vscale_pool,
                    const int *positions, const int *slot_ids,
                    const int *block_table, int max_blocks,
                    int S, float *part_acc, float *part_ml) {
    extern __shared__ uint32_t sm_u32[];
    char *sm = (char *)sm_u32;
    float *qn = (float *)(sm + OFF_QN);
    float *sch = (float *)(sm + OFF_SCH);
    float *ctl = (float *)(sm + OFF_CTL);
    float *m_run = ctl, *l_run = ctl + GQA, *alpha_s = ctl + 3 * GQA;
    const int kv_head = blockIdx.x, col = blockIdx.y, split = blockIdx.z, tid = threadIdx.x;
    const int pos = positions[col];
    const int slot = slot_ids[col];
    for (int g = tid; g < GQA * 256; g += 256) qn[g] = 0.001f * (g & 255);
    if (tid < GQA) { m_run[tid] = -3.402823466e38f; l_run[tid] = 0.0f; }
    __syncthreads();
    const size_t bt_base = (size_t)slot * max_blocks;
    const size_t k4_pp = (size_t)(NKV * HD) >> 1, kq4s_pp = (size_t)NKV * 16, v8_pp = (size_t)NKV * HD;
    const int warp = tid >> 5, lane = tid & 31;
    const float scale = rsqrtf((float)HD);
    float acc[GQA];
    #pragma unroll
    for (int g = 0; g < GQA; g++) acc[g] = 0.0f;
    const int total = pos + 1;
    const int per = (total + S - 1) / S;
    const int lo = split * per;
    int hi = lo + per; if (hi > total) hi = total;
    if (lo >= hi) return;
    for (int c0 = lo; c0 < hi; c0 += CH) {
        int clen = hi - c0; if (clen > CH) clen = CH;
        for (int j = warp; j < clen; j += 8) {
            int t = c0 + j;
            int phys = block_table[bt_base + (t >> PAGE_LOG)];
            size_t pr = (size_t)phys * PAGE + (t & (PAGE - 1));
            const uint8_t *krow = k4_pool + pr * k4_pp + (size_t)kv_head * (HD >> 1);
            const uint16_t *ksz = kq4s_pool + pr * kq4s_pp + (size_t)kv_head * 16;
            float kd[8];
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                int d = lane + 32 * e;
                uint8_t byte = __ldg(&krow[d >> 1]);
                float code = (float)((d & 1) ? (byte >> 4) : (byte & 15));
                float sc = __half2float(((const __half *)ksz)[2 * e]);
                float zp = __half2float(((const __half *)ksz)[2 * e + 1]);
                kd[e] = (code - zp) * sc;
            }
            #pragma unroll
            for (int g = 0; g < GQA; g++) {
                float dot = 0.0f;
                #pragma unroll
                for (int e = 0; e < 8; e++) dot += qn[g * 256 + lane + 32 * e] * kd[e];
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) dot += __shfl_down_sync(0xffffffffu, dot, off);
                if (lane == 0) sch[g * CH + j] = dot * scale;
            }
        }
        __syncthreads();
        if (warp < GQA) {
            const int g = warp;
            float v0 = (lane < clen) ? sch[g * CH + lane] : -3.402823466e38f;
            float m = v0;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
            float m_old = m_run[g], m_new = fmaxf(m_old, m);
            float p0 = (lane < clen) ? expf(v0 - m_new) : 0.0f;
            if (lane < clen) sch[g * CH + lane] = p0;
            if (lane == 0) { alpha_s[g] = expf(m_old - m_new); m_run[g] = m_new; l_run[g] = l_run[g] * alpha_s[g]; }
        }
        __syncthreads();
        #pragma unroll
        for (int g = 0; g < GQA; g++) acc[g] *= alpha_s[g];
        for (int j = 0; j < clen; j++) {
            int t = c0 + j;
            int phys = block_table[bt_base + (t >> PAGE_LOG)];
            size_t pr = (size_t)phys * PAGE + (t & (PAGE - 1));
            float vcode = c_e4m3_lut[__ldg(v8_pool + pr * v8_pp + (size_t)kv_head * HD + tid)];
            float vscl = __ldg(vscale_pool + pr * NKV + kv_head);
            #pragma unroll
            for (int g = 0; g < GQA; g++) acc[g] += sch[g * CH + j] * vcode * vscl;
        }
        __syncthreads();
    }
    #pragma unroll
    for (int g = 0; g < GQA; g++) {
        int head = kv_head * GQA + g;
        size_t pidx = ((size_t)(head * gridDim.y + col) * S + split);
        part_acc[pidx * HD + tid] = acc[g];
        if (tid == 0) { part_ml[pidx * 2 + 0] = m_run[g]; part_ml[pidx * 2 + 1] = 0.0f; }
    }
}

static int flow_S(int N, int total) {
    int sm = 170, blocks0 = NKV * N;
    int s_occ = (2 * sm + blocks0 - 1) / blocks0;
    if (N >= 2) {
        int s_flow = (total + 383) / 384;
        int S = s_occ > s_flow ? s_occ : s_flow;
        return S > 192 ? 192 : S;
    }
    int s_work = total / 512;
    int S = s_occ < s_work ? s_occ : s_work;
    if (S < 1) S = 1;
    return S > 96 ? 96 : S;
}

int main(int argc, char **argv) {
    int iters = argc > 1 ? std::atoi(argv[1]) : 200;
    int rows = argc > 2 ? std::atoi(argv[2]) : 32769;
    int N = argc > 3 ? std::atoi(argv[3]) : 1;
    std::string only = argc > 4 ? argv[4] : "";
    const int LAYERS = 16;

    float lut[256];
    for (int i = 0; i < 256; i++) {
        int s = (i >> 7) & 1, e = (i >> 3) & 0xF, m = i & 7;
        float v = e ? ldexpf(1.0f + m / 8.0f, e - 7) : ldexpf(m / 8.0f, -6);
        lut[i] = s ? -v : v;
    }
    CUDA_OK(cudaMemcpyToSymbol(c_e4m3_lut, lut, sizeof(lut)));

    const int max_blocks = (rows + PAGE - 1) / PAGE + 1;
    const size_t pool_rows = (size_t)max_blocks * N * PAGE;
    uint8_t *k4; uint16_t *kq4s; uint8_t *v8; float *vs;
    CUDA_OK(cudaMalloc(&k4, pool_rows * NKV * HD / 2));
    CUDA_OK(cudaMalloc(&kq4s, pool_rows * NKV * 16 * 2));
    CUDA_OK(cudaMalloc(&v8, pool_rows * NKV * HD));
    CUDA_OK(cudaMalloc(&vs, pool_rows * NKV * 4));
    CUDA_OK(cudaMemset(k4, 0x53, pool_rows * NKV * HD / 2));
    CUDA_OK(cudaMemset(v8, 0x41, pool_rows * NKV * HD));
    {   // sane fp16 scales + fp32 vscales
        std::vector<uint16_t> hs(pool_rows * NKV * 16, 0x3C00);   // 1.0h
        CUDA_OK(cudaMemcpy(kq4s, hs.data(), hs.size() * 2, cudaMemcpyHostToDevice));
        std::vector<float> hv(pool_rows * NKV, 0.01f);
        CUDA_OK(cudaMemcpy(vs, hv.data(), hv.size() * 4, cudaMemcpyHostToDevice));
    }
    std::vector<int> h_bt(N * max_blocks), h_pos(N), h_slot(N);
    for (int n = 0; n < N; n++) {
        for (int b = 0; b < max_blocks; b++) h_bt[n * max_blocks + b] = n * max_blocks + b;
        h_pos[n] = rows - 1; h_slot[n] = n;
    }
    int *bt, *pos, *slot;
    CUDA_OK(cudaMalloc(&bt, h_bt.size() * 4));
    CUDA_OK(cudaMalloc(&pos, N * 4));
    CUDA_OK(cudaMalloc(&slot, N * 4));
    CUDA_OK(cudaMemcpy(bt, h_bt.data(), h_bt.size() * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(pos, h_pos.data(), N * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(slot, h_slot.data(), N * 4, cudaMemcpyHostToDevice));
    float *q_proj; uint16_t *qnw;
    CUDA_OK(cudaMalloc(&q_proj, (size_t)N * QM * 4));
    CUDA_OK(cudaMemset(q_proj, 0, (size_t)N * QM * 4));
    CUDA_OK(cudaMalloc(&qnw, HD * 2));
    CUDA_OK(cudaMemset(qnw, 0, HD * 2));

    const int S = flow_S(N, rows);
    float *pacc, *pml, *sink;
    CUDA_OK(cudaMalloc(&pacc, (size_t)24 * N * S * HD * 4));
    CUDA_OK(cudaMalloc(&pml, (size_t)24 * N * S * 2 * 4));
    CUDA_OK(cudaMalloc(&sink, (size_t)S * NKV * 256 * 4));

    const double kv_bytes = (double)rows * N * NKV * (128 + 32 + 256 + 4) * LAYERS;
    dim3 grid(NKV, N, S), blk(256, 1, 1);
    cudaStream_t st; CUDA_OK(cudaStreamCreate(&st));
    cudaEvent_t a, b; CUDA_OK(cudaEventCreate(&a)); CUDA_OK(cudaEventCreate(&b));

    auto bench = [&](const char *name, auto launch) {
        if (!only.empty() && only != name) return;
        for (int w = 0; w < 20; w++) launch();
        CUDA_OK(cudaStreamSynchronize(st));
        CUDA_OK(cudaEventRecord(a, st));
        for (int i = 0; i < iters; i++) launch();
        CUDA_OK(cudaEventRecord(b, st));
        CUDA_OK(cudaStreamSynchronize(st));
        float ms = 0.f; CUDA_OK(cudaEventElapsedTime(&ms, a, b));
        double per_burst_us = (double)ms * 1000.0 / iters;
        double gbps = kv_bytes / (per_burst_us * 1e-6) / 1e9;
        std::printf("{\"variant\":\"%s\",\"rows\":%d,\"N\":%d,\"S\":%d,"
                    "\"burst16_us\":%.2f,\"us_per_layer\":%.2f,\"gb_s\":%.1f}\n",
                    name, rows, N, S, per_burst_us, per_burst_us / LAYERS, gbps);
    };

    bench("clone", [&]{ for (int l = 0; l < LAYERS; l++)
        k_probe_v2<0><<<grid, blk, SMEM_SZ, st>>>(q_proj, qnw, k4, kq4s, v8, vs, pos, slot, bt, max_blocks, S, pacc, pml, nullptr); });
    bench("noprep", [&]{ for (int l = 0; l < LAYERS; l++)
        k_probe_v2<1><<<grid, blk, SMEM_SZ, st>>>(q_proj, qnw, k4, kq4s, v8, vs, pos, slot, bt, max_blocks, S, pacc, pml, nullptr); });
    bench("nocompute", [&]{ for (int l = 0; l < LAYERS; l++)
        k_probe_v2<2><<<grid, blk, SMEM_SZ, st>>>(q_proj, qnw, k4, kq4s, v8, vs, pos, slot, bt, max_blocks, S, pacc, pml, sink); });
    bench("scoresonly", [&]{ for (int l = 0; l < LAYERS; l++)
        k_probe_v2<4><<<grid, blk, SMEM_SZ, st>>>(q_proj, qnw, k4, kq4s, v8, vs, pos, slot, bt, max_blocks, S, pacc, pml, sink); });
    bench("foldonly", [&]{ for (int l = 0; l < LAYERS; l++)
        k_probe_v2<5><<<grid, blk, SMEM_SZ, st>>>(q_proj, qnw, k4, kq4s, v8, vs, pos, slot, bt, max_blocks, S, pacc, pml, nullptr); });
    bench("ilv2", [&]{ for (int l = 0; l < LAYERS; l++)
        k_probe_v2<6><<<grid, blk, SMEM_SZ, st>>>(q_proj, qnw, k4, kq4s, v8, vs, pos, slot, bt, max_blocks, S, pacc, pml, nullptr); });
    bench("nodeq", [&]{ for (int l = 0; l < LAYERS; l++)
        k_probe_v2<7><<<grid, blk, SMEM_SZ, st>>>(q_proj, qnw, k4, kq4s, v8, vs, pos, slot, bt, max_blocks, S, pacc, pml, nullptr); });
    bench("noshfl", [&]{ for (int l = 0; l < LAYERS; l++)
        k_probe_v2<8><<<grid, blk, SMEM_SZ, st>>>(q_proj, qnw, k4, kq4s, v8, vs, pos, slot, bt, max_blocks, S, pacc, pml, nullptr); });
    bench("hoistbt", [&]{ for (int l = 0; l < LAYERS; l++)
        k_probe_v2<3><<<grid, blk, SMEM_SZ, st>>>(q_proj, qnw, k4, kq4s, v8, vs, pos, slot, bt, max_blocks, S, pacc, pml, nullptr); });
    bench("direct", [&]{ for (int l = 0; l < LAYERS; l++)
        k_probe_direct<<<grid, blk, SMEM_SZ, st>>>(q_proj, qnw, k4, kq4s, v8, vs, pos, slot, bt, max_blocks, S, pacc, pml); });
    return 0;
}
