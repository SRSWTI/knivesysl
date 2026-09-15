// Benchmark-only engine translation unit: compile this file instead of
// src/forward_qwen.cu, never alongside it or linked to another engine instance.
#include <stdint.h>
#include "../src/forward_qwen.cu"

// Advance the real paged engine once, then copy its raw FP32 logits [N][V].
// Rows follow slots[] order, not physical slot indices. float_capacity counts
// float elements; no truncation is permitted. This copy is for numerical gates,
// not timed inference. The caller must serialize all operations on this engine.
extern "C" int qwn_bench_paged_step_logits(
    const int *tokens, const int *slots, const int *positions, int N,
    int *out_tokens, float *host_logits, size_t float_capacity) {
    if (!tokens || !slots || !positions || !out_tokens || !host_logits || N < 1)
        return -200;
    if (!g_qwen.initialized || !g_pg_ready || !g_pg_logits || g_qwen.V < 1)
        return -201;
    if (N > g_pg_maxslots) return -200;
    // The alternative per-column argmax path never publishes g_pg_logits.
    if (g_qwen.tie_word_embeddings || !g_qwen.lm_head.e2m3 ||
        g_qwen.lm_head.e2m3_byte || g_qwen.lm_head.word_major ||
        !can_use_qmma_sf_weight(&g_qwen.lm_head) || g_qwen.lm_head.row_major ||
        g_qwen.lm_head.M != g_qwen.V)
        return -202;
    const size_t vocab = (size_t)g_qwen.V;
    if (vocab > (SIZE_MAX / sizeof(float)) / (size_t)N) return -203;
    const size_t count = (size_t)N * vocab;
    if (float_capacity < count) return -203;

    int rc = qwn_paged_decode_step(tokens, slots, positions, N, out_tokens);
    if (rc != 0) return rc;
    cudaError_t err = cudaMemcpyAsync(host_logits, g_pg_logits,
                                      count * sizeof(float), cudaMemcpyDeviceToHost,
                                      g_qwen.stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "qwn_bench_paged_step_logits: logits copy failed: %s\n",
                cudaGetErrorString(err));
        return -204;
    }
    err = cudaStreamSynchronize(g_qwen.stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "qwn_bench_paged_step_logits: logits sync failed: %s\n",
                cudaGetErrorString(err));
        return -205;
    }
    return 0;
}
