#pragma once

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

// Only this explicit status may cross an optional-cache boundary as fatal.
static constexpr int TQ_CUDA_FATAL = -121;

static inline int tq_cuda_check(cudaError_t err, const char *operation,
                                size_t bytes = 0) {
    if (err == cudaSuccess) return 0;
    size_t available = 0, total = 0;
    cudaError_t info = cudaMemGetInfo(&available, &total);
    fprintf(stderr, "[cuda] %s: %s (code=%d requested=%zu bytes",
            operation, cudaGetErrorString(err), (int)err, bytes);
    if (info == cudaSuccess)
        fprintf(stderr, " free=%zu total=%zu bytes", available, total);
    fprintf(stderr, ")\n");
    return TQ_CUDA_FATAL;
}

// A failed optional allocation is recoverable; an unrelated pending error is not.
// Successful CUDA calls do NOT clear an earlier cudaErrorMemoryAllocation.
static inline bool tq_cuda_recover_oom(cudaError_t err) {
    if (err != cudaErrorMemoryAllocation) return false;
    cudaError_t pending = cudaPeekAtLastError();
    if (pending != cudaSuccess && pending != cudaErrorMemoryAllocation) return false;
    if (pending == cudaErrorMemoryAllocation) cudaGetLastError();
    return true;
}

// Used at startup, before any slot can own in-flight work. Allocate first so a
// failed growth never leaves a freed pointer paired with a nonzero capacity.
template <typename T, typename Count>
static int tq_cuda_reserve(T **ptr, Count *capacity, size_t count, const char *name) {
    if ((size_t)*capacity >= count) return 0;
    if (count > SIZE_MAX / sizeof(T) || (size_t)(Count)count != count)
        return TQ_CUDA_FATAL;
    T *replacement = nullptr;
    cudaError_t err = cudaMalloc(&replacement, count * sizeof(T));
    if (err != cudaSuccess) return tq_cuda_check(err, name, count * sizeof(T));
    if (*ptr) {
        err = cudaFree(*ptr);
        if (err != cudaSuccess) {
            cudaFree(replacement);
            return tq_cuda_check(err, name);
        }
    }
    *ptr = replacement;
    *capacity = (Count)count;
    return 0;
}
