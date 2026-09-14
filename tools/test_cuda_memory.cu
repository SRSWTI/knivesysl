// Build: nvcc -std=c++17 -arch=sm_120f -I. tools/test_cuda_memory.cu -o /tmp/test_cuda_memory
// Run on an idle GPU; no model, mocks, or fault-injection allocator is used.
#include "src/cuda_memory.h"
#include <assert.h>

__global__ void increment(int *value) { ++*value; }

int main() {
    size_t available = 0, total = 0;
    assert(cudaMemGetInfo(&available, &total) == cudaSuccess);
    assert(cudaGetLastError() == cudaSuccess);
    int *value = nullptr;
    size_t capacity = 0;
    assert(tq_cuda_reserve(&value, &capacity, 1, "initial allocation") == 0);
    assert(cudaMemset(value, 0, sizeof(int)) == cudaSuccess);
    // Match an already-running model: lazy kernel/module initialization may
    // itself consume CUDA's last-error slot on the first launch.
    increment<<<1, 1>>>(value);
    assert(cudaStreamSynchronize(nullptr) == cudaSuccess);
    assert(cudaGetLastError() == cudaSuccess);
    assert(cudaMemset(value, 0, sizeof(int)) == cudaSuccess);
    int *original = value;

    // Exactly the production failure: OOM, a successful smaller retry, successful
    // GPU work, then a last-error check which still sees the recovered OOM.
    void *impossible = nullptr;
    cudaError_t err = cudaMalloc(&impossible, total + 1);
    assert(err == cudaErrorMemoryAllocation);
    void *small = nullptr;
    assert(cudaMalloc(&small, 4) == cudaSuccess);
    increment<<<1, 1>>>(value);
    assert(cudaStreamSynchronize(nullptr) == cudaSuccess);
    assert(cudaPeekAtLastError() == cudaErrorMemoryAllocation);
    assert(tq_cuda_recover_oom(err));
    assert(cudaGetLastError() == cudaSuccess);
    assert(cudaFree(small) == cudaSuccess);
    int host = 0;
    assert(cudaMemcpy(&host, value, sizeof(host), cudaMemcpyDeviceToHost) == cudaSuccess);
    assert(host == 1);
    puts("PASS: recovered OOM does not poison the next successful kernel");

    // Failed growth must preserve both the old pointer and its usable contents.
    assert(tq_cuda_reserve(&value, &capacity, total / sizeof(int) + 1,
                           "intentional oversized growth") == TQ_CUDA_FATAL);
    assert(value == original && capacity == 1);
    assert(tq_cuda_recover_oom(cudaPeekAtLastError()));
    assert(cudaMemcpy(&host, value, sizeof(host), cudaMemcpyDeviceToHost) == cudaSuccess);
    assert(host == 1);
    assert(tq_cuda_reserve(&value, &capacity, 1, "reuse allocation") == 0);
    assert(value == original && capacity == 1);
    puts("PASS: failed growth preserves the existing workspace");

    // Do not erase an unrelated real CUDA error while handling an OOM.
    err = cudaSetDevice(-1);
    assert(err != cudaSuccess && err != cudaErrorMemoryAllocation);
    assert(cudaPeekAtLastError() == err);
    assert(!tq_cuda_recover_oom(cudaErrorMemoryAllocation));
    assert(cudaPeekAtLastError() == err);
    assert(!tq_cuda_recover_oom(cudaErrorIllegalAddress));
    assert(cudaGetLastError() == err);  // explicit test cleanup, not recovery
    assert(cudaFree(value) == cudaSuccess);
    assert(cudaDeviceReset() == cudaSuccess);
    puts("PASS: unrelated and fatal CUDA errors are not suppressed");
}
