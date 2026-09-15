// Real-CUDA model allocation ownership regression; compile instead of the engine.
// Build from repo root: nvcc -O3 -std=c++17 -gencode=arch=compute_120f,code=sm_120f tools/test_model_ownership.cu -lcuda -o /tmp/test_model_ownership
// Run on an idle GPU: TQ_W_NVFP4=all /tmp/test_model_ownership MODEL.tqf 2
// Tracked model allocations must return to zero after every teardown. CUDA driver
// and runtime residency is diagnostic only: no fixed memory target or device reset.
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <map>
#include <unordered_map>
#include <string>
#include <utility>

namespace allocation_probe {
struct Allocation {
    size_t bytes;
    const char *file;
    int line;
};
static std::unordered_map<void *, Allocation> live;
static size_t live_bytes = 0;
static size_t peak_live_bytes = 0;
static size_t baseline_free = 0;
static unsigned long long sequence = 0;
static bool observer_failed = false;

static void snapshot(const char *stage) {
    size_t available = 0, total = 0;
    cudaError_t rc = ::cudaMemGetInfo(&available, &total);
    if (rc != cudaSuccess) observer_failed = true;
    const long long delta = baseline_free
        ? (long long)baseline_free - (long long)available : 0;
    fprintf(stderr,
        "ALLOC_PROBE snapshot stage=%s rc=%d error=%s free_bytes=%zu total_bytes=%zu "
        "delta_from_baseline_bytes=%lld tracked_live_bytes=%zu tracked_peak_bytes=%zu "
        "untracked_delta_bytes=%lld live_allocations=%zu\n",
        stage, (int)rc, cudaGetErrorString(rc), available, total, delta,
        live_bytes, peak_live_bytes, delta - (long long)live_bytes, live.size());
}

template<class T>
static cudaError_t allocate(T **destination, size_t bytes, const char *file, int line) {
    // Preserve CUDA's allocation and failure semantics. Do not read the output
    // pointer on failure, when CUDA may have left its previous value untouched.
    cudaError_t rc = ::cudaMalloc(reinterpret_cast<void **>(destination), bytes);
    void *pointer = rc == cudaSuccess ? static_cast<void *>(*destination) : nullptr;
    if (rc != cudaSuccess) observer_failed = true;
    if (rc == cudaSuccess && pointer) {
        auto inserted = live.emplace(pointer, Allocation{bytes, file, line});
        if (!inserted.second) {
            observer_failed = true;
            fprintf(stderr,
                "ALLOC_PROBE observer_error duplicate_live_pointer=%p file=%s line=%d\n",
                pointer, file, line);
        } else {
            live_bytes += bytes;
            peak_live_bytes = std::max(peak_live_bytes, live_bytes);
        }
    }
    ++sequence;
    if (rc != cudaSuccess) {
        fprintf(stderr,
            "ALLOC_PROBE malloc seq=%llu file=%s line=%d pointer=%p bytes=%zu "
            "rc=%d error=%s tracked_live_bytes=%zu\n",
            sequence, file, line, pointer, bytes, (int)rc, cudaGetErrorString(rc), live_bytes);
    }
    return rc;
}

static cudaError_t release(void *pointer, const char *file, int line) {
    const auto found = live.find(pointer);
    const bool tracked = found != live.end();
    const size_t bytes = tracked ? found->second.bytes : 0;
    cudaError_t rc = ::cudaFree(pointer);
    if (rc != cudaSuccess) observer_failed = true;
    if (rc == cudaSuccess && tracked) {
        live_bytes -= bytes;
        live.erase(found);
    }
    ++sequence;
    if (rc != cudaSuccess) {
        fprintf(stderr,
            "ALLOC_PROBE free seq=%llu file=%s line=%d pointer=%p bytes=%zu tracked=%d "
            "rc=%d error=%s tracked_live_bytes=%zu\n",
            sequence, file, line, pointer, bytes, tracked ? 1 : 0,
            (int)rc, cudaGetErrorString(rc), live_bytes);
    }
    return rc;
}

static bool synchronized_snapshot(const char *stage) {
    const cudaError_t rc = ::cudaDeviceSynchronize();
    if (rc != cudaSuccess) observer_failed = true;
    fprintf(stderr, "ALLOC_PROBE synchronize stage=%s rc=%d error=%s\n",
            stage, (int)rc, cudaGetErrorString(rc));
    snapshot(stage);
    return rc == cudaSuccess;
}

static void report_live(const char *stage) {
    // Deterministic source-site totals, plus exact pointers for ownership tracing.
    std::map<std::pair<std::string, int>, std::pair<size_t, size_t>> sites;
    for (const auto &entry : live) {
        const Allocation &a = entry.second;
        fprintf(stderr, "ALLOC_PROBE live stage=%s file=%s line=%d pointer=%p bytes=%zu\n",
                stage, a.file, a.line, entry.first, a.bytes);
        auto &site = sites[std::make_pair(std::string(a.file), a.line)];
        site.first += a.bytes;
        site.second++;
    }
    for (const auto &entry : sites) {
        fprintf(stderr, "ALLOC_PROBE site stage=%s file=%s line=%d bytes=%zu allocations=%zu\n",
                stage, entry.first.first.c_str(), entry.first.second,
                entry.second.first, entry.second.second);
    }
}
} // namespace allocation_probe

// CUDA headers are already parsed. All explicit engine cudaMalloc/cudaFree calls
// (including inline cuda_memory.h helpers) now delegate to the real CUDA API above.
#define cudaMalloc(destination, bytes) allocation_probe::allocate((destination), (bytes), __FILE__, __LINE__)
#define cudaFree(pointer) allocation_probe::release((pointer), __FILE__, __LINE__)
#include "../src/forward_qwen.cu"
#undef cudaMalloc
#undef cudaFree

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "usage: %s MODEL.tqf [cycles]\n", argv[0]);
        return 2;
    }
    const int cycles = argc == 3 ? atoi(argv[2]) : 1;
    if (cycles < 1 || cycles > 8) {
        fprintf(stderr, "ALLOC_PROBE cycles must be between 1 and 8\n");
        return 2;
    }
    setvbuf(stderr, nullptr, _IOLBF, 0);
    // Establish a context before the allocation baseline, matching the benchmarks.
    cudaError_t rc = ::cudaFree(nullptr);
    if (rc != cudaSuccess) {
        fprintf(stderr, "ALLOC_PROBE context_init rc=%d error=%s\n", (int)rc, cudaGetErrorString(rc));
        return 3;
    }
    if (!allocation_probe::synchronized_snapshot("context_initialized")) return 3;
    size_t total = 0;
    rc = ::cudaMemGetInfo(&allocation_probe::baseline_free, &total);
    if (rc != cudaSuccess) {
        fprintf(stderr, "ALLOC_PROBE baseline rc=%d error=%s\n", (int)rc, cudaGetErrorString(rc));
        return 3;
    }
    for (int cycle = 0; cycle < cycles; ++cycle) {
        fprintf(stderr, "ALLOC_PROBE cycle=%d begin\n", cycle + 1);
        allocation_probe::snapshot("before_qwn_init");
        const int init_rc = qwn_init(argv[1]);
        fprintf(stderr, "ALLOC_PROBE qwn_init rc=%d device_bytes_counter=%zu\n", init_rc, qwn_device_bytes());
        const bool init_sync_ok = allocation_probe::synchronized_snapshot("after_qwn_init");
        if (init_rc != 0) allocation_probe::report_live("after_failed_qwn_init");
        // No paged pool, forward pass, model mutation, or manual buffer reclamation.
        qwn_free();
        const bool free_sync_ok = allocation_probe::synchronized_snapshot("after_qwn_free");
        allocation_probe::report_live("after_qwn_free");
        fprintf(stderr, "ALLOC_PROBE cycle=%d end remaining_bytes=%zu remaining_allocations=%zu\n",
                cycle + 1, allocation_probe::live_bytes, allocation_probe::live.size());
        if (init_rc != 0 || !init_sync_ok || !free_sync_ok || allocation_probe::observer_failed ||
            !allocation_probe::live.empty() || allocation_probe::live_bytes != 0) return 1;
    }
    return 0;
}
