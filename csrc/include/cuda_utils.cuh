#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cuda_fp16.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <memory>
#include <vector>
#include <mutex>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
//  Error-checking macros
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                        \
        cublasStatus_t stat = (call);                                          \
        if (stat != CUBLAS_STATUS_SUCCESS) {                                   \
            fprintf(stderr, "cuBLAS error at %s:%d — code %d\n",              \
                    __FILE__, __LINE__, (int)stat);                             \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CURAND_CHECK(call)                                                     \
    do {                                                                        \
        curandStatus_t stat = (call);                                          \
        if (stat != CURAND_STATUS_SUCCESS) {                                   \
            fprintf(stderr, "cuRAND error at %s:%d — code %d\n",              \
                    __FILE__, __LINE__, (int)stat);                             \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
//  RAII wrappers
// ---------------------------------------------------------------------------

/// Device (GPU) memory buffer with automatic lifetime management.
template <typename T>
struct DeviceBuffer {
    T*     ptr   = nullptr;
    size_t count = 0;

    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t n) : count(n) {
        if (n > 0) CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T)));
    }
    ~DeviceBuffer() { reset(); }

    // Move-only
    DeviceBuffer(DeviceBuffer&& o) noexcept : ptr(o.ptr), count(o.count) {
        o.ptr = nullptr; o.count = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) { reset(); ptr = o.ptr; count = o.count; o.ptr = nullptr; o.count = 0; }
        return *this;
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void allocate(size_t n) { reset(); count = n; if (n > 0) CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T))); }
    void reset() { if (ptr) { cudaFree(ptr); ptr = nullptr; } count = 0; }
    size_t bytes() const { return count * sizeof(T); }
    operator T*() { return ptr; }
    operator const T*() const { return ptr; }
};

/// Pinned (page-locked) host memory buffer.
template <typename T>
struct PinnedBuffer {
    T*     ptr   = nullptr;
    size_t count = 0;

    PinnedBuffer() = default;
    explicit PinnedBuffer(size_t n) : count(n) {
        if (n > 0) CUDA_CHECK(cudaMallocHost(&ptr, n * sizeof(T)));
    }
    ~PinnedBuffer() { reset(); }

    PinnedBuffer(PinnedBuffer&& o) noexcept : ptr(o.ptr), count(o.count) {
        o.ptr = nullptr; o.count = 0;
    }
    PinnedBuffer& operator=(PinnedBuffer&& o) noexcept {
        if (this != &o) { reset(); ptr = o.ptr; count = o.count; o.ptr = nullptr; o.count = 0; }
        return *this;
    }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    void allocate(size_t n) { reset(); count = n; if (n > 0) CUDA_CHECK(cudaMallocHost(&ptr, n * sizeof(T))); }
    void reset() { if (ptr) { cudaFreeHost(ptr); ptr = nullptr; } count = 0; }
    size_t bytes() const { return count * sizeof(T); }
    operator T*() { return ptr; }
    operator const T*() const { return ptr; }
};

/// RAII wrapper for cudaStream_t.
struct CudaStream {
    cudaStream_t s = nullptr;
    CudaStream()  { CUDA_CHECK(cudaStreamCreate(&s)); }
    explicit CudaStream(unsigned flags) { CUDA_CHECK(cudaStreamCreateWithFlags(&s, flags)); }
    ~CudaStream() { if (s) cudaStreamDestroy(s); }
    CudaStream(CudaStream&& o) noexcept : s(o.s) { o.s = nullptr; }
    CudaStream& operator=(CudaStream&& o) noexcept {
        if (this != &o) { if (s) cudaStreamDestroy(s); s = o.s; o.s = nullptr; }
        return *this;
    }
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    operator cudaStream_t() const { return s; }
    void sync() const { CUDA_CHECK(cudaStreamSynchronize(s)); }
};

/// RAII wrapper for cudaEvent_t.
struct CudaEvent {
    cudaEvent_t e = nullptr;
    CudaEvent() { CUDA_CHECK(cudaEventCreate(&e)); }
    explicit CudaEvent(unsigned flags) { CUDA_CHECK(cudaEventCreateWithFlags(&e, flags)); }
    ~CudaEvent() { if (e) cudaEventDestroy(e); }
    CudaEvent(CudaEvent&& o) noexcept : e(o.e) { o.e = nullptr; }
    CudaEvent& operator=(CudaEvent&& o) noexcept {
        if (this != &o) { if (e) cudaEventDestroy(e); e = o.e; o.e = nullptr; }
        return *this;
    }
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    operator cudaEvent_t() const { return e; }
    void record(cudaStream_t stream = nullptr) { CUDA_CHECK(cudaEventRecord(e, stream)); }
    void sync()   const { CUDA_CHECK(cudaEventSynchronize(e)); }
    float elapsed(const CudaEvent& start) const {
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start.e, e));
        return ms;
    }
};

/// RAII wrapper for cublasHandle_t.
struct CublasHandle {
    cublasHandle_t h = nullptr;
    CublasHandle()  { CUBLAS_CHECK(cublasCreate(&h)); }
    ~CublasHandle() { if (h) cublasDestroy(h); }
    CublasHandle(CublasHandle&& o) noexcept : h(o.h) { o.h = nullptr; }
    CublasHandle& operator=(CublasHandle&& o) noexcept {
        if (this != &o) { if (h) cublasDestroy(h); h = o.h; o.h = nullptr; }
        return *this;
    }
    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;
    operator cublasHandle_t() const { return h; }
    void set_stream(cudaStream_t s) { CUBLAS_CHECK(cublasSetStream(h, s)); }
};

// ---------------------------------------------------------------------------
//  Utility helpers
// ---------------------------------------------------------------------------

/// Ceiling division.
inline constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

/// Align value up to the nearest multiple of alignment.
inline constexpr size_t align_up(size_t value, size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

/// Print GPU device info banner.
inline void print_device_info(int device = 0) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("┌─────────────────────────────────────────────────┐\n");
    printf("│  GPU: %-42s│\n", prop.name);
    printf("│  VRAM: %.1f GB    SMs: %-3d   Compute: %d.%d       │\n",
           prop.totalGlobalMem / 1073741824.0, prop.multiProcessorCount,
           prop.major, prop.minor);
    printf("│  Max threads/block: %-4d   Shared mem: %zu KB     │\n",
           prop.maxThreadsPerBlock, prop.sharedMemPerBlock / 1024);
    printf("└─────────────────────────────────────────────────┘\n");
}

/// Returns total and free GPU memory in bytes.
inline std::pair<size_t, size_t> gpu_memory_info() {
    size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    return {total_bytes, free_bytes};
}
