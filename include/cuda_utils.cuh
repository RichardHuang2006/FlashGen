#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

// ── Error-checking macros ────────────────────────────────────────────────────
#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t _e = (expr);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "[CUDA] %s:%d  %s\n",                              \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                 \
            std::abort();                                                       \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(expr)                                                      \
    do {                                                                        \
        cublasStatus_t _s = (expr);                                             \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                      \
            fprintf(stderr, "[cuBLAS] %s:%d  status=%d\n",                     \
                    __FILE__, __LINE__, (int)_s);                               \
            std::abort();                                                       \
        }                                                                       \
    } while (0)

// ── CUDA device query helpers ────────────────────────────────────────────────
namespace flashgen {

inline int get_sm_count(int device = 0) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    return prop.multiProcessorCount;
}

inline size_t get_free_vram_bytes(int device = 0) {
    CUDA_CHECK(cudaSetDevice(device));
    size_t free_bytes, total_bytes;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    return free_bytes;
}

// ── Pinned host memory RAII ──────────────────────────────────────────────────
template<typename T>
struct PinnedBuffer {
    T*     ptr  = nullptr;
    size_t size = 0;

    explicit PinnedBuffer(size_t n) : size(n) {
        CUDA_CHECK(cudaMallocHost(&ptr, n * sizeof(T)));
    }
    ~PinnedBuffer() {
        if (ptr) cudaFreeHost(ptr);
    }
    PinnedBuffer(const PinnedBuffer&)            = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    T& operator[](size_t i)       { return ptr[i]; }
    const T& operator[](size_t i) const { return ptr[i]; }
};

// ── Device memory RAII ───────────────────────────────────────────────────────
template<typename T>
struct DeviceBuffer {
    T*     ptr  = nullptr;
    size_t size = 0;

    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t n) : size(n) {
        CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T)));
        CUDA_CHECK(cudaMemset(ptr, 0, n * sizeof(T)));
    }
    ~DeviceBuffer() {
        if (ptr) cudaFree(ptr);
    }
    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    // Move semantics
    DeviceBuffer(DeviceBuffer&& o) noexcept : ptr(o.ptr), size(o.size) {
        o.ptr  = nullptr;
        o.size = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) {
            if (ptr) cudaFree(ptr);
            ptr    = o.ptr;
            size   = o.size;
            o.ptr  = nullptr;
            o.size = 0;
        }
        return *this;
    }

    void resize(size_t n) {
        if (ptr) cudaFree(ptr);
        size = n;
        CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T)));
        CUDA_CHECK(cudaMemset(ptr, 0, n * sizeof(T)));
    }

    size_t bytes() const { return size * sizeof(T); }
};

// ── CUDA Event RAII ──────────────────────────────────────────────────────────
struct CudaEvent {
    cudaEvent_t ev;
    explicit CudaEvent(unsigned flags = cudaEventDefault) {
        CUDA_CHECK(cudaEventCreateWithFlags(&ev, flags));
    }
    ~CudaEvent() { cudaEventDestroy(ev); }
    void record(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(ev, stream));
    }
    void synchronize() { CUDA_CHECK(cudaEventSynchronize(ev)); }
    float elapsed_ms(const CudaEvent& start) const {
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start.ev, ev));
        return ms;
    }
};

// ── CUDA Stream RAII ─────────────────────────────────────────────────────────
struct CudaStream {
    cudaStream_t stream;
    CudaStream() { CUDA_CHECK(cudaStreamCreate(&stream)); }
    ~CudaStream() { cudaStreamDestroy(stream); }
    void synchronize() { CUDA_CHECK(cudaStreamSynchronize(stream)); }
    operator cudaStream_t() const { return stream; }
};

// ── Kernel launch bounds helper ──────────────────────────────────────────────
inline dim3 grid1d(int n, int block) {
    return dim3((n + block - 1) / block);
}

} // namespace flashgen
