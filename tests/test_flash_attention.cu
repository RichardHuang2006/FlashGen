/*
 * test_flash_attention.cu
 *
 * Correctness and performance tests for the FlashAttention kernel.
 *
 * Test 1: Numerical correctness against a naive O(N^2) attention reference.
 *   - Tolerance: max absolute error < 1e-3 (FP32)
 *
 * Test 2: Causal masking correctness.
 *   - Verify that output[i] doesn't depend on any token j > i.
 *
 * Test 3: FP16 vs FP32 precision comparison.
 *
 * Test 4: Throughput measurement (tokens/s, effective FLOP/s).
 */

#include "flash_attention.cuh"
#include "cuda_utils.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

using namespace flashgen;

// -- Naive reference attention (CPU) -----------------------------------------

static void naive_attention_cpu(
    const float* Q, const float* K, const float* V, float* O,
    int batch, int n_heads, int seq, int head_dim,
    float scale, bool causal
) {
    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < n_heads; h++) {
            for (int qi = 0; qi < seq; qi++) {
                // Compute scores
                std::vector<float> s(seq);
                for (int ki = 0; ki < seq; ki++) {
                    if (causal && ki > qi) { s[ki] = -1e9f; continue; }
                    float dot = 0.f;
                    for (int d = 0; d < head_dim; d++) {
                        const long long base = ((long long)b * n_heads + h) * seq * head_dim;
                        dot += Q[base + qi * head_dim + d] * K[base + ki * head_dim + d];
                    }
                    s[ki] = dot * scale;
                }
                // Softmax
                float m = *std::max_element(s.begin(), s.end());
                float z = 0.f;
                for (float& si : s) { si = std::exp(si - m); z += si; }
                for (float& si : s) si /= z;
                // Weighted sum of V
                for (int d = 0; d < head_dim; d++) {
                    const long long base = ((long long)b * n_heads + h) * seq * head_dim;
                    float acc = 0.f;
                    for (int vi = 0; vi < seq; vi++)
                        acc += s[vi] * V[base + vi * head_dim + d];
                    O[base + qi * head_dim + d] = acc;
                }
            }
        }
    }
}

// -- Test helpers -------------------------------------------------------------

static float max_abs_error(const float* a, const float* b, size_t n) {
    float err = 0.f;
    for (size_t i = 0; i < n; i++) err = std::max(err, std::abs(a[i] - b[i]));
    return err;
}

static float mean_abs_error(const float* a, const float* b, size_t n) {
    float s = 0.f;
    for (size_t i = 0; i < n; i++) s += std::abs(a[i] - b[i]);
    return s / (float)n;
}

static void fill_random(float* buf, size_t n, float scale = 0.1f) {
    static std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.f, scale);
    for (size_t i = 0; i < n; i++) buf[i] = dist(rng);
}

static bool check(bool cond, const char* msg) {
    if (!cond) {
        printf("  FAIL: %s\n", msg);
        return false;
    }
    printf("  PASS: %s\n", msg);
    return true;
}

// ============================================================================
//  Test 1: Correctness vs. naive attention (FP32, non-causal)
// ============================================================================

static bool test_correctness_fp32() {
    printf("\n[Test 1] FlashAttention FP32 correctness (non-causal)\n");
    bool all_pass = true;

    struct Case { int batch, heads, seq, hd; };
    std::vector<Case> cases = {
        {1, 1,  64,  64},
        {1, 4, 128,  64},
        {2, 8, 256,  64},
        {1, 1, 512, 128},
        {4, 12, 64,  64},
    };

    for (auto& c : cases) {
        const size_t n = (size_t)c.batch * c.heads * c.seq * c.hd;
        std::vector<float> hQ(n), hK(n), hV(n), hO_ref(n), hO_flash(n);
        fill_random(hQ.data(), n);
        fill_random(hK.data(), n);
        fill_random(hV.data(), n);

        // CPU reference
        naive_attention_cpu(hQ.data(), hK.data(), hV.data(), hO_ref.data(),
                            c.batch, c.heads, c.seq, c.hd,
                            1.f / sqrtf((float)c.hd), false);

        // GPU FlashAttention
        DeviceBuffer<float> dQ(n), dK(n), dV(n), dO(n);
        CUDA_CHECK(cudaMemcpy(dQ.ptr, hQ.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dK.ptr, hK.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dV.ptr, hV.data(), n * sizeof(float), cudaMemcpyHostToDevice));

        flash_attention_forward(dQ.ptr, dK.ptr, dV.ptr, dO.ptr, nullptr,
                                c.batch, c.heads, c.seq, c.hd,
                                1.f / sqrtf((float)c.hd), false, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(hO_flash.data(), dO.ptr, n * sizeof(float), cudaMemcpyDeviceToHost));

        float max_err  = max_abs_error(hO_ref.data(), hO_flash.data(), n);
        float mean_err = mean_abs_error(hO_ref.data(), hO_flash.data(), n);

        char msg[128];
        snprintf(msg, sizeof(msg),
                 "batch=%d heads=%d seq=%d hd=%d  max_err=%.2e  mean_err=%.2e",
                 c.batch, c.heads, c.seq, c.hd, max_err, mean_err);
        all_pass &= check(max_err < 1e-3f, msg);
    }
    return all_pass;
}

// ============================================================================
//  Test 2: Causal masking correctness
// ============================================================================

static bool test_causal_masking() {
    printf("\n[Test 2] Causal masking correctness\n");

    const int batch = 1, heads = 4, seq = 128, hd = 64;
    const size_t n = (size_t)batch * heads * seq * hd;
    std::vector<float> hQ(n), hK(n), hV(n), hO_causal(n), hO_ref(n);

    fill_random(hQ.data(), n);
    fill_random(hK.data(), n);
    fill_random(hV.data(), n);

    naive_attention_cpu(hQ.data(), hK.data(), hV.data(), hO_ref.data(),
                        batch, heads, seq, hd, 1.f / sqrtf((float)hd), true);

    DeviceBuffer<float> dQ(n), dK(n), dV(n), dO(n);
    CUDA_CHECK(cudaMemcpy(dQ.ptr, hQ.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK.ptr, hK.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV.ptr, hV.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    flash_attention_forward(dQ.ptr, dK.ptr, dV.ptr, dO.ptr, nullptr,
                            batch, heads, seq, hd,
                            1.f / sqrtf((float)hd), true, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hO_causal.data(), dO.ptr, n * sizeof(float), cudaMemcpyDeviceToHost));

    float max_err = max_abs_error(hO_ref.data(), hO_causal.data(), n);
    char msg[64];
    snprintf(msg, sizeof(msg), "max_err=%.2e", max_err);
    return check(max_err < 1e-3f, msg);
}

// ============================================================================
//  Test 3: FP16 precision
// ============================================================================

static bool test_fp16_precision() {
    printf("\n[Test 3] FP16 precision vs FP32 reference\n");

    const int batch = 1, heads = 8, seq = 256, hd = 64;
    const size_t n = (size_t)batch * heads * seq * hd;

    std::vector<float> hQf(n), hKf(n), hVf(n), hOf_ref(n), hOf_fp16(n);
    fill_random(hQf.data(), n, 0.05f); // small values for better FP16 repr
    fill_random(hKf.data(), n, 0.05f);
    fill_random(hVf.data(), n, 0.05f);

    // FP32 reference on GPU
    DeviceBuffer<float> dQf(n), dKf(n), dVf(n), dOf(n);
    CUDA_CHECK(cudaMemcpy(dQf.ptr, hQf.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dKf.ptr, hKf.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dVf.ptr, hVf.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    flash_attention_forward(dQf.ptr, dKf.ptr, dVf.ptr, dOf.ptr, nullptr,
                            batch, heads, seq, hd, 1.f/sqrtf((float)hd), false, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hOf_ref.data(), dOf.ptr, n * sizeof(float), cudaMemcpyDeviceToHost));

    // FP16 on GPU
    std::vector<__half> hQh(n), hKh(n), hVh(n), hOh(n);
    for (size_t i = 0; i < n; i++) {
        hQh[i] = __float2half(hQf[i]);
        hKh[i] = __float2half(hKf[i]);
        hVh[i] = __float2half(hVf[i]);
    }
    DeviceBuffer<__half> dQh(n), dKh(n), dVh(n), dOh(n);
    CUDA_CHECK(cudaMemcpy(dQh.ptr, hQh.data(), n * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dKh.ptr, hKh.data(), n * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dVh.ptr, hVh.data(), n * sizeof(__half), cudaMemcpyHostToDevice));

    flash_attention_forward_fp16(dQh.ptr, dKh.ptr, dVh.ptr, dOh.ptr, nullptr,
                                 batch, heads, seq, hd, 1.f/sqrtf((float)hd), false, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hOh.data(), dOh.ptr, n * sizeof(__half), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < n; i++) hOf_fp16[i] = __half2float(hOh[i]);

    float max_err  = max_abs_error(hOf_ref.data(), hOf_fp16.data(), n);
    float mean_err = mean_abs_error(hOf_ref.data(), hOf_fp16.data(), n);
    char msg[64];
    snprintf(msg, sizeof(msg), "FP16 vs FP32: max_err=%.2e mean_err=%.2e", max_err, mean_err);
    // FP16 has ~3 decimal digits; tolerate slightly larger error
    return check(max_err < 5e-3f, msg);
}

// ============================================================================
//  Test 4: Throughput benchmark
// ============================================================================

static void test_throughput() {
    printf("\n[Test 4] FlashAttention throughput\n");

    struct Case { int batch, heads, seq, hd; const char* label; };
    std::vector<Case> cases = {
        {1,  12, 512,  64, "GPT-2 small  seq=512"},
        {1,  12,1024,  64, "GPT-2 small  seq=1024"},
        {4,  16, 512,  64, "GPT-2 medium seq=512  bs=4"},
        {1,  25,1024,  64, "GPT-2 XL     seq=1024"},
    };

    for (auto& c : cases) {
        const size_t n = (size_t)c.batch * c.heads * c.seq * c.hd;
        DeviceBuffer<float> dQ(n), dK(n), dV(n), dO(n);

        // Warmup
        for (int i = 0; i < 3; i++) {
            flash_attention_forward(dQ.ptr, dK.ptr, dV.ptr, dO.ptr, nullptr,
                                    c.batch, c.heads, c.seq, c.hd,
                                    1.f / sqrtf((float)c.hd), true, 0);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        const int N = 20;
        CudaEvent start, end;
        start.record();
        for (int i = 0; i < N; i++) {
            flash_attention_forward(dQ.ptr, dK.ptr, dV.ptr, dO.ptr, nullptr,
                                    c.batch, c.heads, c.seq, c.hd,
                                    1.f / sqrtf((float)c.hd), true, 0);
        }
        end.record();
        CUDA_CHECK(cudaDeviceSynchronize());
        float ms = end.elapsed_ms(start) / N;

        // FLOP count: 4 * B * H * N^2 * d  (two GEMMs, softmax is negligible)
        double flops = 4.0 * c.batch * c.heads * (double)c.seq * c.seq * c.hd;
        double tflops = (flops / (ms * 1e-3)) / 1e12;
        printf("  %-35s  %6.2f ms  %5.2f TFLOP/s\n", c.label, ms, tflops);
    }
}

// -- Entry point --------------------------------------------------------------

int main() {
    printf("==========================================\n");
    printf("  FlashAttention Test Suite\n");
    printf("==========================================\n");

    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("Device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    bool pass = true;
    pass &= test_correctness_fp32();
    pass &= test_causal_masking();
    pass &= test_fp16_precision();
    test_throughput();

    printf("\n==========================================\n");
    printf("  Result: %s\n", pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    printf("==========================================\n");

    return pass ? 0 : 1;
}
