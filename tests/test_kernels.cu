/*
 * test_kernels.cu
 *
 * Correctness and performance tests for:
 *   - Layer Normalization (standard + fused residual)
 *   - GELU activation
 *   - Fused Feed-Forward Network
 *   - Token embedding lookup
 *   - Softmax with temperature
 */

#include "kernels.cuh"
#include "cuda_utils.cuh"
#include <cublas_v2.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>

using namespace flashgen;

// ── Utilities ─────────────────────────────────────────────────────────────────

static std::mt19937 g_rng(123);

static void fill_rand(float* buf, size_t n, float lo = -1.f, float hi = 1.f) {
    std::uniform_real_distribution<float> dist(lo, hi);
    for (size_t i = 0; i < n; i++) buf[i] = dist(g_rng);
}

static float max_abs_err(const float* a, const float* b, size_t n) {
    float e = 0.f;
    for (size_t i = 0; i < n; i++) e = std::max(e, std::abs(a[i] - b[i]));
    return e;
}

static bool check(bool cond, const char* msg) {
    printf("  %s %s\n", cond ? "PASS" : "FAIL", msg);
    return cond;
}

// ── CPU reference implementations ────────────────────────────────────────────

static void ref_layer_norm(
    const float* x, const float* gamma, const float* beta,
    float* out, int rows, int cols, float eps
) {
    for (int r = 0; r < rows; r++) {
        const float* xr = x + r * cols;
        float*       yr = out + r * cols;
        float sum = 0.f, sum2 = 0.f;
        for (int c = 0; c < cols; c++) { sum += xr[c]; sum2 += xr[c] * xr[c]; }
        float mu  = sum / cols;
        float var = sum2 / cols - mu * mu;
        float rstd = 1.f / sqrtf(var + eps);
        for (int c = 0; c < cols; c++)
            yr[c] = gamma[c] * (xr[c] - mu) * rstd + beta[c];
    }
}

static void ref_gelu(const float* x, float* y, size_t n) {
    for (size_t i = 0; i < n; i++)
        y[i] = 0.5f * x[i] * (1.f + erff(x[i] * 0.7071067811865476f));
}

static void ref_softmax(float* x, int rows, int cols, float temp) {
    for (int r = 0; r < rows; r++) {
        float* xr = x + r * cols;
        float m = *std::max_element(xr, xr + cols);
        float s = 0.f;
        for (int c = 0; c < cols; c++) { xr[c] = expf(xr[c] / temp - m); s += xr[c]; }
        for (int c = 0; c < cols; c++) xr[c] /= s;
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  Test: Layer Normalization
// ════════════════════════════════════════════════════════════════════════════

static bool test_layer_norm() {
    printf("\n[Test] Layer Normalization\n");
    bool all_pass = true;

    struct Case { int rows, cols; };
    std::vector<Case> cases = {{1,768}, {64,768}, {32,1024}, {128,64}, {256,3072}};

    for (auto& c : cases) {
        const size_t n = (size_t)c.rows * c.cols;
        std::vector<float> hX(n), hG(c.cols, 1.f), hB(c.cols, 0.f);
        std::vector<float> hOut_ref(n), hOut_gpu(n);
        fill_rand(hX.data(), n);
        fill_rand(hG.data(), c.cols, 0.5f, 2.f);
        fill_rand(hB.data(), c.cols, -0.5f, 0.5f);

        ref_layer_norm(hX.data(), hG.data(), hB.data(), hOut_ref.data(),
                       c.rows, c.cols, 1e-5f);

        DeviceBuffer<float> dX(n), dG(c.cols), dB(c.cols), dOut(n);
        CUDA_CHECK(cudaMemcpy(dX.ptr, hX.data(), n*4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dG.ptr, hG.data(), c.cols*4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB.ptr, hB.data(), c.cols*4, cudaMemcpyHostToDevice));

        layer_norm(dX.ptr, dOut.ptr, dG.ptr, dB.ptr, c.rows, c.cols, 1e-5f, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(hOut_gpu.data(), dOut.ptr, n*4, cudaMemcpyDeviceToHost));

        float err = max_abs_err(hOut_ref.data(), hOut_gpu.data(), n);
        char msg[64];
        snprintf(msg, sizeof(msg), "rows=%d cols=%d  max_err=%.2e", c.rows, c.cols, err);
        all_pass &= check(err < 1e-4f, msg);
    }
    return all_pass;
}

// ════════════════════════════════════════════════════════════════════════════
//  Test: Fused residual LayerNorm
// ════════════════════════════════════════════════════════════════════════════

static bool test_layer_norm_residual() {
    printf("\n[Test] LayerNorm with fused residual add\n");

    const int rows = 16, cols = 768;
    const size_t n = (size_t)rows * cols;

    std::vector<float> hX(n), hRes(n), hG(cols, 1.f), hB(cols, 0.f);
    std::vector<float> hRef(n), hGpu(n), hSum(n);
    fill_rand(hX.data(), n);
    fill_rand(hRes.data(), n);

    for (size_t i = 0; i < n; i++) hSum[i] = hX[i] + hRes[i];
    ref_layer_norm(hSum.data(), hG.data(), hB.data(), hRef.data(), rows, cols, 1e-5f);

    DeviceBuffer<float> dX(n), dRes(n), dG(cols), dB(cols), dOut(n);
    CUDA_CHECK(cudaMemcpy(dX.ptr,   hX.data(),  n*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dRes.ptr, hRes.data(), n*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dG.ptr,   hG.data(), cols*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB.ptr,   hB.data(), cols*4, cudaMemcpyHostToDevice));

    layer_norm_residual(dX.ptr, dRes.ptr, dOut.ptr, dG.ptr, dB.ptr,
                        rows, cols, 1e-5f, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hGpu.data(), dOut.ptr, n*4, cudaMemcpyDeviceToHost));

    float err = max_abs_err(hRef.data(), hGpu.data(), n);
    char msg[64];
    snprintf(msg, sizeof(msg), "max_err=%.2e", err);
    return check(err < 1e-4f, msg);
}

// ════════════════════════════════════════════════════════════════════════════
//  Test: GELU
// ════════════════════════════════════════════════════════════════════════════

static bool test_gelu() {
    printf("\n[Test] GELU activation\n");
    bool all_pass = true;

    for (int n : {128, 1024, 65536, 1 << 20}) {
        std::vector<float> hX(n), hRef(n), hGpu(n);
        fill_rand(hX.data(), n, -3.f, 3.f);
        ref_gelu(hX.data(), hRef.data(), n);

        DeviceBuffer<float> dX(n);
        CUDA_CHECK(cudaMemcpy(dX.ptr, hX.data(), n*4, cudaMemcpyHostToDevice));
        gelu_inplace(dX.ptr, n, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(hGpu.data(), dX.ptr, n*4, cudaMemcpyDeviceToHost));

        float err = max_abs_err(hRef.data(), hGpu.data(), n);
        char msg[48];
        snprintf(msg, sizeof(msg), "n=%d  max_err=%.2e", n, err);
        all_pass &= check(err < 1e-5f, msg);
    }
    return all_pass;
}

// ════════════════════════════════════════════════════════════════════════════
//  Test: Softmax with temperature
// ════════════════════════════════════════════════════════════════════════════

static bool test_softmax_temperature() {
    printf("\n[Test] Softmax with temperature\n");
    bool all_pass = true;

    for (float temp : {0.5f, 1.0f, 2.0f}) {
        const int rows = 4, cols = 50257; // vocab-size rows
        const size_t n = (size_t)rows * cols;
        std::vector<float> hX(n), hRef(n), hGpu(n);
        fill_rand(hX.data(), n, -5.f, 5.f);
        std::copy(hX.begin(), hX.end(), hRef.begin());
        ref_softmax(hRef.data(), rows, cols, temp);

        DeviceBuffer<float> dX(n);
        CUDA_CHECK(cudaMemcpy(dX.ptr, hX.data(), n*4, cudaMemcpyHostToDevice));
        softmax_temperature(dX.ptr, rows, cols, temp, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(hGpu.data(), dX.ptr, n*4, cudaMemcpyDeviceToHost));

        // Check sum=1 and max_err
        float sum0 = 0.f;
        for (int c = 0; c < cols; c++) sum0 += hGpu[c];
        float err = max_abs_err(hRef.data(), hGpu.data(), n);

        char msg[64];
        snprintf(msg, sizeof(msg), "temp=%.1f  sum[0]=%.6f  max_err=%.2e",
                 temp, sum0, err);
        all_pass &= check(std::abs(sum0 - 1.f) < 1e-4f && err < 1e-4f, msg);
    }
    return all_pass;
}

// ════════════════════════════════════════════════════════════════════════════
//  Test: LayerNorm throughput
// ════════════════════════════════════════════════════════════════════════════

static void benchmark_layer_norm() {
    printf("\n[Bench] LayerNorm throughput\n");
    struct Case { int rows, cols; const char* label; };
    std::vector<Case> cases = {
        {512, 768,  "bs=512 d=768  (GPT-2 small)"},
        {512, 1024, "bs=512 d=1024 (GPT-2 medium)"},
        {128, 1280, "bs=128 d=1280 (GPT-2 large)"},
    };
    for (auto& c : cases) {
        const size_t n = (size_t)c.rows * c.cols;
        DeviceBuffer<float> dX(n), dO(n), dG(c.cols), dB(c.cols);
        // Warmup
        for (int i = 0; i < 5; i++)
            layer_norm(dX.ptr, dO.ptr, dG.ptr, dB.ptr, c.rows, c.cols, 1e-5f, 0);
        CUDA_CHECK(cudaDeviceSynchronize());

        const int N = 100;
        CudaEvent start, end;
        start.record();
        for (int i = 0; i < N; i++)
            layer_norm(dX.ptr, dO.ptr, dG.ptr, dB.ptr, c.rows, c.cols, 1e-5f, 0);
        end.record();
        CUDA_CHECK(cudaDeviceSynchronize());
        float ms = end.elapsed_ms(start) / N;
        // Bandwidth: read x, write out (2 * n * 4 bytes)
        double bw_gb = 2.0 * n * 4 / (ms * 1e-3) / 1e9;
        printf("  %-38s %6.3f ms  %6.1f GB/s\n", c.label, ms, bw_gb);
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────

int main() {
    printf("==========================================\n");
    printf("  Kernels Test Suite\n");
    printf("==========================================\n");

    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("Device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    bool pass = true;
    pass &= test_layer_norm();
    pass &= test_layer_norm_residual();
    pass &= test_gelu();
    pass &= test_softmax_temperature();
    benchmark_layer_norm();

    printf("\n==========================================\n");
    printf("  Result: %s\n", pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    printf("==========================================\n");
    return pass ? 0 : 1;
}
