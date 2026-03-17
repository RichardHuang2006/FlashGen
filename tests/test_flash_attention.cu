#include "flash_attention.cuh"
#include "cuda_utils.cuh"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) printf("  %-50s", name); fflush(stdout)
#define PASS() do { printf("[PASS]\n"); ++tests_passed; } while(0)
#define FAIL(msg) do { printf("[FAIL] %s\n", msg); ++tests_failed; } while(0)

// Naive reference attention (CPU)
static void ref_attention(const float* Q, const float* K, const float* V,
                          float* O, int N, int Hd, bool causal) {
    float scale = 1.0f / sqrtf((float)Hd);
    for (int i = 0; i < N; ++i) {
        float max_s = -1e30f;
        int kv_end = causal ? i + 1 : N;
        for (int j = 0; j < kv_end; ++j) {
            float s = 0.f;
            for (int d = 0; d < Hd; ++d)
                s += Q[i * Hd + d] * K[j * Hd + d];
            s *= scale;
            if (s > max_s) max_s = s;
        }
        float sum = 0.f;
        std::vector<float> w(N, 0.f);
        for (int j = 0; j < kv_end; ++j) {
            float s = 0.f;
            for (int d = 0; d < Hd; ++d)
                s += Q[i * Hd + d] * K[j * Hd + d];
            w[j] = expf(s * scale - max_s);
            sum += w[j];
        }
        for (int d = 0; d < Hd; ++d) {
            float val = 0.f;
            for (int j = 0; j < kv_end; ++j)
                val += (w[j] / sum) * V[j * Hd + d];
            O[i * Hd + d] = val;
        }
    }
}

// ---------------------------------------------------------------------------
//  Tests
// ---------------------------------------------------------------------------

void test_correctness_causal() {
    TEST("FlashAttention causal correctness");

    int B = 1, H = 4, N = 128, Hd = 64;
    int total = B * H * N * Hd;

    std::vector<float> hQ(total), hK(total), hV(total), hO(total), hRef(total);
    srand(123);
    for (int i = 0; i < total; ++i) {
        hQ[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.3f;
        hK[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.3f;
        hV[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.3f;
    }

    DeviceBuffer<float> dQ(total), dK(total), dV(total), dO(total);
    CUDA_CHECK(cudaMemcpy(dQ, hQ.data(), total * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, hK.data(), total * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, hV.data(), total * 4, cudaMemcpyHostToDevice));

    CudaStream s;
    flash_attention::forward(dQ, dK, dV, dO, nullptr, B, H, N, Hd, true, s);
    s.sync();
    CUDA_CHECK(cudaMemcpy(hO.data(), dO, total * 4, cudaMemcpyDeviceToHost));

    for (int b = 0; b < B; ++b)
        for (int h = 0; h < H; ++h) {
            int off = (b * H + h) * N * Hd;
            ref_attention(hQ.data() + off, hK.data() + off,
                          hV.data() + off, hRef.data() + off, N, Hd, true);
        }

    float max_err = 0.f;
    for (int i = 0; i < total; ++i)
        max_err = fmaxf(max_err, fabsf(hO[i] - hRef[i]));

    if (max_err < 1e-2f) {
        printf("(err=%.6f) ", max_err);
        PASS();
    } else {
        char buf[64]; snprintf(buf, 64, "err=%.4f", max_err);
        FAIL(buf);
    }
}

void test_correctness_noncausal() {
    TEST("FlashAttention non-causal correctness");

    int B = 1, H = 2, N = 64, Hd = 64;
    int total = B * H * N * Hd;

    std::vector<float> hQ(total), hK(total), hV(total), hO(total), hRef(total);
    srand(456);
    for (int i = 0; i < total; ++i) {
        hQ[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.3f;
        hK[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.3f;
        hV[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.3f;
    }

    DeviceBuffer<float> dQ(total), dK(total), dV(total), dO(total);
    CUDA_CHECK(cudaMemcpy(dQ, hQ.data(), total * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, hK.data(), total * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, hV.data(), total * 4, cudaMemcpyHostToDevice));

    CudaStream s;
    flash_attention::forward(dQ, dK, dV, dO, nullptr, B, H, N, Hd, false, s);
    s.sync();
    CUDA_CHECK(cudaMemcpy(hO.data(), dO, total * 4, cudaMemcpyDeviceToHost));

    for (int b = 0; b < B; ++b)
        for (int h = 0; h < H; ++h) {
            int off = (b * H + h) * N * Hd;
            ref_attention(hQ.data() + off, hK.data() + off,
                          hV.data() + off, hRef.data() + off, N, Hd, false);
        }

    float max_err = 0.f;
    for (int i = 0; i < total; ++i)
        max_err = fmaxf(max_err, fabsf(hO[i] - hRef[i]));

    if (max_err < 1e-2f) {
        printf("(err=%.6f) ", max_err);
        PASS();
    } else {
        char buf[64]; snprintf(buf, 64, "err=%.4f", max_err);
        FAIL(buf);
    }
}

void test_decode_kernel() {
    TEST("FlashAttention decode (single query)");

    int B = 2, H = 4, KV = 128, Hd = 64;
    int q_total = B * H * 1 * Hd;
    int kv_total = B * H * KV * Hd;

    std::vector<float> hQ(q_total), hK(kv_total), hV(kv_total), hO(q_total);
    srand(789);
    for (int i = 0; i < q_total; ++i)
        hQ[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.3f;
    for (int i = 0; i < kv_total; ++i) {
        hK[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.3f;
        hV[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.3f;
    }

    DeviceBuffer<float> dQ(q_total), dK(kv_total), dV(kv_total), dO(q_total);
    CUDA_CHECK(cudaMemcpy(dQ, hQ.data(), q_total * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, hK.data(), kv_total * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, hV.data(), kv_total * 4, cudaMemcpyHostToDevice));

    CudaStream s;
    flash_attention::decode(dQ, dK, dV, dO, B, H, KV, Hd, s);
    s.sync();
    CUDA_CHECK(cudaMemcpy(hO.data(), dO, q_total * 4, cudaMemcpyDeviceToHost));

    // Verify output is finite and reasonable
    bool ok = true;
    for (int i = 0; i < q_total; ++i) {
        if (!std::isfinite(hO[i])) { ok = false; break; }
    }

    if (ok) PASS(); else FAIL("non-finite output");
}

void test_different_head_dims() {
    TEST("FlashAttention head_dim=128");

    int B = 1, H = 2, N = 64, Hd = 128;
    int total = B * H * N * Hd;

    std::vector<float> hQ(total), hK(total), hV(total), hO(total), hRef(total);
    srand(999);
    for (int i = 0; i < total; ++i) {
        hQ[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.2f;
        hK[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.2f;
        hV[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.2f;
    }

    DeviceBuffer<float> dQ(total), dK(total), dV(total), dO(total);
    CUDA_CHECK(cudaMemcpy(dQ, hQ.data(), total * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, hK.data(), total * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, hV.data(), total * 4, cudaMemcpyHostToDevice));

    CudaStream s;
    flash_attention::forward(dQ, dK, dV, dO, nullptr, B, H, N, Hd, true, s);
    s.sync();
    CUDA_CHECK(cudaMemcpy(hO.data(), dO, total * 4, cudaMemcpyDeviceToHost));

    for (int b = 0; b < B; ++b)
        for (int h = 0; h < H; ++h) {
            int off = (b * H + h) * N * Hd;
            ref_attention(hQ.data() + off, hK.data() + off,
                          hV.data() + off, hRef.data() + off, N, Hd, true);
        }

    float max_err = 0.f;
    for (int i = 0; i < total; ++i)
        max_err = fmaxf(max_err, fabsf(hO[i] - hRef[i]));

    if (max_err < 5e-2f) {
        printf("(err=%.6f) ", max_err);
        PASS();
    } else {
        char buf[64]; snprintf(buf, 64, "err=%.4f", max_err);
        FAIL(buf);
    }
}

void test_throughput_benchmark() {
    TEST("FlashAttention throughput benchmark");

    int B = 4, H = 12, N = 512, Hd = 64;
    int total = B * H * N * Hd;
    int runs = 5;

    DeviceBuffer<float> dQ(total), dK(total), dV(total), dO(total);
    CUDA_CHECK(cudaMemset(dQ, 0, total * 4));
    CUDA_CHECK(cudaMemset(dK, 0, total * 4));
    CUDA_CHECK(cudaMemset(dV, 0, total * 4));

    CudaStream s;
    // Warmup
    flash_attention::forward(dQ, dK, dV, dO, nullptr, B, H, N, Hd, true, s);
    s.sync();

    CudaEvent start, stop;
    start.record(s);
    for (int r = 0; r < runs; ++r) {
        flash_attention::forward(dQ, dK, dV, dO, nullptr, B, H, N, Hd, true, s);
    }
    stop.record(s);
    s.sync();

    float ms = stop.elapsed(start) / runs;
    // FLOPS: 4 * B * H * N * N * Hd (Q*K + softmax + attn*V)
    double flops = 4.0 * B * H * (double)N * N * Hd;
    double tflops = flops / (ms * 1e9);

    printf("(%.2f ms, %.2f TFLOP/s) ", ms, tflops);
    PASS();
}

int main() {
    printf("═══ FlashAttention Tests ═══\n\n");

    test_correctness_causal();
    test_correctness_noncausal();
    test_decode_kernel();
    test_different_head_dims();
    test_throughput_benchmark();

    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
