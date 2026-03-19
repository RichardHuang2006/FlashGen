#include "kernels.cuh"
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

// CPU references
static void ref_layer_norm(const float* x, const float* g, const float* b,
                           float* out, int cols, float eps) {
    float mean = 0.f;
    for (int i = 0; i < cols; ++i) mean += x[i];
    mean /= cols;
    float var = 0.f;
    for (int i = 0; i < cols; ++i) var += (x[i] - mean) * (x[i] - mean);
    var /= cols;
    float inv = 1.f / sqrtf(var + eps);
    for (int i = 0; i < cols; ++i) {
        out[i] = g[i] * (x[i] - mean) * inv + b[i];
    }
}

static float ref_gelu(float x) {
    return 0.5f * x * (1.f + erff(x * 0.7071067811865476f));
}

// ---------------------------------------------------------------------------
//  Tests
// ---------------------------------------------------------------------------

void test_layer_norm() {
    TEST("LayerNorm correctness");

    int rows = 8, cols = 256;
    float eps = 1e-5f;

    std::vector<float> hx(rows * cols), hg(cols, 1.f), hb(cols, 0.f);
    std::vector<float> ho(rows * cols), href(rows * cols);
    srand(42);
    for (int i = 0; i < rows * cols; ++i)
        hx[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.f;

    DeviceBuffer<float> dx(rows * cols), dg(cols), db(cols), dout(rows * cols);
    CUDA_CHECK(cudaMemcpy(dx, hx.data(), rows * cols * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dg, hg.data(), cols * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, hb.data(), cols * 4, cudaMemcpyHostToDevice));

    CudaStream s;
    kernels::layer_norm(dx, dg, db, dout, rows, cols, eps, s);
    s.sync();
    CUDA_CHECK(cudaMemcpy(ho.data(), dout, rows * cols * 4, cudaMemcpyDeviceToHost));

    for (int r = 0; r < rows; ++r) {
        ref_layer_norm(hx.data() + r * cols, hg.data(), hb.data(),
                       href.data() + r * cols, cols, eps);
    }

    float max_err = 0.f;
    for (int i = 0; i < rows * cols; ++i)
        max_err = fmaxf(max_err, fabsf(ho[i] - href[i]));

    if (max_err < 1e-4f) {
        printf("(err=%.7f) ", max_err);
        PASS();
    } else {
        char buf[64]; snprintf(buf, 64, "err=%.5f", max_err);
        FAIL(buf);
    }
}

void test_layer_norm_residual() {
    TEST("LayerNorm with residual");

    int rows = 4, cols = 128;
    float eps = 1e-5f;

    std::vector<float> hx(rows * cols), hr(rows * cols);
    std::vector<float> hg(cols, 1.f), hb(cols, 0.f);
    std::vector<float> ho(rows * cols), href(rows * cols);
    srand(77);
    for (int i = 0; i < rows * cols; ++i) {
        hx[i] = ((float)rand() / RAND_MAX - 0.5f);
        hr[i] = ((float)rand() / RAND_MAX - 0.5f);
    }

    // Reference: LN(x + residual)
    std::vector<float> hsum(rows * cols);
    for (int i = 0; i < rows * cols; ++i) hsum[i] = hx[i] + hr[i];
    for (int r = 0; r < rows; ++r) {
        ref_layer_norm(hsum.data() + r * cols, hg.data(), hb.data(),
                       href.data() + r * cols, cols, eps);
    }

    DeviceBuffer<float> dx(rows * cols), dr(rows * cols), dg(cols), db(cols), dout(rows * cols);
    CUDA_CHECK(cudaMemcpy(dx, hx.data(), rows * cols * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dr, hr.data(), rows * cols * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dg, hg.data(), cols * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, hb.data(), cols * 4, cudaMemcpyHostToDevice));

    CudaStream s;
    kernels::layer_norm_residual(dx, dr, dg, db, dout, rows, cols, eps, s);
    s.sync();
    CUDA_CHECK(cudaMemcpy(ho.data(), dout, rows * cols * 4, cudaMemcpyDeviceToHost));

    float max_err = 0.f;
    for (int i = 0; i < rows * cols; ++i)
        max_err = fmaxf(max_err, fabsf(ho[i] - href[i]));

    if (max_err < 1e-4f) PASS();
    else { char buf[64]; snprintf(buf, 64, "err=%.5f", max_err); FAIL(buf); }
}

void test_gelu() {
    TEST("GELU activation");

    int n = 1024;
    std::vector<float> hx(n), href(n);
    srand(55);
    for (int i = 0; i < n; ++i) {
        hx[i] = ((float)rand() / RAND_MAX - 0.5f) * 4.f;
        href[i] = ref_gelu(hx[i]);
    }

    DeviceBuffer<float> dx(n);
    CUDA_CHECK(cudaMemcpy(dx, hx.data(), n * 4, cudaMemcpyHostToDevice));

    CudaStream s;
    kernels::gelu(dx, n, s);
    s.sync();

    CUDA_CHECK(cudaMemcpy(hx.data(), dx, n * 4, cudaMemcpyDeviceToHost));

    float max_err = 0.f;
    for (int i = 0; i < n; ++i)
        max_err = fmaxf(max_err, fabsf(hx[i] - href[i]));

    if (max_err < 1e-5f) PASS();
    else { char buf[64]; snprintf(buf, 64, "err=%.6f", max_err); FAIL(buf); }
}

void test_softmax_temperature() {
    TEST("Softmax with temperature");

    int rows = 4, cols = 100;
    float temp = 0.5f;

    std::vector<float> hx(rows * cols);
    srand(88);
    for (int i = 0; i < rows * cols; ++i)
        hx[i] = ((float)rand() / RAND_MAX - 0.5f) * 3.f;

    DeviceBuffer<float> dx(rows * cols);
    CUDA_CHECK(cudaMemcpy(dx, hx.data(), rows * cols * 4, cudaMemcpyHostToDevice));

    CudaStream s;
    kernels::softmax_temperature(dx, rows, cols, temp, s);
    s.sync();

    std::vector<float> ho(rows * cols);
    CUDA_CHECK(cudaMemcpy(ho.data(), dx, rows * cols * 4, cudaMemcpyDeviceToHost));

    // Check: rows should sum to ~1, all values >= 0
    bool ok = true;
    for (int r = 0; r < rows; ++r) {
        float sum = 0.f;
        for (int c = 0; c < cols; ++c) {
            float v = ho[r * cols + c];
            if (v < -1e-6f) { ok = false; break; }
            sum += v;
        }
        if (fabsf(sum - 1.f) > 1e-3f) ok = false;
    }

    if (ok) PASS(); else FAIL("row sums != 1 or negative values");
}

void test_argmax() {
    TEST("Argmax");

    int rows = 8, cols = 50;
    std::vector<float> hx(rows * cols, 0.f);
    std::vector<int> expected(rows);

    srand(33);
    for (int r = 0; r < rows; ++r) {
        int best = rand() % cols;
        expected[r] = best;
        for (int c = 0; c < cols; ++c)
            hx[r * cols + c] = (c == best) ? 10.f : -1.f;
    }

    DeviceBuffer<float> dx(rows * cols);
    DeviceBuffer<int> dout(rows);
    CUDA_CHECK(cudaMemcpy(dx, hx.data(), rows * cols * 4, cudaMemcpyHostToDevice));

    CudaStream s;
    kernels::argmax(dx, dout, rows, cols, s);
    s.sync();

    std::vector<int> ho(rows);
    CUDA_CHECK(cudaMemcpy(ho.data(), dout, rows * 4, cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int r = 0; r < rows; ++r) {
        if (ho[r] != expected[r]) { ok = false; break; }
    }

    if (ok) PASS(); else FAIL("incorrect argmax");
}

void test_rms_norm() {
    TEST("RMSNorm correctness");

    int rows = 4, cols = 256;
    float eps = 1e-5f;

    std::vector<float> hx(rows * cols), hg(cols, 1.f);
    std::vector<float> ho(rows * cols), href(rows * cols);
    srand(44);
    for (int i = 0; i < rows * cols; ++i)
        hx[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.f;

    // CPU reference
    for (int r = 0; r < rows; ++r) {
        float ss = 0.f;
        for (int c = 0; c < cols; ++c)
            ss += hx[r * cols + c] * hx[r * cols + c];
        float inv = 1.f / sqrtf(ss / cols + eps);
        for (int c = 0; c < cols; ++c)
            href[r * cols + c] = hg[c] * hx[r * cols + c] * inv;
    }

    DeviceBuffer<float> dx(rows * cols), dg(cols), dout(rows * cols);
    CUDA_CHECK(cudaMemcpy(dx, hx.data(), rows * cols * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dg, hg.data(), cols * 4, cudaMemcpyHostToDevice));

    CudaStream s;
    kernels::rms_norm(dx, dg, dout, rows, cols, eps, s);
    s.sync();
    CUDA_CHECK(cudaMemcpy(ho.data(), dout, rows * cols * 4, cudaMemcpyDeviceToHost));

    float max_err = 0.f;
    for (int i = 0; i < rows * cols; ++i)
        max_err = fmaxf(max_err, fabsf(ho[i] - href[i]));

    if (max_err < 1e-4f) {
        printf("(err=%.7f) ", max_err);
        PASS();
    } else {
        char buf[64]; snprintf(buf, 64, "err=%.5f", max_err);
        FAIL(buf);
    }
}

void test_silu() {
    TEST("SiLU activation");

    int n = 512;
    std::vector<float> hx(n), href(n);
    srand(66);
    for (int i = 0; i < n; ++i) {
        hx[i] = ((float)rand() / RAND_MAX - 0.5f) * 4.f;
        float v = hx[i];
        href[i] = v / (1.f + expf(-v));
    }

    DeviceBuffer<float> dx(n);
    CUDA_CHECK(cudaMemcpy(dx, hx.data(), n * 4, cudaMemcpyHostToDevice));

    CudaStream s;
    kernels::silu(dx, n, s);
    s.sync();

    CUDA_CHECK(cudaMemcpy(hx.data(), dx, n * 4, cudaMemcpyDeviceToHost));

    float max_err = 0.f;
    for (int i = 0; i < n; ++i)
        max_err = fmaxf(max_err, fabsf(hx[i] - href[i]));

    if (max_err < 1e-5f) PASS();
    else { char buf[64]; snprintf(buf, 64, "err=%.6f", max_err); FAIL(buf); }
}

int main() {
    printf("═══ Kernel Tests ═══\n\n");

    test_layer_norm();
    test_layer_norm_residual();
    test_gelu();
    test_silu();
    test_rms_norm();
    test_softmax_temperature();
    test_argmax();

    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
