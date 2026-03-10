/*
 * test_pipeline.cpp
 *
 * End-to-end pipeline tests:
 *
 *  Test 1 – Synchronous single request (verifies run_sync returns output).
 *  Test 2 – Async multi-request throughput (N concurrent submissions).
 *  Test 3 – Pipeline statistics accumulation (tokens generated counter).
 *  Test 4 – Shutdown under load (graceful drain without hang).
 *  Test 5 – BenchmarkResult structure (basic sanity of benchmark() helper).
 */

#include "pipeline.hpp"
#include "transformer.hpp"
#include "model_config.hpp"
#include "cuda_utils.cuh"
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <memory>
#include <thread>
#include <vector>

using namespace flashgen;

// ── Tiny model config for fast testing ───────────────────────────────────────
// 2 layers, 2 heads, d_model=64, d_ff=256  — fits on any GPU
static ModelConfig tiny_config() {
    ModelConfig cfg;
    cfg.vocab_size  = 256;   // small vocab for speed
    cfg.max_seq_len = 128;
    cfg.d_model     = 64;
    cfg.n_heads     = 2;
    cfg.n_layers    = 2;
    cfg.d_ff        = 256;
    return cfg;
}

static bool check(bool cond, const char* msg) {
    printf("  %s %s\n", cond ? "PASS" : "FAIL", msg);
    return cond;
}

// ════════════════════════════════════════════════════════════════════════════
//  Test 1: Synchronous inference produces non-empty output
// ════════════════════════════════════════════════════════════════════════════

static bool test_sync_inference() {
    printf("\n[Test 1] Synchronous inference\n");

    auto mcfg = tiny_config();
    InferenceConfig icfg;
    icfg.batch_size     = 1;
    icfg.max_new_tokens = 10;
    icfg.greedy         = true;

    auto model    = std::make_shared<Transformer>(mcfg, icfg);
    AsyncPipeline pipeline(model);

    InferenceRequest req;
    req.id = 1;
    req.cfg = icfg;
    for (int i = 0; i < 8; i++) req.prompt_ids.push_back(65 + i); // "ABCDEFGH"

    auto resp = pipeline.run_sync(req);

    bool pass = true;
    pass &= check(resp.success, "response.success == true");
    pass &= check(resp.request_id == 1, "request_id preserved");
    pass &= check(!resp.generated_ids.empty(), "generated non-empty token sequence");
    pass &= check(resp.latency_ms > 0.f, "latency_ms > 0");
    pass &= check(resp.tpot_ms > 0.f,    "tpot_ms > 0");

    printf("  Tokens generated : %zu\n", resp.generated_ids.size());
    printf("  Latency          : %.2f ms\n", resp.latency_ms);
    printf("  TPOT             : %.2f ms/token\n", resp.tpot_ms);

    return pass;
}

// ════════════════════════════════════════════════════════════════════════════
//  Test 2: Multiple synchronous requests (sequential)
// ════════════════════════════════════════════════════════════════════════════

static bool test_multiple_requests() {
    printf("\n[Test 2] Multiple sequential requests\n");

    auto mcfg = tiny_config();
    InferenceConfig icfg;
    icfg.batch_size     = 1;
    icfg.max_new_tokens = 5;
    icfg.greedy         = true;

    auto model    = std::make_shared<Transformer>(mcfg, icfg);
    AsyncPipeline pipeline(model);

    const int N = 5;
    bool all_pass = true;
    for (int i = 0; i < N; i++) {
        InferenceRequest req;
        req.id = i;
        req.cfg = icfg;
        for (int j = 0; j < 4; j++) req.prompt_ids.push_back(10 + j);

        auto resp = pipeline.run_sync(req);
        char msg[64];
        snprintf(msg, sizeof(msg), "request %d: success && tokens>0", i);
        all_pass &= check(resp.success && !resp.generated_ids.empty(), msg);
    }
    return all_pass;
}

// ════════════════════════════════════════════════════════════════════════════
//  Test 3: Async submission and callback
// ════════════════════════════════════════════════════════════════════════════

static bool test_async_submission() {
    printf("\n[Test 3] Async submission with callback\n");

    auto mcfg = tiny_config();
    InferenceConfig icfg;
    icfg.batch_size     = 1;
    icfg.max_new_tokens = 4;
    icfg.greedy         = true;

    auto model    = std::make_shared<Transformer>(mcfg, icfg);
    AsyncPipeline pipeline(model);

    std::atomic<int> callbacks_fired{0};
    const int N = 3;

    for (int i = 0; i < N; i++) {
        InferenceRequest req;
        req.id = i;
        req.cfg = icfg;
        req.prompt_ids = {1, 2, 3};
        req.callback = [&callbacks_fired](const std::string& /*text*/) {
            callbacks_fired.fetch_add(1, std::memory_order_relaxed);
        };

        // Keep trying until queue accepts
        while (!pipeline.submit(req))
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // Wait for all callbacks (max 10 seconds)
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (callbacks_fired.load() < N &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    char msg[64];
    snprintf(msg, sizeof(msg), "all %d callbacks fired (got %d)", N, callbacks_fired.load());
    return check(callbacks_fired.load() == N, msg);
}

// ════════════════════════════════════════════════════════════════════════════
//  Test 4: Pipeline statistics
// ════════════════════════════════════════════════════════════════════════════

static bool test_statistics() {
    printf("\n[Test 4] Pipeline statistics\n");

    auto mcfg = tiny_config();
    InferenceConfig icfg;
    icfg.batch_size     = 1;
    icfg.max_new_tokens = 8;
    icfg.greedy         = true;

    auto model    = std::make_shared<Transformer>(mcfg, icfg);
    AsyncPipeline pipeline(model);

    const int N = 4;
    for (int i = 0; i < N; i++) {
        InferenceRequest req;
        req.id  = i;
        req.cfg = icfg;
        req.prompt_ids = {1, 2, 3, 4};
        pipeline.run_sync(req);
    }

    const auto& s = pipeline.stats();
    bool pass = true;
    pass &= check(s.requests_completed.load() >= (uint64_t)N,
                  "requests_completed >= N");
    pass &= check(s.total_tokens_generated.load() > 0,
                  "total_tokens_generated > 0");

    printf("  Requests completed  : %llu\n",
           (unsigned long long)s.requests_completed.load());
    printf("  Tokens generated    : %llu\n",
           (unsigned long long)s.total_tokens_generated.load());

    return pass;
}

// ════════════════════════════════════════════════════════════════════════════
//  Test 5: Graceful shutdown under load
// ════════════════════════════════════════════════════════════════════════════

static bool test_shutdown() {
    printf("\n[Test 5] Graceful shutdown under load\n");

    auto mcfg = tiny_config();
    InferenceConfig icfg;
    icfg.max_new_tokens = 2;
    icfg.greedy         = true;

    auto model = std::make_shared<Transformer>(mcfg, icfg);
    bool shutdown_clean = false;

    {
        AsyncPipeline pipeline(model);

        // Submit a few requests without waiting for them
        for (int i = 0; i < 4; i++) {
            InferenceRequest req;
            req.id  = i;
            req.cfg = icfg;
            req.prompt_ids = {1, 2};
            pipeline.submit(req);
        }

        // Destructor calls shutdown() — should not hang
        auto t0 = std::chrono::steady_clock::now();
        // pipeline goes out of scope here, destructor runs
        pipeline.shutdown();
        auto t1 = std::chrono::steady_clock::now();
        float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
        shutdown_clean = (ms < 5000.f);
        printf("  Shutdown took %.1f ms\n", ms);
    }

    return check(shutdown_clean, "shutdown completed in < 5s");
}

// ════════════════════════════════════════════════════════════════════════════
//  Test 6: Benchmark helper returns sane values
// ════════════════════════════════════════════════════════════════════════════

static bool test_benchmark_helper() {
    printf("\n[Test 6] Benchmark helper sanity check\n");

    auto mcfg = tiny_config();
    InferenceConfig icfg;
    auto model = std::make_shared<Transformer>(mcfg, icfg);

    auto r = benchmark(*model, /*seq_len=*/32, /*batch_size=*/1,
                       /*num_runs=*/10, /*warmup=*/true);

    bool pass = true;
    pass &= check(r.mean_latency_ms > 0.f,      "mean_latency_ms > 0");
    pass &= check(r.p99_latency_ms >= r.p95_latency_ms &&
                  r.p95_latency_ms >= r.p50_latency_ms,
                  "p50 <= p95 <= p99");
    pass &= check(r.throughput_tokens_per_sec > 0.f, "throughput > 0");

    printf("  Mean latency : %.2f ms\n",  r.mean_latency_ms);
    printf("  P99  latency : %.2f ms\n",  r.p99_latency_ms);
    printf("  Throughput   : %.1f tok/s\n", r.throughput_tokens_per_sec);

    return pass;
}

// ── Entry point ───────────────────────────────────────────────────────────────

int main() {
    printf("==========================================\n");
    printf("  Pipeline / End-to-End Test Suite\n");
    printf("==========================================\n");

    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("Device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    bool pass = true;
    pass &= test_sync_inference();
    pass &= test_multiple_requests();
    pass &= test_async_submission();
    pass &= test_statistics();
    pass &= test_shutdown();
    pass &= test_benchmark_helper();

    printf("\n==========================================\n");
    printf("  Result: %s\n", pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    printf("==========================================\n");
    return pass ? 0 : 1;
}
