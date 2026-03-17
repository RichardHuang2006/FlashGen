#include "pipeline.hpp"
#include <cassert>
#include <cstdio>
#include <future>
#include <numeric>
#include <thread>
#include <vector>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) printf("  %-50s", name); fflush(stdout)
#define PASS() do { printf("[PASS]\n"); ++tests_passed; } while(0)
#define FAIL(msg) do { printf("[FAIL] %s\n", msg); ++tests_failed; } while(0)

static EngineConfig make_test_config() {
    EngineConfig cfg;
    cfg.model = ModelConfig::gpt2();
    cfg.cache.block_size = 16;
    cfg.cache.num_gpu_blocks = 128;
    cfg.cache.enable_prefix_caching = true;
    cfg.scheduler.max_num_seqs = 16;
    cfg.scheduler.max_num_tokens = 2048;
    return cfg;
}

// ---------------------------------------------------------------------------
//  Tests
// ---------------------------------------------------------------------------

void test_engine_construction() {
    TEST("Engine construction and destruction");

    auto cfg = make_test_config();
    {
        InferenceEngine engine(cfg);
        assert(!engine.is_running());
    }
    // Destructor should not crash

    PASS();
}

void test_engine_start_stop() {
    TEST("Engine start and shutdown");

    auto cfg = make_test_config();
    InferenceEngine engine(cfg);

    engine.start();
    assert(engine.is_running());

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    engine.shutdown();
    assert(!engine.is_running());

    PASS();
}

void test_submit_with_callback() {
    TEST("Submit request with callback");

    auto cfg = make_test_config();
    InferenceEngine engine(cfg);
    engine.start();

    std::promise<InferenceResponse> promise;
    auto future = promise.get_future();

    InferenceRequest req;
    req.prompt_token_ids = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    req.sampling.max_tokens = 5;
    req.sampling.greedy = true;
    req.callback = [&promise](const InferenceResponse& resp) {
        promise.set_value(resp);
    };

    int rid = engine.submit(std::move(req));
    assert(rid >= 0);

    // Wait with timeout
    auto status = future.wait_for(std::chrono::seconds(10));
    engine.shutdown();

    if (status == std::future_status::ready) {
        auto resp = future.get();
        printf("(rid=%d, tokens=%d) ", resp.request_id, resp.generated_tokens);
        PASS();
    } else {
        FAIL("timeout waiting for response");
    }
}

void test_sync_generate() {
    TEST("Synchronous generate");

    auto cfg = make_test_config();
    InferenceEngine engine(cfg);

    InferenceRequest req;
    req.prompt_token_ids.resize(16, 42);
    req.sampling.max_tokens = 3;
    req.sampling.greedy = true;

    auto resp = engine.generate(std::move(req));

    if (resp.generated_tokens > 0) {
        printf("(tokens=%d, %.1fms) ", resp.generated_tokens, resp.latency_ms);
        PASS();
    } else {
        FAIL("no tokens generated");
    }
}

void test_engine_stats() {
    TEST("Engine statistics tracking");

    auto cfg = make_test_config();
    InferenceEngine engine(cfg);
    engine.start();

    // Submit a few requests
    int n_requests = 3;
    std::vector<std::future<InferenceResponse>> futures;

    for (int i = 0; i < n_requests; ++i) {
        std::promise<InferenceResponse> p;
        futures.push_back(p.get_future());

        InferenceRequest req;
        req.prompt_token_ids.resize(16, i + 1);
        req.sampling.max_tokens = 2;
        req.sampling.greedy = true;
        req.callback = [pr = std::make_shared<std::promise<InferenceResponse>>(std::move(p))]
                       (const InferenceResponse& resp) {
            pr->set_value(resp);
        };

        engine.submit(std::move(req));
    }

    // Wait for all
    for (auto& f : futures) {
        f.wait_for(std::chrono::seconds(10));
    }

    engine.shutdown();

    auto& stats = engine.stats();
    printf("(completed=%lld, tokens=%lld) ",
           (long long)stats.requests_completed.load(),
           (long long)stats.tokens_generated.load());

    if (stats.requests_completed.load() >= n_requests) {
        PASS();
    } else {
        FAIL("not all requests completed");
    }
}

void test_benchmark_prefill() {
    TEST("Benchmark prefill runs without crash");

    auto cfg = make_test_config();
    cfg.cache.num_gpu_blocks = 256;
    InferenceEngine engine(cfg);

    // Should not crash
    engine.benchmark_prefill(/*seq_len=*/32, /*batch=*/2, /*runs=*/2);

    PASS();
}

int main() {
    printf("═══ Pipeline / Engine Tests ═══\n\n");

    test_engine_construction();
    test_engine_start_stop();
    test_submit_with_callback();
    test_sync_generate();
    test_engine_stats();
    test_benchmark_prefill();

    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
