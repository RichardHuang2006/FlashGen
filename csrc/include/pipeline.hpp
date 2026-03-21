#pragma once

#include "model_config.hpp"
#include "request.hpp"
#include "block_allocator.hpp"
#include "paged_kv_cache.cuh"
#include "prefix_cache.hpp"
#include "scheduler.hpp"
#include "transformer.hpp"

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <string>

// ---------------------------------------------------------------------------
//  Inference engine — production-grade serving pipeline
//
//  Integrates all components:
//    Transformer (model execution)
//    BlockAllocator (GPU memory management)
//    PagedKVCache (block-table-based KV storage)
//    PrefixCache (radix-tree prefix sharing)
//    Scheduler (continuous batching)
//
//  The engine runs a main loop that:
//    1. Calls scheduler to form the next batch
//    2. Executes prefill / decode forward passes
//    3. Samples next tokens
//    4. Updates sequence state and delivers responses
// ---------------------------------------------------------------------------

struct EngineStats {
    std::atomic<int64_t> requests_completed{0};
    std::atomic<int64_t> tokens_generated{0};
    std::atomic<int64_t> prefill_tokens{0};
    std::atomic<double>  total_latency_ms{0.0};
    std::atomic<int64_t> iterations{0};

    double avg_latency_ms() const {
        int64_t n = requests_completed.load();
        return n > 0 ? total_latency_ms.load() / n : 0.0;
    }
    double throughput_tps() const {
        double t = total_latency_ms.load();
        return t > 0.0 ? tokens_generated.load() / (t / 1000.0) : 0.0;
    }
};

class InferenceEngine {
public:
    explicit InferenceEngine(const EngineConfig& config);
    ~InferenceEngine();

    InferenceEngine(const InferenceEngine&) = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;

    // ── Lifecycle ───────────────────────────────────────────────────────

    /// Start the engine loop (background thread).
    void start();

    /// Graceful shutdown: finish in-flight requests, then stop.
    void shutdown();

    bool is_running() const { return running_.load(); }

    // ── Request submission ──────────────────────────────────────────────

    /// Async: submit a request with callback. Returns request ID.
    int submit(InferenceRequest req);

    /// Sync: submit and block until response is ready.
    InferenceResponse generate(InferenceRequest req);

    // ── Stats ───────────────────────────────────────────────────────────

    const EngineStats& stats() const { return stats_; }

    // ── Benchmarking ────────────────────────────────────────────────────

    /// Run a benchmark: prefill throughput (tokens/sec).
    void benchmark_prefill(int seq_len, int batch_size, int num_runs);

    /// Run a benchmark: decode throughput (tokens/sec).
    void benchmark_decode(int num_seqs, int context_len, int decode_steps);

private:
    EngineConfig config_;

    // Core components
    std::unique_ptr<Transformer>    model_;
    std::unique_ptr<BlockAllocator> allocator_;
    std::unique_ptr<PagedKVCache>   kv_cache_;
    std::unique_ptr<PrefixCache>    prefix_cache_;
    std::unique_ptr<Scheduler>      scheduler_;

    // Engine thread
    std::thread         engine_thread_;
    std::atomic<bool>   running_{false};
    std::atomic<bool>   shutdown_requested_{false};

    // Request queue (thread-safe ingestion)
    std::queue<std::shared_ptr<SequenceGroup>> incoming_;
    std::mutex              incoming_mu_;
    std::condition_variable incoming_cv_;
    std::atomic<int>        next_request_id_{0};

    // GPU resources
    CudaStream   compute_stream_;
    CudaStream   copy_stream_;
    DeviceBuffer<float> logits_buf_;

    // Stats
    EngineStats stats_;

    // ── Internal ────────────────────────────────────────────────────────

    void engine_loop();
    void step();
    void drain_incoming();
    void sample_tokens(const float* logits, int num_seqs,
                       const std::vector<SequenceGroup*>& batch,
                       std::vector<int>& out_tokens);
    void deliver_responses(const std::vector<SequenceGroup*>& finished);
};
