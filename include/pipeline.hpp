#pragma once
#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>
#include "model_config.hpp"
#include "cuda_utils.cuh"

namespace flashgen {

// ── Request / response types ─────────────────────────────────────────────────
struct InferenceRequest {
    int                      id;             // caller-assigned request ID
    std::vector<int>         prompt_ids;     // tokenized prompt
    InferenceConfig          cfg;            // per-request config
    std::function<void(const std::string&)> callback; // called when done
};

struct InferenceResponse {
    int              request_id;
    std::string      generated_text;
    std::vector<int> generated_ids;
    float            latency_ms;    // end-to-end wall time
    float            tpot_ms;       // time per output token
    bool             success;
    std::string      error;
};

// ── Bounded lock-free-like SPSC slot for pipeline stages ────────────────────
template<typename T, int N>
class BoundedQueue {
    static_assert((N & (N - 1)) == 0, "N must be power of 2");
public:
    BoundedQueue() : head_(0), tail_(0) {}

    bool push(T item) {
        size_t h = head_.load(std::memory_order_relaxed);
        size_t next = (h + 1) & mask_;
        if (next == tail_.load(std::memory_order_acquire)) return false; // full
        slots_[h] = std::move(item);
        head_.store(next, std::memory_order_release);
        return true;
    }

    bool pop(T& item) {
        size_t t = tail_.load(std::memory_order_relaxed);
        if (t == head_.load(std::memory_order_acquire)) return false; // empty
        item = std::move(slots_[t]);
        tail_.store((t + 1) & mask_, std::memory_order_release);
        return true;
    }

    bool empty() const {
        return tail_.load(std::memory_order_acquire) ==
               head_.load(std::memory_order_acquire);
    }

private:
    static constexpr size_t mask_ = N - 1;
    T                       slots_[N];
    std::atomic<size_t>     head_;
    std::atomic<size_t>     tail_;
};

// ── Stage 1: preprocessing (CPU) ─────────────────────────────────────────────
// Tokenization → embedding lookup (host side) → H2D transfer to pinned buf
struct PreprocessedBatch {
    std::shared_ptr<InferenceRequest> request;
    PinnedBuffer<int>                 ids;
    int                               seq_len;
    CudaEvent                         h2d_done;

    PreprocessedBatch(std::shared_ptr<InferenceRequest> req, int seq)
        : request(std::move(req)), ids(seq), seq_len(seq) {}
};

// ── Stage 2: GPU compute ──────────────────────────────────────────────────────
struct ComputedBatch {
    std::shared_ptr<InferenceRequest> request;
    PinnedBuffer<float>               logits;  // [vocab_size] per request step
    int                               vocab_size;
    int                               steps_done;
    CudaEvent                         compute_done;

    ComputedBatch(std::shared_ptr<InferenceRequest> req, int vsz, int steps)
        : request(std::move(req)), logits(vsz), vocab_size(vsz),
          steps_done(steps) {}
};

// ── Async three-stage pipeline ────────────────────────────────────────────────
//
// Stage 1 (CPU thread): tokenize, pin memory, H2D transfer
// Stage 2 (GPU thread): run transformer forward pass (streaming)
// Stage 3 (CPU thread): D2H transfer, decode tokens, invoke callback
//
// The pipeline is started once at construction and runs until shutdown().
class AsyncPipeline {
public:
    explicit AsyncPipeline(
        std::shared_ptr<class Transformer> model,
        int queue_depth = 8
    );
    ~AsyncPipeline();

    // Non-blocking: enqueue request, callback called on completion
    // Returns false if the input queue is full
    bool submit(InferenceRequest req);

    // Blocking: submit + wait for response
    InferenceResponse run_sync(InferenceRequest req);

    // Drain all in-flight requests and stop worker threads gracefully
    void shutdown();

    // Statistics
    struct Stats {
        std::atomic<uint64_t> requests_completed{0};
        std::atomic<uint64_t> total_tokens_generated{0};
        std::atomic<double>   total_latency_ms{0.0};
    };
    const Stats& stats() const { return stats_; }

private:
    std::shared_ptr<Transformer> model_;
    Stats                        stats_;
    std::atomic<bool>            running_{true};

    // Inter-stage queues  (power-of-two capacities)
    static constexpr int kQueueSize = 16;
    std::queue<std::shared_ptr<InferenceRequest>> input_queue_;
    std::mutex  input_mutex_;
    std::condition_variable input_cv_;

    std::queue<std::shared_ptr<PreprocessedBatch>> preproc_queue_;
    std::mutex  preproc_mutex_;
    std::condition_variable preproc_cv_;

    std::queue<std::shared_ptr<ComputedBatch>> postproc_queue_;
    std::mutex  postproc_mutex_;
    std::condition_variable postproc_cv_;

    // Worker threads
    std::thread preproc_thread_;
    std::thread compute_thread_;
    std::thread postproc_thread_;

    // Dedicated CUDA streams per stage
    CudaStream h2d_stream_;
    CudaStream compute_stream_;
    CudaStream d2h_stream_;

    // Worker implementations
    void preproc_worker();
    void compute_worker();
    void postproc_worker();

    // Tokenizer (simple BPE stub — replace with tiktoken binding)
    std::vector<int> tokenize(const std::string& text) const;
    std::string      detokenize(const std::vector<int>& ids) const;

    // Sampling
    int sample_next_token(
        const float* logits, int vocab_size,
        float temperature, float top_p, bool greedy
    );
};

// ── Simple benchmark harness ─────────────────────────────────────────────────
struct BenchmarkResult {
    int    seq_len;
    int    batch_size;
    int    num_runs;
    float  mean_latency_ms;
    float  p50_latency_ms;
    float  p95_latency_ms;
    float  p99_latency_ms;
    float  throughput_tokens_per_sec;
};

BenchmarkResult benchmark(
    Transformer&    model,
    int             seq_len,
    int             batch_size,
    int             num_runs,
    bool            warmup = true
);

} // namespace flashgen
