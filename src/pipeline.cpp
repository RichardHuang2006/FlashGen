/*
 * pipeline.cpp
 *
 * Asynchronous three-stage CPU-GPU inference pipeline.
 *
 * Stage 1 (preproc_thread_):
 *   - Receives InferenceRequest from input_queue_
 *   - Tokenizes prompt text
 *   - Copies token IDs to pinned host memory
 *   - Enqueues PreprocessedBatch to preproc_queue_
 *
 * Stage 2 (compute_thread_):
 *   - Dequeues PreprocessedBatch
 *   - Runs Transformer::prefill then iterative ::decode steps
 *   - Streams logits to pinned output buffer
 *   - Enqueues ComputedBatch to postproc_queue_
 *
 * Stage 3 (postproc_thread_):
 *   - Decodes token IDs to text (greedy or top-p sampling)
 *   - Invokes the per-request callback with InferenceResponse
 *   - Updates pipeline statistics
 */

#include "pipeline.hpp"
#include "transformer.hpp"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <future>
#include <numeric>
#include <random>
#include <sstream>

namespace flashgen {

// ════════════════════════════════════════════════════════════════════════════
//  Construction / destruction
// ════════════════════════════════════════════════════════════════════════════

AsyncPipeline::AsyncPipeline(
    std::shared_ptr<Transformer> model, int /*queue_depth*/
) : model_(std::move(model)) {
    preproc_thread_ = std::thread([this]{ preproc_worker(); });
    compute_thread_ = std::thread([this]{ compute_worker(); });
    postproc_thread_ = std::thread([this]{ postproc_worker(); });
}

AsyncPipeline::~AsyncPipeline() {
    shutdown();
}

void AsyncPipeline::shutdown() {
    running_.store(false, std::memory_order_release);
    input_cv_.notify_all();
    preproc_cv_.notify_all();
    postproc_cv_.notify_all();

    if (preproc_thread_.joinable())  preproc_thread_.join();
    if (compute_thread_.joinable())  compute_thread_.join();
    if (postproc_thread_.joinable()) postproc_thread_.join();
}

// ════════════════════════════════════════════════════════════════════════════
//  Public submit
// ════════════════════════════════════════════════════════════════════════════

bool AsyncPipeline::submit(InferenceRequest req) {
    std::unique_lock<std::mutex> lk(input_mutex_);
    if (input_queue_.size() >= kQueueSize) return false;
    input_queue_.push(std::make_shared<InferenceRequest>(std::move(req)));
    input_cv_.notify_one();
    return true;
}

InferenceResponse AsyncPipeline::run_sync(InferenceRequest req) {
    std::promise<InferenceResponse> promise;
    auto future = promise.get_future();

    req.callback = [&promise](const std::string& /*text*/) {
        // The actual response is delivered below via the dedicated field.
        // (In a full implementation the callback receives InferenceResponse.)
    };

    InferenceResponse resp;
    resp.request_id = req.id;

    // Run synchronously by bypassing the queue
    auto start = std::chrono::steady_clock::now();
    const int vocab_size = model_->config().vocab_size;
    const int max_new    = req.cfg.max_new_tokens;

    std::vector<float> logits(vocab_size);
    std::vector<int>   generated;

    // Prefill
    model_->prefill(req.prompt_ids.data(), logits.data(), 1,
                    (int)req.prompt_ids.size());

    // Autoregressive decode
    std::vector<int> last_token(1);
    last_token[0] = sample_next_token(
        logits.data(), vocab_size,
        req.cfg.temperature, req.cfg.top_p, req.cfg.greedy);
    generated.push_back(last_token[0]);

    for (int step = 1; step < max_new; step++) {
        model_->decode(last_token.data(), logits.data(), 1);
        last_token[0] = sample_next_token(
            logits.data(), vocab_size,
            req.cfg.temperature, req.cfg.top_p, req.cfg.greedy);
        generated.push_back(last_token[0]);
        if (last_token[0] == 50256) break; // GPT-2 EOS token
    }
    model_->reset_cache();

    auto end = std::chrono::steady_clock::now();
    float elapsed_ms = std::chrono::duration<float, std::milli>(end - start).count();

    resp.generated_ids   = generated;
    resp.generated_text  = detokenize(generated);
    resp.latency_ms      = elapsed_ms;
    resp.tpot_ms         = elapsed_ms / (float)generated.size();
    resp.success         = true;

    stats_.requests_completed.fetch_add(1, std::memory_order_relaxed);
    stats_.total_tokens_generated.fetch_add(generated.size(), std::memory_order_relaxed);
    stats_.total_latency_ms.store(
        stats_.total_latency_ms.load(std::memory_order_relaxed) + elapsed_ms,
        std::memory_order_relaxed);

    return resp;
}

// ════════════════════════════════════════════════════════════════════════════
//  Stage 1: preprocessing worker
// ════════════════════════════════════════════════════════════════════════════

void AsyncPipeline::preproc_worker() {
    while (running_.load(std::memory_order_acquire)) {
        std::shared_ptr<InferenceRequest> req;
        {
            std::unique_lock<std::mutex> lk(input_mutex_);
            input_cv_.wait(lk, [this]{
                return !input_queue_.empty() || !running_.load();
            });
            if (!running_ && input_queue_.empty()) break;
            req = input_queue_.front();
            input_queue_.pop();
        }

        // Tokenize (if prompt_ids not already filled)
        if (req->prompt_ids.empty()) {
            // In a real system, call tiktoken / sentencepiece here.
            // For the demo we just use the IDs the caller provided.
        }

        const int seq_len = (int)req->prompt_ids.size();
        auto batch = std::make_shared<PreprocessedBatch>(req, seq_len);
        std::copy(req->prompt_ids.begin(), req->prompt_ids.end(), batch->ids.ptr);

        {
            std::lock_guard<std::mutex> lk(preproc_mutex_);
            preproc_queue_.push(batch);
        }
        preproc_cv_.notify_one();
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  Stage 2: GPU compute worker
// ════════════════════════════════════════════════════════════════════════════

void AsyncPipeline::compute_worker() {
    const int vocab_size = model_->config().vocab_size;
    const int max_new    = 128; // default; use per-request cfg in production

    while (running_.load(std::memory_order_acquire)) {
        std::shared_ptr<PreprocessedBatch> pp;
        {
            std::unique_lock<std::mutex> lk(preproc_mutex_);
            preproc_cv_.wait(lk, [this]{
                return !preproc_queue_.empty() || !running_.load();
            });
            if (!running_ && preproc_queue_.empty()) break;
            pp = preproc_queue_.front();
            preproc_queue_.pop();
        }

        const InferenceConfig& cfg = pp->request->cfg;
        const int actual_max = cfg.max_new_tokens > 0 ? cfg.max_new_tokens : max_new;

        // Allocate output logits on pinned host
        auto cb = std::make_shared<ComputedBatch>(pp->request, vocab_size, actual_max);

        std::vector<float> logits(vocab_size);
        std::vector<int>   generated;
        generated.reserve(actual_max);

        // Prefill phase
        model_->prefill(pp->ids.ptr, logits.data(), 1, pp->seq_len);

        std::vector<int> last(1);
        last[0] = sample_next_token(logits.data(), vocab_size,
                                    cfg.temperature, cfg.top_p, cfg.greedy);
        generated.push_back(last[0]);

        // Decode loop
        for (int step = 1; step < actual_max; step++) {
            model_->decode(last.data(), logits.data(), 1);
            last[0] = sample_next_token(logits.data(), vocab_size,
                                        cfg.temperature, cfg.top_p, cfg.greedy);
            generated.push_back(last[0]);
            if (last[0] == 50256) break;
        }
        model_->reset_cache();

        // Store generated token IDs in pinned buffer
        cb->steps_done = (int)generated.size();
        std::copy(generated.begin(), generated.end(), cb->logits.ptr); // reuse buffer as int

        {
            std::lock_guard<std::mutex> lk(postproc_mutex_);
            postproc_queue_.push(cb);
        }
        postproc_cv_.notify_one();
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  Stage 3: postprocessing worker
// ════════════════════════════════════════════════════════════════════════════

void AsyncPipeline::postproc_worker() {
    while (running_.load(std::memory_order_acquire)) {
        std::shared_ptr<ComputedBatch> cb;
        {
            std::unique_lock<std::mutex> lk(postproc_mutex_);
            postproc_cv_.wait(lk, [this]{
                return !postproc_queue_.empty() || !running_.load();
            });
            if (!running_ && postproc_queue_.empty()) break;
            cb = postproc_queue_.front();
            postproc_queue_.pop();
        }

        std::vector<int> ids(cb->steps_done);
        for (int i = 0; i < cb->steps_done; i++)
            ids[i] = (int)cb->logits.ptr[i];

        std::string text = detokenize(ids);

        if (cb->request->callback)
            cb->request->callback(text);

        stats_.requests_completed.fetch_add(1, std::memory_order_relaxed);
        stats_.total_tokens_generated.fetch_add(ids.size(), std::memory_order_relaxed);
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  Tokenizer stubs (replace with tiktoken or sentencepiece)
// ════════════════════════════════════════════════════════════════════════════

std::vector<int> AsyncPipeline::tokenize(const std::string& text) const {
    // Stub: encode each character as its ASCII code (not real BPE).
    // Replace with a proper tokenizer library in production.
    std::vector<int> ids;
    ids.reserve(text.size());
    for (unsigned char c : text) ids.push_back((int)c);
    return ids;
}

std::string AsyncPipeline::detokenize(const std::vector<int>& ids) const {
    // Stub: interpret each ID as an ASCII character.
    std::string out;
    out.reserve(ids.size());
    for (int id : ids) {
        if (id >= 32 && id < 127) out += (char)id;
    }
    return out;
}

// ════════════════════════════════════════════════════════════════════════════
//  Sampling
// ════════════════════════════════════════════════════════════════════════════

int AsyncPipeline::sample_next_token(
    const float* logits, int vocab_size,
    float temperature, float top_p, bool greedy
) {
    if (greedy) {
        return (int)(std::max_element(logits, logits + vocab_size) - logits);
    }

    // Temperature scaling
    std::vector<float> probs(logits, logits + vocab_size);
    const float inv_temp = 1.f / std::max(temperature, 1e-6f);
    float max_logit = *std::max_element(probs.begin(), probs.end());
    float sum = 0.f;
    for (auto& p : probs) { p = std::exp((p - max_logit) * inv_temp); sum += p; }
    for (auto& p : probs) p /= sum;

    // Top-p (nucleus) sampling
    std::vector<int> idx(vocab_size);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b){ return probs[a] > probs[b]; });

    float cumsum = 0.f;
    int cutoff = vocab_size;
    for (int i = 0; i < vocab_size; i++) {
        cumsum += probs[idx[i]];
        if (cumsum >= top_p) { cutoff = i + 1; break; }
    }

    // Renormalise truncated distribution
    std::vector<float> trunc_probs(cutoff);
    float trunc_sum = 0.f;
    for (int i = 0; i < cutoff; i++) trunc_sum += probs[idx[i]];
    for (int i = 0; i < cutoff; i++) trunc_probs[i] = probs[idx[i]] / trunc_sum;

    // Sample
    static thread_local std::mt19937 rng(std::random_device{}());
    std::discrete_distribution<int> dist(trunc_probs.begin(), trunc_probs.end());
    return idx[dist(rng)];
}

// ════════════════════════════════════════════════════════════════════════════
//  Benchmark harness
// ════════════════════════════════════════════════════════════════════════════

BenchmarkResult benchmark(
    Transformer& model, int seq_len, int batch_size, int num_runs, bool warmup
) {
    const int vocab_size = model.config().vocab_size;

    // Fake input IDs
    std::vector<int> ids(batch_size * seq_len, 1);

    std::vector<float> logits_out((size_t)batch_size * vocab_size);
    std::vector<float> latencies;
    latencies.reserve(num_runs);

    if (warmup) {
        // Two warmup runs
        model.prefill(ids.data(), logits_out.data(), batch_size, seq_len);
        model.reset_cache();
        model.prefill(ids.data(), logits_out.data(), batch_size, seq_len);
        model.reset_cache();
    }

    for (int run = 0; run < num_runs; run++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        model.prefill(ids.data(), logits_out.data(), batch_size, seq_len);
        auto t1 = std::chrono::high_resolution_clock::now();
        model.reset_cache();
        float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
        latencies.push_back(ms);
    }

    std::sort(latencies.begin(), latencies.end());
    float mean = std::accumulate(latencies.begin(), latencies.end(), 0.f) / num_runs;
    float p50  = latencies[num_runs * 50 / 100];
    float p95  = latencies[num_runs * 95 / 100];
    float p99  = latencies[num_runs * 99 / 100];
    float tput = (float)(batch_size * seq_len) / (mean * 1e-3f);

    return { seq_len, batch_size, num_runs, mean, p50, p95, p99, tput };
}

} // namespace flashgen
