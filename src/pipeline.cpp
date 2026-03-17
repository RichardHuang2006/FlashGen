#include "pipeline.hpp"
#include "kernels.cuh"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <future>
#include <numeric>

// ===========================================================================
//  InferenceEngine — production-grade continuous batching pipeline
// ===========================================================================

InferenceEngine::InferenceEngine(const EngineConfig& config)
    : config_(config)
{
    CUDA_CHECK(cudaSetDevice(config.device_id));
    print_device_info(config.device_id);

    // Initialize model
    model_ = std::make_unique<Transformer>(config.model);

    // Load weights if path provided
    if (!config.weight_path.empty()) {
        CudaStream load_stream;
        model_->load_weights(config.weight_path, load_stream);
        load_stream.sync();
    }

    // Compute number of KV cache blocks
    CacheConfig cache_cfg = config.cache;
    if (cache_cfg.num_gpu_blocks < 0) {
        auto [total, free] = gpu_memory_info();
        cache_cfg.num_gpu_blocks = cache_cfg.auto_num_blocks(config.model, free);
        printf("Auto-configured %d KV cache blocks (%.1f GB)\n",
               cache_cfg.num_gpu_blocks,
               (double)cache_cfg.num_gpu_blocks * cache_cfg.bytes_per_block(config.model) / 1073741824.0);
    }

    // Initialize block allocator
    allocator_ = std::make_unique<BlockAllocator>(
        cache_cfg.num_gpu_blocks,
        cache_cfg.block_size,
        config.model.n_layers,
        config.model.actual_kv_heads(),
        config.model.head_dim(),
        cache_cfg.kv_quant);

    // Initialize paged KV cache
    kv_cache_ = std::make_unique<PagedKVCache>(*allocator_, config.model);

    // Initialize prefix cache
    prefix_cache_ = std::make_unique<PrefixCache>(
        *allocator_, cache_cfg.block_size, config.model.n_layers);
    prefix_cache_->set_enabled(cache_cfg.enable_prefix_caching);

    // Initialize scheduler
    scheduler_ = std::make_unique<Scheduler>(
        config.scheduler, *allocator_, *kv_cache_, *prefix_cache_);

    // Allocate logits buffer
    logits_buf_.allocate((size_t)config.scheduler.max_num_seqs * config.model.vocab_size);

    printf("Engine initialized: %s, %d layers, %d heads, d=%d\n",
           config.model.name.c_str(), config.model.n_layers,
           config.model.n_heads, config.model.d_model);
    printf("  KV cache: %d blocks x %d tokens/block, %s quantization\n",
           cache_cfg.num_gpu_blocks, cache_cfg.block_size,
           cache_cfg.kv_quant == KVQuantType::NONE ? "none" :
           cache_cfg.kv_quant == KVQuantType::INT8 ? "INT8" : "FP8");
    printf("  Scheduler: max_seqs=%d, max_tokens=%d, prefix_cache=%s\n",
           config.scheduler.max_num_seqs, config.scheduler.max_num_tokens,
           cache_cfg.enable_prefix_caching ? "on" : "off");
}

InferenceEngine::~InferenceEngine() {
    shutdown();
}

void InferenceEngine::start() {
    if (running_.load()) return;
    running_ = true;
    shutdown_requested_ = false;
    engine_thread_ = std::thread(&InferenceEngine::engine_loop, this);
    printf("Engine started\n");
}

void InferenceEngine::shutdown() {
    if (!running_.load()) return;

    shutdown_requested_ = true;
    incoming_cv_.notify_all();

    if (engine_thread_.joinable()) {
        engine_thread_.join();
    }
    running_ = false;
    printf("Engine shut down (%lld requests completed, %lld tokens generated)\n",
           (long long)stats_.requests_completed.load(),
           (long long)stats_.tokens_generated.load());
}

int InferenceEngine::submit(InferenceRequest req) {
    int rid = next_request_id_++;
    req.request_id = rid;

    // Wrap into SequenceGroup
    auto sg = std::make_shared<SequenceGroup>();
    sg->request_id  = rid;
    sg->sampling    = req.sampling;
    sg->is_prefill  = true;
    sg->priority    = req.priority;
    sg->arrival_time = req.arrival_time;
    sg->callback    = std::move(req.callback);

    SequenceData seq;
    seq.prompt_tokens = std::move(req.prompt_token_ids);
    seq.status = SequenceStatus::WAITING;
    sg->sequences.push_back(std::move(seq));

    {
        std::lock_guard<std::mutex> lock(incoming_mu_);
        incoming_.push(std::move(sg));
    }
    incoming_cv_.notify_one();

    return rid;
}

InferenceResponse InferenceEngine::generate(InferenceRequest req) {
    std::promise<InferenceResponse> promise;
    auto future = promise.get_future();

    req.callback = [&promise](const InferenceResponse& resp) {
        promise.set_value(resp);
    };

    // If engine isn't running, run synchronously
    if (!running_.load()) {
        start();
        submit(std::move(req));
        auto response = future.get();
        shutdown();
        return response;
    }

    submit(std::move(req));
    return future.get();
}

void InferenceEngine::engine_loop() {
    while (!shutdown_requested_.load() || scheduler_->has_pending()) {
        // Wait for incoming requests or pending work
        {
            std::unique_lock<std::mutex> lock(incoming_mu_);
            incoming_cv_.wait_for(lock, std::chrono::milliseconds(1), [this] {
                return !incoming_.empty() || shutdown_requested_.load();
            });
        }

        // Drain incoming requests into scheduler
        drain_incoming();

        // Run one scheduling + execution step
        if (scheduler_->has_pending()) {
            step();
        }
    }
}

void InferenceEngine::drain_incoming() {
    std::lock_guard<std::mutex> lock(incoming_mu_);
    while (!incoming_.empty()) {
        scheduler_->add_request(std::move(incoming_.front()));
        incoming_.pop();
    }
}

void InferenceEngine::step() {
    auto t0 = std::chrono::steady_clock::now();

    // 1. Schedule next batch
    SchedulerOutput sched = scheduler_->schedule();
    if (sched.empty()) return;

    // 2. Collect batch information
    std::vector<SequenceGroup*> batch;
    batch.insert(batch.end(), sched.scheduled_prefills.begin(),
                 sched.scheduled_prefills.end());
    batch.insert(batch.end(), sched.scheduled_decodes.begin(),
                 sched.scheduled_decodes.end());

    // 3. Run forward pass
    if (!sched.scheduled_prefills.empty()) {
        // Pack prefill tokens
        std::vector<int> packed_tokens;
        std::vector<int> seq_lens;
        std::vector<int> seq_ids;

        for (auto* sg : sched.scheduled_prefills) {
            auto& seq = sg->first_seq();
            int start = seq.num_computed_tokens;
            int end   = seq.prompt_len();
            packed_tokens.insert(packed_tokens.end(),
                                 seq.prompt_tokens.begin() + start,
                                 seq.prompt_tokens.begin() + end);
            seq_lens.push_back(end - start);
            seq_ids.push_back(seq.seq_id);
        }

        model_->batch_prefill(packed_tokens, seq_lens, *kv_cache_,
                              seq_ids, logits_buf_, compute_stream_);
    }

    if (!sched.scheduled_decodes.empty()) {
        std::vector<int> token_ids;
        std::vector<int> seq_lens;
        std::vector<int> seq_ids;

        for (auto* sg : sched.scheduled_decodes) {
            auto& seq = sg->first_seq();
            // Last generated token (or last prompt token if just finished prefill)
            int last_token = seq.output_tokens.empty()
                ? seq.prompt_tokens.back()
                : seq.output_tokens.back();
            token_ids.push_back(last_token);
            seq_lens.push_back(seq.total_len());
            seq_ids.push_back(seq.seq_id);
        }

        int logits_offset = (int)sched.scheduled_prefills.size() * config_.model.vocab_size;
        model_->batch_decode(token_ids, seq_lens, *kv_cache_,
                             seq_ids,
                             logits_buf_.ptr + logits_offset,
                             compute_stream_);
    }

    compute_stream_.sync();

    // 4. Sample next tokens
    std::vector<int> new_tokens;
    sample_tokens(logits_buf_, (int)batch.size(), batch, new_tokens);

    // 5. Update scheduler state
    scheduler_->post_step(batch, new_tokens);

    // 6. Deliver finished responses
    std::vector<SequenceGroup*> finished;
    for (auto* sg : batch) {
        if (sg->first_seq().is_finished()) {
            finished.push_back(sg);
        }
    }
    deliver_responses(finished);

    // 7. Update stats
    auto t1 = std::chrono::steady_clock::now();
    double step_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    stats_.iterations++;
    stats_.tokens_generated += (int64_t)new_tokens.size();
    stats_.prefill_tokens += sched.num_prefill_tokens;
}

void InferenceEngine::sample_tokens(
    const float* logits, int num_seqs,
    const std::vector<SequenceGroup*>& batch,
    std::vector<int>& out_tokens)
{
    out_tokens.resize(num_seqs);

    // Copy logits to host for sampling
    std::vector<float> h_logits((size_t)num_seqs * config_.model.vocab_size);
    CUDA_CHECK(cudaMemcpy(h_logits.data(), logits,
                          h_logits.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    for (int i = 0; i < num_seqs; ++i) {
        const float* row = h_logits.data() + (size_t)i * config_.model.vocab_size;
        const auto& sampling = batch[i]->sampling;

        if (sampling.greedy || sampling.temperature < 1e-6f) {
            // Greedy: argmax
            int best = 0;
            float best_val = row[0];
            for (int v = 1; v < config_.model.vocab_size; ++v) {
                if (row[v] > best_val) {
                    best_val = row[v];
                    best = v;
                }
            }
            out_tokens[i] = best;
        } else {
            // Temperature + top-p sampling
            std::vector<std::pair<float, int>> probs(config_.model.vocab_size);
            float max_val = *std::max_element(row, row + config_.model.vocab_size);

            float sum = 0.f;
            for (int v = 0; v < config_.model.vocab_size; ++v) {
                float p = expf((row[v] - max_val) / sampling.temperature);
                probs[v] = {p, v};
                sum += p;
            }

            // Normalize
            for (auto& [p, _] : probs) p /= sum;

            // Sort descending for top-p
            std::sort(probs.begin(), probs.end(),
                      [](auto& a, auto& b) { return a.first > b.first; });

            // Top-p filtering
            float cumsum = 0.f;
            int cutoff = config_.model.vocab_size;
            for (int v = 0; v < config_.model.vocab_size; ++v) {
                cumsum += probs[v].first;
                if (cumsum >= sampling.top_p) {
                    cutoff = v + 1;
                    break;
                }
            }

            // Renormalize
            sum = 0.f;
            for (int v = 0; v < cutoff; ++v) sum += probs[v].first;

            // Sample
            float r = (float)rand() / RAND_MAX * sum;
            float acc = 0.f;
            out_tokens[i] = probs[0].second;
            for (int v = 0; v < cutoff; ++v) {
                acc += probs[v].first;
                if (acc >= r) {
                    out_tokens[i] = probs[v].second;
                    break;
                }
            }
        }
    }
}

void InferenceEngine::deliver_responses(const std::vector<SequenceGroup*>& finished) {
    for (auto* sg : finished) {
        auto& seq = sg->first_seq();
        auto now = std::chrono::steady_clock::now();
        float latency = std::chrono::duration<float, std::milli>(
            now - sg->arrival_time).count();

        InferenceResponse resp;
        resp.request_id       = sg->request_id;
        resp.output_token_ids = seq.output_tokens;
        resp.finish_reason    = seq.finish_reason;
        resp.latency_ms       = latency;
        resp.prompt_tokens    = seq.prompt_len();
        resp.generated_tokens = seq.output_len();
        resp.tpot_ms          = seq.output_len() > 0
                                ? latency / seq.output_len() : 0.f;
        resp.success          = true;

        stats_.requests_completed++;
        stats_.total_latency_ms.store(
            stats_.total_latency_ms.load() + (double)latency);

        if (sg->callback) {
            sg->callback(resp);
        }
    }
}

void InferenceEngine::benchmark_prefill(int seq_len, int batch_size, int num_runs) {
    printf("\n=== Prefill Benchmark ===\n");
    printf("  seq_len=%d  batch_size=%d  runs=%d\n", seq_len, batch_size, num_runs);

    // Create dummy token IDs
    std::vector<int> packed_tokens(seq_len * batch_size, 1);
    std::vector<int> seq_lens(batch_size, seq_len);
    std::vector<int> seq_ids(batch_size);
    std::iota(seq_ids.begin(), seq_ids.end(), 0);

    // Allocate KV blocks
    for (int s = 0; s < batch_size; ++s) {
        kv_cache_->allocate_for_prefill(s, seq_len);
    }

    DeviceBuffer<float> logits((size_t)batch_size * config_.model.vocab_size);

    // Warmup
    model_->batch_prefill(packed_tokens, seq_lens, *kv_cache_,
                          seq_ids, logits, compute_stream_);
    compute_stream_.sync();

    // Benchmark
    CudaEvent start, end;
    start.record(compute_stream_);

    for (int r = 0; r < num_runs; ++r) {
        model_->batch_prefill(packed_tokens, seq_lens, *kv_cache_,
                              seq_ids, logits, compute_stream_);
    }

    end.record(compute_stream_);
    compute_stream_.sync();

    float total_ms = end.elapsed(start);
    float avg_ms   = total_ms / num_runs;
    int total_tokens = seq_len * batch_size;
    float tps = total_tokens / (avg_ms / 1000.f);

    printf("  Avg latency: %.2f ms\n", avg_ms);
    printf("  Throughput:  %.0f tokens/sec\n", tps);

    // Cleanup
    for (int s = 0; s < batch_size; ++s) {
        kv_cache_->free_sequence(s);
    }
}

void InferenceEngine::benchmark_decode(int num_seqs, int context_len, int decode_steps) {
    printf("\n=== Decode Benchmark ===\n");
    printf("  num_seqs=%d  context=%d  steps=%d\n", num_seqs, context_len, decode_steps);

    // Allocate KV cache for context
    std::vector<int> seq_ids(num_seqs);
    std::iota(seq_ids.begin(), seq_ids.end(), 0);

    for (int s = 0; s < num_seqs; ++s) {
        kv_cache_->allocate_for_prefill(s, context_len);
    }

    DeviceBuffer<float> logits((size_t)num_seqs * config_.model.vocab_size);
    std::vector<int> token_ids(num_seqs, 1);
    std::vector<int> seq_lens(num_seqs, context_len);

    // Warmup
    model_->batch_decode(token_ids, seq_lens, *kv_cache_,
                         seq_ids, logits, compute_stream_);
    compute_stream_.sync();

    // Benchmark
    CudaEvent start, end;
    start.record(compute_stream_);

    for (int step = 0; step < decode_steps; ++step) {
        for (int s = 0; s < num_seqs; ++s) {
            kv_cache_->extend_one_token(s);
            seq_lens[s]++;
        }
        model_->batch_decode(token_ids, seq_lens, *kv_cache_,
                             seq_ids, logits, compute_stream_);
    }

    end.record(compute_stream_);
    compute_stream_.sync();

    float total_ms = end.elapsed(start);
    float avg_ms   = total_ms / decode_steps;
    float tps = (float)(num_seqs * decode_steps) / (total_ms / 1000.f);

    printf("  Avg step latency: %.2f ms\n", avg_ms);
    printf("  Throughput:       %.0f tokens/sec\n", tps);

    // Cleanup
    for (int s = 0; s < num_seqs; ++s) {
        kv_cache_->free_sequence(s);
    }
}
