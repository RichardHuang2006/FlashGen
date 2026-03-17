#pragma once

#include <cstdint>
#include <chrono>
#include <functional>
#include <string>
#include <vector>
#include <atomic>

// ---------------------------------------------------------------------------
//  Sampling parameters
// ---------------------------------------------------------------------------

struct SamplingParams {
    float temperature       = 1.0f;
    float top_p             = 0.9f;
    int   top_k             = -1;          // -1 = disabled
    float repetition_penalty = 1.0f;
    bool  greedy            = false;
    int   max_tokens        = 128;
    std::vector<int> stop_token_ids = {50256};  // EOS for GPT-2
};

// ---------------------------------------------------------------------------
//  Sequence status
// ---------------------------------------------------------------------------

enum class SequenceStatus {
    WAITING,      // queued, not yet scheduled
    RUNNING,      // actively generating tokens
    PREEMPTED,    // evicted to reclaim KV blocks
    FINISHED,     // generation complete
    ABORTED,      // cancelled
};

enum class FinishReason {
    NONE,
    EOS_TOKEN,
    MAX_TOKENS,
    STOP_STRING,
    ABORT,
};

// ---------------------------------------------------------------------------
//  Sequence data — tracks one generation sequence
// ---------------------------------------------------------------------------

struct SequenceData {
    int                   seq_id          = -1;
    std::vector<int>      prompt_tokens;
    std::vector<int>      output_tokens;
    SequenceStatus        status          = SequenceStatus::WAITING;
    FinishReason          finish_reason   = FinishReason::NONE;
    int                   num_computed_tokens = 0; // KV entries in cache

    int prompt_len()  const { return (int)prompt_tokens.size(); }
    int output_len()  const { return (int)output_tokens.size(); }
    int total_len()   const { return prompt_len() + output_len(); }

    /// Number of KV blocks this sequence needs.
    int num_blocks_needed(int block_size) const {
        return (total_len() + block_size - 1) / block_size;
    }

    /// Tokens that still need prefill (haven't been computed yet).
    int remaining_prefill() const {
        return prompt_len() - num_computed_tokens;
    }

    bool is_finished() const {
        return status == SequenceStatus::FINISHED ||
               status == SequenceStatus::ABORTED;
    }
};

// ---------------------------------------------------------------------------
//  Inference response
// ---------------------------------------------------------------------------

struct InferenceResponse {
    int              request_id       = -1;
    std::vector<int> output_token_ids;
    std::string      generated_text;
    FinishReason     finish_reason    = FinishReason::NONE;
    float            latency_ms      = 0.f;
    float            tpot_ms         = 0.f;  // time per output token
    int              prompt_tokens   = 0;
    int              generated_tokens = 0;
    bool             success         = true;
    std::string      error;
};

// ---------------------------------------------------------------------------
//  Inference request — submitted by the client
// ---------------------------------------------------------------------------

struct InferenceRequest {
    int                 request_id   = -1;
    std::vector<int>    prompt_token_ids;
    SamplingParams      sampling;
    float               priority     = 0.f;   // higher = more important
    std::chrono::steady_clock::time_point arrival_time =
        std::chrono::steady_clock::now();

    using Callback = std::function<void(const InferenceResponse&)>;
    Callback callback;
};

// ---------------------------------------------------------------------------
//  Sequence group — one request may produce multiple sequences (beam search)
// ---------------------------------------------------------------------------

struct SequenceGroup {
    int                         request_id   = -1;
    std::vector<SequenceData>   sequences;        // usually 1 for greedy/sample
    SamplingParams              sampling;
    bool                        is_prefill   = true;  // first forward pass?
    float                       priority     = 0.f;
    std::chrono::steady_clock::time_point arrival_time;
    InferenceRequest::Callback  callback;

    int num_unfinished() const {
        int n = 0;
        for (auto& s : sequences)
            if (!s.is_finished()) ++n;
        return n;
    }

    int max_seq_len() const {
        int mx = 0;
        for (auto& s : sequences)
            mx = std::max(mx, s.total_len());
        return mx;
    }

    SequenceData& first_seq() { return sequences[0]; }
    const SequenceData& first_seq() const { return sequences[0]; }
};
