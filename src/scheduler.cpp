#include "scheduler.hpp"
#include <algorithm>
#include <cassert>

// ===========================================================================
//  Continuous batching scheduler — vLLM-style three-queue scheduling
// ===========================================================================

Scheduler::Scheduler(const SchedulerConfig& config, BlockAllocator& allocator,
                     PagedKVCache& kv_cache, PrefixCache& prefix_cache)
    : config_(config),
      allocator_(allocator),
      kv_cache_(kv_cache),
      prefix_cache_(prefix_cache)
{}

void Scheduler::add_request(std::shared_ptr<SequenceGroup> sg) {
    std::lock_guard<std::mutex> lock(mu_);

    // Assign sequence IDs
    for (auto& seq : sg->sequences) {
        seq.seq_id = next_seq_id_++;
    }

    waiting_.push_back(std::move(sg));
}

void Scheduler::abort_request(int request_id) {
    std::lock_guard<std::mutex> lock(mu_);

    // Check waiting queue
    for (auto it = waiting_.begin(); it != waiting_.end(); ++it) {
        if ((*it)->request_id == request_id) {
            waiting_.erase(it);
            return;
        }
    }

    // Check running queue — free blocks
    for (auto it = running_.begin(); it != running_.end(); ++it) {
        if ((*it)->request_id == request_id) {
            for (auto& seq : (*it)->sequences) {
                kv_cache_.free_sequence(seq.seq_id);
                seq.status = SequenceStatus::ABORTED;
            }
            running_.erase(it);
            return;
        }
    }
}

SchedulerOutput Scheduler::schedule() {
    std::lock_guard<std::mutex> lock(mu_);

    SchedulerOutput output;
    int token_budget = config_.max_num_tokens;

    // Phase 1: Resume preempted sequences (LIFO order)
    try_resume_preempted(output, token_budget);

    // Phase 2: Admit new sequences from waiting queue (FCFS)
    try_admit_waiting(output, token_budget);

    // Phase 3: Continue running decode sequences
    schedule_decodes(output, token_budget);

    return output;
}

void Scheduler::try_resume_preempted(SchedulerOutput& output, int& token_budget) {
    while (!preempted_.empty()) {
        auto& sg = preempted_.back();
        auto& seq = sg->first_seq();

        // For RECOMPUTE preemption: need to re-prefill the entire sequence
        int tokens_needed = seq.total_len();
        int blocks_needed = seq.num_blocks_needed(allocator_.block_size());

        if (!can_allocate_blocks(blocks_needed * kv_cache_.get_state(seq.seq_id).block_tables.size()) ||
            tokens_needed > token_budget) {
            break;  // can't resume yet
        }

        if ((int)(output.scheduled_prefills.size() + output.scheduled_decodes.size())
            >= config_.max_num_seqs) {
            break;
        }

        // Re-allocate blocks and schedule for re-prefill
        sg->is_prefill = true;
        seq.num_computed_tokens = 0;  // recompute from scratch
        seq.status = SequenceStatus::RUNNING;

        if (kv_cache_.allocate_for_prefill(seq.seq_id, seq.total_len())) {
            output.scheduled_prefills.push_back(sg.get());
            output.num_prefill_tokens += tokens_needed;
            token_budget -= tokens_needed;
            running_.push_back(std::move(sg));
            preempted_.pop_back();
        } else {
            break;
        }
    }
}

void Scheduler::try_admit_waiting(SchedulerOutput& output, int& token_budget) {
    while (!waiting_.empty()) {
        auto& sg = waiting_.front();
        auto& seq = sg->first_seq();

        if ((int)(output.scheduled_prefills.size() + output.scheduled_decodes.size())
            >= config_.max_num_seqs) {
            break;
        }

        // Check prefix cache for reusable KV blocks
        int prefill_tokens = seq.prompt_len();
        PrefixMatchResult prefix_match;

        if (prefix_cache_.enabled()) {
            prefix_match = prefix_cache_.match_prefix(seq.prompt_tokens);
            prefill_tokens -= prefix_match.tokens_matched;
        }

        // Chunked prefill: limit tokens per iteration
        if (config_.enable_chunked_prefill) {
            prefill_tokens = std::min(prefill_tokens, config_.max_prefill_tokens);
        }

        if (prefill_tokens > token_budget) break;

        // Check block availability
        int total_blocks_needed = seq.num_blocks_needed(allocator_.block_size());
        int prefix_blocks = (int)(prefix_match.block_indices.empty() ? 0 :
                                  prefix_match.block_indices[0].size());
        int new_blocks_needed = (total_blocks_needed - prefix_blocks);

        // Need new_blocks per layer
        int total_new = new_blocks_needed * (int)kv_cache_.get_state(0).block_tables.size();
        // Simplified: check total blocks
        if (new_blocks_needed > 0 && !can_allocate_blocks(new_blocks_needed)) {
            // Try preemption
            if (!running_.empty()) {
                preempt_last(output);
                continue;  // retry
            }
            break;  // truly out of memory
        }

        // Admit this request
        sg->is_prefill = true;
        seq.status = SequenceStatus::RUNNING;

        // Attach prefix blocks
        if (prefix_match.tokens_matched > 0) {
            kv_cache_.attach_prefix_blocks(seq.seq_id,
                                            prefix_match.block_indices,
                                            prefix_match.tokens_matched);
            seq.num_computed_tokens = prefix_match.tokens_matched;
        }

        // Allocate remaining blocks
        int remaining = seq.prompt_len() - prefix_match.tokens_matched;
        if (remaining > 0) {
            kv_cache_.allocate_for_prefill(seq.seq_id, seq.prompt_len());
        }

        output.scheduled_prefills.push_back(sg.get());
        output.num_prefill_tokens += prefill_tokens;
        token_budget -= prefill_tokens;

        running_.push_back(std::move(sg));
        waiting_.pop_front();
    }
}

void Scheduler::schedule_decodes(SchedulerOutput& output, int& token_budget) {
    for (auto it = running_.begin(); it != running_.end(); ) {
        auto& sg = *it;

        // Skip sequences that are already scheduled for prefill
        bool in_prefill = false;
        for (auto* p : output.scheduled_prefills) {
            if (p == sg.get()) { in_prefill = true; break; }
        }
        if (in_prefill) { ++it; continue; }

        auto& seq = sg->first_seq();
        if (seq.is_finished()) { ++it; continue; }

        if (token_budget <= 0) break;
        if ((int)(output.scheduled_prefills.size() + output.scheduled_decodes.size())
            >= config_.max_num_seqs) {
            break;
        }

        // Each decode step needs 1 token and possibly 1 new block
        if (!kv_cache_.extend_one_token(seq.seq_id)) {
            // Out of blocks — preempt something
            if (running_.size() > 1) {
                preempt_last(output);
                continue;
            }
            break;
        }

        sg->is_prefill = false;
        output.scheduled_decodes.push_back(sg.get());
        output.num_decode_tokens += 1;
        token_budget -= 1;
        ++it;
    }
}

void Scheduler::post_step(const std::vector<SequenceGroup*>& batch,
                          const std::vector<int>& new_token_ids) {
    std::lock_guard<std::mutex> lock(mu_);

    int token_idx = 0;
    for (auto* sg : batch) {
        auto& seq = sg->first_seq();

        if (token_idx < (int)new_token_ids.size()) {
            int new_token = new_token_ids[token_idx++];
            seq.output_tokens.push_back(new_token);

            if (sg->is_prefill) {
                seq.num_computed_tokens = seq.prompt_len();
                sg->is_prefill = false;
            }

            // Check finish conditions
            for (int stop_id : sg->sampling.stop_token_ids) {
                if (new_token == stop_id) {
                    seq.status = SequenceStatus::FINISHED;
                    seq.finish_reason = FinishReason::EOS_TOKEN;
                    break;
                }
            }
            if (seq.output_len() >= sg->sampling.max_tokens) {
                seq.status = SequenceStatus::FINISHED;
                seq.finish_reason = FinishReason::MAX_TOKENS;
            }
        }
    }

    // Move finished sequences out of running queue
    for (auto it = running_.begin(); it != running_.end(); ) {
        if ((*it)->first_seq().is_finished()) {
            // Register blocks in prefix cache before freeing
            auto& seq = (*it)->first_seq();
            if (prefix_cache_.enabled() && kv_cache_.has_sequence(seq.seq_id)) {
                auto& state = kv_cache_.get_state(seq.seq_id);
                prefix_cache_.insert(seq.prompt_tokens, state.block_tables);
            }
            kv_cache_.free_sequence(seq.seq_id);
            it = running_.erase(it);
        } else {
            ++it;
        }
    }
}

void Scheduler::preempt_last(SchedulerOutput& output) {
    if (running_.empty()) return;

    auto sg = std::move(running_.back());
    running_.pop_back();

    auto& seq = sg->first_seq();
    seq.status = SequenceStatus::PREEMPTED;

    // Free KV blocks (recompute preemption)
    if (config_.preemption_mode == SchedulerConfig::PreemptionMode::RECOMPUTE) {
        kv_cache_.free_sequence(seq.seq_id);
    }

    output.preempted.push_back(sg.get());
    preempted_.push_back(std::move(sg));
}

bool Scheduler::can_allocate_blocks(int num_blocks) const {
    return allocator_.num_free() >= num_blocks;
}

int Scheduler::num_waiting() const {
    std::lock_guard<std::mutex> lock(mu_);
    return (int)waiting_.size();
}

int Scheduler::num_running() const {
    std::lock_guard<std::mutex> lock(mu_);
    return (int)running_.size();
}

int Scheduler::num_preempted() const {
    std::lock_guard<std::mutex> lock(mu_);
    return (int)preempted_.size();
}

bool Scheduler::has_pending() const {
    std::lock_guard<std::mutex> lock(mu_);
    return !waiting_.empty() || !running_.empty() || !preempted_.empty();
}
