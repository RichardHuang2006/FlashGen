#pragma once

#include "model_config.hpp"
#include "request.hpp"
#include "block_allocator.hpp"
#include "paged_kv_cache.cuh"
#include "prefix_cache.hpp"

#include <deque>
#include <memory>
#include <mutex>
#include <vector>

// ---------------------------------------------------------------------------
//  Continuous batching scheduler — vLLM-style iteration-level scheduling
//
//  Three queues:
//    WAITING   — newly submitted, not yet scheduled
//    RUNNING   — actively generating tokens (have KV blocks allocated)
//    PREEMPTED — evicted to free memory, can be resumed
//
//  Each call to schedule() produces a SchedulerOutput describing which
//  sequences to prefill/decode in the current iteration, and what block
//  operations to perform beforehand.
//
//  Scheduling algorithm per iteration:
//    1. Resume preempted sequences (LIFO) if blocks available
//    2. Admit waiting sequences (FCFS) with prefix cache lookup
//    3. Continue running decode sequences
//    4. If out of blocks, preempt lowest-priority running sequence
// ---------------------------------------------------------------------------

struct SchedulerOutput {
    std::vector<SequenceGroup*> scheduled_prefills;
    std::vector<SequenceGroup*> scheduled_decodes;
    std::vector<SequenceGroup*> preempted;
    std::vector<SequenceGroup*> finished;

    int num_prefill_tokens = 0;
    int num_decode_tokens  = 0;
    int total_tokens()     const { return num_prefill_tokens + num_decode_tokens; }
    int num_scheduled()    const {
        return (int)(scheduled_prefills.size() + scheduled_decodes.size());
    }
    bool empty() const { return num_scheduled() == 0; }
};

class Scheduler {
public:
    Scheduler(const SchedulerConfig& config, BlockAllocator& allocator,
              PagedKVCache& kv_cache, PrefixCache& prefix_cache);

    // ── Request management ──────────────────────────────────────────────

    /// Add a new inference request to the waiting queue.
    void add_request(std::shared_ptr<SequenceGroup> sg);

    /// Abort a request by ID. Frees any allocated blocks.
    void abort_request(int request_id);

    // ── Core scheduling ─────────────────────────────────────────────────

    /// Run one scheduling iteration. Returns the batch to execute.
    SchedulerOutput schedule();

    /// After the forward pass: append new tokens, check for completion.
    /// finished_out receives shared_ptrs to completed sequences (kept alive
    /// so callers can safely access them after they leave the running queue).
    void post_step(const std::vector<SequenceGroup*>& batch,
                   const std::vector<int>& new_token_ids,
                   std::vector<std::shared_ptr<SequenceGroup>>& finished_out);

    /// Convenience overload — discards finished sequence references.
    void post_step(const std::vector<SequenceGroup*>& batch,
                   const std::vector<int>& new_token_ids) {
        std::vector<std::shared_ptr<SequenceGroup>> dummy;
        post_step(batch, new_token_ids, dummy);
    }

    // ── Queries ─────────────────────────────────────────────────────────

    int  num_waiting()   const;
    int  num_running()   const;
    int  num_preempted() const;
    bool has_pending()   const;

private:
    SchedulerConfig  config_;
    BlockAllocator&  allocator_;
    PagedKVCache&    kv_cache_;
    PrefixCache&     prefix_cache_;

    std::deque<std::shared_ptr<SequenceGroup>> waiting_;
    std::deque<std::shared_ptr<SequenceGroup>> running_;
    std::deque<std::shared_ptr<SequenceGroup>> preempted_;

    mutable std::mutex mu_;
    int next_seq_id_ = 0;

    bool can_allocate_blocks(int num_blocks) const;
    void preempt_last(SchedulerOutput& output);
    void try_admit_waiting(SchedulerOutput& output, int& token_budget);
    void try_resume_preempted(SchedulerOutput& output, int& token_budget);
    void schedule_decodes(SchedulerOutput& output, int& token_budget);
};
