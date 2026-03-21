#pragma once

#include "cuda_utils.cuh"
#include "model_config.hpp"
#include "block_allocator.hpp"

#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
//  Paged KV cache — vLLM-style block-table-based KV management
//
//  Each sequence owns a block table: an array of PhysicalBlockIdx values
//  that map logical token positions to physical blocks in the GPU pool.
//
//  logical_block = token_pos / block_size
//  slot_in_block = token_pos % block_size
//
//  The GPU pool is managed by BlockAllocator; this class manages the
//  per-sequence block table bookkeeping and prepares GPU-side tensors
//  for the paged attention kernel.
// ---------------------------------------------------------------------------

/// Per-sequence KV state.
struct SequenceKVState {
    int                              seq_id      = -1;
    int                              current_len = 0;    // tokens stored
    // block_tables[layer][logical_block_idx] = physical block index
    std::vector<std::vector<PhysicalBlockIdx>> block_tables;
};

/// Parameters passed to paged attention kernels.
struct PagedAttentionParams {
    void*          kv_pool;                // block allocator GPU pool base
    const int32_t* block_tables_gpu;       // [num_seqs, max_blocks_per_seq]
    const int32_t* seq_lens_gpu;           // [num_seqs]
    int            num_seqs;
    int            max_blocks_per_seq;
    int            block_size;
    int            num_kv_heads;
    int            head_dim;
    float          scale;                  // 1/sqrt(head_dim)
    size_t         pool_block_stride;      // bytes between physical blocks
    size_t         layer_stride;           // bytes between layers in a block
    size_t         head_stride;            // bytes between heads in a layer
    size_t         kv_stride;              // bytes for K (or V) in a head
};

class PagedKVCache {
public:
    PagedKVCache(BlockAllocator& allocator, const ModelConfig& cfg);

    // ── Sequence lifecycle ──────────────────────────────────────────────

    /// Allocate blocks for a new sequence's prompt.
    /// Returns false if insufficient blocks.
    bool allocate_for_prefill(int seq_id, int prompt_len);

    /// Extend sequence by one token (decode step).
    /// Allocates a new block if the current last block is full.
    bool extend_one_token(int seq_id);

    /// Free all blocks owned by a sequence.
    void free_sequence(int seq_id);

    /// Fork: new_seq shares all blocks with src_seq via refcounting.
    bool fork_sequence(int src_seq_id, int new_seq_id);

    // ── Prefix reuse ────────────────────────────────────────────────────

    /// Attach pre-matched prefix blocks (from PrefixCache) to a sequence.
    /// The caller has already incremented refcounts for these blocks.
    void attach_prefix_blocks(int seq_id,
                              const std::vector<std::vector<PhysicalBlockIdx>>& prefix_blocks,
                              int prefix_tokens);

    // ── GPU tensor preparation ──────────────────────────────────────────

    /// Upload block tables and seq lengths to GPU for a batch of sequences.
    /// Returns PagedAttentionParams ready for kernel launch.
    PagedAttentionParams prepare_batch(const std::vector<int>& seq_ids,
                                       int layer,
                                       cudaStream_t stream);

    // ── Queries ─────────────────────────────────────────────────────────

    bool has_sequence(int seq_id) const;
    int  sequence_length(int seq_id) const;
    const SequenceKVState& get_state(int seq_id) const;

private:
    BlockAllocator& allocator_;
    ModelConfig     cfg_;
    int             block_size_;
    int             num_layers_;

    std::unordered_map<int, SequenceKVState> states_;

    // Staging buffers for GPU upload (reused across calls)
    PinnedBuffer<int32_t> h_block_tables_;
    PinnedBuffer<int32_t> h_seq_lens_;
    DeviceBuffer<int32_t> d_block_tables_;
    DeviceBuffer<int32_t> d_seq_lens_;
    int staging_capacity_ = 0;

    void ensure_staging(int num_seqs, int max_blocks);
};

// ---------------------------------------------------------------------------
//  Paged attention kernel launchers
// ---------------------------------------------------------------------------

namespace paged_attention {

/// Paged flash attention — prefill (multi-token query, tiled).
/// Q: [total_tokens, num_heads, head_dim]
/// O: [total_tokens, num_heads, head_dim]
void prefill(const float* Q, float* O,
             const PagedAttentionParams& params,
             const int32_t* query_start_locs,   // [num_seqs + 1]
             int total_tokens, int num_heads,
             int layer, cudaStream_t stream);

/// Paged attention — decode (one query token per sequence).
/// Q: [num_seqs, num_heads, head_dim]
/// O: [num_seqs, num_heads, head_dim]
void decode(const float* Q, float* O,
            const PagedAttentionParams& params,
            int num_heads, int layer,
            cudaStream_t stream);

} // namespace paged_attention
