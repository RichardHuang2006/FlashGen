#pragma once

#include "cuda_utils.cuh"
#include "model_config.hpp"

#include <vector>
#include <mutex>
#include <cstdint>
#include <atomic>

// ---------------------------------------------------------------------------
//  Block allocator — vLLM-style GPU memory pool for KV cache
//
//  The allocator manages a flat GPU buffer carved into fixed-size blocks.
//  Each physical block stores K and V data for one block_size worth of tokens
//  across all layers and KV heads.
//
//  Physical layout per block:
//    [layer0_head0_K][layer0_head0_V][layer0_head1_K][layer0_head1_V]...
//    [layer1_head0_K][layer1_head0_V]...
//
//  Each K/V segment is block_size * head_dim * element_size bytes.
//
//  Blocks are reference-counted for prefix sharing and copy-on-write.
// ---------------------------------------------------------------------------

using PhysicalBlockIdx = int32_t;
static constexpr PhysicalBlockIdx kNullBlock = -1;

class BlockAllocator {
public:
    BlockAllocator(int num_blocks, int block_size, int num_layers,
                   int num_kv_heads, int head_dim, KVQuantType quant);
    ~BlockAllocator();

    BlockAllocator(const BlockAllocator&) = delete;
    BlockAllocator& operator=(const BlockAllocator&) = delete;

    // ── Core allocation ─────────────────────────────────────────────────

    /// Allocate a single block. Returns kNullBlock if pool exhausted.
    PhysicalBlockIdx allocate();

    /// Allocate n blocks atomically. Returns empty vector on failure.
    std::vector<PhysicalBlockIdx> allocate_n(int n);

    /// Release a block (decrements refcount; freed when it reaches 0).
    void free(PhysicalBlockIdx idx);

    /// Release multiple blocks.
    void free_all(const std::vector<PhysicalBlockIdx>& blocks);

    // ── Reference counting (prefix sharing) ─────────────────────────────

    /// Increment reference count (for sharing a block between sequences).
    void add_ref(PhysicalBlockIdx idx);

    /// Current reference count.
    int ref_count(PhysicalBlockIdx idx) const;

    // ── Copy-on-write ───────────────────────────────────────────────────

    /// If refcount > 1, allocate a new block, copy data, return new idx.
    /// If refcount == 1, return the same idx (no copy needed).
    PhysicalBlockIdx cow_copy(PhysicalBlockIdx src, cudaStream_t stream);

    // ── Accessors ───────────────────────────────────────────────────────

    /// Raw device pointer to the start of block `idx`.
    void* block_ptr(PhysicalBlockIdx idx) const;

    /// Device pointer into a specific layer's K or V segment within a block.
    /// is_value: false = K, true = V.
    void* block_ptr(PhysicalBlockIdx idx, int layer, int head, bool is_value) const;

    /// Base pointer to the entire GPU pool.
    void* pool_ptr() const { return pool_; }

    // ── Statistics ──────────────────────────────────────────────────────

    int  num_free()         const;
    int  num_total()        const { return num_blocks_; }
    int  block_size()       const { return block_size_; }
    size_t block_bytes()    const { return block_bytes_; }
    float utilization()     const;

    // ── Layout helpers (for kernels) ────────────────────────────────────

    /// Stride in bytes between consecutive physical blocks in the pool.
    size_t pool_block_stride()  const { return block_bytes_; }

    /// Stride in bytes between layers within one block.
    size_t layer_stride()       const { return layer_stride_; }

    /// Stride in bytes between KV heads within one layer-block.
    size_t head_stride()        const { return head_stride_; }

    /// Stride in bytes between K and V within one head-block.
    size_t kv_stride()          const { return kv_stride_; }

private:
    void* pool_ = nullptr;                     // GPU memory pool
    int   num_blocks_;
    int   block_size_;                         // tokens per block
    int   num_layers_;
    int   num_kv_heads_;
    int   head_dim_;
    int   elem_size_;                          // bytes per element

    size_t block_bytes_;                       // total bytes per block
    size_t layer_stride_;                      // bytes per layer in one block
    size_t head_stride_;                       // bytes per head in one layer-block
    size_t kv_stride_;                         // bytes for K (or V) in one head-block

    std::vector<PhysicalBlockIdx> free_list_;
    std::vector<int>              ref_counts_;
    mutable std::mutex            mu_;
};
