#include "block_allocator.hpp"
#include <algorithm>
#include <cassert>
#include <cstring>

// ---------------------------------------------------------------------------
//  BlockAllocator implementation
// ---------------------------------------------------------------------------

BlockAllocator::BlockAllocator(int num_blocks, int block_size, int num_layers,
                               int num_kv_heads, int head_dim, KVQuantType quant)
    : num_blocks_(num_blocks),
      block_size_(block_size),
      num_layers_(num_layers),
      num_kv_heads_(num_kv_heads),
      head_dim_(head_dim),
      elem_size_(kv_quant_element_size(quant))
{
    // Compute layout strides
    // Per-head K or V segment: block_size * head_dim * elem_size
    kv_stride_    = (size_t)block_size * head_dim * elem_size_;
    // Per-head pair (K + V): 2 * kv_stride
    head_stride_  = kv_stride_ * 2;
    // Per-layer: num_kv_heads * head_stride
    layer_stride_ = head_stride_ * num_kv_heads;
    // Per-block: num_layers * layer_stride
    block_bytes_  = layer_stride_ * num_layers;

    // Allocate GPU pool
    size_t total_bytes = (size_t)num_blocks * block_bytes_;
    if (total_bytes > 0) {
        CUDA_CHECK(cudaMalloc(&pool_, total_bytes));
        CUDA_CHECK(cudaMemset(pool_, 0, total_bytes));
    }

    // Initialize free list (all blocks available)
    free_list_.resize(num_blocks);
    for (int i = 0; i < num_blocks; ++i) {
        free_list_[i] = num_blocks - 1 - i;  // stack: pop from back
    }

    // Initialize reference counts
    ref_counts_.resize(num_blocks, 0);
}

BlockAllocator::~BlockAllocator() {
    if (pool_) {
        cudaFree(pool_);
        pool_ = nullptr;
    }
}

PhysicalBlockIdx BlockAllocator::allocate() {
    std::lock_guard<std::mutex> lock(mu_);
    if (free_list_.empty()) return kNullBlock;

    PhysicalBlockIdx idx = free_list_.back();
    free_list_.pop_back();
    ref_counts_[idx] = 1;
    return idx;
}

std::vector<PhysicalBlockIdx> BlockAllocator::allocate_n(int n) {
    std::lock_guard<std::mutex> lock(mu_);
    if ((int)free_list_.size() < n) return {};

    std::vector<PhysicalBlockIdx> result(n);
    for (int i = 0; i < n; ++i) {
        result[i] = free_list_.back();
        free_list_.pop_back();
        ref_counts_[result[i]] = 1;
    }
    return result;
}

void BlockAllocator::free(PhysicalBlockIdx idx) {
    if (idx == kNullBlock) return;
    std::lock_guard<std::mutex> lock(mu_);
    assert(idx >= 0 && idx < num_blocks_);
    assert(ref_counts_[idx] > 0);

    --ref_counts_[idx];
    if (ref_counts_[idx] == 0) {
        free_list_.push_back(idx);
    }
}

void BlockAllocator::free_all(const std::vector<PhysicalBlockIdx>& blocks) {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto idx : blocks) {
        if (idx == kNullBlock) continue;
        assert(idx >= 0 && idx < num_blocks_);
        assert(ref_counts_[idx] > 0);
        --ref_counts_[idx];
        if (ref_counts_[idx] == 0) {
            free_list_.push_back(idx);
        }
    }
}

void BlockAllocator::add_ref(PhysicalBlockIdx idx) {
    if (idx == kNullBlock) return;
    std::lock_guard<std::mutex> lock(mu_);
    assert(idx >= 0 && idx < num_blocks_);
    assert(ref_counts_[idx] > 0);
    ++ref_counts_[idx];
}

int BlockAllocator::ref_count(PhysicalBlockIdx idx) const {
    if (idx == kNullBlock) return 0;
    std::lock_guard<std::mutex> lock(mu_);
    return ref_counts_[idx];
}

PhysicalBlockIdx BlockAllocator::cow_copy(PhysicalBlockIdx src, cudaStream_t stream) {
    if (src == kNullBlock) return kNullBlock;

    std::lock_guard<std::mutex> lock(mu_);
    assert(src >= 0 && src < num_blocks_);

    // No copy needed if this is the sole owner
    if (ref_counts_[src] == 1) return src;

    // Allocate new block
    if (free_list_.empty()) return kNullBlock;
    PhysicalBlockIdx dst = free_list_.back();
    free_list_.pop_back();
    ref_counts_[dst] = 1;

    // Copy data on GPU
    void* src_ptr = static_cast<char*>(pool_) + (size_t)src * block_bytes_;
    void* dst_ptr = static_cast<char*>(pool_) + (size_t)dst * block_bytes_;
    CUDA_CHECK(cudaMemcpyAsync(dst_ptr, src_ptr, block_bytes_,
                               cudaMemcpyDeviceToDevice, stream));

    // Decrement source ref
    --ref_counts_[src];
    if (ref_counts_[src] == 0) {
        free_list_.push_back(src);
    }

    return dst;
}

void* BlockAllocator::block_ptr(PhysicalBlockIdx idx) const {
    assert(idx >= 0 && idx < num_blocks_);
    return static_cast<char*>(pool_) + (size_t)idx * block_bytes_;
}

void* BlockAllocator::block_ptr(PhysicalBlockIdx idx, int layer,
                                 int head, bool is_value) const {
    assert(idx >= 0 && idx < num_blocks_);
    size_t offset = (size_t)idx * block_bytes_
                  + (size_t)layer * layer_stride_
                  + (size_t)head  * head_stride_
                  + (is_value ? kv_stride_ : 0);
    return static_cast<char*>(pool_) + offset;
}

int BlockAllocator::num_free() const {
    std::lock_guard<std::mutex> lock(mu_);
    return (int)free_list_.size();
}

float BlockAllocator::utilization() const {
    std::lock_guard<std::mutex> lock(mu_);
    int used = num_blocks_ - (int)free_list_.size();
    return num_blocks_ > 0 ? (float)used / num_blocks_ : 0.f;
}
