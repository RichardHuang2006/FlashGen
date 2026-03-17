#include "paged_kv_cache.cuh"
#include "quantization.cuh"

#include <algorithm>
#include <cassert>
#include <cstring>

// ===========================================================================
//  PagedKVCache implementation
// ===========================================================================

PagedKVCache::PagedKVCache(BlockAllocator& allocator, const ModelConfig& cfg)
    : allocator_(allocator),
      cfg_(cfg),
      block_size_(cfg.max_seq_len > 0 ? 16 : 16),  // default block size
      num_layers_(cfg.n_layers)
{
    block_size_ = allocator.block_size();
}

bool PagedKVCache::allocate_for_prefill(int seq_id, int prompt_len) {
    int num_blocks = (prompt_len + block_size_ - 1) / block_size_;

    auto blocks = allocator_.allocate_n(num_blocks * num_layers_);
    if (blocks.empty() && num_blocks > 0) return false;

    SequenceKVState state;
    state.seq_id      = seq_id;
    state.current_len = prompt_len;
    state.block_tables.resize(num_layers_);

    int idx = 0;
    for (int l = 0; l < num_layers_; ++l) {
        state.block_tables[l].resize(num_blocks);
        for (int b = 0; b < num_blocks; ++b) {
            state.block_tables[l][b] = blocks[idx++];
        }
    }

    states_[seq_id] = std::move(state);
    return true;
}

bool PagedKVCache::extend_one_token(int seq_id) {
    auto it = states_.find(seq_id);
    assert(it != states_.end());
    auto& state = it->second;

    int new_len    = state.current_len + 1;
    int new_blocks = (new_len + block_size_ - 1) / block_size_;
    int old_blocks = (int)state.block_tables[0].size();

    if (new_blocks > old_blocks) {
        // Need one more block per layer
        auto blocks = allocator_.allocate_n(num_layers_);
        if (blocks.empty()) return false;

        for (int l = 0; l < num_layers_; ++l) {
            state.block_tables[l].push_back(blocks[l]);
        }
    }

    state.current_len = new_len;
    return true;
}

void PagedKVCache::free_sequence(int seq_id) {
    auto it = states_.find(seq_id);
    if (it == states_.end()) return;

    for (auto& table : it->second.block_tables) {
        allocator_.free_all(table);
    }
    states_.erase(it);
}

bool PagedKVCache::fork_sequence(int src_seq_id, int new_seq_id) {
    auto it = states_.find(src_seq_id);
    if (it == states_.end()) return false;
    const auto& src = it->second;

    SequenceKVState dst;
    dst.seq_id      = new_seq_id;
    dst.current_len = src.current_len;
    dst.block_tables = src.block_tables;

    // Increment refcounts for all shared blocks
    for (auto& table : dst.block_tables) {
        for (auto idx : table) {
            allocator_.add_ref(idx);
        }
    }

    states_[new_seq_id] = std::move(dst);
    return true;
}

void PagedKVCache::attach_prefix_blocks(
    int seq_id,
    const std::vector<std::vector<PhysicalBlockIdx>>& prefix_blocks,
    int prefix_tokens)
{
    auto it = states_.find(seq_id);
    if (it == states_.end()) {
        // Create new state with prefix blocks
        SequenceKVState state;
        state.seq_id      = seq_id;
        state.current_len = prefix_tokens;
        state.block_tables = prefix_blocks;
        states_[seq_id] = std::move(state);
    } else {
        // Prepend prefix blocks (should be called before any allocation)
        auto& state = it->second;
        for (int l = 0; l < num_layers_; ++l) {
            auto& table = state.block_tables[l];
            table.insert(table.begin(),
                         prefix_blocks[l].begin(), prefix_blocks[l].end());
        }
        state.current_len += prefix_tokens;
    }
}

void PagedKVCache::ensure_staging(int num_seqs, int max_blocks) {
    int needed = num_seqs * max_blocks;
    if (needed <= staging_capacity_) return;

    staging_capacity_ = needed;
    h_block_tables_.allocate(needed);
    h_seq_lens_.allocate(num_seqs);
    d_block_tables_.allocate(needed);
    d_seq_lens_.allocate(num_seqs);
}

PagedAttentionParams PagedKVCache::prepare_batch(
    const std::vector<int>& seq_ids, int layer, cudaStream_t stream)
{
    int num_seqs = (int)seq_ids.size();

    // Find max blocks per sequence in this batch
    int max_blocks = 0;
    for (int sid : seq_ids) {
        auto it = states_.find(sid);
        assert(it != states_.end());
        max_blocks = std::max(max_blocks, (int)it->second.block_tables[layer].size());
    }

    ensure_staging(num_seqs, max_blocks);

    // Fill host staging buffers
    std::memset(h_block_tables_.ptr, 0xFF,
                num_seqs * max_blocks * sizeof(int32_t));  // -1 = null

    for (int i = 0; i < num_seqs; ++i) {
        auto& state = states_[seq_ids[i]];
        h_seq_lens_.ptr[i] = state.current_len;

        const auto& table = state.block_tables[layer];
        for (int b = 0; b < (int)table.size(); ++b) {
            h_block_tables_.ptr[i * max_blocks + b] = table[b];
        }
    }

    // Async upload to GPU
    CUDA_CHECK(cudaMemcpyAsync(d_block_tables_.ptr, h_block_tables_.ptr,
                               num_seqs * max_blocks * sizeof(int32_t),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_seq_lens_.ptr, h_seq_lens_.ptr,
                               num_seqs * sizeof(int32_t),
                               cudaMemcpyHostToDevice, stream));

    PagedAttentionParams params;
    params.kv_pool            = allocator_.pool_ptr();
    params.block_tables_gpu   = d_block_tables_.ptr;
    params.seq_lens_gpu       = d_seq_lens_.ptr;
    params.num_seqs           = num_seqs;
    params.max_blocks_per_seq = max_blocks;
    params.block_size         = block_size_;
    params.num_kv_heads       = cfg_.actual_kv_heads();
    params.head_dim           = cfg_.head_dim();
    params.scale              = 1.0f / sqrtf((float)cfg_.head_dim());
    params.pool_block_stride  = allocator_.pool_block_stride();
    params.layer_stride       = allocator_.layer_stride();
    params.head_stride        = allocator_.head_stride();
    params.kv_stride          = allocator_.kv_stride();

    return params;
}

bool PagedKVCache::has_sequence(int seq_id) const {
    return states_.count(seq_id) > 0;
}

int PagedKVCache::sequence_length(int seq_id) const {
    auto it = states_.find(seq_id);
    return it != states_.end() ? it->second.current_len : 0;
}

const SequenceKVState& PagedKVCache::get_state(int seq_id) const {
    return states_.at(seq_id);
}

// ===========================================================================
//  Paged attention CUDA kernels
// ===========================================================================

// ── Paged attention decode kernel ──────────────────────────────────────
//
// One warp per (sequence, head) pair. Each warp iterates over all KV
// blocks in the sequence's block table, computing online softmax attention.
//
// Grid:  (num_seqs, num_heads, 1)
// Block: (WARP_SIZE=32, 1, 1)

template <int HEAD_DIM, int BLOCK_SIZE>
__global__ void paged_attention_decode_kernel(
    float*       __restrict__ O,               // [num_seqs, num_heads, HEAD_DIM]
    const float* __restrict__ Q,               // [num_seqs, num_heads, HEAD_DIM]
    const char*  __restrict__ kv_pool,
    const int32_t* __restrict__ block_tables,  // [num_seqs, max_blocks_per_seq]
    const int32_t* __restrict__ seq_lens,      // [num_seqs]
    int num_kv_heads,
    int max_blocks_per_seq,
    float scale,
    size_t pool_block_stride,
    size_t layer_stride,
    size_t head_stride,
    size_t kv_stride,
    int layer)
{
    int seq_idx  = blockIdx.x;
    int head_idx = blockIdx.y;
    int tid      = threadIdx.x;

    int seq_len = seq_lens[seq_idx];
    if (seq_len == 0) return;

    int kv_head = head_idx / (gridDim.y / num_kv_heads);  // GQA mapping

    // Load query vector into registers
    float q_reg[HEAD_DIM];
    const float* q_ptr = Q + ((size_t)seq_idx * gridDim.y + head_idx) * HEAD_DIM;
    for (int d = tid; d < HEAD_DIM; d += 32) {
        q_reg[d] = q_ptr[d];
    }

    float m_prev = -1e30f;  // running max
    float l_prev = 0.f;     // running sum of exp
    float acc[HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; ++d) acc[d] = 0.f;

    int num_blocks = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int b = 0; b < num_blocks; ++b) {
        int32_t phys_block = block_tables[seq_idx * max_blocks_per_seq + b];
        if (phys_block < 0) continue;

        int tokens_in_block = min(BLOCK_SIZE, seq_len - b * BLOCK_SIZE);

        // Compute base pointer for this block's K and V
        const char* block_base = kv_pool + (size_t)phys_block * pool_block_stride
                               + (size_t)layer * layer_stride
                               + (size_t)kv_head * head_stride;
        const float* K_block = (const float*)(block_base);
        const float* V_block = (const float*)(block_base + kv_stride);

        for (int t = 0; t < tokens_in_block; ++t) {
            // Dot product: q · k[t]
            float dot = 0.f;
            for (int d = tid; d < HEAD_DIM; d += 32) {
                dot += q_reg[d] * K_block[t * HEAD_DIM + d];
            }
            // Warp reduction
            for (int offset = 16; offset > 0; offset >>= 1)
                dot += __shfl_down_sync(0xffffffff, dot, offset);
            dot = __shfl_sync(0xffffffff, dot, 0) * scale;

            // Online softmax update
            float m_new = fmaxf(m_prev, dot);
            float exp_prev = expf(m_prev - m_new);
            float exp_cur  = expf(dot - m_new);

            float l_new = l_prev * exp_prev + exp_cur;

            // Rescale accumulator and add new V contribution
            float rescale = l_prev * exp_prev / fmaxf(l_new, 1e-10f);
            float weight  = exp_cur / fmaxf(l_new, 1e-10f);

            for (int d = tid; d < HEAD_DIM; d += 32) {
                acc[d] = acc[d] * rescale + weight * V_block[t * HEAD_DIM + d];
            }

            m_prev = m_new;
            l_prev = l_new;
        }
    }

    // Write output
    float* o_ptr = O + ((size_t)seq_idx * gridDim.y + head_idx) * HEAD_DIM;
    for (int d = tid; d < HEAD_DIM; d += 32) {
        o_ptr[d] = acc[d];
    }
}

// ── Paged attention prefill kernel ─────────────────────────────────────
//
// Tiled flash attention reading KV through block tables.
// One CUDA block per (sequence, head, Q-tile).
// Br=64 query rows, Bc=64 KV columns.

static constexpr int PA_BR = 64;  // query tile rows
static constexpr int PA_BC = 64;  // KV tile cols

template <int HEAD_DIM>
__global__ void paged_attention_prefill_kernel(
    float*       __restrict__ O,
    const float* __restrict__ Q,
    const char*  __restrict__ kv_pool,
    const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ query_start_locs,
    int num_kv_heads,
    int num_q_heads,
    int max_blocks_per_seq,
    float scale,
    size_t pool_block_stride,
    size_t layer_stride,
    size_t head_stride,
    size_t kv_stride,
    int layer)
{
    int seq_idx  = blockIdx.x;
    int head_idx = blockIdx.y;
    int tile_idx = blockIdx.z;
    int tid      = threadIdx.x;

    int seq_len      = seq_lens[seq_idx];
    int query_start  = query_start_locs[seq_idx];
    int query_len    = query_start_locs[seq_idx + 1] - query_start;
    int kv_head      = head_idx / (num_q_heads / num_kv_heads);

    // This tile processes Q rows [tile_start, tile_end)
    int tile_start = tile_idx * PA_BR;
    int tile_end   = min(tile_start + PA_BR, query_len);
    if (tile_start >= query_len) return;

    int tile_rows = tile_end - tile_start;

    // Shared memory for Q tile and running statistics
    extern __shared__ float smem[];
    float* s_Q = smem;                                    // [PA_BR, HEAD_DIM]
    float* s_m = s_Q + PA_BR * HEAD_DIM;                  // [PA_BR]
    float* s_l = s_m + PA_BR;                             // [PA_BR]
    float* s_O = s_l + PA_BR;                             // [PA_BR, HEAD_DIM]
    float* s_K = s_O + PA_BR * HEAD_DIM;                  // [PA_BC, HEAD_DIM]
    float* s_V = s_K + PA_BC * HEAD_DIM;                  // [PA_BC, HEAD_DIM]
    float* s_S = s_V + PA_BC * HEAD_DIM;                  // [PA_BR, PA_BC]

    // Load Q tile into shared memory
    for (int i = tid; i < tile_rows * HEAD_DIM; i += blockDim.x) {
        int r = i / HEAD_DIM;
        int d = i % HEAD_DIM;
        int global_pos = query_start + tile_start + r;
        s_Q[r * HEAD_DIM + d] = Q[(size_t)global_pos * num_q_heads * HEAD_DIM
                                  + head_idx * HEAD_DIM + d];
    }

    // Initialize running max and sum
    for (int i = tid; i < tile_rows; i += blockDim.x) {
        s_m[i] = -1e30f;
        s_l[i] = 0.f;
    }
    for (int i = tid; i < tile_rows * HEAD_DIM; i += blockDim.x) {
        s_O[i] = 0.f;
    }
    __syncthreads();

    // Iterate over KV in tiles of PA_BC
    // For causal: only attend to positions <= query position
    int kv_len = seq_len;  // full KV length (may include prefix)
    int num_kv_tiles = (kv_len + PA_BC - 1) / PA_BC;

    for (int kv_tile = 0; kv_tile < num_kv_tiles; ++kv_tile) {
        int kv_start = kv_tile * PA_BC;
        int kv_end   = min(kv_start + PA_BC, kv_len);
        int kv_count = kv_end - kv_start;

        // Load K and V tiles from paged KV cache
        for (int i = tid; i < kv_count * HEAD_DIM; i += blockDim.x) {
            int t = i / HEAD_DIM;
            int d = i % HEAD_DIM;
            int global_kv_pos = kv_start + t;

            // Block table lookup
            int logical_block = global_kv_pos / block_tables[0];  // block_size from params
            int slot_in_block = global_kv_pos % max_blocks_per_seq;
            // Simplified: compute from actual block size
            int blk_idx  = global_kv_pos / 16;  // assuming block_size=16
            int slot     = global_kv_pos % 16;
            int32_t phys = block_tables[seq_idx * max_blocks_per_seq + blk_idx];

            if (phys >= 0) {
                const char* base = kv_pool + (size_t)phys * pool_block_stride
                                 + (size_t)layer * layer_stride
                                 + (size_t)kv_head * head_stride;
                s_K[t * HEAD_DIM + d] = ((const float*)base)[slot * HEAD_DIM + d];
                s_V[t * HEAD_DIM + d] = ((const float*)(base + kv_stride))[slot * HEAD_DIM + d];
            } else {
                s_K[t * HEAD_DIM + d] = 0.f;
                s_V[t * HEAD_DIM + d] = 0.f;
            }
        }
        __syncthreads();

        // Compute S = Q * K^T * scale
        for (int i = tid; i < tile_rows * kv_count; i += blockDim.x) {
            int r = i / kv_count;
            int c = i % kv_count;

            // Causal mask: query at position (tile_start + r) can attend
            // to KV at position (kv_start + c) only if kv_pos <= q_pos
            int q_pos  = tile_start + r;
            int kv_pos = kv_start + c;

            float dot = 0.f;
            for (int d = 0; d < HEAD_DIM; ++d) {
                dot += s_Q[r * HEAD_DIM + d] * s_K[c * HEAD_DIM + d];
            }
            dot *= scale;

            // Apply causal mask
            if (kv_pos > q_pos + (kv_len - query_len)) {
                dot = -1e30f;
            }
            s_S[r * kv_count + c] = dot;
        }
        __syncthreads();

        // Online softmax update: for each query row, update m, l, O
        for (int r = tid; r < tile_rows; r += blockDim.x) {
            float m_old = s_m[r];
            float l_old = s_l[r];

            // Find max in this tile
            float m_new = m_old;
            for (int c = 0; c < kv_count; ++c) {
                m_new = fmaxf(m_new, s_S[r * kv_count + c]);
            }

            // Compute exp and sum
            float exp_scale = expf(m_old - m_new);
            float l_new = l_old * exp_scale;
            for (int c = 0; c < kv_count; ++c) {
                l_new += expf(s_S[r * kv_count + c] - m_new);
            }

            // Rescale output and add new V contribution
            float rescale = (l_old * exp_scale) / fmaxf(l_new, 1e-10f);
            for (int d = 0; d < HEAD_DIM; ++d) {
                float o_val = s_O[r * HEAD_DIM + d] * rescale;
                float v_acc = 0.f;
                for (int c = 0; c < kv_count; ++c) {
                    float w = expf(s_S[r * kv_count + c] - m_new) /
                              fmaxf(l_new, 1e-10f);
                    v_acc += w * s_V[c * HEAD_DIM + d];
                }
                s_O[r * HEAD_DIM + d] = o_val + v_acc;
            }

            s_m[r] = m_new;
            s_l[r] = l_new;
        }
        __syncthreads();
    }

    // Write output
    for (int i = tid; i < tile_rows * HEAD_DIM; i += blockDim.x) {
        int r = i / HEAD_DIM;
        int d = i % HEAD_DIM;
        int global_pos = query_start + tile_start + r;
        O[(size_t)global_pos * num_q_heads * HEAD_DIM + head_idx * HEAD_DIM + d] = s_O[r * HEAD_DIM + d];
    }
}

// ── KV write kernel ────────────────────────────────────────────────────
// Write computed K/V into paged cache blocks.

__global__ void write_kv_to_paged_cache_kernel(
    const float* __restrict__ K,       // [num_tokens, num_kv_heads, head_dim]
    const float* __restrict__ V,
    char*        __restrict__ kv_pool,
    const int32_t* __restrict__ block_tables,  // [num_seqs, max_blocks_per_seq]
    const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ token_to_seq,  // [num_tokens] -> seq index
    const int32_t* __restrict__ token_pos,     // [num_tokens] -> position in seq
    int num_kv_heads,
    int head_dim,
    int max_blocks_per_seq,
    int block_size,
    size_t pool_block_stride,
    size_t layer_stride,
    size_t head_stride,
    size_t kv_stride,
    int layer)
{
    int token_idx = blockIdx.x;
    int head      = blockIdx.y;
    int d         = threadIdx.x;

    if (d >= head_dim) return;

    int seq  = token_to_seq[token_idx];
    int pos  = token_pos[token_idx];
    int blk  = pos / block_size;
    int slot = pos % block_size;

    int32_t phys = block_tables[seq * max_blocks_per_seq + blk];
    if (phys < 0) return;

    char* base = kv_pool + (size_t)phys * pool_block_stride
               + (size_t)layer * layer_stride
               + (size_t)head * head_stride;

    float* K_dst = (float*)base;
    float* V_dst = (float*)(base + kv_stride);

    size_t src_offset = (size_t)token_idx * num_kv_heads * head_dim + head * head_dim + d;
    K_dst[slot * head_dim + d] = K[src_offset];
    V_dst[slot * head_dim + d] = V[src_offset];
}

// ===========================================================================
//  Kernel launchers
// ===========================================================================

namespace paged_attention {

void decode(const float* Q, float* O,
            const PagedAttentionParams& p,
            int num_heads, int layer,
            cudaStream_t stream)
{
    dim3 grid(p.num_seqs, num_heads, 1);
    dim3 block(32, 1, 1);  // one warp per (seq, head)

    // Dispatch on head_dim
    switch (p.head_dim) {
    case 64:
        paged_attention_decode_kernel<64, 16><<<grid, block, 0, stream>>>(
            O, Q, (const char*)p.kv_pool, p.block_tables_gpu, p.seq_lens_gpu,
            p.num_kv_heads, p.max_blocks_per_seq, p.scale,
            p.pool_block_stride, p.layer_stride, p.head_stride, p.kv_stride,
            layer);
        break;
    case 80:
        paged_attention_decode_kernel<80, 16><<<grid, block, 0, stream>>>(
            O, Q, (const char*)p.kv_pool, p.block_tables_gpu, p.seq_lens_gpu,
            p.num_kv_heads, p.max_blocks_per_seq, p.scale,
            p.pool_block_stride, p.layer_stride, p.head_stride, p.kv_stride,
            layer);
        break;
    case 96:
        paged_attention_decode_kernel<96, 16><<<grid, block, 0, stream>>>(
            O, Q, (const char*)p.kv_pool, p.block_tables_gpu, p.seq_lens_gpu,
            p.num_kv_heads, p.max_blocks_per_seq, p.scale,
            p.pool_block_stride, p.layer_stride, p.head_stride, p.kv_stride,
            layer);
        break;
    case 128:
        paged_attention_decode_kernel<128, 16><<<grid, block, 0, stream>>>(
            O, Q, (const char*)p.kv_pool, p.block_tables_gpu, p.seq_lens_gpu,
            p.num_kv_heads, p.max_blocks_per_seq, p.scale,
            p.pool_block_stride, p.layer_stride, p.head_stride, p.kv_stride,
            layer);
        break;
    default:
        fprintf(stderr, "Unsupported head_dim=%d for paged attention\n", p.head_dim);
        exit(1);
    }
    CUDA_CHECK(cudaGetLastError());
}

void prefill(const float* Q, float* O,
             const PagedAttentionParams& p,
             const int32_t* query_start_locs,
             int total_tokens, int num_heads,
             int layer, cudaStream_t stream)
{
    // Find max query length to determine number of tiles
    // For simplicity, use total_tokens / num_seqs as approx
    int max_query_tiles = (total_tokens + PA_BR - 1) / PA_BR;

    dim3 grid(p.num_seqs, num_heads, max_query_tiles);
    dim3 block(128, 1, 1);

    // Shared memory: Q tile + m + l + O tile + K tile + V tile + S tile
    size_t smem = 0;
    switch (p.head_dim) {
    case 64:
        smem = (PA_BR * 64 + PA_BR + PA_BR + PA_BR * 64 + PA_BC * 64 + PA_BC * 64 + PA_BR * PA_BC) * sizeof(float);
        paged_attention_prefill_kernel<64><<<grid, block, smem, stream>>>(
            O, Q, (const char*)p.kv_pool, p.block_tables_gpu, p.seq_lens_gpu,
            query_start_locs, p.num_kv_heads, num_heads, p.max_blocks_per_seq,
            p.scale, p.pool_block_stride, p.layer_stride, p.head_stride,
            p.kv_stride, layer);
        break;
    case 128:
        smem = (PA_BR * 128 + PA_BR + PA_BR + PA_BR * 128 + PA_BC * 128 + PA_BC * 128 + PA_BR * PA_BC) * sizeof(float);
        paged_attention_prefill_kernel<128><<<grid, block, smem, stream>>>(
            O, Q, (const char*)p.kv_pool, p.block_tables_gpu, p.seq_lens_gpu,
            query_start_locs, p.num_kv_heads, num_heads, p.max_blocks_per_seq,
            p.scale, p.pool_block_stride, p.layer_stride, p.head_stride,
            p.kv_stride, layer);
        break;
    default:
        fprintf(stderr, "Unsupported head_dim=%d for paged prefill\n", p.head_dim);
        exit(1);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace paged_attention
