#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "model_config.hpp"

namespace flashgen {

// ── Tile dimensions (must divide evenly into shared memory) ──────────────────
// Br: rows of Q processed per block  (query tile)
// Bc: cols of K/V processed per inner loop iteration (key-value tile)
// These must satisfy: (Br + 2*Bc) * d * sizeof(float) <= shared memory limit
static constexpr int kFlashBr = 64;   // query block rows
static constexpr int kFlashBc = 64;   // key/value block cols

// ── FlashAttention-2 forward pass  (FP32) ────────────────────────────────────
//
// Q, K, V : [batch, n_heads, seq_len, head_dim]  (row-major)
// O       : [batch, n_heads, seq_len, head_dim]  (output)
// L       : [batch, n_heads, seq_len]             (logsumexp, for backward)
//
// scale   : 1 / sqrt(head_dim)
// causal  : if true, apply causal (lower-triangular) mask
//
void flash_attention_forward(
    const float* Q,
    const float* K,
    const float* V,
    float*       O,
    float*       L,        // can be nullptr if logsumexp not needed
    int          batch,
    int          n_heads,
    int          seq_len,
    int          head_dim,
    float        scale,
    bool         causal,
    cudaStream_t stream
);

// ── FlashAttention-2 forward pass  (FP16 I/O, FP32 accumulation) ─────────────
void flash_attention_forward_fp16(
    const __half* Q,
    const __half* K,
    const __half* V,
    __half*       O,
    float*        L,
    int           batch,
    int           n_heads,
    int           seq_len,
    int           head_dim,
    float         scale,
    bool          causal,
    cudaStream_t  stream
);

// ── KV-cache incremental decoding step ──────────────────────────────────────
//
// Appends the new K/V vectors for the current token to the rolling cache,
// then runs FlashAttention over the full cached sequence.
//
// q_new       : [batch, n_heads, 1, head_dim]  new query (one step)
// k_cache     : [batch, n_heads, max_seq, head_dim]  mutable KV cache
// v_cache     : [batch, n_heads, max_seq, head_dim]
// k_new, v_new: [batch, n_heads, 1, head_dim]  new key and value to append
// cache_len   : current number of cached tokens (before this step)
//
void flash_attention_decode(
    const float* q_new,
    float*       k_cache,
    float*       v_cache,
    const float* k_new,
    const float* v_new,
    float*       O,
    int          batch,
    int          n_heads,
    int          cache_len,
    int          head_dim,
    float        scale,
    cudaStream_t stream
);

} // namespace flashgen
