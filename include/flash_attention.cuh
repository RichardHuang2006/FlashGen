#pragma once

#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
//  FlashAttention-2 — standard (contiguous KV) variants
//
//  These kernels implement IO-aware tiled attention with online softmax,
//  achieving O(N*d) GPU memory usage instead of O(N^2).
//
//  Tile sizes: Br=64 (query), Bc=64 (KV).
//  One CUDA block per (batch, head, Q-tile) triple.
//  FP32 accumulation regardless of I/O precision.
// ---------------------------------------------------------------------------

namespace flash_attention {

/// Full attention (prefill) — all Q tokens attend to all prior K/V tokens.
///   Q, K, V: [batch, n_heads, seq_len, head_dim]
///   O:       [batch, n_heads, seq_len, head_dim]
///   L:       [batch, n_heads, seq_len] (logsumexp, optional — can be nullptr)
void forward(const float* Q, const float* K, const float* V,
             float* O, float* L,
             int batch, int n_heads, int seq_len, int head_dim,
             bool causal, cudaStream_t stream);

/// FP16 I/O variant (FP32 accumulation internally).
void forward_fp16(const __half* Q, const __half* K, const __half* V,
                  __half* O, float* L,
                  int batch, int n_heads, int seq_len, int head_dim,
                  bool causal, cudaStream_t stream);

/// Incremental decode — single new query token attending to full KV cache.
///   Q:     [batch, n_heads, 1, head_dim]
///   K, V:  [batch, n_heads, kv_len, head_dim]
///   O:     [batch, n_heads, 1, head_dim]
void decode(const float* Q, const float* K, const float* V,
            float* O, int batch, int n_heads, int kv_len, int head_dim,
            cudaStream_t stream);

} // namespace flash_attention
