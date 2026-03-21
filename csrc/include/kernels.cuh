#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

// ---------------------------------------------------------------------------
//  Fused transformer kernels
// ---------------------------------------------------------------------------

namespace kernels {

// ── Layer normalization ─────────────────────────────────────────────────

/// LayerNorm: out[i] = gamma * (x[i] - mean) / sqrt(var + eps) + beta
/// One warp per row, Welford online algorithm.
void layer_norm(const float* x, const float* gamma, const float* beta,
                float* out, int rows, int cols, float eps,
                cudaStream_t stream);

/// Fused residual + LayerNorm: out = LN(x + residual)
void layer_norm_residual(const float* x, const float* residual,
                         const float* gamma, const float* beta,
                         float* out, int rows, int cols, float eps,
                         cudaStream_t stream);

/// RMSNorm: out[i] = gamma[i] * x[i] / sqrt(mean(x^2) + eps)
void rms_norm(const float* x, const float* gamma, float* out,
              int rows, int cols, float eps, cudaStream_t stream);

// ── Activations ─────────────────────────────────────────────────────────

/// GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
void gelu(float* x, int n, cudaStream_t stream);

/// SiLU (Swish): x * sigmoid(x)
void silu(float* x, int n, cudaStream_t stream);

/// Fused SiLU + element-wise multiply (for LLaMA-style gated FFN):
///   out[i] = silu(gate[i]) * up[i]
void silu_multiply(const float* gate, const float* up, float* out,
                   int n, cudaStream_t stream);

// ── Feed-forward network ────────────────────────────────────────────────

/// Fused FFN: out = W2 * GELU(W1 * x + b1) + b2
/// workspace must hold at least rows * d_ff floats.
void fused_ffn(const float* x, const float* W1, const float* b1,
               const float* W2, const float* b2,
               float* out, float* workspace,
               int rows, int d_model, int d_ff,
               cublasHandle_t cublas, cudaStream_t stream);

// ── Embedding ───────────────────────────────────────────────────────────

/// Token + positional embedding lookup (fused, vectorized).
/// out[i] = token_emb[token_ids[i]] + pos_emb[positions[i]]
void embedding(const int* token_ids, const int* positions,
               const float* token_emb, const float* pos_emb,
               float* out, int num_tokens, int d_model,
               cudaStream_t stream);

/// Token embedding lookup only (no positional — for RoPE models).
void token_embedding(const int* token_ids, const float* token_emb,
                     float* out, int num_tokens, int d_model,
                     cudaStream_t stream);

// ── Rotary positional encoding (RoPE) ───────────────────────────────────

/// Apply RoPE in-place to Q and K tensors.
/// Q, K: [num_tokens, num_heads, head_dim]
/// positions: [num_tokens]
void apply_rope(float* Q, float* K, const int* positions,
                int num_tokens, int num_q_heads, int num_kv_heads,
                int head_dim, float theta, cudaStream_t stream);

// ── Softmax & sampling ──────────────────────────────────────────────────

/// Temperature-scaled softmax (in-place).
/// logits: [rows, cols] — divided by temperature, then softmax.
void softmax_temperature(float* logits, int rows, int cols,
                         float temperature, cudaStream_t stream);

/// Top-k filtering: set logits outside top-k to -inf.
void top_k_filter(float* logits, int rows, int cols, int k,
                  cudaStream_t stream);

/// Top-p (nucleus) filtering: set logits outside nucleus to -inf.
void top_p_filter(float* logits, int rows, int cols, float p,
                  cudaStream_t stream);

/// Argmax over each row.
void argmax(const float* logits, int* out, int rows, int cols,
            cudaStream_t stream);

// ── Linear projection (cuBLAS wrapper) ──────────────────────────────────

/// out = x @ W^T + bias
/// x: [M, K], W: [N, K], bias: [N] (can be nullptr), out: [M, N]
void linear(const float* x, const float* W, const float* bias,
            float* out, int M, int N, int K,
            cublasHandle_t cublas, cudaStream_t stream);

} // namespace kernels
