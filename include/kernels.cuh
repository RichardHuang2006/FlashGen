#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace flashgen {

// ════════════════════════════════════════════════════════════════════════════
//  Layer Normalization
// ════════════════════════════════════════════════════════════════════════════

// Standard layer norm:  y = gamma * (x - mean) / sqrt(var + eps) + beta
//
// x     : [rows, cols]  (input, modified in-place if residual != nullptr)
// out   : [rows, cols]  (output)
// gamma : [cols]
// beta  : [cols]
// residual : [rows, cols] or nullptr  — if provided, fuses x += residual
//
// One warp per row; warp-level reduction for mean/variance.
void layer_norm(
    const float* x,
    float*       out,
    const float* gamma,
    const float* beta,
    int          rows,
    int          cols,
    float        eps,
    cudaStream_t stream
);

// Fused variant: out = LayerNorm(x + residual)
void layer_norm_residual(
    const float* x,
    const float* residual,
    float*       out,
    const float* gamma,
    const float* beta,
    int          rows,
    int          cols,
    float        eps,
    cudaStream_t stream
);

// FP16 version
void layer_norm_fp16(
    const __half* x,
    __half*       out,
    const __half* gamma,
    const __half* beta,
    int           rows,
    int           cols,
    float         eps,
    cudaStream_t  stream
);

// ════════════════════════════════════════════════════════════════════════════
//  GELU activation  (exact Gaussian CDF form, not tanh approximation)
//
//  GELU(x) = x * Φ(x)  where Φ is the standard normal CDF
//  Approximation used: 0.5 * x * (1 + erf(x / sqrt(2)))
// ════════════════════════════════════════════════════════════════════════════

// Element-wise fused GELU applied to pre-allocated buffer
void gelu_inplace(float* x, int n, cudaStream_t stream);

// FP16 version
void gelu_inplace_fp16(__half* x, int n, cudaStream_t stream);

// ════════════════════════════════════════════════════════════════════════════
//  Fused Feed-Forward Network
//
//  FFN(x) = GELU(x · W1 + b1) · W2 + b2
//
//  Uses cuBLAS GEMMs for W1/W2, custom fused GELU between them.
//  Workspace must be pre-allocated to at least rows * d_ff * sizeof(float).
// ════════════════════════════════════════════════════════════════════════════
struct FFNWeights {
    const float* W1;   // [d_ff, d_model]  (cuBLAS row-major = col-major BLAS)
    const float* b1;   // [d_ff]
    const float* W2;   // [d_model, d_ff]
    const float* b2;   // [d_model]
};

void fused_ffn(
    const float* x,         // [rows, d_model]
    float*       out,        // [rows, d_model]
    float*       workspace,  // [rows, d_ff]  scratch buffer
    const FFNWeights& w,
    int           rows,
    int           d_model,
    int           d_ff,
    void*         cublas_handle,
    cudaStream_t  stream
);

// ════════════════════════════════════════════════════════════════════════════
//  Embedding lookup + positional encoding
// ════════════════════════════════════════════════════════════════════════════

// Token embedding: out[i] = token_emb[ids[i]] + pos_emb[offset + i]
// out     : [batch * seq_len, d_model]
// token_emb : [vocab_size, d_model]
// pos_emb   : [max_seq_len, d_model]
// ids       : [batch * seq_len]
// offset    : starting position index (for KV-cache decoding)
void embed_tokens(
    const int*   ids,
    float*       out,
    const float* token_emb,
    const float* pos_emb,
    int          batch,
    int          seq_len,
    int          d_model,
    int          offset,
    cudaStream_t stream
);

// ════════════════════════════════════════════════════════════════════════════
//  Output projection + softmax (final LM head)
// ════════════════════════════════════════════════════════════════════════════

// logits = x · lm_head^T   then optional temperature-scaled softmax
// x       : [batch, d_model]   (last token only at decoding time)
// logits  : [batch, vocab_size]
// lm_head : [vocab_size, d_model]
void lm_head_project(
    const float* x,
    float*       logits,
    const float* lm_head,
    int          batch,
    int          d_model,
    int          vocab_size,
    void*        cublas_handle,
    cudaStream_t stream
);

// In-place bias addition: adds bias[cols] to every row of x[rows, cols]
void bias_add(float* x, const float* bias, int rows, int cols, cudaStream_t stream);

// In-place temperature scaling + softmax over last dimension
// logits : [rows, cols]
void softmax_temperature(
    float*       logits,
    int          rows,
    int          cols,
    float        temperature,
    cudaStream_t stream
);

} // namespace flashgen
