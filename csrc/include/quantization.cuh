#pragma once

#include "cuda_utils.cuh"
#include "model_config.hpp"

// ---------------------------------------------------------------------------
//  KV cache quantization — INT8 symmetric per-token quantization
//
//  Each token's K (or V) vector of [head_dim] floats is quantized to INT8
//  with a single per-token scale factor:
//      q[i] = round(clamp(x[i] / scale, -128, 127))
//      scale = max(|x[i]|) / 127
//
//  The scale factor is stored alongside the quantized data.
//  Dequantization is fused into the paged attention kernel to avoid
//  materializing full-precision intermediate buffers.
// ---------------------------------------------------------------------------

namespace quantization {

// ── Host-callable kernel launchers ──────────────────────────────────────

/// Quantize a contiguous [num_tokens, head_dim] FP32 buffer to INT8.
/// dst:    [num_tokens, head_dim] int8_t output
/// scales: [num_tokens] float output (one scale per token-vector)
void quantize_kv_int8(const float* src, int8_t* dst, float* scales,
                      int num_tokens, int head_dim, cudaStream_t stream);

/// Dequantize INT8 back to FP32 (for debugging / testing).
void dequantize_kv_int8(const int8_t* src, const float* scales, float* dst,
                        int num_tokens, int head_dim, cudaStream_t stream);

// ── Device-inline helpers (called from within paged attention kernel) ───

/// Compute dot(q, dequant(k)) = sum_i( q[i] * k_int8[i] ) * scale
/// Used for S = Q * K^T in paged attention.
__device__ __forceinline__
float dequant_dot_int8(const float* q, const int8_t* k, float scale, int dim) {
    float acc = 0.f;
    for (int i = 0; i < dim; ++i) {
        acc += q[i] * (float)k[i];
    }
    return acc * scale;
}

/// Accumulate dequantized V into output: acc[i] += weight * v_int8[i] * scale
/// Used for O += softmax_weight * V in paged attention.
__device__ __forceinline__
void dequant_accum_v_int8(float* acc, const int8_t* v, float scale,
                          float weight, int dim) {
    float w_s = weight * scale;
    for (int i = 0; i < dim; ++i) {
        acc[i] += w_s * (float)v[i];
    }
}

} // namespace quantization
