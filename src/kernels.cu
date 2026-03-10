/*
 * kernels.cu
 *
 * Fused CUDA kernels for:
 *   - Layer Normalization  (with optional fused residual add)
 *   - GELU activation      (in-place, exact erf form)
 *   - Feed-Forward Network (two GEMMs + fused GELU)
 *   - Token embedding lookup + positional encoding
 *   - LM-head projection
 *   - Temperature-scaled softmax
 */

#include "kernels.cuh"
#include "cuda_utils.cuh"
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cfloat>

namespace flashgen {

// ════════════════════════════════════════════════════════════════════════════
//  Warp utilities
// ════════════════════════════════════════════════════════════════════════════

static __device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, mask);
    return v;
}

static __device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, mask));
    return v;
}

// ════════════════════════════════════════════════════════════════════════════
//  Layer Normalization
// ════════════════════════════════════════════════════════════════════════════
//
// One warp (32 threads) per row.  Welford online algorithm for mean+variance.
// Supports cols up to 32 * any integer (handles cols > 32 via loop).

template<bool HAS_RESIDUAL>
__global__ void layer_norm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ residual,  // nullptr if HAS_RESIDUAL==false
    float*       __restrict__ out,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    int          rows,
    int          cols,
    float        eps
) {
    // One block = one warp = 32 threads, handles one row
    const int row = blockIdx.x;
    if (row >= rows) return;

    const float* x_row   = x   + row * cols;
    float*       out_row = out + row * cols;
    const float* res_row = HAS_RESIDUAL ? (residual + row * cols) : nullptr;

    // Welford online mean & variance
    float mean = 0.f, M2 = 0.f;
    int   count = 0;
    const int lane = threadIdx.x;  // 0..31

    for (int i = lane; i < cols; i += 32) {
        float xi = x_row[i];
        if constexpr (HAS_RESIDUAL) xi += res_row[i];
        count++;
        float delta  = xi - mean;
        mean += delta / count;
        float delta2 = xi - mean;
        M2   += delta * delta2;
    }
    // Reduction across warp: mean
    // We need sum over all threads, then divide by cols.
    // Simpler: accumulate partial sums, then reduce.
    // Re-derive as simple sum approach (cols known at runtime):
    float local_sum  = 0.f;
    float local_sum2 = 0.f;
    for (int i = lane; i < cols; i += 32) {
        float xi = x_row[i];
        if constexpr (HAS_RESIDUAL) xi += res_row[i];
        local_sum  += xi;
        local_sum2 += xi * xi;
    }
    local_sum  = warp_reduce_sum(local_sum);
    local_sum2 = warp_reduce_sum(local_sum2);

    const float mu    = local_sum / cols;
    const float var   = local_sum2 / cols - mu * mu;
    const float rstd  = rsqrtf(var + eps);

    // Normalize and apply affine parameters
    for (int i = lane; i < cols; i += 32) {
        float xi = x_row[i];
        if constexpr (HAS_RESIDUAL) xi += res_row[i];
        out_row[i] = gamma[i] * (xi - mu) * rstd + beta[i];
    }
}

void layer_norm(
    const float* x, float* out,
    const float* gamma, const float* beta,
    int rows, int cols, float eps, cudaStream_t stream
) {
    // One warp per row
    dim3 grid(rows), block(32);
    layer_norm_kernel<false><<<grid, block, 0, stream>>>(
        x, nullptr, out, gamma, beta, rows, cols, eps);
    CUDA_CHECK(cudaGetLastError());
}

void layer_norm_residual(
    const float* x, const float* residual, float* out,
    const float* gamma, const float* beta,
    int rows, int cols, float eps, cudaStream_t stream
) {
    dim3 grid(rows), block(32);
    layer_norm_kernel<true><<<grid, block, 0, stream>>>(
        x, residual, out, gamma, beta, rows, cols, eps);
    CUDA_CHECK(cudaGetLastError());
}

// FP16 layer norm (FP32 accumulation internally)
__global__ void layer_norm_fp16_kernel(
    const __half* __restrict__ x,
    __half*       __restrict__ out,
    const __half* __restrict__ gamma,
    const __half* __restrict__ beta,
    int rows, int cols, float eps
) {
    const int row  = blockIdx.x;
    if (row >= rows) return;
    const __half* x_row   = x   + row * cols;
    __half*       out_row = out + row * cols;
    const int     lane    = threadIdx.x;

    float local_sum = 0.f, local_sum2 = 0.f;
    for (int i = lane; i < cols; i += 32) {
        float xi = __half2float(x_row[i]);
        local_sum  += xi;
        local_sum2 += xi * xi;
    }
    local_sum  = warp_reduce_sum(local_sum);
    local_sum2 = warp_reduce_sum(local_sum2);

    const float mu   = local_sum / cols;
    const float var  = local_sum2 / cols - mu * mu;
    const float rstd = rsqrtf(var + eps);

    for (int i = lane; i < cols; i += 32) {
        float xi = __half2float(x_row[i]);
        float g  = __half2float(gamma[i]);
        float b  = __half2float(beta[i]);
        out_row[i] = __float2half(g * (xi - mu) * rstd + b);
    }
}

void layer_norm_fp16(
    const __half* x, __half* out,
    const __half* gamma, const __half* beta,
    int rows, int cols, float eps, cudaStream_t stream
) {
    dim3 grid(rows), block(32);
    layer_norm_fp16_kernel<<<grid, block, 0, stream>>>(
        x, out, gamma, beta, rows, cols, eps);
    CUDA_CHECK(cudaGetLastError());
}

// ════════════════════════════════════════════════════════════════════════════
//  GELU — exact form: 0.5*x*(1 + erf(x/sqrt(2)))
// ════════════════════════════════════════════════════════════════════════════

__global__ void gelu_kernel(float* __restrict__ x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i];
    x[i] = 0.5f * v * (1.f + erff(v * 0.7071067811865476f));
}

void gelu_inplace(float* x, int n, cudaStream_t stream) {
    const int block = 256;
    gelu_kernel<<<(n + block - 1) / block, block, 0, stream>>>(x, n);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void gelu_fp16_kernel(__half* __restrict__ x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = __half2float(x[i]);
    x[i] = __float2half(0.5f * v * (1.f + erff(v * 0.7071067811865476f)));
}

void gelu_inplace_fp16(__half* x, int n, cudaStream_t stream) {
    const int block = 256;
    gelu_fp16_kernel<<<(n + block - 1) / block, block, 0, stream>>>(x, n);
    CUDA_CHECK(cudaGetLastError());
}

// ════════════════════════════════════════════════════════════════════════════
//  Bias-add kernel (adds a bias vector to every row of a 2D tensor)
// ════════════════════════════════════════════════════════════════════════════

__global__ void bias_add_kernel(
    float* __restrict__ x,
    const float* __restrict__ bias,
    int rows, int cols
) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    if (col < cols && row < rows)
        x[row * cols + col] += bias[col];
}

void bias_add(float* x, const float* bias, int rows, int cols, cudaStream_t stream) {
    dim3 block(256);
    dim3 grid((cols + 255) / 256, rows);
    bias_add_kernel<<<grid, block, 0, stream>>>(x, bias, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

// ════════════════════════════════════════════════════════════════════════════
//  Fused Feed-Forward Network
//
//  Step 1:  hidden = x · W1ᵀ + b1        cuBLAS SGEMM
//  Step 2:  hidden = GELU(hidden)          custom kernel
//  Step 3:  out    = hidden · W2ᵀ + b2    cuBLAS SGEMM
// ════════════════════════════════════════════════════════════════════════════

void fused_ffn(
    const float* x, float* out, float* workspace,
    const FFNWeights& w,
    int rows, int d_model, int d_ff,
    void* cublas_handle_v, cudaStream_t stream
) {
    cublasHandle_t cublas = reinterpret_cast<cublasHandle_t>(cublas_handle_v);
    CUBLAS_CHECK(cublasSetStream(cublas, stream));

    const float alpha = 1.f, beta_zero = 0.f, beta_one = 1.f;

    // ── Step 1: hidden = x @ W1^T  [rows, d_model] x [d_model, d_ff]^T ──────
    // cuBLAS: C = alpha * A * B + beta * C
    // A = x     [rows x d_model]  (treated as row-major = column-major transposed)
    // B = W1    [d_ff x d_model]
    // We want hidden = x * W1^T, so in column-major: hidden^T = W1 * x^T
    // cublasSgemm(handle, transB, transA, n, m, k, ...)
    CUBLAS_CHECK(cublasSgemm(
        cublas,
        CUBLAS_OP_T,   // op(B) = W1^T  (in col-major: W1 is [d_model x d_ff])
        CUBLAS_OP_N,   // op(A) = x
        d_ff,          // n = number of output cols
        rows,          // m = number of rows
        d_model,       // k = inner dim
        &alpha,
        w.W1, d_model, // lda = d_model (W1: [d_ff, d_model] row-major)
        x,    d_model, // ldb = d_model (x:  [rows, d_model] row-major)
        &beta_zero,
        workspace, d_ff // ldc = d_ff
    ));

    // Add b1 bias and apply GELU
    if (w.b1) bias_add(workspace, w.b1, rows, d_ff, stream);
    gelu_inplace(workspace, rows * d_ff, stream);

    // ── Step 2: out = hidden @ W2^T  [rows, d_ff] x [d_ff, d_model]^T ────────
    CUBLAS_CHECK(cublasSgemm(
        cublas,
        CUBLAS_OP_T,    // W2^T
        CUBLAS_OP_N,
        d_model,        // n
        rows,           // m
        d_ff,           // k
        &alpha,
        w.W2,      d_ff,    // W2: [d_model, d_ff] row-major, lda = d_ff
        workspace, d_ff,
        &beta_zero,
        out, d_model
    ));

    if (w.b2) bias_add(out, w.b2, rows, d_model, stream);
}

// ════════════════════════════════════════════════════════════════════════════
//  Token embedding + positional encoding
// ════════════════════════════════════════════════════════════════════════════

__global__ void embed_tokens_kernel(
    const int*   __restrict__ ids,
    float*       __restrict__ out,
    const float* __restrict__ token_emb,
    const float* __restrict__ pos_emb,
    int seq_len, int d_model, int offset
) {
    // Grid: (seq_len * batch, d_model / 4)  — process 4 floats per thread
    const int token_idx = blockIdx.x;           // flat index into (batch * seq_len)
    const int d_base    = threadIdx.x * 4;
    if (d_base >= d_model) return;

    const int token_id = ids[token_idx];
    const int pos      = offset + (token_idx % seq_len);

    const float* te = token_emb + (long long)token_id * d_model + d_base;
    const float* pe = pos_emb   + (long long)pos       * d_model + d_base;
    float*       ot = out        + (long long)token_idx * d_model + d_base;

    // Vectorised 4-element load
    float4 t4 = *reinterpret_cast<const float4*>(te);
    float4 p4 = *reinterpret_cast<const float4*>(pe);
    float4 o4 = { t4.x + p4.x, t4.y + p4.y, t4.z + p4.z, t4.w + p4.w };
    *reinterpret_cast<float4*>(ot) = o4;
}

void embed_tokens(
    const int* ids, float* out,
    const float* token_emb, const float* pos_emb,
    int batch, int seq_len, int d_model, int offset, cudaStream_t stream
) {
    // d_model must be divisible by 4 for vectorised load; assert in debug
    const int total_tokens = batch * seq_len;
    dim3 grid(total_tokens);
    dim3 block(d_model / 4);
    embed_tokens_kernel<<<grid, block, 0, stream>>>(
        ids, out, token_emb, pos_emb, seq_len, d_model, offset);
    CUDA_CHECK(cudaGetLastError());
}

// ════════════════════════════════════════════════════════════════════════════
//  LM-head projection
// ════════════════════════════════════════════════════════════════════════════

void lm_head_project(
    const float* x, float* logits, const float* lm_head,
    int batch, int d_model, int vocab_size,
    void* cublas_handle_v, cudaStream_t stream
) {
    cublasHandle_t cublas = reinterpret_cast<cublasHandle_t>(cublas_handle_v);
    CUBLAS_CHECK(cublasSetStream(cublas, stream));
    const float alpha = 1.f, beta = 0.f;
    // logits = x @ lm_head^T   [batch, d_model] @ [d_model, vocab_size]^T
    CUBLAS_CHECK(cublasSgemm(
        cublas,
        CUBLAS_OP_T, CUBLAS_OP_N,
        vocab_size, batch, d_model,
        &alpha,
        lm_head, d_model,
        x,        d_model,
        &beta,
        logits, vocab_size
    ));
}

// ════════════════════════════════════════════════════════════════════════════
//  Temperature-scaled softmax  (in-place, one warp per row)
// ════════════════════════════════════════════════════════════════════════════

__global__ void softmax_temperature_kernel(
    float* __restrict__ logits,
    int rows, int cols, float inv_temp
) {
    const int row  = blockIdx.x;
    if (row >= rows) return;
    float* row_ptr = logits + (long long)row * cols;
    const int lane = threadIdx.x;

    // Find max
    float m = -FLT_MAX;
    for (int i = lane; i < cols; i += 32)
        m = fmaxf(m, row_ptr[i] * inv_temp);
    m = warp_reduce_max(m);

    // Exp and sum
    float s = 0.f;
    for (int i = lane; i < cols; i += 32) {
        float v = expf(row_ptr[i] * inv_temp - m);
        row_ptr[i] = v;
        s += v;
    }
    s = warp_reduce_sum(s);

    // Normalise
    for (int i = lane; i < cols; i += 32)
        row_ptr[i] /= s;
}

void softmax_temperature(
    float* logits, int rows, int cols, float temperature, cudaStream_t stream
) {
    const float inv_temp = (temperature > 0.f) ? 1.f / temperature : 1.f;
    softmax_temperature_kernel<<<rows, 32, 0, stream>>>(
        logits, rows, cols, inv_temp);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace flashgen
