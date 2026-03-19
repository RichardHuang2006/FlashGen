#include "quantization.cuh"
#include <cfloat>

// ---------------------------------------------------------------------------
//  INT8 symmetric per-token quantization kernels
// ---------------------------------------------------------------------------

namespace quantization {

/// Quantize kernel: one block per token row.
/// Each row of [head_dim] floats → [head_dim] int8 + 1 scale.
__global__ void quantize_int8_kernel(
    const float* __restrict__ src,   // [num_tokens, head_dim]
    int8_t*      __restrict__ dst,   // [num_tokens, head_dim]
    float*       __restrict__ scales, // [num_tokens]
    int head_dim)
{
    int token = blockIdx.x;
    int tid   = threadIdx.x;

    const float* row = src + (size_t)token * head_dim;

    // Phase 1: find max absolute value via warp reduction
    float local_max = 0.f;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(row[d]));
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));

    // Shared memory for cross-warp reduction
    __shared__ float s_max[32];
    int warp_id = tid / 32;
    int lane    = tid % 32;
    if (lane == 0) s_max[warp_id] = local_max;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane < (blockDim.x + 31) / 32) ? s_max[lane] : 0.f;
        for (int offset = 16; offset > 0; offset >>= 1)
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
        if (lane == 0) s_max[0] = val;
    }
    __syncthreads();

    float amax  = s_max[0];
    float scale = amax / 127.f;
    if (scale < 1e-10f) scale = 1e-10f;
    float inv_scale = 127.f / fmaxf(amax, 1e-10f);

    // Phase 2: quantize
    int8_t* out_row = dst + (size_t)token * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float val = row[d] * inv_scale;
        val = fmaxf(-128.f, fminf(127.f, rintf(val)));
        out_row[d] = (int8_t)val;
    }

    if (tid == 0) scales[token] = scale;
}

/// Dequantize kernel: reverse of above.
__global__ void dequantize_int8_kernel(
    const int8_t* __restrict__ src,
    const float*  __restrict__ scales,
    float*        __restrict__ dst,
    int head_dim)
{
    int token = blockIdx.x;
    int tid   = threadIdx.x;

    float scale = scales[token];
    const int8_t* in_row  = src + (size_t)token * head_dim;
    float*        out_row = dst + (size_t)token * head_dim;

    for (int d = tid; d < head_dim; d += blockDim.x) {
        out_row[d] = (float)in_row[d] * scale;
    }
}

// ── Host API ────────────────────────────────────────────────────────────

void quantize_kv_int8(const float* src, int8_t* dst, float* scales,
                      int num_tokens, int head_dim, cudaStream_t stream) {
    if (num_tokens == 0) return;
    int threads = std::min(256, ((head_dim + 31) / 32) * 32);
    quantize_int8_kernel<<<num_tokens, threads, 0, stream>>>(
        src, dst, scales, head_dim);
    CUDA_CHECK(cudaGetLastError());
}

void dequantize_kv_int8(const int8_t* src, const float* scales, float* dst,
                        int num_tokens, int head_dim, cudaStream_t stream) {
    if (num_tokens == 0) return;
    int threads = std::min(256, ((head_dim + 31) / 32) * 32);
    dequantize_int8_kernel<<<num_tokens, threads, 0, stream>>>(
        src, scales, dst, head_dim);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace quantization
