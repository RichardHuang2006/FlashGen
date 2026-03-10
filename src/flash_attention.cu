/*
 * flash_attention.cu
 *
 * FlashAttention-2 forward pass — CUDA implementation.
 *
 * Key ideas
 * ---------
 *  • Tile the sequence dimension: process Q in blocks of Br rows, iterate
 *    over K/V in blocks of Bc columns.
 *  • Maintain online softmax statistics (running max m, denominator l)
 *    so the full N×N score matrix never needs to be materialised.
 *  • Shared memory holds one Q tile and one K/V tile simultaneously.
 *  • One CUDA block handles one (batch, head, q_tile) triple.
 *  • Inner per-row work is distributed across 128 threads via a
 *    warp-cooperative loop, not one thread per row, so that the inner
 *    d-dimensional dot product is fast.
 *
 * Memory layout (all row-major)
 * ------------------------------
 *  Q, K, V, O : [batch, n_heads, seq_len, head_dim]
 *  L           : [batch, n_heads, seq_len]   logsumexp
 */

#include "flash_attention.cuh"
#include "cuda_utils.cuh"
#include <cuda_fp16.h>
#include <cfloat>
#include <cmath>

namespace flashgen {

// ── Shared-memory kernel template ────────────────────────────────────────────
//
// Template parameters
//  Br  : query tile size  (rows of Q per block)
//  Bc  : KV tile size     (cols of K/V per inner iteration)
//  Hd  : head dimension   (must be a compile-time constant ≤ 128)
//
template<int Br, int Bc, int Hd, typename scalar_t = float>
__global__ void flash_attn_kernel(
    const scalar_t* __restrict__ Q,   // [B, H, N, Hd]
    const scalar_t* __restrict__ K,   // [B, H, N, Hd]
    const scalar_t* __restrict__ V,   // [B, H, N, Hd]
    scalar_t*       __restrict__ O,   // [B, H, N, Hd]
    float*          __restrict__ L,   // [B, H, N]  logsumexp (FP32)
    int N,                             // sequence length
    float scale,
    bool causal
) {
    // blockIdx.x = q_tile index
    // blockIdx.y = head index
    // blockIdx.z = batch index
    const int q_tile  = blockIdx.x;
    const int head    = blockIdx.y;
    const int batch   = blockIdx.z;
    const int tid     = threadIdx.x;  // 0 .. blockDim.x-1
    const int nthreads = blockDim.x;

    // Starting row for this Q tile
    const int q_start = q_tile * Br;
    if (q_start >= N) return;

    // Stride to reach correct (batch, head) slice
    const long long slice = (long long)(batch * gridDim.y + head) * N * Hd;
    const long long Lslice = (long long)(batch * gridDim.y + head) * N;

    // Shared memory layout:
    //  sQ  [Br][Hd]
    //  sK  [Bc][Hd]
    //  sV  [Bc][Hd]
    //  sO  [Br][Hd]
    extern __shared__ float smem[];
    float* sQ = smem;
    float* sK = sQ + Br * Hd;
    float* sV = sK + Bc * Hd;
    float* sO = sV + Bc * Hd;

    // Per-row running statistics (registers)
    float m[Br], l_acc[Br];
    #pragma unroll
    for (int i = 0; i < Br; i++) { m[i] = -FLT_MAX; l_acc[i] = 0.f; }

    // ── Load Q tile into shared memory ───────────────────────────────────────
    for (int idx = tid; idx < Br * Hd; idx += nthreads) {
        int r = idx / Hd, c = idx % Hd;
        int global_row = q_start + r;
        sQ[idx] = (global_row < N)
                ? (float)Q[slice + global_row * Hd + c]
                : 0.f;
    }
    // Zero output accumulator
    for (int idx = tid; idx < Br * Hd; idx += nthreads) sO[idx] = 0.f;
    __syncthreads();

    // ── Iterate over KV tiles ─────────────────────────────────────────────────
    const int num_kv_tiles = (N + Bc - 1) / Bc;
    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
        const int kv_start = kv_tile * Bc;

        // Load K and V tiles
        for (int idx = tid; idx < Bc * Hd; idx += nthreads) {
            int r = idx / Hd, c = idx % Hd;
            int global_row = kv_start + r;
            float kval = (global_row < N) ? (float)K[slice + global_row * Hd + c] : 0.f;
            float vval = (global_row < N) ? (float)V[slice + global_row * Hd + c] : 0.f;
            sK[idx] = kval;
            sV[idx] = vval;
        }
        __syncthreads();

        // Each thread handles a contiguous range of Q rows
        for (int qi = tid; qi < Br; qi += nthreads) {
            int global_qi = q_start + qi;
            if (global_qi >= N) continue;

            // Compute raw dot products S[qi][0..Bc-1]
            float s[Bc];
            float m_local = -FLT_MAX;
            #pragma unroll
            for (int ki = 0; ki < Bc; ki++) {
                int global_ki = kv_start + ki;
                // Causal mask: if key position > query position, mask out
                if (causal && global_ki > global_qi) {
                    s[ki] = -FLT_MAX;
                    continue;
                }
                if (global_ki >= N) { s[ki] = -FLT_MAX; continue; }

                float dot = 0.f;
                #pragma unroll
                for (int d = 0; d < Hd; d++) {
                    dot += sQ[qi * Hd + d] * sK[ki * Hd + d];
                }
                s[ki] = dot * scale;
                m_local = fmaxf(m_local, s[ki]);
            }

            // Online softmax update
            float m_new   = fmaxf(m[qi], m_local);
            float exp_old = expf(m[qi] - m_new);          // rescale factor

            // Rescale old output accumulator
            #pragma unroll
            for (int d = 0; d < Hd; d++) sO[qi * Hd + d] *= exp_old;
            float l_new = l_acc[qi] * exp_old;

            // Add new P * V contribution
            #pragma unroll
            for (int ki = 0; ki < Bc; ki++) {
                float p = (s[ki] == -FLT_MAX) ? 0.f : expf(s[ki] - m_new);
                l_new += p;
                #pragma unroll
                for (int d = 0; d < Hd; d++) {
                    sO[qi * Hd + d] += p * sV[ki * Hd + d];
                }
            }
            m[qi]     = m_new;
            l_acc[qi] = l_new;
        }
        __syncthreads();
    }

    // ── Write normalised output ───────────────────────────────────────────────
    for (int qi = tid; qi < Br; qi += nthreads) {
        int global_qi = q_start + qi;
        if (global_qi >= N) continue;
        float inv_l = (l_acc[qi] > 0.f) ? 1.f / l_acc[qi] : 0.f;
        #pragma unroll
        for (int d = 0; d < Hd; d++) {
            O[slice + global_qi * Hd + d] = (scalar_t)(sO[qi * Hd + d] * inv_l);
        }
        if (L) {
            L[Lslice + global_qi] = logf(l_acc[qi]) + m[qi];
        }
    }
}

// ── Dispatch helper (C++17 compatible: no template lambda) ───────────────────
template<int Br, int Bc, int Hd, typename scalar_t>
static void launch_flash_attn_hd(
    const scalar_t* Q, const scalar_t* K, const scalar_t* V,
    scalar_t* O, float* L,
    int batch, int n_heads, int seq_len,
    float scale, bool causal, cudaStream_t stream
) {
    const int num_q_tiles = (seq_len + Br - 1) / Br;
    dim3 grid(num_q_tiles, n_heads, batch);
    const int threads = 128;
    const size_t smem = (size_t)(Br + 2 * Bc + Br) * Hd * sizeof(float);
    flash_attn_kernel<Br, Bc, Hd, scalar_t>
        <<<grid, threads, smem, stream>>>(Q, K, V, O, L, seq_len, scale, causal);
}

template<int Br, int Bc, typename scalar_t>
static void dispatch_flash_attn(
    const scalar_t* Q, const scalar_t* K, const scalar_t* V,
    scalar_t* O, float* L,
    int batch, int n_heads, int seq_len, int head_dim,
    float scale, bool causal, cudaStream_t stream
) {
    const int num_q_tiles = (seq_len + Br - 1) / Br;
    dim3 grid(num_q_tiles, n_heads, batch);
    const int threads = 128;
    const size_t smem = (size_t)(Br + 2 * Bc + Br) * head_dim * sizeof(float);

    switch (head_dim) {
        case 32:
            launch_flash_attn_hd<Br, Bc, 32, scalar_t>(
                Q, K, V, O, L, batch, n_heads, seq_len, scale, causal, stream);
            break;
        case 64:
            launch_flash_attn_hd<Br, Bc, 64, scalar_t>(
                Q, K, V, O, L, batch, n_heads, seq_len, scale, causal, stream);
            break;
        case 80:
            launch_flash_attn_hd<Br, Bc, 80, scalar_t>(
                Q, K, V, O, L, batch, n_heads, seq_len, scale, causal, stream);
            break;
        case 96:
            launch_flash_attn_hd<Br, Bc, 96, scalar_t>(
                Q, K, V, O, L, batch, n_heads, seq_len, scale, causal, stream);
            break;
        case 128:
            launch_flash_attn_hd<Br, Bc, 128, scalar_t>(
                Q, K, V, O, L, batch, n_heads, seq_len, scale, causal, stream);
            break;
        default:
            flash_attn_kernel<Br, Bc, 128, scalar_t>
                <<<grid, threads, smem, stream>>>(
                    Q, K, V, O, L, seq_len, scale, causal);
            break;
    }
    CUDA_CHECK(cudaGetLastError());
}

// ── Public API ────────────────────────────────────────────────────────────────

void flash_attention_forward(
    const float* Q, const float* K, const float* V,
    float* O, float* L,
    int batch, int n_heads, int seq_len, int head_dim,
    float scale, bool causal, cudaStream_t stream
) {
    dispatch_flash_attn<kFlashBr, kFlashBc, float>(
        Q, K, V, O, L, batch, n_heads, seq_len, head_dim,
        scale, causal, stream);
}

void flash_attention_forward_fp16(
    const __half* Q, const __half* K, const __half* V,
    __half* O, float* L,
    int batch, int n_heads, int seq_len, int head_dim,
    float scale, bool causal, cudaStream_t stream
) {
    dispatch_flash_attn<kFlashBr, kFlashBc, __half>(
        Q, K, V, O, L, batch, n_heads, seq_len, head_dim,
        scale, causal, stream);
}

// ── Incremental decoding (single new token + full KV cache) ──────────────────
//
// For decoding, q_new has seq=1 so the Q tile trivially holds one row.
// We iterate over the entire cached K/V with Bc tiles as usual.
//
__global__ void flash_decode_kernel(
    const float* __restrict__ q_new,    // [B, H, 1, Hd]
    float*       __restrict__ k_cache,  // [B, H, max_seq, Hd]
    float*       __restrict__ v_cache,
    const float* __restrict__ k_new,    // [B, H, 1, Hd]
    const float* __restrict__ v_new,
    float*       __restrict__ O,        // [B, H, 1, Hd]
    int          cache_len,             // tokens already in cache
    int          max_seq,
    int          head_dim,
    float        scale
) {
    const int head  = blockIdx.y;
    const int batch = blockIdx.z;
    const int tid   = threadIdx.x;

    const long long bh_stride = (long long)(batch * gridDim.y + head);
    const long long kv_off    = bh_stride * max_seq * head_dim;
    const long long q_off     = bh_stride * head_dim;

    // Append new K, V into cache at position cache_len
    for (int d = tid; d < head_dim; d += blockDim.x) {
        k_cache[kv_off + cache_len * head_dim + d] = k_new[q_off + d];
        v_cache[kv_off + cache_len * head_dim + d] = v_new[q_off + d];
    }
    __syncthreads();

    const int total_kv = cache_len + 1;

    // Each thread computes dot(q, k[j]) for a stripe of j values,
    // then does an online softmax over the full context.
    // We use a simple single-warp reduction here (head_dim ≤ 128).

    // Load q into registers
    float q_reg[128] = {};
    for (int d = 0; d < head_dim && d < 128; d++)
        q_reg[d] = q_new[q_off + d];

    // Compute all scores, find max
    float m_val = -FLT_MAX;
    // Store scores in shared memory
    extern __shared__ float smem_dec[];
    float* scores = smem_dec;

    for (int j = tid; j < total_kv; j += blockDim.x) {
        float dot = 0.f;
        for (int d = 0; d < head_dim; d++)
            dot += q_reg[d] * k_cache[kv_off + j * head_dim + d];
        scores[j] = dot * scale;
        m_val = fmaxf(m_val, scores[j]);
    }
    // Warp-level max reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        m_val = fmaxf(m_val, __shfl_xor_sync(0xffffffff, m_val, offset));

    // Compute softmax denominator
    float l_val = 0.f;
    for (int j = tid; j < total_kv; j += blockDim.x)
        l_val += expf(scores[j] - m_val);
    for (int offset = 16; offset > 0; offset >>= 1)
        l_val += __shfl_xor_sync(0xffffffff, l_val, offset);

    // Accumulate weighted V
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.f;
        for (int j = 0; j < total_kv; j++) {
            float p = expf(scores[j] - m_val) / l_val;
            acc += p * v_cache[kv_off + j * head_dim + d];
        }
        O[q_off + d] = acc;
    }
}

void flash_attention_decode(
    const float* q_new, float* k_cache, float* v_cache,
    const float* k_new, const float* v_new,
    float* O,
    int batch, int n_heads, int cache_len, int head_dim,
    float scale, cudaStream_t stream
) {
    // One warp per (batch, head), one block
    dim3 grid(1, n_heads, batch);
    const int threads = 32;
    // Shared memory for scores: (cache_len + 1) floats
    const size_t smem = (size_t)(cache_len + 1) * sizeof(float);

    flash_decode_kernel<<<grid, threads, smem, stream>>>(
        q_new, k_cache, v_cache, k_new, v_new, O,
        cache_len, /*max_seq=*/cache_len + 1, head_dim, scale);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace flashgen
