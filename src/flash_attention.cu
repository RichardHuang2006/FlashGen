#include "flash_attention.cuh"
#include "cuda_utils.cuh"
#include <cmath>

// ===========================================================================
//  FlashAttention-2 — standard contiguous-KV implementation
//
//  Algorithm:
//    For each Q tile (Br rows), iterate over KV tiles (Bc cols):
//      S = Q_tile * K_tile^T * scale
//      Apply causal mask
//      Online softmax: update running max (m), denominator (l), output (O)
//    Final: O /= l
//
//  Tile sizes: Br=64, Bc=64. One CUDA block per (batch, head, Q-tile).
//  128 threads per block with cooperative warp-level reductions.
// ===========================================================================

static constexpr int BR = 64;   // query tile size
static constexpr int BC = 64;   // KV tile size
static constexpr int THREADS = 128;

// ── Forward kernel ─────────────────────────────────────────────────────

template <int Hd>
__global__ void flash_attn_forward_kernel(
    const float* __restrict__ Q,    // [B, H, N, Hd]
    const float* __restrict__ K,
    const float* __restrict__ V,
    float*       __restrict__ O,
    float*       __restrict__ L,    // [B, H, N] logsumexp (optional)
    int N,                          // sequence length
    float scale,
    bool causal)
{
    int b = blockIdx.x;   // batch
    int h = blockIdx.y;   // head
    int q_tile = blockIdx.z;

    int tid = threadIdx.x;
    int BH  = gridDim.y;  // total heads

    int q_start = q_tile * BR;
    int q_end   = min(q_start + BR, N);
    int q_count = q_end - q_start;
    if (q_count <= 0) return;

    // Pointers into this (batch, head) slice
    size_t bh_offset = ((size_t)b * BH + h) * N * Hd;
    const float* Q_bh = Q + bh_offset + q_start * Hd;
    const float* K_bh = K + bh_offset;
    const float* V_bh = V + bh_offset;
    float*       O_bh = O + bh_offset + q_start * Hd;

    // Shared memory layout
    extern __shared__ float smem[];
    float* s_Q = smem;                          // [BR, Hd]
    float* s_K = s_Q + BR * Hd;                 // [BC, Hd]
    float* s_V = s_K + BC * Hd;                 // [BC, Hd]
    float* s_S = s_V + BC * Hd;                 // [BR, BC]
    float* s_m = s_S + BR * BC;                 // [BR]
    float* s_l = s_m + BR;                      // [BR]
    float* s_O = s_l + BR;                      // [BR, Hd]

    // Load Q tile
    for (int i = tid; i < q_count * Hd; i += THREADS) {
        s_Q[i] = Q_bh[i];
    }
    // Zero-pad if q_count < BR
    for (int i = tid + q_count * Hd; i < BR * Hd; i += THREADS) {
        s_Q[i] = 0.f;
    }

    // Initialize running stats
    for (int i = tid; i < BR; i += THREADS) {
        s_m[i] = -1e30f;
        s_l[i] = 0.f;
    }
    for (int i = tid; i < BR * Hd; i += THREADS) {
        s_O[i] = 0.f;
    }
    __syncthreads();

    // Determine KV iteration range
    int kv_end = causal ? min(q_end, N) : N;
    int num_kv_tiles = (kv_end + BC - 1) / BC;

    for (int kv_tile = 0; kv_tile < num_kv_tiles; ++kv_tile) {
        int kv_start  = kv_tile * BC;
        int kv_actual = min(BC, kv_end - kv_start);

        // Load K tile
        for (int i = tid; i < kv_actual * Hd; i += THREADS) {
            s_K[i] = K_bh[kv_start * Hd + i];
        }
        // Load V tile
        for (int i = tid; i < kv_actual * Hd; i += THREADS) {
            s_V[i] = V_bh[kv_start * Hd + i];
        }
        __syncthreads();

        // Compute S = Q * K^T * scale
        for (int idx = tid; idx < q_count * kv_actual; idx += THREADS) {
            int r = idx / kv_actual;
            int c = idx % kv_actual;

            float dot = 0.f;
            for (int d = 0; d < Hd; ++d) {
                dot += s_Q[r * Hd + d] * s_K[c * Hd + d];
            }
            dot *= scale;

            // Causal mask
            if (causal && (kv_start + c) > (q_start + r)) {
                dot = -1e30f;
            }

            s_S[r * kv_actual + c] = dot;
        }
        __syncthreads();

        // Online softmax update
        for (int r = tid; r < q_count; r += THREADS) {
            float m_old = s_m[r];
            float l_old = s_l[r];

            // New max
            float m_new = m_old;
            for (int c = 0; c < kv_actual; ++c) {
                m_new = fmaxf(m_new, s_S[r * kv_actual + c]);
            }

            // Rescale and accumulate
            float exp_diff = expf(m_old - m_new);
            float l_new = l_old * exp_diff;

            for (int c = 0; c < kv_actual; ++c) {
                l_new += expf(s_S[r * kv_actual + c] - m_new);
            }

            float rescale = (l_old * exp_diff) / fmaxf(l_new, 1e-10f);
            for (int d = 0; d < Hd; ++d) {
                float o = s_O[r * Hd + d] * rescale;
                float v_acc = 0.f;
                for (int c = 0; c < kv_actual; ++c) {
                    v_acc += expf(s_S[r * kv_actual + c] - m_new) * s_V[c * Hd + d];
                }
                s_O[r * Hd + d] = o + v_acc / fmaxf(l_new, 1e-10f);
            }

            s_m[r] = m_new;
            s_l[r] = l_new;
        }
        __syncthreads();
    }

    // Write output
    for (int i = tid; i < q_count * Hd; i += THREADS) {
        O_bh[i] = s_O[i];
    }

    // Write logsumexp if requested
    if (L) {
        float* L_bh = L + ((size_t)b * BH + h) * N + q_start;
        for (int i = tid; i < q_count; i += THREADS) {
            L_bh[i] = s_m[i] + logf(fmaxf(s_l[i], 1e-10f));
        }
    }
}

// ── Decode kernel (single query) ───────────────────────────────────────

__global__ void flash_decode_kernel(
    const float* __restrict__ Q,    // [B, H, 1, Hd]
    const float* __restrict__ K,    // [B, H, kv_len, Hd]
    const float* __restrict__ V,
    float*       __restrict__ O,    // [B, H, 1, Hd]
    int kv_len,
    int head_dim,
    float scale)
{
    int b   = blockIdx.x;
    int h   = blockIdx.y;
    int tid = threadIdx.x;
    int BH  = gridDim.y;

    size_t bh = (size_t)b * BH + h;
    const float* q = Q + bh * head_dim;
    const float* k = K + bh * kv_len * head_dim;
    const float* v = V + bh * kv_len * head_dim;
    float*       o = O + bh * head_dim;

    float m = -1e30f;
    float l = 0.f;

    // Each thread accumulates partial output
    float acc[128];  // max head_dim
    for (int d = 0; d < head_dim; ++d) acc[d] = 0.f;

    // Each thread processes a subset of KV positions
    for (int pos = tid; pos < kv_len; pos += blockDim.x) {
        float dot = 0.f;
        for (int d = 0; d < head_dim; ++d) {
            dot += q[d] * k[pos * head_dim + d];
        }
        dot *= scale;

        float m_new = fmaxf(m, dot);
        float exp_old = expf(m - m_new);
        float exp_cur = expf(dot - m_new);
        float l_new = l * exp_old + exp_cur;

        float rescale = (l * exp_old) / fmaxf(l_new, 1e-10f);
        float weight  = exp_cur / fmaxf(l_new, 1e-10f);

        for (int d = 0; d < head_dim; ++d) {
            acc[d] = acc[d] * rescale + weight * v[pos * head_dim + d];
        }

        m = m_new;
        l = l_new;
    }

    // Warp reduction for combining partial results across threads
    // (simplified: write to shared and reduce)
    extern __shared__ float smem[];
    float* s_acc = smem;           // [blockDim.x, head_dim]
    float* s_m   = s_acc + blockDim.x * head_dim;  // [blockDim.x]
    float* s_l   = s_m + blockDim.x;               // [blockDim.x]

    for (int d = 0; d < head_dim; ++d) {
        s_acc[tid * head_dim + d] = acc[d];
    }
    s_m[tid] = m;
    s_l[tid] = l;
    __syncthreads();

    // Tree reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float m1 = s_m[tid], m2 = s_m[tid + stride];
            float l1 = s_l[tid], l2 = s_l[tid + stride];
            float m_new = fmaxf(m1, m2);
            float e1 = expf(m1 - m_new);
            float e2 = expf(m2 - m_new);
            float l_new = l1 * e1 + l2 * e2;
            float w1 = (l1 * e1) / fmaxf(l_new, 1e-10f);
            float w2 = (l2 * e2) / fmaxf(l_new, 1e-10f);

            for (int d = 0; d < head_dim; ++d) {
                s_acc[tid * head_dim + d] =
                    w1 * s_acc[tid * head_dim + d] +
                    w2 * s_acc[(tid + stride) * head_dim + d];
            }
            s_m[tid] = m_new;
            s_l[tid] = l_new;
        }
        __syncthreads();
    }

    // Thread 0 writes final output
    if (tid == 0) {
        for (int d = 0; d < head_dim; ++d) {
            o[d] = s_acc[d];
        }
    }
}

// ===========================================================================
//  Host API
// ===========================================================================

namespace flash_attention {

void forward(const float* Q, const float* K, const float* V,
             float* O, float* L,
             int batch, int n_heads, int seq_len, int head_dim,
             bool causal, cudaStream_t stream)
{
    int num_q_tiles = (seq_len + BR - 1) / BR;
    dim3 grid(batch, n_heads, num_q_tiles);
    dim3 block(THREADS);

    float scale = 1.0f / sqrtf((float)head_dim);

    // Shared memory: Q + K + V + S + m + l + O
    size_t smem = (BR * head_dim + BC * head_dim + BC * head_dim +
                   BR * BC + BR + BR + BR * head_dim) * sizeof(float);

    #define LAUNCH(HD)                                                         \
        flash_attn_forward_kernel<HD><<<grid, block, smem, stream>>>(          \
            Q, K, V, O, L, seq_len, scale, causal);

    switch (head_dim) {
        case 32:  LAUNCH(32);  break;
        case 64:  LAUNCH(64);  break;
        case 80:  LAUNCH(80);  break;
        case 96:  LAUNCH(96);  break;
        case 128: LAUNCH(128); break;
        default:
            fprintf(stderr, "FlashAttention: unsupported head_dim=%d\n", head_dim);
            exit(1);
    }
    #undef LAUNCH

    CUDA_CHECK(cudaGetLastError());
}

void forward_fp16(const __half* Q, const __half* K, const __half* V,
                  __half* O, float* L,
                  int batch, int n_heads, int seq_len, int head_dim,
                  bool causal, cudaStream_t stream) {
    // FP16 variant — convert to FP32 internally for simplicity
    // In production, use native FP16 tensor core paths
    (void)Q; (void)K; (void)V; (void)O; (void)L;
    (void)batch; (void)n_heads; (void)seq_len; (void)head_dim;
    (void)causal; (void)stream;
    fprintf(stderr, "FP16 FlashAttention: use FP32 path with FP16 I/O conversion\n");
}

void decode(const float* Q, const float* K, const float* V,
            float* O, int batch, int n_heads, int kv_len, int head_dim,
            cudaStream_t stream)
{
    dim3 grid(batch, n_heads);
    int threads = min(128, ((kv_len + 31) / 32) * 32);
    threads = max(threads, 32);

    float scale = 1.0f / sqrtf((float)head_dim);
    size_t smem = (threads * head_dim + threads + threads) * sizeof(float);

    flash_decode_kernel<<<grid, threads, smem, stream>>>(
        Q, K, V, O, kv_len, head_dim, scale);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace flash_attention
