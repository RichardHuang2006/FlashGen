#include "transformer.hpp"
#include "flash_attention.cuh"
#include "kernels.cuh"
#include "paged_kv_cache.cuh"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <numeric>

// ===========================================================================
//  ForwardBuffers — pre-allocated scratch memory
// ===========================================================================

void ForwardBuffers::allocate(const ModelConfig& cfg, int max_tokens, int max_seqs) {
    int d = cfg.d_model;
    int kv_dim = cfg.actual_kv_heads() * cfg.head_dim();

    hidden.allocate((size_t)max_tokens * d);
    residual.allocate((size_t)max_tokens * d);
    norm_out.allocate((size_t)max_tokens * d);
    // Q: d_model, K: kv_dim, V: kv_dim
    qkv.allocate((size_t)max_tokens * (d + 2 * kv_dim));
    attn_out.allocate((size_t)max_tokens * d);
    ffn_workspace.allocate((size_t)max_tokens * cfg.d_ff);
    logits.allocate((size_t)max_seqs * cfg.vocab_size);
    positions.allocate(max_tokens);
    token_ids.allocate(max_tokens);
    query_start_locs.allocate(max_seqs + 1);
}

// ===========================================================================
//  Transformer
// ===========================================================================

Transformer::Transformer(const ModelConfig& cfg)
    : cfg_(cfg)
{
    cublas_.set_stream(nullptr);
    int max_tokens = cfg.max_seq_len;
    int max_seqs   = 256;  // reasonable default
    buffers_.allocate(cfg, max_tokens, max_seqs);
}

Transformer::~Transformer() = default;

void Transformer::load_weights(const std::string& path, cudaStream_t stream) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "Failed to open weights file: %s\n", path.c_str());
        return;
    }

    // Determine file size
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    // Read entire file into host memory
    std::vector<float> host_data(file_size / sizeof(float));
    size_t read = fread(host_data.data(), sizeof(float), host_data.size(), f);
    fclose(f);

    if (read != host_data.size()) {
        fprintf(stderr, "Warning: read %zu/%zu floats from weights file\n",
                read, host_data.size());
    }

    // Allocate and copy weights to GPU
    // Layout: token_emb, pos_emb, [per-layer weights], ln_f_gamma, ln_f_beta, lm_head
    size_t offset = 0;
    auto alloc_and_copy = [&](float*& ptr, size_t count) {
        if (offset + count > host_data.size()) {
            fprintf(stderr, "Weight file too small at offset %zu\n", offset);
            return;
        }
        CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(float)));
        CUDA_CHECK(cudaMemcpyAsync(ptr, host_data.data() + offset,
                                   count * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));
        offset += count;
    };

    int d = cfg_.d_model;
    int kv_dim = cfg_.actual_kv_heads() * cfg_.head_dim();

    alloc_and_copy(weights_.token_emb, (size_t)cfg_.vocab_size * d);
    alloc_and_copy(weights_.pos_emb,   (size_t)cfg_.max_seq_len * d);

    weights_.layers.resize(cfg_.n_layers);
    for (int l = 0; l < cfg_.n_layers; ++l) {
        auto& lw = weights_.layers[l];
        alloc_and_copy(lw.ln1_gamma, d);
        alloc_and_copy(lw.ln1_beta,  d);
        alloc_and_copy(lw.Wq,       (size_t)d * d);
        alloc_and_copy(lw.Wk,       (size_t)kv_dim * d);
        alloc_and_copy(lw.Wv,       (size_t)kv_dim * d);
        alloc_and_copy(lw.Wo,       (size_t)d * d);
        alloc_and_copy(lw.bq,       d);
        alloc_and_copy(lw.bk,       kv_dim);
        alloc_and_copy(lw.bv,       kv_dim);
        alloc_and_copy(lw.bo,       d);
        alloc_and_copy(lw.ln2_gamma, d);
        alloc_and_copy(lw.ln2_beta,  d);
        alloc_and_copy(lw.W1,       (size_t)cfg_.d_ff * d);
        alloc_and_copy(lw.b1,       cfg_.d_ff);
        alloc_and_copy(lw.W2,       (size_t)d * cfg_.d_ff);
        alloc_and_copy(lw.b2,       d);
    }

    alloc_and_copy(weights_.ln_f_gamma, d);
    alloc_and_copy(weights_.ln_f_beta,  d);
    alloc_and_copy(weights_.lm_head,    (size_t)cfg_.vocab_size * d);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    printf("Loaded %zu parameters (%.1f MB)\n",
           offset, offset * sizeof(float) / 1048576.0);
}

void Transformer::compute_qkv(int layer_idx, const float* hidden,
                               int num_tokens,
                               float* Q, float* K, float* V,
                               cudaStream_t stream) {
    auto& lw = weights_.layers[layer_idx];
    int d = cfg_.d_model;
    int kv_dim = cfg_.actual_kv_heads() * cfg_.head_dim();

    cublas_.set_stream(stream);

    // Q = hidden @ Wq^T
    kernels::linear(hidden, lw.Wq, lw.bq, Q, num_tokens, d, d, cublas_, stream);

    // K = hidden @ Wk^T
    kernels::linear(hidden, lw.Wk, lw.bk, K, num_tokens, kv_dim, d, cublas_, stream);

    // V = hidden @ Wv^T
    kernels::linear(hidden, lw.Wv, lw.bv, V, num_tokens, kv_dim, d, cublas_, stream);
}

void Transformer::layer_forward(
    int layer_idx, float* hidden, int num_tokens,
    const int* positions,
    PagedKVCache& kv_cache,
    const std::vector<int>& seq_ids,
    const std::vector<int>& seq_lens,
    bool is_prefill,
    const int32_t* query_start_locs,
    cudaStream_t stream)
{
    auto& lw = weights_.layers[layer_idx];
    int d      = cfg_.d_model;
    int kv_dim = cfg_.actual_kv_heads() * cfg_.head_dim();

    float* norm_out  = buffers_.norm_out;
    float* residual  = buffers_.residual;
    float* qkv_buf   = buffers_.qkv;
    float* attn_out  = buffers_.attn_out;
    float* ffn_ws    = buffers_.ffn_workspace;

    // Save residual
    CUDA_CHECK(cudaMemcpyAsync(residual, hidden,
                               (size_t)num_tokens * d * sizeof(float),
                               cudaMemcpyDeviceToDevice, stream));

    // Pre-attention LayerNorm
    kernels::layer_norm(hidden, lw.ln1_gamma, lw.ln1_beta,
                        norm_out, num_tokens, d, cfg_.layer_norm_eps, stream);

    // QKV projections
    float* Q = qkv_buf;
    float* K = Q + (size_t)num_tokens * d;
    float* V = K + (size_t)num_tokens * kv_dim;
    compute_qkv(layer_idx, norm_out, num_tokens, Q, K, V, stream);

    // Apply RoPE if no positional embeddings
    if (!weights_.pos_emb) {
        kernels::apply_rope(Q, K, positions, num_tokens,
                            cfg_.n_heads, cfg_.actual_kv_heads(),
                            cfg_.head_dim(), 10000.f, stream);
    }

    // Write K, V to paged cache
    // (In production, this would use write_kv_to_cache with proper mapping)
    // For now, we prepare block tables and use paged attention

    // Prepare paged attention params
    auto params = kv_cache.prepare_batch(seq_ids, layer_idx, stream);

    // Run paged attention
    if (is_prefill) {
        paged_attention::prefill(Q, attn_out, params, query_start_locs,
                                 num_tokens, cfg_.n_heads, layer_idx, stream);
    } else {
        paged_attention::decode(Q, attn_out, params,
                                cfg_.n_heads, layer_idx, stream);
    }

    // Output projection: attn_out = attn_out @ Wo^T
    kernels::linear(attn_out, lw.Wo, lw.bo, hidden,
                    num_tokens, d, d, cublas_, stream);

    // Residual connection
    // hidden = hidden + residual
    // Then pre-FFN LayerNorm
    kernels::layer_norm_residual(hidden, residual, lw.ln2_gamma, lw.ln2_beta,
                                 norm_out, num_tokens, d,
                                 cfg_.layer_norm_eps, stream);

    // Save new residual (hidden + old_residual)
    // hidden currently = Wo(attn), add residual
    // Actually: norm_out = LN(hidden + residual), so save (hidden + residual) as new residual
    // For GPT-2: residual = hidden + old_residual
    // Then hidden = norm_out after FFN + residual
    CUDA_CHECK(cudaMemcpyAsync(residual, hidden,
                               (size_t)num_tokens * d * sizeof(float),
                               cudaMemcpyDeviceToDevice, stream));

    // FFN
    kernels::fused_ffn(norm_out, lw.W1, lw.b1, lw.W2, lw.b2,
                       hidden, ffn_ws,
                       num_tokens, d, cfg_.d_ff, cublas_, stream);

    // Final residual: hidden = FFN_out + residual
    // (fused_ffn writes to hidden, add residual in-place)
    // Simplified: use a kernel or cublasSaxpy
    float alpha = 1.f;
    CUBLAS_CHECK(cublasSetStream(cublas_, stream));
    CUBLAS_CHECK(cublasSaxpy(cublas_, num_tokens * d, &alpha,
                             residual, 1, hidden, 1));
}

void Transformer::batch_prefill(
    const std::vector<int>& packed_tokens,
    const std::vector<int>& seq_lens,
    PagedKVCache& kv_cache,
    const std::vector<int>& seq_ids,
    float* logits_out,
    cudaStream_t stream)
{
    int num_seqs     = (int)seq_lens.size();
    int total_tokens = (int)packed_tokens.size();
    int d = cfg_.d_model;

    cublas_.set_stream(stream);

    // Upload token IDs and compute positions
    CUDA_CHECK(cudaMemcpyAsync(buffers_.token_ids.ptr, packed_tokens.data(),
                               total_tokens * sizeof(int),
                               cudaMemcpyHostToDevice, stream));

    // Compute positions and query_start_locs on host
    std::vector<int> positions(total_tokens);
    std::vector<int32_t> query_starts(num_seqs + 1, 0);
    int offset = 0;
    for (int s = 0; s < num_seqs; ++s) {
        query_starts[s] = offset;
        for (int t = 0; t < seq_lens[s]; ++t) {
            positions[offset + t] = t;
        }
        offset += seq_lens[s];
    }
    query_starts[num_seqs] = total_tokens;

    CUDA_CHECK(cudaMemcpyAsync(buffers_.positions.ptr, positions.data(),
                               total_tokens * sizeof(int),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(buffers_.query_start_locs.ptr, query_starts.data(),
                               (num_seqs + 1) * sizeof(int32_t),
                               cudaMemcpyHostToDevice, stream));

    // If weights are not loaded, return zero logits
    if (!weights_.token_emb) {
        CUDA_CHECK(cudaMemsetAsync(logits_out,
                                  0,
                                  (size_t)num_seqs * cfg_.vocab_size * sizeof(float),
                                  stream));
        return;
    }

    // Token + positional embedding
    if (weights_.pos_emb) {
        kernels::embedding(buffers_.token_ids, buffers_.positions,
                           weights_.token_emb, weights_.pos_emb,
                           buffers_.hidden, total_tokens, d, stream);
    } else {
        kernels::token_embedding(buffers_.token_ids, weights_.token_emb,
                                 buffers_.hidden, total_tokens, d, stream);
    }

    // Run through all layers
    for (int l = 0; l < cfg_.n_layers; ++l) {
        layer_forward(l, buffers_.hidden, total_tokens,
                      buffers_.positions, kv_cache, seq_ids, seq_lens,
                      /*is_prefill=*/true, buffers_.query_start_locs,
                      stream);
    }

    // Final layer norm
    kernels::layer_norm(buffers_.hidden, weights_.ln_f_gamma, weights_.ln_f_beta,
                        buffers_.norm_out, total_tokens, d,
                        cfg_.layer_norm_eps, stream);

    // LM head: extract last token per sequence, project to vocab
    // For each sequence, take the last token's hidden state
    float alpha = 1.f, beta = 0.f;
    for (int s = 0; s < num_seqs; ++s) {
        int last_pos = query_starts[s + 1] - 1;
        const float* h = buffers_.norm_out.ptr + (size_t)last_pos * d;
        float* out = logits_out + (size_t)s * cfg_.vocab_size;

        CUBLAS_CHECK(cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                                 cfg_.vocab_size, 1, d,
                                 &alpha, weights_.lm_head, d, h, d,
                                 &beta, out, cfg_.vocab_size));
    }
}

void Transformer::batch_decode(
    const std::vector<int>& token_ids,
    const std::vector<int>& seq_lens,
    PagedKVCache& kv_cache,
    const std::vector<int>& seq_ids,
    float* logits_out,
    cudaStream_t stream)
{
    int num_seqs = (int)token_ids.size();
    int d = cfg_.d_model;

    cublas_.set_stream(stream);

    // Upload token IDs
    CUDA_CHECK(cudaMemcpyAsync(buffers_.token_ids.ptr, token_ids.data(),
                               num_seqs * sizeof(int),
                               cudaMemcpyHostToDevice, stream));

    // Positions: seq_len - 1 for each sequence (0-indexed position of new token)
    std::vector<int> positions(num_seqs);
    for (int s = 0; s < num_seqs; ++s) {
        positions[s] = seq_lens[s] - 1;
    }
    CUDA_CHECK(cudaMemcpyAsync(buffers_.positions.ptr, positions.data(),
                               num_seqs * sizeof(int),
                               cudaMemcpyHostToDevice, stream));

    // If weights are not loaded, return zero logits
    if (!weights_.token_emb) {
        CUDA_CHECK(cudaMemsetAsync(logits_out,
                                  0,
                                  (size_t)num_seqs * cfg_.vocab_size * sizeof(float),
                                  stream));
        return;
    }

    // Embedding
    if (weights_.pos_emb) {
        kernels::embedding(buffers_.token_ids, buffers_.positions,
                           weights_.token_emb, weights_.pos_emb,
                           buffers_.hidden, num_seqs, d, stream);
    } else {
        kernels::token_embedding(buffers_.token_ids, weights_.token_emb,
                                 buffers_.hidden, num_seqs, d, stream);
    }

    // Run through all layers (decode mode: 1 token per seq)
    std::vector<int32_t> query_starts(num_seqs + 1);
    std::iota(query_starts.begin(), query_starts.end(), 0);
    CUDA_CHECK(cudaMemcpyAsync(buffers_.query_start_locs.ptr, query_starts.data(),
                               (num_seqs + 1) * sizeof(int32_t),
                               cudaMemcpyHostToDevice, stream));

    for (int l = 0; l < cfg_.n_layers; ++l) {
        layer_forward(l, buffers_.hidden, num_seqs,
                      buffers_.positions, kv_cache, seq_ids, seq_lens,
                      /*is_prefill=*/false, buffers_.query_start_locs,
                      stream);
    }

    // Final LN + LM head
    kernels::layer_norm(buffers_.hidden, weights_.ln_f_gamma, weights_.ln_f_beta,
                        buffers_.norm_out, num_seqs, d,
                        cfg_.layer_norm_eps, stream);

    float alpha = 1.f, beta = 0.f;
    CUBLAS_CHECK(cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                             cfg_.vocab_size, num_seqs, d,
                             &alpha, weights_.lm_head, d,
                             buffers_.norm_out, d,
                             &beta, logits_out, cfg_.vocab_size));
}
