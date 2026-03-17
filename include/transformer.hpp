#pragma once

#include "cuda_utils.cuh"
#include "model_config.hpp"
#include "paged_kv_cache.cuh"

#include <string>
#include <vector>

// ---------------------------------------------------------------------------
//  Transformer model with paged KV cache support
//
//  Implements a GPT-2 / LLaMA-style decoder-only transformer.
//  KV cache is managed externally via PagedKVCache and BlockAllocator;
//  the model reads/writes KV through block table indirection.
//
//  Two execution paths:
//    batch_prefill() — parallel processing of prompt tokens
//    batch_decode()  — autoregressive single-token-per-sequence
// ---------------------------------------------------------------------------

/// Per-layer weight tensors (device pointers).
struct LayerWeights {
    // Attention
    float* ln1_gamma  = nullptr;   // [d_model]
    float* ln1_beta   = nullptr;   // [d_model]
    float* Wq         = nullptr;   // [d_model, d_model]
    float* Wk         = nullptr;   // [n_kv_heads * head_dim, d_model]
    float* Wv         = nullptr;   // [n_kv_heads * head_dim, d_model]
    float* Wo         = nullptr;   // [d_model, d_model]
    float* bq         = nullptr;   // [d_model]       (nullptr for no-bias models)
    float* bk         = nullptr;   // [n_kv_heads * head_dim]
    float* bv         = nullptr;   // [n_kv_heads * head_dim]
    float* bo         = nullptr;   // [d_model]

    // FFN
    float* ln2_gamma  = nullptr;   // [d_model]
    float* ln2_beta   = nullptr;   // [d_model]
    float* W1         = nullptr;   // [d_ff, d_model]
    float* b1         = nullptr;   // [d_ff]
    float* W2         = nullptr;   // [d_model, d_ff]
    float* b2         = nullptr;   // [d_model]
};

/// All model weights.
struct TransformerWeights {
    // Embeddings
    float* token_emb  = nullptr;   // [vocab_size, d_model]
    float* pos_emb    = nullptr;   // [max_seq_len, d_model]  (nullptr for RoPE)

    // Layers
    std::vector<LayerWeights> layers;

    // Final layer norm + LM head
    float* ln_f_gamma = nullptr;   // [d_model]
    float* ln_f_beta  = nullptr;   // [d_model]
    float* lm_head    = nullptr;   // [vocab_size, d_model]
};

/// Scratch buffers for one forward pass (pre-allocated, reused).
struct ForwardBuffers {
    DeviceBuffer<float> hidden;        // [max_tokens, d_model]
    DeviceBuffer<float> residual;      // [max_tokens, d_model]
    DeviceBuffer<float> norm_out;      // [max_tokens, d_model]
    DeviceBuffer<float> qkv;           // [max_tokens, (n_heads + 2*n_kv_heads) * head_dim]
    DeviceBuffer<float> attn_out;      // [max_tokens, d_model]
    DeviceBuffer<float> ffn_workspace; // [max_tokens, d_ff]
    DeviceBuffer<float> logits;        // [max_tokens, vocab_size] (only last token per seq)
    DeviceBuffer<int>   positions;     // [max_tokens]
    DeviceBuffer<int>   token_ids;     // [max_tokens]
    DeviceBuffer<int32_t> query_start_locs; // [max_seqs + 1]

    void allocate(const ModelConfig& cfg, int max_tokens, int max_seqs);
};

class Transformer {
public:
    Transformer(const ModelConfig& cfg);
    ~Transformer();

    /// Load weights from binary file.
    void load_weights(const std::string& path, cudaStream_t stream);

    /// Batch prefill: process variable-length prompts packed contiguously.
    /// packed_tokens: concatenated token IDs for all sequences.
    /// seq_lens:      number of tokens per sequence.
    /// Writes logits for the *last token* of each sequence to logits_out.
    void batch_prefill(const std::vector<int>& packed_tokens,
                       const std::vector<int>& seq_lens,
                       PagedKVCache& kv_cache,
                       const std::vector<int>& seq_ids,
                       float* logits_out,     // device: [num_seqs, vocab_size]
                       cudaStream_t stream);

    /// Batch decode: one new token per sequence.
    /// token_ids: [num_seqs] — the most recently generated token.
    /// seq_lens:  [num_seqs] — current total length (including the new token).
    void batch_decode(const std::vector<int>& token_ids,
                      const std::vector<int>& seq_lens,
                      PagedKVCache& kv_cache,
                      const std::vector<int>& seq_ids,
                      float* logits_out,     // device: [num_seqs, vocab_size]
                      cudaStream_t stream);

    const ModelConfig& config() const { return cfg_; }

private:
    ModelConfig        cfg_;
    TransformerWeights weights_;
    ForwardBuffers     buffers_;
    CublasHandle       cublas_;

    // Per-layer forward with paged attention
    void layer_forward(int layer_idx, float* hidden, int num_tokens,
                       const int* positions,
                       PagedKVCache& kv_cache,
                       const std::vector<int>& seq_ids,
                       const std::vector<int>& seq_lens,
                       bool is_prefill,
                       const int32_t* query_start_locs,
                       cudaStream_t stream);

    void compute_qkv(int layer_idx, const float* hidden, int num_tokens,
                     float* Q, float* K, float* V, cudaStream_t stream);

    void write_kv_to_cache(int layer_idx, const float* K, const float* V,
                           int num_tokens, PagedKVCache& kv_cache,
                           const std::vector<int>& seq_ids,
                           const std::vector<int>& seq_lens,
                           cudaStream_t stream);
};
