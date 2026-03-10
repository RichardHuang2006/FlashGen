#pragma once
#include <memory>
#include <vector>
#include <cublas_v2.h>
#include "model_config.hpp"
#include "cuda_utils.cuh"

namespace flashgen {

// ── Weight tensors for a single transformer layer ────────────────────────────
struct LayerWeights {
    // Pre-attention layer norm
    DeviceBuffer<float> ln1_gamma;   // [d_model]
    DeviceBuffer<float> ln1_beta;    // [d_model]

    // Attention projections  (all stored as [d_model, d_model])
    DeviceBuffer<float> Wq, Wk, Wv; // query / key / value
    DeviceBuffer<float> bq, bk, bv; // biases (optional)
    DeviceBuffer<float> Wo;          // output projection [d_model, d_model]
    DeviceBuffer<float> bo;

    // Pre-FFN layer norm
    DeviceBuffer<float> ln2_gamma;   // [d_model]
    DeviceBuffer<float> ln2_beta;    // [d_model]

    // FFN
    DeviceBuffer<float> W1;          // [d_ff, d_model]
    DeviceBuffer<float> b1;          // [d_ff]
    DeviceBuffer<float> W2;          // [d_model, d_ff]
    DeviceBuffer<float> b2;          // [d_model]

    // Allocate all buffers for a given config
    void allocate(const ModelConfig& cfg);
    // Load weights from a flat float array (in the order above)
    void load_from_host(const float* data, const ModelConfig& cfg, cudaStream_t stream);
};

// ── KV cache for a single layer ──────────────────────────────────────────────
struct KVCache {
    DeviceBuffer<float> k;  // [batch, n_heads, max_seq_len, head_dim]
    DeviceBuffer<float> v;
    int current_len = 0;    // number of tokens currently cached

    void allocate(int batch, int n_heads, int max_seq_len, int head_dim);
    void reset() { current_len = 0; }
};

// ── Activation scratchpad for one layer ─────────────────────────────────────
struct LayerActivations {
    DeviceBuffer<float> ln_out;    // [batch * seq, d_model]  post-LN buffer
    DeviceBuffer<float> qkv;       // [batch * seq, 3 * d_model]  packed QKV
    DeviceBuffer<float> attn_out;  // [batch * seq, d_model]
    DeviceBuffer<float> proj_out;  // [batch * seq, d_model]
    DeviceBuffer<float> ffn_hidden;// [batch * seq, d_ff]  FFN workspace
    DeviceBuffer<float> ffn_out;   // [batch * seq, d_model]

    void allocate(const ModelConfig& cfg, int max_batch, int max_seq);
};

// ── Single transformer block ─────────────────────────────────────────────────
class TransformerLayer {
public:
    TransformerLayer(const ModelConfig& cfg, int layer_idx, cublasHandle_t cublas, int max_batch);
    ~TransformerLayer() = default;

    // Full-sequence forward pass (prefill)
    // x_in  : [batch, seq_len, d_model]
    // x_out : [batch, seq_len, d_model]
    void forward(
        const float* x_in,
        float*       x_out,
        int          batch,
        int          seq_len,
        KVCache*     kv_cache,    // nullptr = no caching
        cudaStream_t stream
    );

    // Incremental decoding step (one new token per batch element)
    // x_in  : [batch, 1, d_model]
    // x_out : [batch, 1, d_model]
    void decode_step(
        const float* x_in,
        float*       x_out,
        int          batch,
        KVCache&     kv_cache,
        cudaStream_t stream
    );

    LayerWeights&    weights()     { return weights_; }
    LayerActivations& activations(){ return acts_; }

private:
    ModelConfig    cfg_;
    int            layer_idx_;
    cublasHandle_t cublas_;
    LayerWeights   weights_;
    LayerActivations acts_;

    // Project x → Q, K, V via batched GEMM
    void project_qkv(
        const float* x, float* qkv,
        int batch, int seq, cudaStream_t stream
    );
    // Reshape packed QKV into [batch, n_heads, seq, head_dim] views
    void split_heads(
        const float* qkv,
        float* Q, float* K, float* V,
        int batch, int seq
    );
};

// ── Full GPT-2 style model ────────────────────────────────────────────────────
class Transformer {
public:
    explicit Transformer(const ModelConfig& cfg, const InferenceConfig& icfg);
    ~Transformer();

    // Load weights from binary file
    void load_weights(const std::string& path);

    // Prefill: process a full prompt, populate KV cache, return logits
    // input_ids : [batch, seq_len]  (host pointer, will be copied)
    // logits    : [batch, vocab_size]  (host output)
    void prefill(
        const int* input_ids,
        float*     logits,
        int        batch,
        int        seq_len
    );

    // Decode one new token per batch element
    // input_ids : [batch]  (last token IDs, host pointer)
    // logits    : [batch, vocab_size]  (host output)
    void decode(const int* input_ids, float* logits, int batch);

    // Reset KV caches (start a new conversation)
    void reset_cache();

    const ModelConfig& config() const { return cfg_; }

private:
    ModelConfig    cfg_;
    InferenceConfig icfg_;
    cublasHandle_t  cublas_  = nullptr;
    CudaStream      stream_;

    // Token + positional embeddings
    DeviceBuffer<float> token_emb_;   // [vocab_size, d_model]
    DeviceBuffer<float> pos_emb_;     // [max_seq_len, d_model]

    // Final layer norm + LM head
    DeviceBuffer<float> final_ln_gamma_;
    DeviceBuffer<float> final_ln_beta_;
    DeviceBuffer<float> lm_head_;     // [vocab_size, d_model]

    // Transformer layers
    std::vector<std::unique_ptr<TransformerLayer>> layers_;

    // KV caches: one per layer
    std::vector<KVCache> kv_caches_;

    // Intermediate activations for full model
    DeviceBuffer<float> hidden_;       // [batch * seq, d_model]
    DeviceBuffer<float> hidden_tmp_;   // [batch * seq, d_model]
    DeviceBuffer<float> logits_d_;     // [batch, vocab_size]  device side

    // Host output buffer
    std::vector<float> logits_host_;

    void allocate_buffers(int max_batch, int max_seq);
};

} // namespace flashgen
