#pragma once

#include <string>
#include <cstdint>
#include <algorithm>

// ---------------------------------------------------------------------------
//  Quantization type for KV cache storage
// ---------------------------------------------------------------------------

enum class KVQuantType : int {
    NONE    = 0,   // FP32 (or FP16 if use_fp16 is set)
    INT8    = 1,   // Per-token symmetric INT8 with FP32 scale
    FP8_E4M3 = 2,  // FP8 E4M3 (Hopper+ native)
};

/// Bytes per element for a given KV quantization type.
inline int kv_quant_element_size(KVQuantType qt) {
    switch (qt) {
        case KVQuantType::NONE:     return 4;   // FP32
        case KVQuantType::INT8:     return 1;
        case KVQuantType::FP8_E4M3: return 1;
    }
    return 4;
}

// ---------------------------------------------------------------------------
//  Model configuration
// ---------------------------------------------------------------------------

struct ModelConfig {
    // Architecture
    std::string name        = "gpt2";
    int         d_model     = 768;
    int         n_heads     = 12;
    int         n_kv_heads  = -1;     // GQA heads; -1 means same as n_heads (MHA)
    int         n_layers    = 12;
    int         d_ff        = 3072;   // feed-forward hidden dim
    int         vocab_size  = 50257;
    int         max_seq_len = 1024;
    float       layer_norm_eps = 1e-5f;

    // Precision
    bool        use_fp16    = false;

    // KV cache quantization
    KVQuantType kv_quant    = KVQuantType::NONE;

    // Derived helpers
    int head_dim()          const { return d_model / n_heads; }
    int actual_kv_heads()   const { return n_kv_heads > 0 ? n_kv_heads : n_heads; }
    int kv_head_dim()       const { return d_model / actual_kv_heads(); }
    int gqa_group_size()    const { return n_heads / actual_kv_heads(); }

    // Presets
    static ModelConfig gpt2()        { return {"gpt2",        768,  12, -1, 12, 3072,  50257, 1024}; }
    static ModelConfig gpt2_medium() { return {"gpt2-medium", 1024, 16, -1, 24, 4096,  50257, 1024}; }
    static ModelConfig gpt2_large()  { return {"gpt2-large",  1280, 20, -1, 36, 5120,  50257, 1024}; }
    static ModelConfig gpt2_xl()     { return {"gpt2-xl",     1600, 25, -1, 48, 6400,  50257, 1024}; }
    static ModelConfig llama_7b()    { return {"llama-7b",    4096, 32, -1, 32, 11008, 32000, 2048}; }
    static ModelConfig llama2_7b()   { return {"llama2-7b",   4096, 32, 32, 32, 11008, 32000, 4096}; }
};

// ---------------------------------------------------------------------------
//  Cache configuration
// ---------------------------------------------------------------------------

struct CacheConfig {
    int         block_size              = 16;        // tokens per KV cache block
    int         num_gpu_blocks          = -1;        // auto-computed if -1
    float       gpu_memory_utilization  = 0.90f;     // fraction of GPU VRAM for blocks
    KVQuantType kv_quant                = KVQuantType::NONE;
    bool        enable_prefix_caching   = true;

    /// Compute bytes per block: stores K and V for all layers and KV heads.
    /// Layout: [block_size * head_dim * elem_size] * 2(K+V) * n_kv_heads * n_layers
    size_t bytes_per_block(const ModelConfig& m) const {
        int elem = kv_quant_element_size(kv_quant);
        size_t per_head = (size_t)block_size * m.head_dim() * elem;
        size_t kv_pair  = per_head * 2;  // K + V
        return kv_pair * m.actual_kv_heads() * m.n_layers;
    }

    /// Auto-compute number of GPU blocks given available memory.
    int auto_num_blocks(const ModelConfig& m, size_t free_gpu_bytes) const {
        size_t usable = (size_t)(free_gpu_bytes * gpu_memory_utilization);
        size_t bpb    = bytes_per_block(m);
        return bpb > 0 ? (int)(usable / bpb) : 0;
    }
};

// ---------------------------------------------------------------------------
//  Scheduler configuration
// ---------------------------------------------------------------------------

struct SchedulerConfig {
    int   max_num_seqs          = 256;      // max concurrent sequences
    int   max_num_tokens        = 8192;     // max total tokens per iteration
    int   max_prefill_tokens    = 4096;     // limit prefill to avoid starving decode
    bool  enable_chunked_prefill = true;    // split long prompts across iterations

    enum class PreemptionMode { RECOMPUTE, SWAP };
    PreemptionMode preemption_mode = PreemptionMode::RECOMPUTE;
};

// ---------------------------------------------------------------------------
//  Engine configuration (combines all sub-configs)
// ---------------------------------------------------------------------------

struct EngineConfig {
    ModelConfig     model;
    CacheConfig     cache;
    SchedulerConfig scheduler;
    std::string     weight_path;
    int             device_id       = 0;
};
