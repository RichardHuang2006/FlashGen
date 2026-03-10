#pragma once
#include <cstddef>
#include <string>

namespace flashgen {

// ── Supported model presets ──────────────────────────────────────────────────
enum class ModelPreset {
    GPT2_SMALL,    // 117M
    GPT2_MEDIUM,   // 345M
    GPT2_LARGE,    // 762M
    GPT2_XL,       // 1.5B
    CUSTOM,
};

// ── Scalar type tag ──────────────────────────────────────────────────────────
enum class DType { FP32, FP16 };

// ── Model hyperparameters ────────────────────────────────────────────────────
struct ModelConfig {
    int   vocab_size    = 50257;
    int   max_seq_len   = 1024;
    int   d_model       = 768;    // embedding / hidden dimension
    int   n_heads       = 12;     // number of attention heads
    int   n_layers      = 12;     // number of transformer blocks
    int   d_ff          = 3072;   // feed-forward intermediate size
    float dropout       = 0.0f;   // 0.0 at inference time
    bool  use_bias      = true;
    DType dtype         = DType::FP32;

    // Derived helpers
    int head_dim() const { return d_model / n_heads; }

    // Construct from preset
    static ModelConfig from_preset(ModelPreset preset) {
        ModelConfig cfg;
        switch (preset) {
            case ModelPreset::GPT2_SMALL:
                cfg.d_model = 768;  cfg.n_heads = 12; cfg.n_layers = 12;
                cfg.d_ff    = 3072;
                break;
            case ModelPreset::GPT2_MEDIUM:
                cfg.d_model = 1024; cfg.n_heads = 16; cfg.n_layers = 24;
                cfg.d_ff    = 4096;
                break;
            case ModelPreset::GPT2_LARGE:
                cfg.d_model = 1280; cfg.n_heads = 20; cfg.n_layers = 36;
                cfg.d_ff    = 5120;
                break;
            case ModelPreset::GPT2_XL:
                cfg.d_model = 1600; cfg.n_heads = 25; cfg.n_layers = 48;
                cfg.d_ff    = 6400;
                break;
            default:
                break;
        }
        return cfg;
    }
};

// ── Runtime inference options ────────────────────────────────────────────────
struct InferenceConfig {
    int   batch_size      = 1;
    int   max_new_tokens  = 128;
    float temperature     = 1.0f;
    float top_p           = 0.9f;
    bool  greedy          = true;   // if true, ignore temperature/top_p
    bool  use_kv_cache    = true;
    int   gpu_id          = 0;
    // Pipeline concurrency
    int   pipeline_stages = 3;      // preprocessing / compute / postprocessing
    std::string weight_path;        // path to .bin weight file
};

} // namespace flashgen
