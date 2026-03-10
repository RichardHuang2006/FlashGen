/*
 * main.cpp
 *
 * FlashGen -- GPT-2 inference demo and benchmark entry point.
 *
 * Usage:
 *   flashgen [options]
 *
 * Options:
 *   --model       <preset>       gpt2 | gpt2-medium | gpt2-large | gpt2-xl
 *   --weights     <path>         path to .bin weight file
 *   --prompt      <string>       input prompt text
 *   --max_tokens  <int>          max new tokens to generate  (default: 128)
 *   --batch_size  <int>          batch size                  (default: 1)
 *   --temperature <float>        sampling temperature        (default: 1.0)
 *   --top_p       <float>        nucleus sampling probability(default: 0.9)
 *   --greedy                     use greedy decoding
 *   --benchmark                  run benchmark mode
 *   --seq_len     <int>          sequence length for benchmark (default: 512)
 *   --num_runs    <int>          number of benchmark runs    (default: 50)
 *   --gpu         <int>          GPU device ID               (default: 0)
 *   --fp16                       use FP16 mode (if compiled with USE_FP16)
 *   --help                       print this help
 */

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "model_config.hpp"
#include "transformer.hpp"
#include "pipeline.hpp"
#include "cuda_utils.cuh"

using namespace flashgen;

// -- Argument parsing ---------------------------------------------------------

struct Args {
    // Model
    std::string model    = "gpt2";
    std::string weights;
    // Inference
    std::string prompt   = "The future of AI is";
    int         max_tokens   = 128;
    int         batch_size   = 1;
    float       temperature  = 1.0f;
    float       top_p        = 0.9f;
    bool        greedy       = false;
    int         gpu          = 0;
    bool        fp16         = false;
    // Benchmark
    bool        benchmark    = false;
    int         seq_len      = 512;
    int         num_runs     = 50;
};

static void print_help(const char* prog) {
    printf("Usage: %s [options]\n\n", prog);
    printf("  --model       <preset>  gpt2 | gpt2-medium | gpt2-large | gpt2-xl\n");
    printf("  --weights     <path>    path to weight .bin file\n");
    printf("  --prompt      <string>  input prompt\n");
    printf("  --max_tokens  <int>     max tokens to generate (default: 128)\n");
    printf("  --batch_size  <int>     batch size (default: 1)\n");
    printf("  --temperature <float>   sampling temperature (default: 1.0)\n");
    printf("  --top_p       <float>   nucleus probability (default: 0.9)\n");
    printf("  --greedy                use greedy decoding\n");
    printf("  --benchmark             run latency benchmark\n");
    printf("  --seq_len     <int>     benchmark sequence length (default: 512)\n");
    printf("  --num_runs    <int>     benchmark iterations (default: 50)\n");
    printf("  --gpu         <int>     CUDA device ID (default: 0)\n");
    printf("  --fp16                  enable FP16 I/O\n");
    printf("  --help                  print this message\n");
}

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; i++) {
        std::string s = argv[i];
        if (s == "--help"   || s == "-h") { print_help(argv[0]); std::exit(0); }
        if (s == "--greedy")    { a.greedy    = true; continue; }
        if (s == "--benchmark") { a.benchmark = true; continue; }
        if (s == "--fp16")      { a.fp16      = true; continue; }

        if (i + 1 >= argc) {
            fprintf(stderr, "Missing argument for %s\n", argv[i]);
            std::exit(1);
        }
        const char* v = argv[++i];
        if      (s == "--model")       a.model       = v;
        else if (s == "--weights")     a.weights     = v;
        else if (s == "--prompt")      a.prompt      = v;
        else if (s == "--max_tokens")  a.max_tokens  = std::atoi(v);
        else if (s == "--batch_size")  a.batch_size  = std::atoi(v);
        else if (s == "--temperature") a.temperature = (float)std::atof(v);
        else if (s == "--top_p")       a.top_p       = (float)std::atof(v);
        else if (s == "--gpu")         a.gpu         = std::atoi(v);
        else if (s == "--seq_len")     a.seq_len     = std::atoi(v);
        else if (s == "--num_runs")    a.num_runs    = std::atoi(v);
        else { fprintf(stderr, "Unknown option: %s\n", s.c_str()); std::exit(1); }
    }
    return a;
}

// -- Model preset lookup ------------------------------------------------------

static ModelPreset preset_from_string(const std::string& name) {
    if (name == "gpt2"        || name == "gpt2-small")  return ModelPreset::GPT2_SMALL;
    if (name == "gpt2-medium")                           return ModelPreset::GPT2_MEDIUM;
    if (name == "gpt2-large")                            return ModelPreset::GPT2_LARGE;
    if (name == "gpt2-xl")                               return ModelPreset::GPT2_XL;
    fprintf(stderr, "Unknown model preset: %s\n", name.c_str());
    std::exit(1);
}

// -- GPU info banner ----------------------------------------------------------

static void print_gpu_info(int device) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("GPU [%d]: %s  |  %.1f GB VRAM  |  %d SMs  |  sm_%d%d\n",
           device, prop.name,
           (double)prop.totalGlobalMem / 1e9,
           prop.multiProcessorCount,
           prop.major, prop.minor);
}

// -- Main ---------------------------------------------------------------------

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    // ASCII-only box (avoids Windows console codepage issues)
    printf("+==================================+\n");
    printf("|  FlashGen Inference Engine v0.1  |\n");
    printf("+==================================+\n\n");

    CUDA_CHECK(cudaSetDevice(args.gpu));
    print_gpu_info(args.gpu);
    printf("\n");

    // Build configs
    ModelConfig mcfg = ModelConfig::from_preset(preset_from_string(args.model));
    if (args.fp16) mcfg.dtype = DType::FP16;

    InferenceConfig icfg;
    icfg.batch_size     = args.batch_size;
    icfg.max_new_tokens = args.max_tokens;
    icfg.temperature    = args.temperature;
    icfg.top_p          = args.top_p;
    icfg.greedy         = args.greedy;
    icfg.gpu_id         = args.gpu;
    icfg.weight_path    = args.weights;

    printf("Model  : %s  (%d layers, %d heads, d_model=%d)\n",
           args.model.c_str(), mcfg.n_layers, mcfg.n_heads, mcfg.d_model);
    printf("Dtype  : %s\n", args.fp16 ? "FP16" : "FP32");
    printf("Batch  : %d\n\n", args.batch_size);

    // Construct model
    auto model = std::make_shared<Transformer>(mcfg, icfg);

    if (!args.weights.empty()) {
        printf("Loading weights from: %s\n", args.weights.c_str());
        model->load_weights(args.weights);
        printf("Weights loaded.\n\n");
    } else {
        printf("[INFO] No weight file provided -- running with random initialisation.\n\n");
    }

    // -- Benchmark mode -------------------------------------------------------
    if (args.benchmark) {
        printf("Running benchmark: seq_len=%d  batch=%d  runs=%d\n\n",
               args.seq_len, args.batch_size, args.num_runs);
        auto r = benchmark(*model, args.seq_len, args.batch_size, args.num_runs);
        printf("Results:\n");
        printf("  Mean latency    : %8.2f ms\n", r.mean_latency_ms);
        printf("  P50  latency    : %8.2f ms\n", r.p50_latency_ms);
        printf("  P95  latency    : %8.2f ms\n", r.p95_latency_ms);
        printf("  P99  latency    : %8.2f ms\n", r.p99_latency_ms);
        printf("  Throughput      : %8.0f tokens/s\n", r.throughput_tokens_per_sec);
        return 0;
    }

    // -- Generation mode ------------------------------------------------------
    AsyncPipeline pipeline(model);

    printf("Prompt : \"%s\"\n", args.prompt.c_str());
    printf("Generating up to %d tokens...\n\n", args.max_tokens);

    InferenceRequest req;
    req.id  = 1;
    // Stub tokenization: encode as character IDs (replace with real tokenizer)
    for (unsigned char c : args.prompt) req.prompt_ids.push_back((int)c);
    req.cfg = icfg;

    auto resp = pipeline.run_sync(req);

    if (resp.success) {
        printf("----------------------------------------\n");
        printf("Generated text:\n%s%s\n",
               args.prompt.c_str(), resp.generated_text.c_str());
        printf("----------------------------------------\n\n");
        printf("Tokens generated : %zu\n", resp.generated_ids.size());
        printf("Latency          : %.2f ms\n",  resp.latency_ms);
        printf("Time/token       : %.2f ms\n",  resp.tpot_ms);
        printf("Throughput       : %.1f tok/s\n",
               1000.f / std::max(resp.tpot_ms, 0.001f));
    } else {
        fprintf(stderr, "Generation failed: %s\n", resp.error.c_str());
        return 1;
    }

    return 0;
}
