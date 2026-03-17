#include "pipeline.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
//  CLI argument parsing
// ---------------------------------------------------------------------------

struct Args {
    std::string model_name  = "gpt2";
    std::string weight_path;
    std::string prompt      = "The future of AI is";
    int         max_tokens  = 64;
    float       temperature = 0.8f;
    float       top_p       = 0.95f;
    bool        greedy      = false;
    int         device      = 0;

    // Benchmark mode
    bool        benchmark   = false;
    int         bench_seq_len   = 512;
    int         bench_batch     = 4;
    int         bench_runs      = 10;
    int         bench_decode_steps = 100;

    // Cache config
    int         block_size       = 16;
    int         num_gpu_blocks   = -1;    // auto
    float       gpu_mem_fraction = 0.9f;
    bool        prefix_caching   = true;
    std::string kv_quant         = "none";

    // Scheduler config
    int         max_num_seqs     = 256;
    int         max_num_tokens   = 8192;
};

static void print_usage(const char* prog) {
    printf("Usage: %s [OPTIONS]\n\n", prog);
    printf("Model:\n");
    printf("  --model NAME       Model preset (gpt2, gpt2-medium, gpt2-large, gpt2-xl, llama-7b)\n");
    printf("  --weights PATH     Path to binary weights file\n");
    printf("  --device ID        GPU device ID (default: 0)\n\n");
    printf("Generation:\n");
    printf("  --prompt TEXT      Input prompt\n");
    printf("  --max-tokens N     Max tokens to generate (default: 64)\n");
    printf("  --temperature F    Sampling temperature (default: 0.8)\n");
    printf("  --top-p F          Top-p nucleus sampling (default: 0.95)\n");
    printf("  --greedy           Use greedy decoding\n\n");
    printf("Cache:\n");
    printf("  --block-size N     Tokens per KV cache block (default: 16)\n");
    printf("  --num-blocks N     Number of GPU blocks (-1 = auto)\n");
    printf("  --gpu-mem-frac F   GPU memory fraction for KV cache (default: 0.9)\n");
    printf("  --prefix-cache     Enable prefix caching (default: on)\n");
    printf("  --no-prefix-cache  Disable prefix caching\n");
    printf("  --kv-quant TYPE    KV quantization: none, int8, fp8 (default: none)\n\n");
    printf("Benchmark:\n");
    printf("  --benchmark        Run benchmark mode\n");
    printf("  --bench-seq N      Benchmark sequence length (default: 512)\n");
    printf("  --bench-batch N    Benchmark batch size (default: 4)\n");
    printf("  --bench-runs N     Number of benchmark runs (default: 10)\n");
}

static Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 < argc) return argv[++i];
            fprintf(stderr, "Missing value for %s\n", arg.c_str());
            exit(1);
        };

        if (arg == "--model")           args.model_name = next();
        else if (arg == "--weights")    args.weight_path = next();
        else if (arg == "--prompt")     args.prompt = next();
        else if (arg == "--max-tokens") args.max_tokens = std::stoi(next());
        else if (arg == "--temperature") args.temperature = std::stof(next());
        else if (arg == "--top-p")      args.top_p = std::stof(next());
        else if (arg == "--greedy")     args.greedy = true;
        else if (arg == "--device")     args.device = std::stoi(next());
        else if (arg == "--benchmark")  args.benchmark = true;
        else if (arg == "--bench-seq")  args.bench_seq_len = std::stoi(next());
        else if (arg == "--bench-batch") args.bench_batch = std::stoi(next());
        else if (arg == "--bench-runs") args.bench_runs = std::stoi(next());
        else if (arg == "--block-size") args.block_size = std::stoi(next());
        else if (arg == "--num-blocks") args.num_gpu_blocks = std::stoi(next());
        else if (arg == "--gpu-mem-frac") args.gpu_mem_fraction = std::stof(next());
        else if (arg == "--prefix-cache") args.prefix_caching = true;
        else if (arg == "--no-prefix-cache") args.prefix_caching = false;
        else if (arg == "--kv-quant")   args.kv_quant = next();
        else if (arg == "--max-seqs")   args.max_num_seqs = std::stoi(next());
        else if (arg == "--max-tokens-batch") args.max_num_tokens = std::stoi(next());
        else if (arg == "--help" || arg == "-h") { print_usage(argv[0]); exit(0); }
        else { fprintf(stderr, "Unknown argument: %s\n", arg.c_str()); exit(1); }
    }
    return args;
}

// ---------------------------------------------------------------------------
//  Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    printf("\n");
    printf("╔═══════════════════════════════════════════════════╗\n");
    printf("║   FlashGen — High-Performance LLM Inference      ║\n");
    printf("║   Paged KV Cache · Continuous Batching · CUDA     ║\n");
    printf("╚═══════════════════════════════════════════════════╝\n\n");

    // Build engine configuration
    EngineConfig ecfg;

    // Model preset
    if (args.model_name == "gpt2")          ecfg.model = ModelConfig::gpt2();
    else if (args.model_name == "gpt2-medium") ecfg.model = ModelConfig::gpt2_medium();
    else if (args.model_name == "gpt2-large")  ecfg.model = ModelConfig::gpt2_large();
    else if (args.model_name == "gpt2-xl")     ecfg.model = ModelConfig::gpt2_xl();
    else if (args.model_name == "llama-7b")    ecfg.model = ModelConfig::llama_7b();
    else if (args.model_name == "llama2-7b")   ecfg.model = ModelConfig::llama2_7b();
    else {
        fprintf(stderr, "Unknown model: %s\n", args.model_name.c_str());
        return 1;
    }

    // Cache config
    ecfg.cache.block_size             = args.block_size;
    ecfg.cache.num_gpu_blocks         = args.num_gpu_blocks;
    ecfg.cache.gpu_memory_utilization = args.gpu_mem_fraction;
    ecfg.cache.enable_prefix_caching  = args.prefix_caching;

    if (args.kv_quant == "int8") ecfg.cache.kv_quant = KVQuantType::INT8;
    else if (args.kv_quant == "fp8") ecfg.cache.kv_quant = KVQuantType::FP8_E4M3;
    else ecfg.cache.kv_quant = KVQuantType::NONE;

    ecfg.model.kv_quant = ecfg.cache.kv_quant;

    // Scheduler config
    ecfg.scheduler.max_num_seqs    = args.max_num_seqs;
    ecfg.scheduler.max_num_tokens  = args.max_num_tokens;

    ecfg.weight_path = args.weight_path;
    ecfg.device_id   = args.device;

    // Create engine
    InferenceEngine engine(ecfg);

    if (args.benchmark) {
        // ── Benchmark mode ──────────────────────────────────────────
        engine.benchmark_prefill(args.bench_seq_len, args.bench_batch,
                                 args.bench_runs);
        engine.benchmark_decode(args.bench_batch, args.bench_seq_len,
                                args.bench_decode_steps);
        return 0;
    }

    // ── Generation mode ─────────────────────────────────────────────
    printf("Prompt: \"%s\"\n", args.prompt.c_str());
    printf("Config: max_tokens=%d, temp=%.2f, top_p=%.2f%s\n",
           args.max_tokens, args.temperature, args.top_p,
           args.greedy ? " (greedy)" : "");

    // Simple tokenization (ASCII char-level — replace with BPE tokenizer)
    std::vector<int> prompt_tokens;
    for (char c : args.prompt) {
        prompt_tokens.push_back((int)(unsigned char)c);
    }

    InferenceRequest req;
    req.prompt_token_ids = prompt_tokens;
    req.sampling.max_tokens   = args.max_tokens;
    req.sampling.temperature  = args.temperature;
    req.sampling.top_p        = args.top_p;
    req.sampling.greedy       = args.greedy;

    auto response = engine.generate(std::move(req));

    if (response.success) {
        // Simple detokenization (ASCII)
        printf("\nGenerated (%d tokens, %.1f ms, %.1f ms/token):\n",
               response.generated_tokens, response.latency_ms, response.tpot_ms);
        for (int tok : response.output_token_ids) {
            char c = (tok >= 32 && tok < 127) ? (char)tok : '?';
            printf("%c", c);
        }
        printf("\n");
    } else {
        fprintf(stderr, "Error: %s\n", response.error.c_str());
        return 1;
    }

    return 0;
}
