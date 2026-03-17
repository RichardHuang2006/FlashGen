<h1 align="center">FlashGen</h1>

<p align="center">
<strong>High-Performance LLM Inference Engine</strong><br>
Paged KV Cache &middot; Continuous Batching &middot; Custom CUDA Kernels
</p>

<p align="center">
<img src="https://img.shields.io/badge/CUDA-12.x-green?logo=nvidia" alt="CUDA 12">
<img src="https://img.shields.io/badge/C%2B%2B-17-blue?logo=cplusplus" alt="C++17">
<img src="https://img.shields.io/badge/GPU-Ampere%2B-orange" alt="GPU Ampere+">
<img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="MIT License">
</p>

---

FlashGen is a from-scratch LLM inference engine written in C++17 and CUDA that combines **vLLM-style paged KV cache management** with **FlashAttention-2 kernels** and a **continuous batching scheduler** to serve decoder-only transformer models efficiently on NVIDIA GPUs.

## Architecture

```
                    ┌──────────────────────┐
                    │   InferenceEngine    │
                    │   (engine loop)      │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
    ┌─────────▼──────┐  ┌─────▼──────┐  ┌──────▼───────┐
    │   Scheduler    │  │ Transformer │  │   Sampling   │
    │ (continuous    │  │ (batched    │  │ (greedy,     │
    │  batching)     │  │  forward)   │  │  top-p/k)    │
    └───┬────┬───────┘  └──────┬──────┘  └──────────────┘
        │    │                 │
        │    │        ┌────────▼────────┐
        │    │        │  Paged Flash    │
        │    │        │  Attention      │
        │    │        │  (CUDA kernel)  │
        │    │        └────────┬────────┘
        │    │                 │
   ┌────▼────▼──┐    ┌────────▼────────┐
   │  Prefix    │    │  Paged KV Cache │
   │  Cache     │    │  (block tables) │
   │ (radix     │    └────────┬────────┘
   │  tree)     │             │
   └────────────┘    ┌────────▼────────┐
                     │ Block Allocator │
                     │ (GPU mem pool,  │
                     │  ref-counted)   │
                     └─────────────────┘
```

## Key Features

### Paged KV Cache (vLLM-style)
- **Block-table indirection**: KV cache is divided into fixed-size blocks (default 16 tokens). Each sequence maintains a block table mapping logical positions to physical GPU blocks, eliminating memory waste from pre-allocated contiguous buffers.
- **Reference-counted blocks**: Blocks track their reference count, enabling copy-on-write semantics and zero-copy prefix sharing between sequences.
- **Dynamic memory**: Sequences allocate blocks on-demand as they grow during decoding, with automatic reclamation on completion.

### Continuous Batching Scheduler
- **Iteration-level scheduling**: After every forward pass, the scheduler re-evaluates which sequences to include in the next batch, admitting new prefills and continuing decode steps concurrently.
- **Three-queue design**: Waiting (new requests), Running (active), and Preempted (evicted) queues following the vLLM scheduling algorithm.
- **Preemption support**: Under memory pressure, the scheduler evicts the lowest-priority running sequence (recompute or swap-to-CPU) to free blocks for higher-priority work.
- **Token budget**: Configurable `max_num_tokens` per iteration prevents long prefills from starving decode latency.

### Prefix-Aware KV Cache Reuse
- **Radix tree index**: Common prompt prefixes (system prompts, few-shot examples) are stored in a radix tree indexed by token sequences. New requests that share a prefix skip prefill for the matched portion.
- **LRU eviction**: Under memory pressure, least-recently-used prefix entries are evicted to free blocks.
- **Transparent integration**: The scheduler automatically checks the prefix cache before admitting new requests.

### Custom CUDA Kernels
- **FlashAttention-2**: Tiled, IO-aware attention with online softmax. O(N d) memory instead of O(N^2). Supports causal masking, multiple head dimensions (32-128), and incremental decode.
- **Paged attention kernels**: Modified FlashAttention that reads K/V through block table indirection, supporting variable-length sequences in a single batch.
- **Fused kernels**: LayerNorm (Welford), RMSNorm, GELU, SiLU, fused residual-add-LayerNorm, fused FFN (cuBLAS GEMM + activation), RoPE, and temperature-scaled softmax.
- **Quantized KV storage**: INT8 symmetric per-token quantization with fused dequantization inside the attention kernel for ~2x KV cache memory savings.

### Model Support
- GPT-2 family (124M to 1.5B parameters)
- LLaMA-style architectures (RoPE, RMSNorm, GQA, SiLU-gated FFN)
- Configurable GQA (grouped-query attention) with arbitrary `n_kv_heads`

## Project Structure

```
FlashGen/
├── include/
│   ├── cuda_utils.cuh         # CUDA error checking, RAII wrappers (streams, events, buffers)
│   ├── model_config.hpp       # Model, cache, scheduler, and engine configuration
│   ├── request.hpp            # Request, sequence, sampling params, response types
│   ├── block_allocator.hpp    # GPU memory pool with ref-counted block management
│   ├── paged_kv_cache.cuh     # Block-table-based KV cache + paged attention params
│   ├── prefix_cache.hpp       # Radix-tree prefix cache for KV block reuse
│   ├── quantization.cuh       # INT8/FP8 KV quantization with fused dequant
│   ├── flash_attention.cuh    # FlashAttention-2 kernel declarations
│   ├── kernels.cuh            # Fused transformer kernel declarations
│   ├── scheduler.hpp          # Continuous batching scheduler
│   ├── transformer.hpp        # Transformer model with paged KV cache
│   └── pipeline.hpp           # Inference engine (integrates all components)
├── src/
│   ├── block_allocator.cpp    # Block allocator: alloc, free, ref count, CoW
│   ├── paged_kv_cache.cu      # Paged KV cache + paged attention CUDA kernels
│   ├── prefix_cache.cpp       # Radix tree insert, match, evict
│   ├── quantization.cu        # INT8 quantize/dequantize kernels
│   ├── flash_attention.cu     # FlashAttention-2 forward + decode kernels
│   ├── kernels.cu             # LayerNorm, GELU, SiLU, RoPE, softmax, FFN, embedding
│   ├── scheduler.cpp          # Three-queue scheduling with preemption
│   ├── transformer.cu         # Batched prefill/decode with paged attention
│   ├── pipeline.cpp           # Engine loop, sampling, benchmarking
│   └── main.cpp               # CLI entry point
├── tests/
│   ├── test_block_allocator.cpp    # Alloc/free, ref counting, CoW, OOM
│   ├── test_paged_attention.cu     # Paged KV lifecycle, fork, attention correctness
│   ├── test_prefix_cache.cpp       # Prefix match, partial match, eviction
│   ├── test_scheduler.cpp          # Scheduling, token budget, finish conditions
│   ├── test_flash_attention.cu     # Correctness vs naive, decode, throughput
│   ├── test_kernels.cu             # LayerNorm, GELU, SiLU, RMSNorm, softmax, argmax
│   └── test_pipeline.cpp           # Engine lifecycle, async submit, stats
├── CMakeLists.txt
└── README.md
```

## Building

### Prerequisites
- NVIDIA GPU (Ampere sm_80+ recommended, Volta sm_70+ minimum)
- CUDA Toolkit 12.x
- CMake 3.20+
- C++17 compiler (GCC 9+, Clang 10+, MSVC 2019+)

### Build

```bash
mkdir build && cd build
cmake .. -DGPU_ARCH=80          # Set your GPU's compute capability
cmake --build . -j$(nproc)
```

Common GPU architectures:

| GPU | Architecture |
|-----|-------------|
| V100 | `70` |
| RTX 3090, A100 | `80` |
| RTX 4090 | `89` |
| H100 | `90` |
| B200 | `120` |

### Run Tests

```bash
cd build
ctest --output-on-failure
```

## Usage

### Generation

```bash
./flashgen \
    --model gpt2 \
    --weights /path/to/weights.bin \
    --prompt "The future of artificial intelligence" \
    --max-tokens 128 \
    --temperature 0.8 \
    --top-p 0.95
```

### Benchmarking

```bash
./flashgen \
    --model gpt2-xl \
    --benchmark \
    --bench-seq 512 \
    --bench-batch 8 \
    --bench-runs 20
```

### Configuration Options

```
Model:
  --model NAME             gpt2, gpt2-medium, gpt2-large, gpt2-xl, llama-7b
  --weights PATH           Binary weights file
  --device ID              GPU device (default: 0)

Generation:
  --prompt TEXT            Input prompt
  --max-tokens N           Max output tokens (default: 64)
  --temperature F          Sampling temperature (default: 0.8)
  --top-p F                Nucleus sampling threshold (default: 0.95)
  --greedy                 Greedy decoding

KV Cache:
  --block-size N           Tokens per cache block (default: 16)
  --num-blocks N           GPU blocks (-1 = auto from available VRAM)
  --gpu-mem-frac F         VRAM fraction for KV cache (default: 0.9)
  --kv-quant TYPE          none | int8 | fp8
  --prefix-cache           Enable prefix caching (default)
  --no-prefix-cache        Disable prefix caching

Scheduler:
  --max-seqs N             Max concurrent sequences (default: 256)
  --max-tokens-batch N     Max tokens per iteration (default: 8192)
```

## Core Components

### Block Allocator

The block allocator pre-allocates a contiguous GPU memory pool and divides it into fixed-size blocks. Each block stores K and V tensors for `block_size` tokens across all layers and KV heads:

```
Physical block layout:
  [Layer 0, Head 0, K: block_size * head_dim * elem_size]
  [Layer 0, Head 0, V: block_size * head_dim * elem_size]
  [Layer 0, Head 1, K: ...]
  ...
  [Layer N, Head M, V: ...]
```

Blocks are managed with a free-list stack (O(1) alloc/free) and atomic reference counts. Copy-on-write is triggered when a shared block (refcount > 1) needs modification — a new block is allocated, data is copied via `cudaMemcpyAsync`, and the old reference is decremented.

### Paged Attention Kernel

The decode kernel assigns one warp (32 threads) per (sequence, head) pair. Each warp iterates over all physical blocks in the sequence's block table, performing online softmax attention:

```
for each block in block_table[seq_idx]:
    phys_block = block_table[seq_idx][logical_block]
    K_ptr = pool + phys_block * stride + layer * layer_stride + head * head_stride
    V_ptr = K_ptr + kv_stride

    for each token in block:
        dot = warp_reduce(Q . K[token])
        online_softmax_update(dot, m, l, acc, V[token])

write O = acc  (already normalized by online softmax)
```

### FlashAttention-2

Tiled, IO-aware attention algorithm avoiding the N^2 attention matrix:

```
for each Q tile (q_start..q_start+Br):
    m = -inf,  l = 0,  O = 0
    for each KV tile (kv_start..kv_start+Bc):
        S  = Q_tile . K^T_tile * scale        # [Br, Bc]
        m' = max(rowmax(S), m)
        P  = exp(S - m')                      # [Br, Bc]
        l' = rowsum(P) + l * exp(m - m')
        O  = O * exp(m - m') + P . V_tile
        m, l = m', l'
    O /= l
```

### Scheduling Algorithm

Each iteration:
1. **Resume preempted** (LIFO): restore evicted sequences if blocks available
2. **Admit waiting** (FCFS): check prefix cache, allocate blocks, respect token budget
3. **Continue decode**: extend running sequences by one token each
4. **Preempt if needed**: evict lowest-priority running sequence to free memory

### INT8 KV Quantization

Per-token symmetric quantization reduces KV cache memory by ~2x:
```
scale = max(|x|) / 127
q[i] = round(clamp(x[i] / scale, -128, 127))
```

Dequantization is fused into the attention kernel's inner loop to avoid materializing full-precision intermediates.

## Supported Models

| Model | Layers | Heads | d_model | d_ff | Parameters |
|-------|--------|-------|---------|------|------------|
| GPT-2 | 12 | 12 | 768 | 3072 | 124M |
| GPT-2 Medium | 24 | 16 | 1024 | 4096 | 355M |
| GPT-2 Large | 36 | 20 | 1280 | 5120 | 774M |
| GPT-2 XL | 48 | 25 | 1600 | 6400 | 1.5B |
| LLaMA 7B | 32 | 32 | 4096 | 11008 | 6.7B |
| LLaMA-2 7B | 32 | 32 (GQA) | 4096 | 11008 | 6.7B |

## Performance Characteristics

| Component | Complexity | Notes |
|-----------|-----------|-------|
| Block alloc/free | O(1) | Stack-based free list |
| Prefix cache lookup | O(L/B) | L=prompt length, B=block size |
| Flash attention (prefill) | O(N^2 d / M) | M=SRAM size, IO-aware |
| Paged attention (decode) | O(N d) | N=context length per seq |
| Scheduling | O(W + R) | W=waiting, R=running queue sizes |

## Roadmap

- [ ] BPE tokenizer integration (tiktoken / SentencePiece)
- [ ] SafeTensors / HuggingFace weight loader
- [ ] Tensor parallelism (multi-GPU)
- [ ] Speculative decoding
- [ ] CUDA graph capture for decode batches
- [ ] FP8 native support on Hopper (sm_90+)
- [ ] OpenAI-compatible HTTP API server
- [ ] Beam search

## License

MIT
