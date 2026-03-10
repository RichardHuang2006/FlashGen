# FlashGen

A high-performance, low-latency transformer inference engine built in C++ and CUDA. FlashGen leverages FlashAttention-2 for memory-efficient attention, fused CUDA kernels for layer normalization and feed-forward networks, and an asynchronous CPU-GPU pipeline to minimize end-to-end inference latency.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        FlashGen Pipeline                        │
│                                                                 │
│  ┌─────────────┐    ┌──────────────────────────┐    ┌────────┐ │
│  │ CPU Thread  │    │      GPU Execution        │    │  CPU   │ │
│  │             │    │                           │    │  Post  │ │
│  │ Tokenize    │───▶│  ┌────────────────────┐  │───▶│Process │ │
│  │ Preprocess  │    │  │  Transformer Block │  │    │        │ │
│  │ H2D Transfer│    │  │  ┌──────────────┐  │  │    │ Decode │ │
│  └─────────────┘    │  │  │FlashAttention│  │  │    │ Output │ │
│                     │  │  └──────────────┘  │  │    └────────┘ │
│                     │  │  ┌──────────────┐  │  │               │
│                     │  │  │  LayerNorm   │  │  │               │
│                     │  │  └──────────────┘  │  │               │
│                     │  │  ┌──────────────┐  │  │               │
│                     │  │  │  FusedFFN    │  │  │               │
│                     │  │  └──────────────┘  │  │               │
│                     │  └────────────────────┘  │               │
│                     │   (×N layers)             │               │
│                     └──────────────────────────┘               │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. FlashAttention-2 Kernel (`src/flash_attention.cu`)

Implements the tiled, IO-aware attention algorithm. Avoids materializing the full N×N attention matrix by processing queries in blocks, iterating over key-value tiles, and maintaining online softmax statistics (running max `m` and denominator `l`).

- **Memory complexity**: O(N·d) vs O(N²) for standard attention
- **Tile sizes**: `Br=64` (Q-block), `Bc=64` (KV-block), templated on head dim `d`
- **Precision**: FP32 accumulation with optional FP16 I/O (`__half`)
- **Causal masking**: In-kernel masking for autoregressive decoding
- **Multi-head**: One CUDA block per (batch, head, Q-tile) triple

Algorithm sketch:
```
for each Q tile (q_start..q_start+Br):
    m = -∞,  l = 0,  O = 0
    for each KV tile (kv_start..kv_start+Bc):
        S  = Q_tile · Kᵀ_tile × scale        # [Br, Bc]
        m' = max(rowmax(S), m)
        P  = exp(S − m')                      # [Br, Bc]
        l' = rowsum(P) + l · exp(m − m')
        O  = O · exp(m − m') + P · V_tile
        m, l = m', l'
    O /= l
```

### 2. Fused Layer Normalization Kernel (`src/kernels.cu`)

Single-pass online algorithm that computes mean and variance in one sweep using Welford's method, then normalizes and applies learned `γ`/`β` parameters. Each warp handles one row of the input tensor.

- Warp-level reductions via `__shfl_xor_sync`
- Fused residual add: `LayerNorm(x + residual)`
- Supports FP16 and FP32

### 3. Fused Feed-Forward Network Kernel (`src/kernels.cu`)

Two linear projections with a GELU activation fused into a minimized kernel sequence:

```
FFN(x) = GELU(x · W₁ + b₁) · W₂ + b₂
```

- cuBLAS GEMM for projection matrices (Tensor Core-accelerated on Ampere+)
- Custom fused GELU kernel eliminates intermediate write-back
- Pre-allocated workspace to avoid runtime memory allocation

### 4. Transformer Block (`src/transformer.cu`)

Combines the above primitives into a complete pre-norm transformer layer:

```
x = x + MultiHeadAttention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

Manages per-layer weight buffers (Q/K/V/O projections, FFN matrices, LN parameters) and CUDA streams for overlapping computation.

### 5. Async CPU-GPU Pipeline (`src/pipeline.cpp`)

Three-stage producer-consumer pipeline using a bounded lock-free queue and CUDA streams with host callbacks:

```
Stage 1 (CPU):  tokenize → embed → H2D transfer (pinned memory)
Stage 2 (GPU):  N × transformer blocks (async on stream)
Stage 3 (CPU):  D2H transfer → logits → greedy/sampling decode
```

- Double-buffering hides H2D/D2H transfer latency
- CUDA events synchronize between stages
- Worker threads pinned to specific cores for NUMA locality

## Performance Techniques

| Technique | Description |
|---|---|
| FlashAttention-2 tiling | O(N) GPU memory, high arithmetic intensity |
| Fused kernels | LayerNorm + residual, GELU in single pass |
| Mixed precision | FP16 I/O, FP32 accumulation |
| Pinned memory | Page-locked host buffers for peak PCIe bandwidth |
| Double buffering | Overlap compute and memory transfers |
| cuBLAS GEMM | Tensor Core WMMA for matrix multiplications |
| Async streams | Overlap transformer layers where possible |

## Project Structure

```
FlashGen/
├── CMakeLists.txt
├── README.md
├── include/
│   ├── model_config.hpp      # GPT-2 / custom model hyperparameters
│   ├── cuda_utils.cuh        # Error-checking macros, CUDA helpers
│   ├── flash_attention.cuh   # FlashAttention kernel declarations
│   ├── kernels.cuh           # LayerNorm, GELU, FFN declarations
│   ├── transformer.hpp       # TransformerLayer class
│   └── pipeline.hpp          # AsyncPipeline class
├── src/
│   ├── flash_attention.cu    # FlashAttention-2 CUDA kernels
│   ├── kernels.cu            # LayerNorm, GELU, FFN CUDA kernels
│   ├── transformer.cu        # Transformer block implementation
│   ├── pipeline.cpp          # Async CPU-GPU pipeline
│   └── main.cpp              # GPT-2 inference demo
└── tests/
    ├── test_flash_attention.cu   # Correctness vs. naive attention
    ├── test_kernels.cu           # LayerNorm / FFN correctness + perf
    └── test_pipeline.cpp         # End-to-end pipeline throughput test
```

## Requirements

- CUDA 11.8+ (12.x recommended)
- cuBLAS (ships with CUDA toolkit)
- CMake 3.20+
- GCC 10+ / Clang 12+ with C++17
- GPU: NVIDIA Volta (sm_70) or newer; Ampere (sm_80) recommended for FP16 Tensor Cores

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCH=80   # sm_80 = A100/RTX 30xx
make -j$(nproc)
```

For older GPUs (e.g. V100):
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCH=70
```

To enable FP16 mode at compile time:
```bash
cmake .. -DUSE_FP16=ON
```

## Usage

### Single-sequence inference

```bash
./flashgen --model gpt2 --prompt "The future of AI is" --max_tokens 128
```

### Batched inference

```bash
./flashgen --model gpt2 --batch_size 8 --prompt "Once upon a time" --max_tokens 64
```

### Benchmark mode

```bash
./flashgen --benchmark --seq_len 512 --batch_size 4 --num_runs 100
```

### Run tests

```bash
cd build
ctest --verbose
# or individually:
./test_flash_attention
./test_kernels
./test_pipeline
```

## Supported Models

| Model | Layers | Heads | d_model | d_ff |
|---|---|---|---|---|
| GPT-2 Small | 12 | 12 | 768 | 3072 |
| GPT-2 Medium | 24 | 16 | 1024 | 4096 |
| GPT-2 Large | 36 | 20 | 1280 | 5120 |
| GPT-2 XL | 48 | 25 | 1600 | 6400 |
| Custom | configurable | configurable | configurable | configurable |

Weights can be loaded from `.bin` files exported via the companion Python script (not included here), or from HuggingFace checkpoints using the provided weight converter.

## License

MIT
