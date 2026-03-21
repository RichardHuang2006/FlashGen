# FlashGen

**End-to-end LLM inference engine** — TensorRT + custom CUDA kernels + vLLM-style paged KV cache.

Inspired by [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) and [vLLM](https://github.com/vllm-project/vllm), this project demonstrates systems-level GPU optimization for large language model serving: from HuggingFace model loading through ONNX export, TensorRT engine building, and high-throughput continuous batching with custom CUDA kernels.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Python API  (flashgen/)                     │
│                                                                 │
│  LLM / AsyncLLMEngine                                          │
│    ├── HFModelLoader      HuggingFace → GPU weight tensors     │
│    ├── ONNXExporter       torch.onnx.export (KV cache as I/O)  │
│    ├── TRTEngineBuilder   ONNX → TensorRT FP16 / INT8          │
│    ├── TRTEngine          Run serialized .trt engine           │
│    ├── ContinuousBatcher  Iteration-level scheduling           │
│    ├── BlockAllocator     GPU memory pool — O(1) alloc/free    │
│    ├── PagedKVCache       Block-table indirection + CoW prefix │
│    ├── Sampler            Greedy / top-k / top-p / temperature │
│    └── StreamingGenerator Real-time token-by-token streaming   │
│                                                                 │
└──────────────────────┬──────────────────────────────────────────┘
                       │  PyBind11  (_C extension)
┌──────────────────────▼──────────────────────────────────────────┐
│                  C++ / CUDA Runtime  (csrc/)                    │
│                                                                 │
│  InferenceEngine  (pipeline.cpp)                               │
│    ├── Transformer        transformer.cu                       │
│    │     ├── FlashAttention-2    flash_attention.cu            │
│    │     ├── Paged attention     paged_kv_cache.cu             │
│    │     ├── RMSNorm / LayerNorm kernels.cu                    │
│    │     ├── RoPE (in-place)     kernels.cu                    │
│    │     └── SwiGLU / GELU FFN  kernels.cu                    │
│    ├── BlockAllocator     block_allocator.cpp                  │
│    ├── PagedKVCache       paged_kv_cache.cu                    │
│    ├── PrefixCache        prefix_cache.cpp  (radix tree LRU)  │
│    └── Scheduler          scheduler.cpp                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Features

### 1. TensorRT Engine (FP16 + INT8)

Export any HuggingFace causal LM to ONNX with **KV cache as explicit graph I/O**,
then build an optimized TensorRT engine:

- **FP16** — halves memory bandwidth pressure; uses Tensor Cores. ~2× vs FP32.
- **INT8** — EntropyCalibrator2 computes per-tensor activation scales.
  ~3–4× vs FP32, <1% perplexity loss.
- Dynamic shape optimization profiles cover prefill and decode separately.
- KV cache is managed outside TRT (paged blocks); TRT handles the transformer body.

### 2. Custom CUDA Kernels

| Kernel | File | Key optimization |
|--------|------|-----------------|
| FlashAttention-2 | `flash_attention.cu` | Tiled Q (Br=64), K/V (Bc=64); online softmax; O(n) HBM reads |
| Paged attention decode | `paged_kv_cache.cu` | Block-table indirection; one warp per (seq, head) |
| Paged attention prefill | `paged_kv_cache.cu` | Multi-token query with tiled KV gather |
| RMSNorm + residual | `kernels.cu` | Fused Welford reduction; one warp per row |
| RoPE | `kernels.cu` | In-place; zero extra allocation |
| SwiGLU FFN | `kernels.cu` | Fused gate × silu(up) multiply |
| INT8 quantization | `quantization.cu` | Per-token symmetric; dequant fused into attention |

### 3. Paged KV Cache (vLLM-style)

```
Traditional:  allocate max_seq_len × kv_size per sequence → 30–60% waste
Paged:        allocate one block (16 tokens) at a time → <1% waste

Block table:  seq_id → [physical_block_0, physical_block_1, ...]
              Attention kernel follows block table indirection to locate K,V.
```

Benefits:
- **+55% more concurrent sequences** in the same GPU memory
- **Zero external fragmentation** between sequences
- **Copy-on-write prefix sharing** via reference counting

### 4. Continuous Batching

```
Static batching:    [AAAA____] [BBBBBBBB]  — GPU idle while short A waits for long B
Continuous:         [AAAACCCC] [BBBBBBBB]  — C starts the moment A finishes
                          ↑ A finishes → C admitted from WAITING queue
```

Three-queue scheduler (WAITING → RUNNING → PREEMPTED):
- FCFS admission with memory-aware preemption (LIFO eviction)
- Token budget prevents prefill from starving decode iterations
- Chunked prefill splits long prompts across iterations

### 5. Streaming Generation

```python
for piece in llm.stream("Tell me about GPU memory", params):
    print(piece, end="", flush=True)   # tokens appear in real-time
```

- **TTFT** (Time-To-First-Token): dominant latency for chat UX
- **TPOT** (Time-Per-Output-Token): streaming smoothness
- SSE format support for HTTP/FastAPI streaming

---

## GPU Execution Tradeoffs

| Axis | Lever | Tradeoff |
|------|-------|----------|
| Memory bandwidth | FP16 weights/activations | 2× BW savings vs FP32; <0.1% accuracy |
| Memory bandwidth | INT8 KV cache | 4× KV memory; ~1% perplexity |
| Compute efficiency | FlashAttention-2 | 2–4× attn kernel; avoids O(n²) memory |
| Memory efficiency | Block size (tokens/block) | Smaller → less waste, more table overhead |
| Latency vs throughput | max_num_tokens | Larger → better GPU utilization, higher TPOT |
| Latency vs throughput | Batch size | Larger → higher throughput, higher tail latency |
| Repeated context | Prefix caching (radix tree) | Eliminates redundant prefill for shared prompts |
| GQA | n_kv_heads < n_heads | Less KV memory; same n_heads query compute |

---

## Supported Models

| Model | HF ID | Architecture |
|-------|-------|--------------|
| GPT-2 | `gpt2` | MHA, learned pos emb, GELU FFN |
| GPT-2 Medium | `gpt2-medium` | MHA, learned pos emb, GELU FFN |
| GPT-2 XL | `gpt2-xl` | MHA, learned pos emb, GELU FFN |
| LLaMA-2 7B | `meta-llama/Llama-2-7b-hf` | MHA, RoPE, RMSNorm, SwiGLU |
| LLaMA-2 13B | `meta-llama/Llama-2-13b-hf` | MHA, RoPE, RMSNorm, SwiGLU |
| Mistral 7B | `mistralai/Mistral-7B-v0.1` | GQA (8 KV heads), RoPE, SwiGLU |

---

## Installation

### Prerequisites

- NVIDIA GPU (Ampere sm_80+ recommended)
- CUDA 12.x + cuDNN
- Python 3.9+
- CMake 3.22+

```bash
# Install Python dependencies
pip install -r requirements.txt

# Build the C++ CUDA extension (flashgen._C)
GPU_ARCH=80 pip install -e .   # A100 / A10G
GPU_ARCH=86 pip install -e .   # RTX 3090 / A6000
GPU_ARCH=89 pip install -e .   # RTX 4090 / L40S
GPU_ARCH=90 pip install -e .   # H100
```

### Manual CMake build (for kernel development)

```bash
mkdir build && cd build
cmake .. -DGPU_ARCH=80 -DCMAKE_BUILD_TYPE=Release
cmake --build . --target flashgen_ext -j$(nproc)
cd ..
# _C.so is placed directly in flashgen/ for immediate import
```

### TensorRT (optional)

```bash
pip install tensorrt --index-url https://pypi.ngc.nvidia.com
# Or via apt: sudo apt-get install tensorrt
```

---

## Quick Start

```python
from flashgen import LLM, SamplingParams

# Load model (no build required for pytorch backend)
llm = LLM("gpt2", backend="pytorch")
params = SamplingParams(temperature=0.8, top_p=0.9, max_tokens=100)

# Single generation
output = llm.generate("The future of artificial intelligence is", params)
print(output.text)
print(f"{output.generated_tokens} tokens | {output.latency_ms:.0f}ms | "
      f"{output.throughput_tps:.0f} tok/s")

# Streaming (tokens appear in real-time)
for piece in llm.stream("Once upon a time in a GPU datacenter", params):
    print(piece, end="", flush=True)

# Batch generation (continuous batching)
outputs = llm.generate_batch(
    ["Hello world", "What is CUDA?", "Explain attention mechanisms"],
    params
)
for o in outputs:
    print(o.text[:80])
```

---

## ONNX Export + TensorRT

```python
llm = LLM("gpt2")

# Step 1: Export to ONNX (KV cache as explicit graph I/O)
llm.export_onnx("engines/gpt2.onnx")

# Step 2: Build TRT engine
llm.build_trt_engine(
    "engines/gpt2.onnx",
    "engines/gpt2_fp16.trt",
    precision="fp16",
    max_batch=32,
    max_seq=1024,
)

# Step 3: Use TRT engine for inference
llm_trt = LLM("gpt2", backend="trt", trt_engine_path="engines/gpt2_fp16.trt")
output = llm_trt.generate("The GPU executes", params)
```

---

## Project Structure

```
FlashGen/
├── flashgen/                     Python package (primary interface)
│   ├── llm.py                    High-level LLM class
│   ├── engine.py                 AsyncLLMEngine orchestration
│   ├── outputs.py                RequestOutput, CompletionOutput
│   ├── core/config.py            ModelConfig, SamplingParams, EngineConfig
│   ├── model_loader/
│   │   ├── hf_loader.py          HuggingFace weight loading + name mapping
│   │   └── tokenizer.py          Tokenizer wrapper
│   ├── onnx_export/exporter.py   ONNX export with KV cache I/O
│   ├── trt_builder/
│   │   ├── builder.py            TensorRT engine builder (FP16/INT8)
│   │   ├── calibrator.py         INT8 EntropyCalibrator2
│   │   └── engine_runner.py      TRT inference runner
│   ├── memory/
│   │   ├── block_allocator.py    GPU block pool (Python)
│   │   └── paged_kv_cache.py     Block-table KV cache
│   ├── scheduler/continuous_batcher.py   3-queue FCFS scheduler
│   ├── sampling/sampler.py       Greedy / top-k / top-p / temperature
│   ├── streaming/generator.py    Sync + async streaming
│   └── profiling/nsight.py       Engine profiling hooks (no-op stubs)
│
├── csrc/                         C++ / CUDA kernel library
│   ├── include/                  Headers (model_config, request, scheduler…)
│   ├── kernels/
│   │   ├── flash_attention.cu    FlashAttention-2 (Br=64, Bc=64)
│   │   ├── paged_kv_cache.cu     Paged attention (prefill + decode)
│   │   ├── kernels.cu            Norm, RoPE, FFN, sampling kernels
│   │   └── quantization.cu       INT8 symmetric per-token quantization
│   ├── runtime/
│   │   ├── block_allocator.cpp   O(1) free-list allocator + refcounting
│   │   ├── prefix_cache.cpp      Radix tree LRU prefix cache
│   │   ├── scheduler.cpp         Continuous batching (WAITING/RUNNING/PREEMPTED)
│   │   ├── transformer.cu        Transformer forward pass + paged attention
│   │   └── pipeline.cpp          Engine loop, sampling, benchmarking
│   └── bindings/flashgen_bindings.cpp   PyBind11 Python ↔ C++ bridge
│
├── CMakeLists.txt
├── setup.py
└── requirements.txt
```

---

## References

- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) — Dao et al., 2023
- [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — Kwon et al., 2023 (vLLM)
- [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu) — Yu et al., 2022
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) — Ainslie et al., 2023
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) — NVIDIA, 2023
