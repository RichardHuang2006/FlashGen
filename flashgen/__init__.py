"""
FlashGen — End-to-end LLM inference engine.

Inspired by TensorRT-LLM and vLLM, this engine delivers high-throughput,
low-latency inference via:
  • TensorRT-optimized transformer execution (FP16 / INT8)
  • Custom CUDA kernels: FlashAttention-2, paged attention
  • Paged KV cache with block-table indirection (vLLM-style)
  • Continuous batching scheduler
  • Streaming token generation

Quick start
-----------
    from flashgen import LLM, SamplingParams

    llm = LLM(model="gpt2")
    params = SamplingParams(temperature=0.8, max_tokens=200)

    # Synchronous generation
    output = llm.generate("The future of AI is", params)
    print(output.text)

    # Streaming
    for token in llm.stream("Once upon a time", params):
        print(token, end="", flush=True)
"""

from flashgen.llm import LLM                         # noqa: F401
from flashgen.engine import AsyncLLMEngine            # noqa: F401
from flashgen.core.config import (                    # noqa: F401
    ModelConfig,
    CacheConfig,
    SchedulerConfig,
    EngineConfig,
    SamplingParams,
)
from flashgen.outputs import RequestOutput            # noqa: F401

__version__ = "0.1.0"
__all__ = [
    "LLM",
    "AsyncLLMEngine",
    "ModelConfig",
    "CacheConfig",
    "SchedulerConfig",
    "EngineConfig",
    "SamplingParams",
    "RequestOutput",
]
