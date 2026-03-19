"""
Throughput benchmark: tokens per second for prefill and decode.

Metrics measured:
  prefill_throughput:  (prompt_tokens * num_requests) / prefill_time  [tok/s]
  decode_throughput:   (output_tokens * num_requests) / decode_time   [tok/s]
  e2e_throughput:      total_tokens / wall_time                       [tok/s]

These map to real workloads as:
  prefill  ↔ batch document processing, long-context queries
  decode   ↔ streaming chat, creative writing
  e2e      ↔ mixed-use production traffic

NVTX annotations allow correlation with Nsight Systems traces.

Usage:
    python -m flashgen.benchmarks.bench_throughput \\
        --model gpt2 \\
        --num-requests 100 \\
        --input-len 128 \\
        --output-len 128 \\
        --backend pytorch
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from typing import List

import torch


@dataclass
class ThroughputResult:
    model: str
    backend: str
    num_requests: int
    input_len: int
    output_len: int
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    elapsed_s: float
    prefill_tps: float
    decode_tps: float
    e2e_tps: float
    gpu_memory_gb: float

    def print(self):
        print(f"\n{'─'*60}")
        print(f"  Throughput benchmark: {self.model} ({self.backend})")
        print(f"{'─'*60}")
        print(f"  Requests           : {self.num_requests}")
        print(f"  Input / Output len : {self.input_len} / {self.output_len} tokens")
        print(f"  Total tokens       : {self.total_tokens:,} ({self.total_input_tokens:,} in + {self.total_output_tokens:,} out)")
        print(f"  Wall time          : {self.elapsed_s:.2f}s")
        print(f"  Prefill throughput : {self.prefill_tps:.0f} tok/s")
        print(f"  Decode throughput  : {self.decode_tps:.0f} tok/s")
        print(f"  E2E throughput     : {self.e2e_tps:.0f} tok/s")
        print(f"  GPU memory (peak)  : {self.gpu_memory_gb:.2f} GB")
        print(f"{'─'*60}\n")


def run_benchmark(
    model: str = "gpt2",
    backend: str = "pytorch",
    num_requests: int = 50,
    input_len: int = 128,
    output_len: int = 128,
    seed: int = 42,
) -> ThroughputResult:
    random.seed(seed)
    torch.manual_seed(seed)

    from flashgen import LLM, SamplingParams

    llm = LLM(model=model, backend=backend)
    tokenizer = llm.tokenizer

    # Generate random prompts of the desired input length
    vocab_size = tokenizer.vocab_size
    prompts: List[List[int]] = [
        [random.randint(0, vocab_size - 1) for _ in range(input_len)]
        for _ in range(num_requests)
    ]
    prompt_texts = [tokenizer.decode(ids) for ids in prompts]

    params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=output_len,
        greedy=True,
    )

    # Warmup
    _ = llm.generate(prompt_texts[0], SamplingParams(max_tokens=8, greedy=True))
    torch.cuda.synchronize()

    # Reset peak memory
    torch.cuda.reset_peak_memory_stats()

    # Prefill timing: run one request, measure prompt processing
    t_prefill_start = time.perf_counter()
    _ = llm.generate(prompt_texts[0], SamplingParams(max_tokens=1, greedy=True))
    torch.cuda.synchronize()
    t_prefill_end = time.perf_counter()
    prefill_time = t_prefill_end - t_prefill_start

    # Full E2E: all requests
    from flashgen.profiling.nsight import NsightProfiler
    profiler = NsightProfiler()
    profiler.start()

    t_start = time.perf_counter()
    results = llm.generate_batch(prompt_texts, params)
    torch.cuda.synchronize()
    t_end = time.perf_counter()

    profiler.stop()
    elapsed = t_end - t_start

    total_output = sum(r.generated_tokens for r in results)
    total_input = num_requests * input_len
    total_tokens = total_input + total_output

    peak_mem = torch.cuda.max_memory_allocated() / 1024**3
    prefill_tps = total_input / prefill_time if prefill_time > 0 else 0
    decode_time = max(elapsed - prefill_time, 1e-6)
    decode_tps = total_output / decode_time
    e2e_tps = total_tokens / elapsed

    return ThroughputResult(
        model=model,
        backend=backend,
        num_requests=num_requests,
        input_len=input_len,
        output_len=output_len,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        total_tokens=total_tokens,
        elapsed_s=elapsed,
        prefill_tps=prefill_tps,
        decode_tps=decode_tps,
        e2e_tps=e2e_tps,
        gpu_memory_gb=peak_mem,
    )


def main():
    parser = argparse.ArgumentParser(description="FlashGen throughput benchmark")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--backend", default="pytorch",
                        choices=["pytorch", "cuda", "trt"])
    parser.add_argument("--num-requests", type=int, default=50)
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = run_benchmark(
        model=args.model,
        backend=args.backend,
        num_requests=args.num_requests,
        input_len=args.input_len,
        output_len=args.output_len,
        seed=args.seed,
    )
    result.print()


if __name__ == "__main__":
    main()
