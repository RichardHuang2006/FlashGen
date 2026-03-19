#!/usr/bin/env python3
"""
Concurrent batch inference with continuous batching.

Demonstrates:
  - Submitting multiple requests simultaneously
  - Continuous batching: finished sequences are replaced immediately
  - Throughput vs latency tradeoffs with varying batch sizes
  - Memory efficiency of paged KV cache

The key insight of continuous batching:
  With static batching, all sequences run together until the longest one
  finishes. With continuous batching, a finished sequence is immediately
  replaced with a new one from the queue — GPU utilization stays high.

Run:
    python examples/batch_inference.py --num-requests 16
    python examples/batch_inference.py --num-requests 64 --input-len 256
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from flashgen import LLM, SamplingParams


PROMPTS = [
    "The theory of general relativity states that",
    "In machine learning, overfitting occurs when",
    "The most efficient sorting algorithm for large arrays is",
    "Climate change is driven primarily by",
    "Quantum computing differs from classical computing because",
    "The human brain contains approximately",
    "Water freezes at 0 degrees Celsius because",
    "The first transistor was invented in",
    "CUDA allows programmers to use GPU for",
    "Attention mechanisms in transformers work by",
    "The Fibonacci sequence is defined as",
    "Natural language processing enables computers to",
    "The speed of light in a vacuum is",
    "GPU memory bandwidth is important because",
    "TensorRT optimizes neural network inference by",
    "The paged KV cache in vLLM improves memory efficiency by",
]


def demonstrate_batch_scaling(llm: LLM, max_new_tokens: int = 64):
    """Show throughput vs batch size tradeoff."""
    print(f"\n{'═'*65}")
    print("  Throughput vs batch size (continuous batching)")
    print(f"{'═'*65}")
    print(f"  {'Batch':>8} {'Requests':>10} {'Wall (s)':>10} {'Tok/s':>10} {'TTFT (ms)':>12}")
    print(f"  {'─'*65}")

    for batch_size in [1, 2, 4, 8, 16]:
        n = min(batch_size, len(PROMPTS))
        batch_prompts = PROMPTS[:n]
        params = SamplingParams(
            temperature=1.0, top_p=1.0,
            max_tokens=max_new_tokens, greedy=True,
        )

        t0 = time.perf_counter()
        results = llm.generate_batch(batch_prompts, params)
        t1 = time.perf_counter()

        elapsed = t1 - t0
        total_out = sum(r.generated_tokens for r in results)
        total_in = sum(r.prompt_tokens for r in results)
        total_tok = total_out + total_in
        tps = total_tok / elapsed if elapsed > 0 else 0

        # Estimate TTFT from latency of first result
        ttft = results[0].ttft_ms if results else 0

        print(f"  {batch_size:>8} {n:>10} {elapsed:>10.2f} {tps:>10.0f} {ttft:>12.0f}")


def demonstrate_variable_lengths(llm: LLM):
    """Show how continuous batching handles variable-length sequences."""
    print(f"\n{'═'*65}")
    print("  Variable-length sequences (continuous batching advantage)")
    print(f"{'═'*65}")

    # Mix of short and long output requests
    mixed_params = [
        ("Short output (16 tok)", SamplingParams(max_tokens=16, greedy=True)),
        ("Medium output (64 tok)", SamplingParams(max_tokens=64, greedy=True)),
        ("Long output (128 tok)", SamplingParams(max_tokens=128, greedy=True)),
        ("Short output (16 tok)", SamplingParams(max_tokens=16, greedy=True)),
        ("Long output (128 tok)", SamplingParams(max_tokens=128, greedy=True)),
        ("Medium output (64 tok)", SamplingParams(max_tokens=64, greedy=True)),
    ]

    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(len(mixed_params))]

    print(f"  Submitting {len(mixed_params)} requests with varied output lengths…")
    t0 = time.perf_counter()

    # Submit all at once
    results = llm.generate_batch(prompts, SamplingParams(max_tokens=128, greedy=True))

    elapsed = time.perf_counter() - t0
    print(f"\n  All {len(results)} requests completed in {elapsed:.2f}s")
    print(f"\n  {'Request':>10} {'Output tokens':>15} {'Latency (ms)':>15}")
    print(f"  {'─'*45}")
    for i, r in enumerate(results):
        print(f"  {i+1:>10} {r.generated_tokens:>15} {r.latency_ms:>15.0f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--num-requests", type=int, default=8)
    parser.add_argument("--input-len", type=int, default=0,
                        help="Fixed input length (0 = use natural prompts)")
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    print(f"\nLoading {args.model}…")
    llm = LLM(model=args.model, backend="pytorch")

    # Basic batch generation
    n = min(args.num_requests, len(PROMPTS))
    prompts = PROMPTS[:n]
    params = SamplingParams(
        temperature=0.8, top_p=0.9, max_tokens=args.max_tokens
    )

    print(f"\nBatch generation: {n} concurrent requests")
    print(f"{'─'*60}")
    t0 = time.perf_counter()
    results = llm.generate_batch(prompts, params)
    elapsed = time.perf_counter() - t0

    total_tokens = sum(r.generated_tokens + r.prompt_tokens for r in results)
    print(f"\nCompleted {n} requests in {elapsed:.2f}s")
    print(f"Throughput: {total_tokens / elapsed:.0f} tok/s")

    for i, r in enumerate(results):
        print(f"\n[{i+1}] {prompts[i]!r}")
        print(f"    → {r.text[:100]}…")

    # Scaling study
    demonstrate_batch_scaling(llm, max_new_tokens=args.max_tokens)
    demonstrate_variable_lengths(llm)


if __name__ == "__main__":
    main()
