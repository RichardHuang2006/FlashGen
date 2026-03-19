"""
Latency benchmark: Time-To-First-Token (TTFT) and Time-Per-Output-Token (TPOT).

Metrics:
  TTFT:  Time from request submission to first output token.
         Dominated by prefill compute: O(prompt_len²) for attention.
         Critical for interactive chat (user sees first word quickly).

  TPOT:  Average time between consecutive output tokens (after first).
         Dominated by decode: one forward pass per token, O(prompt_len) attn.
         Critical for streaming fluency.

  P50/P95/P99 percentiles reveal tail latency behavior under load.

GPU execution tradeoffs captured here:
  - Memory bandwidth: prefill is compute-bound, decode is bandwidth-bound
    (loading KV cache from HBM dominates decode time).
  - Batch size: larger batch → higher TPOT (more KV to load), lower TTFT
    amortization.
  - Context length: longer context → higher TTFT (quadratic attn),
    higher TPOT (more KV to load per decode step).

Usage:
    python -m flashgen.benchmarks.bench_latency \\
        --model gpt2 \\
        --input-len 128 \\
        --output-len 128 \\
        --num-iters 100 \\
        --batch-size 1
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch


@dataclass
class LatencyResult:
    model: str
    backend: str
    batch_size: int
    input_len: int
    output_len: int
    ttft_ms: List[float] = field(default_factory=list)
    tpot_ms: List[float] = field(default_factory=list)

    def print(self):
        def pct(lst, p):
            lst = sorted(lst)
            idx = int(len(lst) * p / 100)
            return lst[min(idx, len(lst) - 1)]

        print(f"\n{'─'*60}")
        print(f"  Latency benchmark: {self.model} ({self.backend})")
        print(f"  batch={self.batch_size}  input={self.input_len}  output={self.output_len}")
        print(f"{'─'*60}")
        if self.ttft_ms:
            print(f"  TTFT (ms):  mean={statistics.mean(self.ttft_ms):.1f}  "
                  f"p50={pct(self.ttft_ms,50):.1f}  "
                  f"p95={pct(self.ttft_ms,95):.1f}  "
                  f"p99={pct(self.ttft_ms,99):.1f}")
        if self.tpot_ms:
            print(f"  TPOT (ms):  mean={statistics.mean(self.tpot_ms):.1f}  "
                  f"p50={pct(self.tpot_ms,50):.1f}  "
                  f"p95={pct(self.tpot_ms,95):.1f}  "
                  f"p99={pct(self.tpot_ms,99):.1f}")
        print(f"{'─'*60}\n")


def run_benchmark(
    model: str = "gpt2",
    backend: str = "pytorch",
    input_len: int = 128,
    output_len: int = 128,
    num_iters: int = 50,
    batch_size: int = 1,
    seed: int = 42,
) -> LatencyResult:
    import random
    random.seed(seed)
    torch.manual_seed(seed)

    from flashgen import LLM, SamplingParams

    llm = LLM(model=model, backend=backend)
    tokenizer = llm.tokenizer
    vocab_size = tokenizer.vocab_size

    result = LatencyResult(model=model, backend=backend,
                           batch_size=batch_size,
                           input_len=input_len, output_len=output_len)
    params = SamplingParams(temperature=1.0, max_tokens=output_len, greedy=True)

    # Warmup
    warmup_ids = [random.randint(0, vocab_size - 1) for _ in range(input_len)]
    warmup_text = tokenizer.decode(warmup_ids)
    for _ in range(3):
        llm.generate(warmup_text, SamplingParams(max_tokens=4, greedy=True))
    torch.cuda.synchronize()

    for _ in range(num_iters):
        token_ids = [random.randint(0, vocab_size - 1) for _ in range(input_len)]
        prompt = tokenizer.decode(token_ids)

        t_submit = time.perf_counter()
        t_first: Optional[float] = None
        token_times: List[float] = []

        # Use streaming to capture per-token timing
        for piece in llm.stream(prompt, params):
            t_now = time.perf_counter()
            if t_first is None:
                t_first = t_now
                result.ttft_ms.append((t_first - t_submit) * 1000)
            else:
                token_times.append(t_now)

        if len(token_times) > 1:
            intervals = [(token_times[i] - token_times[i - 1]) * 1000
                         for i in range(1, len(token_times))]
            result.tpot_ms.extend(intervals)

    return result


def main():
    parser = argparse.ArgumentParser(description="FlashGen latency benchmark")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--backend", default="pytorch",
                        choices=["pytorch", "cuda", "trt"])
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--num-iters", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    result = run_benchmark(
        model=args.model,
        backend=args.backend,
        input_len=args.input_len,
        output_len=args.output_len,
        num_iters=args.num_iters,
        batch_size=args.batch_size,
    )
    result.print()


if __name__ == "__main__":
    main()
