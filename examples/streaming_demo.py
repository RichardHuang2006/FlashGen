#!/usr/bin/env python3
"""
Streaming token generation demo.

Demonstrates real-time token streaming: tokens appear as they are generated,
mimicking the ChatGPT-style response experience.

Key metrics shown:
  TTFT (Time-To-First-Token) — how quickly the user sees the first word
  TPOT (Time-Per-Output-Token) — smoothness of the streaming experience

Run:
    python examples/streaming_demo.py
    python examples/streaming_demo.py --prompt "Write a poem about GPU memory"
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from flashgen import LLM, SamplingParams


def streaming_generate(llm: LLM, prompt: str, params: SamplingParams):
    """Stream tokens to stdout in real-time."""
    print(f"\nPrompt: {prompt!r}")
    print(f"{'─'*60}")
    print("Response (streaming): ", end="", flush=True)

    t_start = time.perf_counter()
    t_first: float = 0
    token_count = 0
    token_times = []

    for piece in llm.stream(prompt, params):
        t_now = time.perf_counter()
        if token_count == 0:
            t_first = t_now
            print()  # newline after "Response (streaming):"
        print(piece, end="", flush=True)
        token_times.append(t_now)
        token_count += 1

    print()  # final newline
    t_end = time.perf_counter()

    ttft_ms = (t_first - t_start) * 1000
    total_ms = (t_end - t_start) * 1000
    tpots = [(token_times[i] - token_times[i-1]) * 1000
             for i in range(1, len(token_times))]
    avg_tpot = sum(tpots) / len(tpots) if tpots else 0

    print(f"\n{'─'*60}")
    print(f"Tokens generated : {token_count}")
    print(f"TTFT             : {ttft_ms:.0f} ms  (time to first token)")
    print(f"TPOT avg         : {avg_tpot:.0f} ms/token")
    print(f"Total latency    : {total_ms:.0f} ms")
    print(f"Throughput       : {token_count / total_ms * 1000:.0f} tok/s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--prompt", default=(
        "The key difference between a GPU and a CPU is that"
    ))
    parser.add_argument("--max-tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    print(f"\nLoading {args.model}…")
    llm = LLM(model=args.model, backend="pytorch")

    params = SamplingParams(
        temperature=args.temperature,
        top_p=0.9,
        max_tokens=args.max_tokens,
    )

    # Single streaming request
    streaming_generate(llm, args.prompt, params)

    # Demonstrate multiple independent streaming sessions
    print(f"\n{'═'*60}")
    print("Multiple streaming sessions (sequential):")
    print(f"{'═'*60}")

    prompts = [
        "In 2030, artificial intelligence will",
        "The most important thing about GPU memory is",
        "When training large language models, the bottleneck is",
    ]
    for p in prompts:
        streaming_generate(llm, p, SamplingParams(max_tokens=60, temperature=0.9))


if __name__ == "__main__":
    main()
