#!/usr/bin/env python3
"""
Basic inference example — simplest path to generation.

Demonstrates:
  - Loading a model from HuggingFace
  - Generating text with various sampling strategies
  - Examining output metadata (tokens, latency)

Run:
    python examples/basic_inference.py
    python examples/basic_inference.py --model gpt2-medium --max-tokens 200
"""

import argparse
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from flashgen import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(description="FlashGen basic inference example")
    parser.add_argument("--model", default="gpt2",
                        help="HuggingFace model ID (e.g., gpt2, gpt2-medium)")
    parser.add_argument("--prompt", default="The future of artificial intelligence is",
                        help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=100,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--greedy", action="store_true",
                        help="Use greedy decoding (deterministic)")
    args = parser.parse_args()

    print(f"\nLoading {args.model}…")
    llm = LLM(model=args.model, backend="pytorch")

    params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        greedy=args.greedy,
    )

    print(f"Prompt: {args.prompt!r}\n")
    print("─" * 60)

    output = llm.generate(args.prompt, params)

    print(f"\nGenerated text:\n{output.text}")
    print(f"\n{'─'*60}")
    print(f"Prompt tokens    : {output.prompt_tokens}")
    print(f"Generated tokens : {output.generated_tokens}")
    print(f"Total latency    : {output.latency_ms:.1f} ms")
    print(f"Time-to-1st-tok  : {output.ttft_ms:.1f} ms")
    print(f"TPOT             : {output.tpot_ms:.1f} ms/token")
    print(f"Throughput       : {output.throughput_tps:.0f} tok/s")

    # ── Different sampling strategies ─────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("Sampling strategy comparison:")
    print(f"{'═'*60}")

    strategies = [
        ("Greedy (argmax)",        SamplingParams(greedy=True, max_tokens=50)),
        ("Temperature=0.3 (sharp)",SamplingParams(temperature=0.3, max_tokens=50)),
        ("Temperature=1.5 (flat)", SamplingParams(temperature=1.5, max_tokens=50)),
        ("Top-K (k=10)",           SamplingParams(top_k=10, max_tokens=50)),
        ("Top-P / nucleus (p=0.8)",SamplingParams(top_p=0.8, max_tokens=50)),
    ]

    for name, sp in strategies:
        result = llm.generate(args.prompt, sp)
        print(f"\n[{name}]")
        print(f"  {result.text[:200]}…")


if __name__ == "__main__":
    main()
