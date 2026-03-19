"""
PyTorch baseline benchmark for comparison against FlashGen.

This benchmark runs vanilla HuggingFace model inference (no paged cache,
no custom kernels, no continuous batching) to provide a baseline that
demonstrates the speedup from FlashGen's optimizations.

Optimizations measured:
  ┌─────────────────────┬──────────────────────────────────────────────────┐
  │ FlashGen Feature     │ Expected speedup vs PyTorch baseline            │
  ├─────────────────────┼──────────────────────────────────────────────────┤
  │ FP16 inference      │ 1.5–2× throughput (memory bandwidth)            │
  │ FlashAttention-2    │ 2–4× attention kernel (reduces HBM reads)       │
  │ Paged KV cache      │ +55% sequences (memory efficiency)              │
  │ Continuous batching │ 2–4× throughput (eliminates idle GPU time)      │
  │ TensorRT FP16       │ 2–3× vs PyTorch FP16 (kernel fusion, Tensor Core)│
  │ TensorRT INT8       │ 3–4× vs PyTorch FP16 (<1% accuracy loss)       │
  └─────────────────────┴──────────────────────────────────────────────────┘

Usage:
    python -m flashgen.benchmarks.bench_pytorch \\
        --model gpt2 \\
        --input-len 256 \\
        --output-len 128 \\
        --compare-backend pytorch
"""

from __future__ import annotations

import argparse
import time
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def baseline_generate(
    model_id: str,
    prompts: List[str],
    max_new_tokens: int = 128,
    dtype: torch.dtype = torch.float16,
) -> dict:
    """Run vanilla HuggingFace generation as the performance baseline."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, device_map="cuda", low_cpu_mem_usage=True
    ).eval()

    # Tokenize
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].cuda()
    attn_mask = inputs["attention_mask"].cuda()
    prompt_len = input_ids.shape[1]

    # Warmup
    with torch.no_grad():
        _ = model.generate(input_ids[:1], attention_mask=attn_mask[:1],
                            max_new_tokens=4, do_sample=False)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Timed run
    t_start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    torch.cuda.synchronize()
    t_end = time.perf_counter()

    elapsed = t_end - t_start
    n_requests = len(prompts)
    total_output_tokens = (outputs.shape[1] - prompt_len) * n_requests
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3

    return {
        "model": model_id,
        "backend": "pytorch_hf",
        "dtype": str(dtype),
        "num_requests": n_requests,
        "prompt_len": prompt_len,
        "output_len": max_new_tokens,
        "elapsed_s": elapsed,
        "decode_tps": total_output_tokens / elapsed,
        "e2e_tps": (n_requests * (prompt_len + max_new_tokens)) / elapsed,
        "peak_memory_gb": peak_mem,
    }


def compare(
    model_id: str,
    num_requests: int = 20,
    input_len: int = 128,
    output_len: int = 64,
):
    """Run both baseline and FlashGen, print comparison table."""
    import random
    random.seed(0)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab = len(tokenizer)

    # Generate random prompts
    token_lists = [[random.randint(0, vocab - 1) for _ in range(input_len)]
                   for _ in range(num_requests)]
    prompts = [tokenizer.decode(ids) for ids in token_lists]

    # Baseline
    print(f"Running PyTorch HuggingFace baseline ({model_id})…")
    baseline = baseline_generate(model_id, prompts, output_len)

    # FlashGen
    print(f"Running FlashGen (pytorch backend)…")
    from flashgen import LLM, SamplingParams
    llm = LLM(model_id, backend="pytorch")
    params = SamplingParams(max_tokens=output_len, greedy=True)

    torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()
    results = llm.generate_batch(prompts, params)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t_start
    peak = torch.cuda.max_memory_allocated() / 1024**3
    total_out = sum(r.generated_tokens for r in results)

    flashgen = {
        "model": model_id,
        "backend": "flashgen_pytorch",
        "num_requests": num_requests,
        "elapsed_s": elapsed,
        "decode_tps": total_out / elapsed,
        "e2e_tps": (num_requests * (input_len + output_len)) / elapsed,
        "peak_memory_gb": peak,
    }

    # Print comparison
    print(f"\n{'═'*70}")
    print(f"  Comparison: {model_id}  ({num_requests} reqs, in={input_len}, out={output_len})")
    print(f"{'═'*70}")
    print(f"  {'Metric':<25} {'PyTorch HF':>15} {'FlashGen':>15} {'Speedup':>10}")
    print(f"  {'─'*65}")

    metrics = [
        ("Wall time (s)", "elapsed_s", "{:.2f}", False),
        ("Decode throughput (tok/s)", "decode_tps", "{:.0f}", True),
        ("E2E throughput (tok/s)", "e2e_tps", "{:.0f}", True),
        ("Peak GPU memory (GB)", "peak_memory_gb", "{:.2f}", False),
    ]
    for label, key, fmt, higher_is_better in metrics:
        b = baseline[key]
        f = flashgen[key]
        speedup = (f / b) if higher_is_better else (b / f)
        direction = "↑" if speedup > 1 else "↓"
        print(f"  {label:<25} {fmt.format(b):>15} {fmt.format(f):>15} "
              f"{speedup:.2f}× {direction}")

    print(f"{'═'*70}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--num-requests", type=int, default=20)
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=64)
    args = parser.parse_args()
    compare(args.model, args.num_requests, args.input_len, args.output_len)


if __name__ == "__main__":
    main()
