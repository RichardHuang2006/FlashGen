#!/usr/bin/env python3
"""
CLI tool to export a HuggingFace model to ONNX and build a TensorRT engine.

Pipeline:
  1. Load model from HuggingFace Hub
  2. Export to ONNX with KV cache as explicit I/O
  3. Build TensorRT engine (FP16 or INT8)
  4. Validate: run one forward pass through the TRT engine

Usage:
    # FP16 engine from GPT-2
    python scripts/build_engine.py --model gpt2 --precision fp16 --output-dir engines/

    # INT8 engine from LLaMA-2-7B (requires calibration data)
    python scripts/build_engine.py \\
        --model meta-llama/Llama-2-7b-hf \\
        --precision int8 \\
        --output-dir engines/ \\
        --max-batch 32 \\
        --max-seq 4096
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Build TensorRT engine from HuggingFace model")
    parser.add_argument("--model", required=True,
                        help="HuggingFace model ID or local path")
    parser.add_argument("--precision", default="fp16", choices=["fp16", "int8"],
                        help="Engine precision (default: fp16)")
    parser.add_argument("--output-dir", default="engines",
                        help="Directory to save ONNX and TRT files")
    parser.add_argument("--max-batch", type=int, default=32,
                        help="Maximum batch size for TRT optimization profiles")
    parser.add_argument("--max-seq", type=int, default=2048,
                        help="Maximum sequence length for TRT optimization profiles")
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset version (default: 17)")
    parser.add_argument("--skip-onnx", action="store_true",
                        help="Skip ONNX export (use existing file)")
    parser.add_argument("--validate", action="store_true",
                        help="Validate TRT engine after building")
    parser.add_argument("--workspace-gb", type=float, default=8.0,
                        help="TRT builder workspace in GB")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_slug = args.model.replace("/", "_").replace("-", "_")
    onnx_path = output_dir / f"{model_slug}.onnx"
    trt_path = output_dir / f"{model_slug}_{args.precision}.trt"

    # ── Step 1: ONNX export ──────────────────────────────────────────────────
    if not args.skip_onnx:
        print(f"\n[1/3] Exporting {args.model} → {onnx_path}")
        from flashgen.onnx_export.exporter import ONNXExporter
        exporter = ONNXExporter(args.model, opset=args.opset)
        exporter.export(str(onnx_path))
    else:
        print(f"\n[1/3] Skipping ONNX export (using {onnx_path})")
        if not onnx_path.exists():
            print(f"ERROR: {onnx_path} does not exist.")
            sys.exit(1)

    # ── Step 2: TRT engine build ─────────────────────────────────────────────
    print(f"\n[2/3] Building TRT engine ({args.precision}) → {trt_path}")

    from transformers import AutoConfig
    hf_cfg = AutoConfig.from_pretrained(args.model)

    # Extract architecture params for TRT profile shapes
    arch = type(hf_cfg).__name__
    if "GPT2" in arch:
        n_layers = hf_cfg.n_layer
        n_kv_heads = hf_cfg.n_head
        head_dim = hf_cfg.n_embd // hf_cfg.n_head
    else:
        n_layers = hf_cfg.num_hidden_layers
        n_kv_heads = getattr(hf_cfg, "num_key_value_heads", hf_cfg.num_attention_heads)
        head_dim = hf_cfg.hidden_size // hf_cfg.num_attention_heads

    calibration_data = None
    if args.precision == "int8":
        print("  Generating calibration data (512 random sequences)…")
        import random
        calibration_data = [
            [random.randint(0, 50256) for _ in range(128)]
            for _ in range(512)
        ]

    from flashgen.trt_builder.builder import TRTEngineBuilder
    builder = TRTEngineBuilder(
        precision=args.precision,
        max_batch=args.max_batch,
        max_seq=args.max_seq,
        workspace_gb=args.workspace_gb,
        calibration_data=calibration_data,
        n_layers=n_layers,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
    )
    builder.build(str(onnx_path), str(trt_path))

    # ── Step 3: Validation ───────────────────────────────────────────────────
    if args.validate:
        print(f"\n[3/3] Validating engine…")
        import torch
        from flashgen.trt_builder.engine_runner import TRTEngine

        engine = TRTEngine(
            str(trt_path),
            n_layers=n_layers,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
        )
        test_ids = torch.zeros(1, 8, dtype=torch.long, device="cuda")
        logits, kv = engine.prefill(
            test_ids,
            torch.ones(1, 8, dtype=torch.int32, device="cuda"),
            torch.arange(8, dtype=torch.int32, device="cuda").unsqueeze(0),
        )
        print(f"  Prefill OK: logits shape = {logits.shape}")
        print(f"  KV cache: {len(kv)} tensors, shape = {kv[0].shape}")
        print("  Validation passed.")
    else:
        print("\n[3/3] Skipping validation (use --validate to enable)")

    print(f"\n✓ Engine ready: {trt_path}")
    print(f"  Size: {trt_path.stat().st_size / 1024**2:.1f} MB")


if __name__ == "__main__":
    main()
