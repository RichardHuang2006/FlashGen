"""
TensorRT engine builder for FlashGen.

Pipeline:
  ONNX file
    → TRT INetworkDefinition (via ONNX parser)
    → IBuilderConfig (FP16 or INT8 with calibration)
    → Optimization profiles (dynamic shapes for batch × seq_len × past_len)
    → Serialized engine bytes (.trt file)

Key GPU-optimization levers exposed here:
  FP16:  Halves activation/weight memory bandwidth; uses Tensor Cores.
         ~2× throughput over FP32 for memory-bandwidth-bound ops (K,V reads).
  INT8:  Halves again (vs FP16); requires per-tensor scale calibration.
         ~1.5–2× over FP16 for compute-bound matmuls at <1% accuracy loss.

Dynamic shape profiles
──────────────────────
TRT compiles a kernel for each optimization profile. We add three:
  min  = (1 batch, 1 seq, 0 past)     — single-token decode, cold start
  opt  = (16 batch, 16 seq, 256 past) — typical continuous batching case
  max  = (max_batch, max_seq, max_ctx) — upper bound for memory allocation

TRT picks the best kernel from the profile whose ranges contain the
runtime shape. Running outside all profiles raises a TRT error.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class TRTEngineBuilder:
    """Build a TensorRT engine from an ONNX graph.

    Example
    -------
    >>> builder = TRTEngineBuilder(precision="fp16", max_batch=32, max_seq=2048)
    >>> engine_bytes = builder.build("model.onnx")
    >>> with open("model.trt", "wb") as f:
    ...     f.write(engine_bytes)
    """

    def __init__(
        self,
        precision: str = "fp16",        # "fp32" | "fp16" | "int8"
        max_batch: int = 32,
        max_seq: int = 2048,
        max_past: int = 4096,
        workspace_gb: float = 8.0,
        calibration_data: Optional[List[List[int]]] = None,
        calibration_cache: Optional[str] = None,
        n_layers: int = 12,
        n_kv_heads: int = 12,
        head_dim: int = 64,
    ):
        self.precision = precision
        self.max_batch = max_batch
        self.max_seq = max_seq
        self.max_past = max_past
        self.workspace_gb = workspace_gb
        self.calibration_data = calibration_data
        self.calibration_cache = calibration_cache
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim

    def build(self, onnx_path: str, output_path: Optional[str] = None) -> bytes:
        """Parse ONNX and build a serialized TRT engine.

        Returns the serialized engine as bytes.
        If output_path is provided, also writes it to disk.
        """
        try:
            import tensorrt as trt
        except ImportError:
            raise ImportError(
                "TensorRT not installed. Install via: "
                "pip install tensorrt --index-url https://pypi.ngc.nvidia.com\n"
                "Or use the CUDA backend (engine.backend='cuda') instead."
            )

        logger.info(f"Building TRT engine from {onnx_path} (precision={self.precision})")
        t0 = time.perf_counter()

        trt_logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(trt_logger)
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        parser = trt.OnnxParser(network, trt_logger)

        with open(onnx_path, "rb") as f:
            onnx_bytes = f.read()
        if not parser.parse(onnx_bytes):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError(f"ONNX parsing failed:\n" + "\n".join(errors))
        logger.info(f"ONNX parsed: {network.num_layers} TRT layers")

        config = builder.create_builder_config()
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            int(self.workspace_gb * 1024**3)
        )

        # Precision flags
        if self.precision in ("fp16", "int8"):
            config.set_flag(trt.BuilderFlag.FP16)
        if self.precision == "int8":
            config.set_flag(trt.BuilderFlag.INT8)
            config.int8_calibrator = self._make_calibrator()

        # Optimization profiles for dynamic shapes
        for profile_spec in self._get_profiles(trt, network):
            profile = builder.create_optimization_profile()
            for tensor_name, (mn, opt, mx) in profile_spec.items():
                profile.set_shape(tensor_name, mn, opt, mx)
            config.add_optimization_profile(profile)

        logger.info("Building TRT engine (this may take several minutes)…")
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TRT engine build failed — check GPU memory and ONNX graph.")

        engine_bytes = bytes(serialized)
        elapsed = time.perf_counter() - t0
        size_mb = len(engine_bytes) / 1024**2
        logger.info(f"Engine built: {size_mb:.1f} MB in {elapsed:.1f}s")

        if output_path:
            Path(output_path).write_bytes(engine_bytes)
            logger.info(f"Engine saved to {output_path}")

        return engine_bytes

    def _get_profiles(self, trt, network) -> List[dict]:
        """Return list of {tensor_name: (min_shape, opt_shape, max_shape)} dicts.

        We create two profiles:
          Profile 0: prefill (seq_len > 1, past_len = 0)
          Profile 1: decode  (seq_len = 1, past_len > 0)
        This avoids shape-conflict issues with a single profile covering both.
        """
        B_min, B_opt, B_max = 1, min(8, self.max_batch), self.max_batch
        S_min, S_opt, S_max = 1, 128, self.max_seq
        P_min, P_opt, P_max = 0, 256, self.max_past
        H, D = self.n_kv_heads, self.head_dim

        def kv_shape(past_len):
            return (B_min, H, past_len, D), (B_opt, H, P_opt, D), (B_max, H, P_max, D)

        prefill_profile: dict = {
            "input_ids":      ((B_min, 1),      (B_opt, S_opt), (B_max, S_max)),
            "attention_mask": ((B_min, 1),      (B_opt, S_opt), (B_max, S_max)),
            "position_ids":   ((B_min, 1),      (B_opt, S_opt), (B_max, S_max)),
        }
        decode_profile: dict = {
            "input_ids":      ((B_min, 1), (B_opt, 1), (B_max, 1)),
            "attention_mask": ((B_min, 1), (B_opt, P_opt + 1), (B_max, P_max + 1)),
            "position_ids":   ((B_min, 1), (B_opt, 1), (B_max, 1)),
        }
        # Add KV cache tensors to both profiles
        for i in range(self.n_layers):
            for name in (f"past_key_{i}", f"past_value_{i}"):
                prefill_profile[name] = (
                    (B_min, H, P_min, D), (B_opt, H, P_opt, D), (B_max, H, P_max, D)
                )
                decode_profile[name] = (
                    (B_min, H, 1, D), (B_opt, H, P_opt, D), (B_max, H, P_max, D)
                )

        return [prefill_profile, decode_profile]

    def _make_calibrator(self):
        from flashgen.trt_builder.calibrator import Int8EntropyCalibrator
        if self.calibration_data is None:
            raise ValueError(
                "INT8 quantization requires calibration_data (list of token ID lists). "
                "Pass calibration_data=<list> to TRTEngineBuilder."
            )
        return Int8EntropyCalibrator(
            data=self.calibration_data,
            cache_file=self.calibration_cache or "calibration.cache",
            max_batch=self.max_batch,
        )
