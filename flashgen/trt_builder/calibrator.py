"""
INT8 calibration for TensorRT.

TensorRT INT8 calibration workflow:
  1. Create a calibrator with a representative dataset (~512 samples).
  2. TRT runs forward passes through the ONNX graph in FP32.
  3. For each activation tensor, it collects statistics and computes
     per-tensor scale factors using Entropy (KL-divergence) minimization.
  4. These scales are cached to disk for subsequent builds.

EntropyCalibrator2 vs MinMax:
  EntropyCalibrator2 minimizes quantization error by finding the threshold
  that reduces the KL-divergence between the FP32 and INT8 distributions.
  This typically outperforms MinMax calibration for language models.
"""

from __future__ import annotations

import logging
import struct
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class Int8EntropyCalibrator:
    """TensorRT IInt8EntropyCalibrator2 implementation.

    The calibrator feeds batches of token sequences through the network.
    TRT records per-tensor activation ranges and computes INT8 scale factors.
    """

    def __init__(
        self,
        data: List[List[int]],     # list of token-id sequences
        cache_file: str = "calibration.cache",
        max_batch: int = 8,
        max_seq: int = 512,
    ):
        try:
            import tensorrt as trt
            self._trt = trt
        except ImportError:
            raise ImportError("TensorRT required for INT8 calibration.")

        self.data = data
        self.cache_file = Path(cache_file)
        self.max_batch = max_batch
        self.max_seq = max_seq

        self._current_idx = 0
        self._device_inputs: Optional[torch.Tensor] = None

    # ── IInt8EntropyCalibrator2 interface ────────────────────────────────────

    def get_batch_size(self) -> int:
        return self.max_batch

    def get_batch(self, names: List[str]):
        """Return next calibration batch as device memory pointers."""
        if self._current_idx >= len(self.data):
            return None  # Signal end of calibration data

        batch = self.data[self._current_idx: self._current_idx + self.max_batch]
        self._current_idx += self.max_batch

        # Pad/truncate to max_seq
        padded = np.zeros((len(batch), self.max_seq), dtype=np.int32)
        for i, seq in enumerate(batch):
            end = min(len(seq), self.max_seq)
            padded[i, :end] = seq[:end]

        self._device_inputs = torch.tensor(padded, dtype=torch.int32, device="cuda")
        return [self._device_inputs.data_ptr()]

    def read_calibration_cache(self) -> Optional[bytes]:
        if self.cache_file.exists():
            logger.info(f"Loading calibration cache: {self.cache_file}")
            return self.cache_file.read_bytes()
        return None

    def write_calibration_cache(self, cache: bytes):
        self.cache_file.write_bytes(cache)
        logger.info(f"Calibration cache written: {self.cache_file} ({len(cache)} bytes)")
