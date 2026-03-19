"""
TensorRT engine runner — executes a serialized .trt engine for LLM inference.

The runner owns:
  - The deserialized CUDA engine + execution context
  - GPU buffers for all I/O tensors (preallocated to max profile shapes)
  - A CUDA stream for async execution

KV cache handling:
  TRT treats past_key/past_value as regular input tensors.
  The runner allocates a GPU KV cache buffer and swaps present → past
  after each decode step, allowing autoregressive generation without
  re-allocating memory.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class TRTEngine:
    """Execute a serialized TensorRT engine.

    Typical usage
    -------------
    >>> engine = TRTEngine("model.trt", n_layers=12, n_kv_heads=12, head_dim=64)
    >>> # Prefill
    >>> logits, kv = engine.prefill(input_ids, attention_mask, position_ids)
    >>> # Decode steps
    >>> for _ in range(max_new_tokens):
    ...     token_id = logits[:, -1, :].argmax(-1)
    ...     logits, kv = engine.decode(token_id, kv, attention_mask, position_ids)
    """

    def __init__(
        self,
        engine_path: str,
        n_layers: int = 12,
        n_kv_heads: int = 12,
        head_dim: int = 64,
        device_id: int = 0,
    ):
        try:
            import tensorrt as trt
            import cuda.cudart as cudart
        except ImportError:
            raise ImportError(
                "TensorRT Python bindings not found. "
                "Install tensorrt and cuda-python packages."
            )

        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.device_id = device_id

        torch.cuda.set_device(device_id)
        self._stream = torch.cuda.Stream()

        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)

        engine_bytes = Path(engine_path).read_bytes()
        self._engine = runtime.deserialize_cuda_engine(engine_bytes)
        self._context = self._engine.create_execution_context()

        self._io_names = [
            self._engine.get_tensor_name(i)
            for i in range(self._engine.num_io_tensors)
        ]
        logger.info(f"TRT engine loaded: {len(self._io_names)} I/O tensors")

    def prefill(
        self,
        input_ids: torch.Tensor,        # [B, S]
        attention_mask: torch.Tensor,   # [B, S]
        position_ids: torch.Tensor,     # [B, S]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Run prefill: process prompt tokens, populate initial KV cache.

        Returns:
            logits: [B, S, vocab]
            kv_cache: flat list of [B, H, S, D] tensors (present K and V for each layer)
        """
        B, S = input_ids.shape
        H, D = self.n_kv_heads, self.head_dim

        # Empty past KV (no prior context)
        past_kv = []
        for _ in range(self.n_layers):
            k = torch.zeros(B, H, 0, D, dtype=torch.float16, device="cuda")
            v = torch.zeros_like(k)
            past_kv.extend([k, v])

        return self._run(input_ids, attention_mask, position_ids, past_kv)

    def decode(
        self,
        token_id: torch.Tensor,          # [B] — last generated token
        past_kv: List[torch.Tensor],     # from previous step
        attention_mask: torch.Tensor,    # [B, past+1]
        position_ids: torch.Tensor,      # [B, 1]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Run one decode step.

        Returns:
            logits: [B, 1, vocab]
            kv_cache: updated KV tensors
        """
        input_ids = token_id.unsqueeze(1)  # [B, 1]
        return self._run(input_ids, attention_mask, position_ids, past_kv)

    def _run(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_kv: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        B, S = input_ids.shape

        # Build input dict
        inputs: Dict[str, torch.Tensor] = {
            "input_ids": input_ids.to(torch.int32),
            "attention_mask": attention_mask.to(torch.int32),
            "position_ids": position_ids.to(torch.int32),
        }
        for i in range(self.n_layers):
            inputs[f"past_key_{i}"] = past_kv[2 * i].contiguous()
            inputs[f"past_value_{i}"] = past_kv[2 * i + 1].contiguous()

        # Set input shapes on context
        for name, tensor in inputs.items():
            self._context.set_input_shape(name, tuple(tensor.shape))
            self._context.set_tensor_address(name, tensor.data_ptr())

        # Allocate outputs (logits + present KV)
        past_len = past_kv[0].shape[2] if past_kv else 0
        vocab_size = self._get_output_vocab_size()
        logits = torch.empty(B, S, vocab_size, dtype=torch.float16, device="cuda")
        self._context.set_tensor_address("logits", logits.data_ptr())

        present_kv = []
        for i in range(self.n_layers):
            k = torch.empty(B, self.n_kv_heads, past_len + S, self.head_dim,
                            dtype=torch.float16, device="cuda")
            v = torch.empty_like(k)
            self._context.set_tensor_address(f"present_key_{i}", k.data_ptr())
            self._context.set_tensor_address(f"present_value_{i}", v.data_ptr())
            present_kv.extend([k, v])

        # Execute async on our stream
        with torch.cuda.stream(self._stream):
            self._context.execute_async_v3(self._stream.cuda_stream)
        self._stream.synchronize()

        return logits, present_kv

    def _get_output_vocab_size(self) -> int:
        # Infer from engine binding shape
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            if name == "logits":
                shape = self._engine.get_tensor_shape(name)
                return shape[-1]
        raise RuntimeError("Could not find 'logits' output in TRT engine.")

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        eos_token_id: int = 50256,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Simple autoregressive generation using the TRT engine.

        Returns generated token IDs: [B, max_new_tokens]
        """
        B, prompt_len = input_ids.shape
        device = input_ids.device

        attn_mask = torch.ones(B, prompt_len, dtype=torch.int32, device=device)
        pos_ids = torch.arange(prompt_len, dtype=torch.int32, device=device).unsqueeze(0)

        logits, kv = self.prefill(input_ids, attn_mask, pos_ids)

        generated = []
        next_token = logits[:, -1, :]
        if temperature != 1.0:
            next_token = next_token / temperature
        token = next_token.argmax(-1)  # greedy for simplicity
        generated.append(token)

        for step in range(1, max_new_tokens):
            if (token == eos_token_id).all():
                break
            total_len = prompt_len + step
            attn_mask = torch.ones(B, total_len, dtype=torch.int32, device=device)
            pos_ids = torch.full((B, 1), total_len - 1, dtype=torch.int32, device=device)

            logits, kv = self.decode(token, kv, attn_mask, pos_ids)
            next_logits = logits[:, -1, :]
            if temperature != 1.0:
                next_logits = next_logits / temperature
            token = next_logits.argmax(-1)
            generated.append(token)

        return torch.stack(generated, dim=1)  # [B, T_gen]
