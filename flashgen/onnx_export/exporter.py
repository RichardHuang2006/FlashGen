"""
ONNX export for transformer models with explicit KV cache I/O.

Why export with KV cache as I/O?
─────────────────────────────────
A naive ONNX export would require feeding the full token sequence on every
decode step (O(n²) FLOPs). By making past_key_values explicit graph I/O,
TensorRT can fuse ops with the always-appending KV cache and only compute
attention over the single new token per decode step.

Graph interface after export:
  Inputs:
    input_ids          [batch, seq_len]
    attention_mask     [batch, total_len]       (optional)
    position_ids       [batch, seq_len]
    past_key_{i}       [batch, n_kv_heads, past_len, head_dim]  for i in layers
    past_value_{i}     [batch, n_kv_heads, past_len, head_dim]

  Outputs:
    logits             [batch, seq_len, vocab_size]
    present_key_{i}    [batch, n_kv_heads, total_len, head_dim]
    present_value_{i}  [batch, n_kv_heads, total_len, head_dim]

Dynamic axes:
  batch        → dim 0 of all tensors
  seq_len      → dim 1 of input_ids, position_ids
  past_len     → dim 2 of past_key/past_value
  total_len    → dim 2 of present_key/present_value, dim 1 of attention_mask
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from flashgen.core.config import ModelConfig

logger = logging.getLogger(__name__)


class _GPT2WithKVCache(nn.Module):
    """Thin wrapper around a HF GPT-2 model exposing KV cache as explicit I/O.

    HuggingFace GPT-2's forward() already supports past_key_values;
    we just need to reshape the output to match our naming convention.
    """

    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model
        self.n_layers = hf_model.config.n_layer

    def forward(
        self,
        input_ids: torch.Tensor,           # [B, S]
        attention_mask: torch.Tensor,       # [B, past+S]
        position_ids: torch.Tensor,         # [B, S]
        *past_kv_flat: torch.Tensor,        # 2*n_layers tensors [B, H, past, D]
    ):
        # Reconstruct tuple-of-tuples past_key_values
        past_kv = None
        if len(past_kv_flat) > 0:
            past_kv = tuple(
                (past_kv_flat[2 * i], past_kv_flat[2 * i + 1])
                for i in range(self.n_layers)
            )

        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_kv,
            use_cache=True,
            return_dict=True,
        )

        # Flatten present_key_values for ONNX (no nested tuples)
        presents_flat = []
        for k, v in out.past_key_values:
            presents_flat.extend([k, v])

        return (out.logits, *presents_flat)


class _LlamaWithKVCache(nn.Module):
    """LLaMA KV cache wrapper (same approach, different HF API shape)."""

    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model
        self.n_layers = hf_model.config.num_hidden_layers

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        *past_kv_flat: torch.Tensor,
    ):
        past_kv = None
        if len(past_kv_flat) > 0:
            past_kv = tuple(
                (past_kv_flat[2 * i], past_kv_flat[2 * i + 1])
                for i in range(self.n_layers)
            )

        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_kv,
            use_cache=True,
            return_dict=True,
        )

        presents_flat = []
        for k, v in out.past_key_values:
            presents_flat.extend([k, v])

        return (out.logits, *presents_flat)


class ONNXExporter:
    """Export a HuggingFace causal LM to ONNX with explicit KV cache I/O.

    Example
    -------
    >>> exporter = ONNXExporter("gpt2")
    >>> exporter.export("model.onnx")
    """

    def __init__(
        self,
        model_id: str,
        model_config: Optional[ModelConfig] = None,
        device: str = "cuda",
        opset: int = 17,
    ):
        self.model_id = model_id
        self.device = device
        self.opset = opset
        self._model_config = model_config

    def export(self, output_path: str, batch_size: int = 1) -> Path:
        """Run export. Returns the path to the ONNX file."""
        from transformers import AutoConfig, AutoModelForCausalLM

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading model for ONNX export: {self.model_id}")
        hf_cfg = AutoConfig.from_pretrained(self.model_id)
        hf_model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(self.device).eval()

        arch = type(hf_cfg).__name__
        if "GPT2" in arch:
            wrapper = _GPT2WithKVCache(hf_model)
            dummy, names, dynamic = self._gpt2_dummy(hf_cfg, batch_size)
        elif "Llama" in arch or "Mistral" in arch:
            wrapper = _LlamaWithKVCache(hf_model)
            dummy, names, dynamic = self._llama_dummy(hf_cfg, batch_size)
        else:
            raise ValueError(f"Unsupported arch for ONNX export: {arch}")

        logger.info(f"Exporting to ONNX (opset {self.opset}): {out}")
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                dummy,
                str(out),
                input_names=names["inputs"],
                output_names=names["outputs"],
                dynamic_axes=dynamic,
                opset_version=self.opset,
                do_constant_folding=True,
                export_params=True,
            )

        # Verify the exported graph is valid
        import onnx
        model_onnx = onnx.load(str(out))
        onnx.checker.check_model(model_onnx)
        logger.info(f"ONNX export complete and validated: {out}")

        return out

    def _gpt2_dummy(
        self, cfg, batch_size: int
    ) -> Tuple[tuple, Dict[str, List[str]], Dict[str, Dict[int, str]]]:
        B, S = batch_size, 1          # decode step: 1 token
        past_len = 16                 # warm-start with some cached tokens
        n_layers = cfg.n_layer
        n_heads = cfg.n_head
        head_dim = cfg.n_embd // n_heads

        input_ids = torch.zeros(B, S, dtype=torch.long, device=self.device)
        attn_mask = torch.ones(B, past_len + S, dtype=torch.long, device=self.device)
        pos_ids = torch.zeros(B, S, dtype=torch.long, device=self.device)

        past_flat = []
        for _ in range(n_layers):
            k = torch.zeros(B, n_heads, past_len, head_dim,
                            dtype=torch.float16, device=self.device)
            v = torch.zeros_like(k)
            past_flat.extend([k, v])

        dummy = (input_ids, attn_mask, pos_ids, *past_flat)

        input_names = ["input_ids", "attention_mask", "position_ids"]
        for i in range(n_layers):
            input_names += [f"past_key_{i}", f"past_value_{i}"]

        output_names = ["logits"]
        for i in range(n_layers):
            output_names += [f"present_key_{i}", f"present_value_{i}"]

        dynamic: Dict[str, Dict[int, str]] = {
            "input_ids": {0: "batch", 1: "seq_len"},
            "attention_mask": {0: "batch", 1: "total_len"},
            "position_ids": {0: "batch", 1: "seq_len"},
            "logits": {0: "batch", 1: "seq_len"},
        }
        for i in range(n_layers):
            dynamic[f"past_key_{i}"] = {0: "batch", 2: "past_len"}
            dynamic[f"past_value_{i}"] = {0: "batch", 2: "past_len"}
            dynamic[f"present_key_{i}"] = {0: "batch", 2: "total_len"}
            dynamic[f"present_value_{i}"] = {0: "batch", 2: "total_len"}

        return dummy, {"inputs": input_names, "outputs": output_names}, dynamic

    def _llama_dummy(
        self, cfg, batch_size: int
    ) -> Tuple[tuple, Dict[str, List[str]], Dict[str, Dict[int, str]]]:
        B, S = batch_size, 1
        past_len = 16
        n_layers = cfg.num_hidden_layers
        n_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        head_dim = cfg.hidden_size // cfg.num_attention_heads

        input_ids = torch.zeros(B, S, dtype=torch.long, device=self.device)
        attn_mask = torch.ones(B, past_len + S, dtype=torch.long, device=self.device)
        pos_ids = torch.arange(past_len, past_len + S, dtype=torch.long,
                               device=self.device).unsqueeze(0).expand(B, -1)

        past_flat = []
        for _ in range(n_layers):
            k = torch.zeros(B, n_kv_heads, past_len, head_dim,
                            dtype=torch.float16, device=self.device)
            v = torch.zeros_like(k)
            past_flat.extend([k, v])

        dummy = (input_ids, attn_mask, pos_ids, *past_flat)

        input_names = ["input_ids", "attention_mask", "position_ids"]
        for i in range(n_layers):
            input_names += [f"past_key_{i}", f"past_value_{i}"]

        output_names = ["logits"]
        for i in range(n_layers):
            output_names += [f"present_key_{i}", f"present_value_{i}"]

        dynamic: Dict[str, Dict[int, str]] = {
            "input_ids": {0: "batch", 1: "seq_len"},
            "attention_mask": {0: "batch", 1: "total_len"},
            "position_ids": {0: "batch", 1: "seq_len"},
            "logits": {0: "batch", 1: "seq_len"},
        }
        for i in range(n_layers):
            dynamic[f"past_key_{i}"] = {0: "batch", 2: "past_len"}
            dynamic[f"past_value_{i}"] = {0: "batch", 2: "past_len"}
            dynamic[f"present_key_{i}"] = {0: "batch", 2: "total_len"}
            dynamic[f"present_value_{i}"] = {0: "batch", 2: "total_len"}

        return dummy, {"inputs": input_names, "outputs": output_names}, dynamic
