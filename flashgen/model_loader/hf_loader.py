"""
HuggingFace model loader for FlashGen.

Responsibilities:
  1. Download / load a model from HuggingFace Hub or a local directory.
  2. Map HF weight tensors → FlashGen C++ TransformerWeights layout.
  3. Return raw GPU tensor pointers that can be handed to the C++ engine.

Weight name mappings
--------------------
GPT-2:
  transformer.wte.weight          → token_emb      [vocab, d_model]
  transformer.wpe.weight          → pos_emb        [max_seq, d_model]
  transformer.h.{i}.ln_1.weight   → ln1_gamma
  transformer.h.{i}.ln_1.bias     → ln1_beta
  transformer.h.{i}.attn.c_attn.weight  → W_qkv   (fused [3*d, d] conv weight)
  transformer.h.{i}.attn.c_proj.weight  → Wo
  transformer.h.{i}.ln_2.weight   → ln2_gamma
  transformer.h.{i}.ln_2.bias     → ln2_beta
  transformer.h.{i}.mlp.c_fc.weight    → W1
  transformer.h.{i}.mlp.c_proj.weight  → W2
  transformer.ln_f.weight         → ln_f_gamma
  transformer.ln_f.bias           → ln_f_beta
  lm_head.weight                  → lm_head  (tied = wte.weight.T for GPT-2)

LLaMA-2:
  model.embed_tokens.weight            → token_emb
  model.layers.{i}.input_layernorm.weight       → ln1_gamma  (RMSNorm, no bias)
  model.layers.{i}.self_attn.q_proj.weight      → Wq
  model.layers.{i}.self_attn.k_proj.weight      → Wk
  model.layers.{i}.self_attn.v_proj.weight      → Wv
  model.layers.{i}.self_attn.o_proj.weight      → Wo
  model.layers.{i}.post_attention_layernorm.weight → ln2_gamma
  model.layers.{i}.mlp.gate_proj.weight         → W1_gate  (SwiGLU gate stream)
  model.layers.{i}.mlp.up_proj.weight           → W1       (SwiGLU up stream)
  model.layers.{i}.mlp.down_proj.weight         → W2
  model.norm.weight                             → ln_f_gamma
  lm_head.weight                                → lm_head
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from flashgen.core.config import ModelConfig, Precision

logger = logging.getLogger(__name__)


class WeightDict:
    """Thin wrapper around a state dict with convenience helpers."""

    def __init__(self, state: Dict[str, torch.Tensor]):
        self._state = state

    def get(self, name: str, dtype: torch.dtype = torch.float16) -> Optional[torch.Tensor]:
        t = self._state.get(name)
        if t is None:
            return None
        return t.to(dtype=dtype, device="cuda").contiguous()

    def require(self, name: str, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        t = self.get(name, dtype)
        if t is None:
            raise KeyError(f"Required weight '{name}' not found in model state dict.")
        return t

    def keys(self):
        return self._state.keys()


class HFModelLoader:
    """Loads a HuggingFace model and converts weights to FlashGen format.

    Example
    -------
    >>> loader = HFModelLoader("gpt2", precision=Precision.FP16)
    >>> model_config, weights = loader.load()
    # weights is a dict of named GPU tensors ready for the C++ engine
    """

    def __init__(
        self,
        model_id: str,
        precision: Precision = Precision.FP16,
        device: str = "cuda",
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = False,
    ):
        self.model_id = model_id
        self.precision = precision
        self.device = device
        self.cache_dir = cache_dir
        self.trust_remote_code = trust_remote_code
        self._torch_dtype = torch.float16 if precision == Precision.FP16 else torch.float32

    def load(self) -> Tuple[ModelConfig, Dict[str, torch.Tensor]]:
        """Download + convert model. Returns (ModelConfig, weight_tensors).

        weight_tensors is a flat dict mapping logical names to CUDA tensors:
          "token_emb", "pos_emb", "lm_head",
          "ln_f_gamma", "ln_f_beta",
          "layers.{i}.ln1_gamma", "layers.{i}.ln1_beta",
          "layers.{i}.Wq", "layers.{i}.Wk", "layers.{i}.Wv", "layers.{i}.Wo",
          "layers.{i}.bq", "layers.{i}.bk", "layers.{i}.bv", "layers.{i}.bo",
          "layers.{i}.ln2_gamma", "layers.{i}.ln2_beta",
          "layers.{i}.W1", "layers.{i}.W1_gate", "layers.{i}.W2",
          "layers.{i}.b1", "layers.{i}.b2",
        """
        from transformers import AutoConfig, AutoModelForCausalLM

        logger.info(f"Loading HuggingFace model: {self.model_id}")

        hf_cfg = AutoConfig.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
            trust_remote_code=self.trust_remote_code,
        )
        model_config = ModelConfig.from_hf_config(hf_cfg)

        # Load the full model on CPU first to avoid OOM on large models,
        # then convert to target dtype and move to GPU.
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
            trust_remote_code=self.trust_remote_code,
            dtype=self._torch_dtype,
            low_cpu_mem_usage=True,
        )
        model.eval()

        state = WeightDict(dict(model.state_dict()))
        arch = type(hf_cfg).__name__

        if "GPT2" in arch:
            weights = self._convert_gpt2(state, model_config)
        elif "Llama" in arch or "Mistral" in arch:
            weights = self._convert_llama(state, model_config)
        else:
            raise ValueError(f"Unsupported architecture: {arch}")

        # Free CPU model to reclaim memory
        del model
        torch.cuda.empty_cache()

        logger.info(
            f"Loaded {model_config.name}: "
            f"{model_config.n_layers} layers, "
            f"d_model={model_config.d_model}, "
            f"vocab={model_config.vocab_size}"
        )
        return model_config, weights

    # ── GPT-2 conversion ─────────────────────────────────────────────────────

    def _convert_gpt2(
        self, state: WeightDict, cfg: ModelConfig
    ) -> Dict[str, torch.Tensor]:
        dt = self._torch_dtype
        weights: Dict[str, torch.Tensor] = {}

        # Embeddings
        weights["token_emb"] = state.require("transformer.wte.weight", dt)
        weights["pos_emb"] = state.require("transformer.wpe.weight", dt)

        # Final norm + head (tied weights: lm_head = wte.T)
        weights["ln_f_gamma"] = state.require("transformer.ln_f.weight", dt)
        weights["ln_f_beta"] = state.require("transformer.ln_f.bias", dt)
        lm_head = state.get("lm_head.weight", dt)
        if lm_head is None:
            # Tied: copy and transpose token_emb
            lm_head = weights["token_emb"].clone()
        weights["lm_head"] = lm_head

        for i in range(cfg.n_layers):
            p = f"transformer.h.{i}"

            # Layer norms
            weights[f"layers.{i}.ln1_gamma"] = state.require(f"{p}.ln_1.weight", dt)
            weights[f"layers.{i}.ln1_beta"] = state.require(f"{p}.ln_1.bias", dt)
            weights[f"layers.{i}.ln2_gamma"] = state.require(f"{p}.ln_2.weight", dt)
            weights[f"layers.{i}.ln2_beta"] = state.require(f"{p}.ln_2.bias", dt)

            # GPT-2 fuses Q, K, V into one conv weight: [3*d_model, d_model]
            # c_attn.weight is stored as [d_model, 3*d_model] in HF (Conv1D)
            c_attn_w = state.require(f"{p}.attn.c_attn.weight", dt)  # [d, 3d]
            c_attn_b = state.require(f"{p}.attn.c_attn.bias", dt)    # [3d]

            d = cfg.d_model
            # Split Q, K, V (HF stores as weight.T for Conv1D: input → output)
            # c_attn_w shape: [d_model, 3*d_model], need to transpose for linear
            Wqkv = c_attn_w.t().contiguous()   # [3d, d]
            Wq, Wk, Wv = Wqkv.split(d, dim=0)
            bq, bk, bv = c_attn_b.split(d, dim=0)

            weights[f"layers.{i}.Wq"] = Wq
            weights[f"layers.{i}.Wk"] = Wk
            weights[f"layers.{i}.Wv"] = Wv
            weights[f"layers.{i}.bq"] = bq
            weights[f"layers.{i}.bk"] = bk
            weights[f"layers.{i}.bv"] = bv

            # Output projection: Conv1D [d, d] → transpose to [d, d] linear
            weights[f"layers.{i}.Wo"] = state.require(f"{p}.attn.c_proj.weight", dt).t().contiguous()
            weights[f"layers.{i}.bo"] = state.require(f"{p}.attn.c_proj.bias", dt)

            # FFN: Conv1D stored as [d, d_ff] — transpose to get [d_ff, d]
            weights[f"layers.{i}.W1"] = state.require(f"{p}.mlp.c_fc.weight", dt).t().contiguous()
            weights[f"layers.{i}.b1"] = state.require(f"{p}.mlp.c_fc.bias", dt)
            weights[f"layers.{i}.W2"] = state.require(f"{p}.mlp.c_proj.weight", dt).t().contiguous()
            weights[f"layers.{i}.b2"] = state.require(f"{p}.mlp.c_proj.bias", dt)

        return weights

    # ── LLaMA / Mistral conversion ────────────────────────────────────────────

    def _convert_llama(
        self, state: WeightDict, cfg: ModelConfig
    ) -> Dict[str, torch.Tensor]:
        dt = self._torch_dtype
        weights: Dict[str, torch.Tensor] = {}

        weights["token_emb"] = state.require("model.embed_tokens.weight", dt)
        weights["ln_f_gamma"] = state.require("model.norm.weight", dt)
        weights["lm_head"] = state.require("lm_head.weight", dt)

        # LLaMA has no positional embedding table (uses RoPE)
        weights["pos_emb"] = None  # type: ignore[assignment]

        for i in range(cfg.n_layers):
            p = f"model.layers.{i}"

            # RMSNorm (no bias)
            weights[f"layers.{i}.ln1_gamma"] = state.require(f"{p}.input_layernorm.weight", dt)
            weights[f"layers.{i}.ln2_gamma"] = state.require(f"{p}.post_attention_layernorm.weight", dt)

            # Separate Q, K, V projections (no bias in LLaMA)
            weights[f"layers.{i}.Wq"] = state.require(f"{p}.self_attn.q_proj.weight", dt)
            weights[f"layers.{i}.Wk"] = state.require(f"{p}.self_attn.k_proj.weight", dt)
            weights[f"layers.{i}.Wv"] = state.require(f"{p}.self_attn.v_proj.weight", dt)
            weights[f"layers.{i}.Wo"] = state.require(f"{p}.self_attn.o_proj.weight", dt)

            # SwiGLU FFN: gate_proj and up_proj are applied in parallel,
            # then multiplied: output = silu(gate_proj(x)) * up_proj(x)
            # Then projected down by down_proj.
            weights[f"layers.{i}.W1_gate"] = state.require(f"{p}.mlp.gate_proj.weight", dt)
            weights[f"layers.{i}.W1"] = state.require(f"{p}.mlp.up_proj.weight", dt)
            weights[f"layers.{i}.W2"] = state.require(f"{p}.mlp.down_proj.weight", dt)

        return weights

    def get_data_ptr(self, t: Optional[torch.Tensor]) -> int:
        """Return GPU data pointer (int) for passing to C++ via PyBind11."""
        return t.data_ptr() if t is not None else 0
