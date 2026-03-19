"""
Central configuration dataclasses for FlashGen.

These Python-side configs mirror the C++ structs in csrc/include/model_config.hpp
and csrc/include/request.hpp, providing a richer interface with validation,
presets, and helper methods.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import List, Optional


# ── Precision ────────────────────────────────────────────────────────────────

class Precision(str, enum.Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"   # weight-only INT8 via TensorRT calibration


# ── Sampling parameters ──────────────────────────────────────────────────────

@dataclass
class SamplingParams:
    """Controls token sampling during autoregressive generation.

    Tradeoffs:
      temperature < 1  → more deterministic (sharper distribution)
      temperature > 1  → more random (flatter distribution)
      top_p (nucleus)  → dynamically filters to the top-P probability mass
      top_k            → hard cap on the candidate vocabulary
      greedy=True      → argmax, fastest but no diversity
    """
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = -1              # -1 = disabled
    repetition_penalty: float = 1.0
    greedy: bool = False
    max_tokens: int = 128
    stop_token_ids: List[int] = field(default_factory=lambda: [50256])  # GPT-2 EOS

    def __post_init__(self):
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if not 0 < self.top_p <= 1:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be > 0, got {self.max_tokens}")


# ── Model configuration ──────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """Architecture hyperparameters for a decoder-only transformer.

    GPU memory tradeoffs:
      d_model × n_layers × 2 (K+V) × n_kv_heads / n_heads × block_size
      determines bytes per KV block. Larger d_model → fewer sequences fit.

    GQA (Grouped Query Attention):
      n_kv_heads < n_heads: multiple query heads share one K/V head.
      E.g. LLaMA-2 uses n_heads=32, n_kv_heads=32 (MHA),
           Mistral uses n_heads=32, n_kv_heads=8 (GQA factor 4).
    """
    name: str = "gpt2"
    d_model: int = 768
    n_heads: int = 12
    n_kv_heads: int = -1        # -1 → same as n_heads (MHA)
    n_layers: int = 12
    d_ff: int = 3072
    vocab_size: int = 50257
    max_seq_len: int = 1024
    layer_norm_eps: float = 1e-5
    rope_theta: float = 10000.0  # RoPE base (LLaMA uses 10000; code Llama 1e6)
    use_rope: bool = False       # False = learned positional embeddings (GPT-2)
    use_rms_norm: bool = False   # False = LayerNorm (GPT-2); True = RMSNorm (LLaMA)
    use_silu_gate: bool = False  # False = GELU FFN; True = SwiGLU (LLaMA)
    tie_embeddings: bool = True  # lm_head shares weights with token embedding

    # Precision handled at engine level, not architecture level
    precision: Precision = Precision.FP16

    # Derived helpers
    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def actual_kv_heads(self) -> int:
        return self.n_kv_heads if self.n_kv_heads > 0 else self.n_heads

    @property
    def gqa_group_size(self) -> int:
        return self.n_heads // self.actual_kv_heads

    @property
    def kv_dim(self) -> int:
        return self.actual_kv_heads * self.head_dim

    # ── Presets ──────────────────────────────────────────────────────────────

    @classmethod
    def gpt2(cls) -> "ModelConfig":
        return cls(name="gpt2", d_model=768, n_heads=12, n_kv_heads=-1,
                   n_layers=12, d_ff=3072, vocab_size=50257, max_seq_len=1024,
                   use_rope=False, use_rms_norm=False, use_silu_gate=False,
                   tie_embeddings=True)

    @classmethod
    def gpt2_medium(cls) -> "ModelConfig":
        return cls(name="gpt2-medium", d_model=1024, n_heads=16, n_kv_heads=-1,
                   n_layers=24, d_ff=4096, vocab_size=50257, max_seq_len=1024,
                   use_rope=False, use_rms_norm=False, use_silu_gate=False,
                   tie_embeddings=True)

    @classmethod
    def gpt2_xl(cls) -> "ModelConfig":
        return cls(name="gpt2-xl", d_model=1600, n_heads=25, n_kv_heads=-1,
                   n_layers=48, d_ff=6400, vocab_size=50257, max_seq_len=1024,
                   use_rope=False, use_rms_norm=False, use_silu_gate=False,
                   tie_embeddings=True)

    @classmethod
    def llama2_7b(cls) -> "ModelConfig":
        return cls(name="llama2-7b", d_model=4096, n_heads=32, n_kv_heads=32,
                   n_layers=32, d_ff=11008, vocab_size=32000, max_seq_len=4096,
                   use_rope=True, use_rms_norm=True, use_silu_gate=True,
                   tie_embeddings=False, layer_norm_eps=1e-5, rope_theta=10000.0)

    @classmethod
    def llama2_13b(cls) -> "ModelConfig":
        return cls(name="llama2-13b", d_model=5120, n_heads=40, n_kv_heads=40,
                   n_layers=40, d_ff=13824, vocab_size=32000, max_seq_len=4096,
                   use_rope=True, use_rms_norm=True, use_silu_gate=True,
                   tie_embeddings=False, layer_norm_eps=1e-5, rope_theta=10000.0)

    @classmethod
    def mistral_7b(cls) -> "ModelConfig":
        """Mistral-7B uses GQA (n_kv_heads=8) for memory efficiency."""
        return cls(name="mistral-7b", d_model=4096, n_heads=32, n_kv_heads=8,
                   n_layers=32, d_ff=14336, vocab_size=32000, max_seq_len=32768,
                   use_rope=True, use_rms_norm=True, use_silu_gate=True,
                   tie_embeddings=False, layer_norm_eps=1e-5, rope_theta=10000.0)

    @classmethod
    def from_hf_config(cls, hf_config) -> "ModelConfig":
        """Build ModelConfig from a HuggingFace AutoConfig object."""
        arch = type(hf_config).__name__

        # GPT-2 family
        if "GPT2" in arch:
            return cls(
                name=hf_config.name_or_path or "gpt2",
                d_model=hf_config.n_embd,
                n_heads=hf_config.n_head,
                n_kv_heads=-1,
                n_layers=hf_config.n_layer,
                d_ff=hf_config.n_inner or 4 * hf_config.n_embd,
                vocab_size=hf_config.vocab_size,
                max_seq_len=hf_config.n_positions,
                use_rope=False,
                use_rms_norm=False,
                use_silu_gate=False,
                tie_embeddings=True,
                layer_norm_eps=hf_config.layer_norm_epsilon,
            )

        # LLaMA / Mistral family
        if "Llama" in arch or "Mistral" in arch:
            return cls(
                name=hf_config._name_or_path or arch,
                d_model=hf_config.hidden_size,
                n_heads=hf_config.num_attention_heads,
                n_kv_heads=getattr(hf_config, "num_key_value_heads",
                                   hf_config.num_attention_heads),
                n_layers=hf_config.num_hidden_layers,
                d_ff=hf_config.intermediate_size,
                vocab_size=hf_config.vocab_size,
                max_seq_len=getattr(hf_config, "max_position_embeddings", 4096),
                use_rope=True,
                use_rms_norm=True,
                use_silu_gate=True,
                tie_embeddings=False,
                layer_norm_eps=hf_config.rms_norm_eps,
                rope_theta=getattr(hf_config, "rope_theta", 10000.0),
            )

        raise ValueError(f"Unsupported HuggingFace architecture: {arch}. "
                         "Supported: GPT-2 family, LLaMA family, Mistral family.")


# ── Cache configuration ───────────────────────────────────────────────────────

@dataclass
class CacheConfig:
    """Controls paged KV cache memory allocation.

    Memory layout per block:
      [block_size tokens × head_dim × 2(K+V) × n_kv_heads × n_layers × elem_bytes]

    Tradeoffs:
      Larger block_size → less fragmentation, but larger minimum allocation.
      Higher gpu_memory_utilization → more sequences, higher OOM risk.
      INT8 KV quantization halves KV memory at ~1% accuracy cost.
    """
    block_size: int = 16
    num_gpu_blocks: int = -1        # -1 → auto-computed from free VRAM
    gpu_memory_utilization: float = 0.90
    kv_dtype: str = "float16"       # "float16", "float32", "int8"
    enable_prefix_caching: bool = True

    def bytes_per_block(self, model: ModelConfig) -> int:
        elem = {"float32": 4, "float16": 2, "int8": 1}[self.kv_dtype]
        per_head = self.block_size * model.head_dim * elem
        kv_pair = per_head * 2          # K + V
        return kv_pair * model.actual_kv_heads * model.n_layers

    def auto_num_blocks(self, model: ModelConfig, free_bytes: int) -> int:
        usable = int(free_bytes * self.gpu_memory_utilization)
        bpb = self.bytes_per_block(model)
        return usable // bpb if bpb > 0 else 0


# ── Scheduler configuration ───────────────────────────────────────────────────

@dataclass
class SchedulerConfig:
    """Controls continuous batching behavior.

    Tradeoffs:
      Larger max_num_seqs → higher throughput at the cost of more GPU memory.
      Larger max_num_tokens → amortizes launch overhead; increases latency tail.
      Smaller max_prefill_tokens → prevents long prompts from starving decodes.
      enable_chunked_prefill → splits prompts across iterations for fairness.
    """
    max_num_seqs: int = 256
    max_num_tokens: int = 8192
    max_prefill_tokens: int = 4096
    enable_chunked_prefill: bool = True
    preemption_mode: str = "recompute"   # "recompute" | "swap"


# ── Engine configuration ──────────────────────────────────────────────────────

@dataclass
class EngineConfig:
    """Top-level engine configuration combining all sub-configs."""
    model: ModelConfig = field(default_factory=ModelConfig.gpt2)
    cache: CacheConfig = field(default_factory=CacheConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    # Paths
    hf_model_id: str = "gpt2"          # HuggingFace model ID or local path
    onnx_path: Optional[str] = None     # Export path for ONNX (None = skip)
    trt_engine_path: Optional[str] = None  # Path to .trt engine file

    # Runtime
    device_id: int = 0
    precision: Precision = Precision.FP16

    # Backend: "cuda" = custom CUDA kernels, "trt" = TensorRT engine
    backend: str = "cuda"
