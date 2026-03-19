"""Output types returned by the LLM engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class FinishReason(str, Enum):
    NONE = "none"
    EOS_TOKEN = "eos_token"
    MAX_TOKENS = "max_tokens"
    STOP_STRING = "stop_string"
    ABORT = "abort"


@dataclass
class CompletionOutput:
    """A single completion from a request (one sequence)."""
    index: int
    text: str
    token_ids: List[int]
    finish_reason: FinishReason = FinishReason.NONE
    cumulative_logprob: Optional[float] = None


@dataclass
class RequestOutput:
    """All outputs for one submitted request."""
    request_id: str
    prompt: str
    prompt_token_ids: List[int]
    outputs: List[CompletionOutput] = field(default_factory=list)
    finished: bool = False

    # Timing
    prompt_tokens: int = 0
    generated_tokens: int = 0
    latency_ms: float = 0.0
    ttft_ms: float = 0.0     # Time-to-first-token
    tpot_ms: float = 0.0     # Time-per-output-token (after first)

    @property
    def text(self) -> str:
        """Convenience: text of the first (and usually only) completion."""
        return self.outputs[0].text if self.outputs else ""

    @property
    def throughput_tps(self) -> float:
        """Total tokens per second (prompt + generated)."""
        total_ms = self.latency_ms
        total_tok = self.prompt_tokens + self.generated_tokens
        return (total_tok / total_ms * 1000) if total_ms > 0 else 0.0
