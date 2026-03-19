"""Tokenizer wrapper that normalizes the HuggingFace tokenizer interface."""

from __future__ import annotations

from typing import List, Optional, Union

import torch


class Tokenizer:
    """Wraps a HuggingFace tokenizer with a simple encode/decode interface.

    Handles the common edge cases:
      - Missing pad token (GPT-2 uses EOS as pad)
      - Batch encoding with padding
      - Return types as plain Python lists (not tensors)
    """

    def __init__(self, model_id: str, cache_dir: Optional[str] = None):
        from transformers import AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            use_fast=True,
        )
        # Many decoder-only models have no pad token — use EOS
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token
            self._tok.pad_token_id = self._tok.eos_token_id

    @property
    def eos_token_id(self) -> int:
        return self._tok.eos_token_id

    @property
    def pad_token_id(self) -> int:
        return self._tok.pad_token_id

    @property
    def vocab_size(self) -> int:
        return len(self._tok)

    def encode(self, text: str) -> List[int]:
        return self._tok.encode(text)

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        return [self._tok.encode(t) for t in texts]

    def decode(self, token_ids: Union[List[int], torch.Tensor]) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self._tok.decode(token_ids, skip_special_tokens=True)

    def decode_batch(self, batch: List[List[int]]) -> List[str]:
        return [self.decode(ids) for ids in batch]

    def decode_token(self, token_id: int) -> str:
        """Decode a single token (may include leading spaces)."""
        return self._tok.decode([token_id])
