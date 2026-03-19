"""
Token sampling strategies for autoregressive generation.

Sampling tradeoffs
──────────────────
Greedy (argmax):
  Always picks highest-probability token. Deterministic, fastest.
  Prone to repetitive, degenerate outputs.

Temperature scaling:
  logits /= T before softmax.
  T < 1 → sharper → more deterministic.
  T → 0 → approaches greedy.
  T > 1 → flatter → more diverse/random.

Top-K filtering:
  Zero out all logits except the top K.
  Simple cutoff; K is fixed regardless of probability mass distribution.

Top-P (nucleus) filtering:
  Zero out logits outside the minimum set whose cumulative probability ≥ P.
  Adaptive: uses more tokens in flat distributions, fewer in peaked ones.
  Usually preferred over Top-K for open-ended generation.

Repetition penalty:
  Divide logit by penalty factor (>1) for tokens already in the sequence.
  Reduces repetitive loops. Applies before temperature and filtering.

GPU implementation notes:
  - All operations are batched over [B, vocab] logit tensors.
  - Filtering is done by setting excluded logits to -inf before softmax.
  - Multinomial sampling uses torch.multinomial (cuRAND under the hood).
  - Greedy uses argmax (no RNG, deterministic).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from flashgen.core.config import SamplingParams


class Sampler:
    """Token sampler supporting greedy, top-k, top-p, and temperature.

    Example
    -------
    >>> sampler = Sampler()
    >>> # logits: [batch, vocab]
    >>> tokens = sampler.sample(logits, [params_for_seq_0, params_for_seq_1])
    >>> # tokens: [batch] int tensor
    """

    def sample(
        self,
        logits: torch.Tensor,          # [B, vocab_size]
        params: List[SamplingParams],
    ) -> torch.Tensor:
        """Sample one token per sequence in the batch.

        Each sequence may have different sampling parameters.
        Returns: [B] int64 token IDs.
        """
        assert logits.dim() == 2
        B, V = logits.shape
        assert len(params) == B

        results = torch.empty(B, dtype=torch.int64, device=logits.device)

        # Fast path: all greedy
        if all(p.greedy for p in params):
            return logits.argmax(dim=-1)

        # Per-sequence sampling (could be batched more aggressively, but
        # different params make vectorization complex)
        for i, p in enumerate(params):
            logit = logits[i]  # [V]

            if p.greedy:
                results[i] = logit.argmax()
                continue

            results[i] = self._sample_one(logit, p)

        return results

    def sample_batch(
        self,
        logits: torch.Tensor,    # [B, vocab_size]
        params: SamplingParams,  # same params for all sequences
    ) -> torch.Tensor:
        """Optimized batch sampling when all sequences share the same params."""
        if params.greedy:
            return logits.argmax(dim=-1)

        return torch.stack([self._sample_one(logits[i], params)
                            for i in range(logits.shape[0])])

    def _sample_one(self, logit: torch.Tensor, p: SamplingParams) -> torch.Tensor:
        """Sample one token from a single logit vector."""
        # 1. Repetition penalty (applied before temperature for stability)
        # NOTE: actual token history needed for rep penalty; done in engine.py

        # 2. Temperature scaling
        if p.temperature != 1.0 and p.temperature > 0:
            logit = logit / p.temperature

        # 3. Top-K filtering
        if p.top_k > 0:
            logit = self._top_k_filter(logit, p.top_k)

        # 4. Top-P (nucleus) filtering
        if 0.0 < p.top_p < 1.0:
            logit = self._top_p_filter(logit, p.top_p)

        # 5. Softmax + multinomial sample
        probs = F.softmax(logit, dim=-1)
        return torch.multinomial(probs.unsqueeze(0), num_samples=1).squeeze()

    def apply_repetition_penalty(
        self,
        logits: torch.Tensor,          # [B, vocab]
        token_histories: List[List[int]],
        penalty: float,
    ) -> torch.Tensor:
        """Apply repetition penalty in-place across a batch.

        For each sequence, divide (not zero) the logit of previously seen
        tokens by `penalty` (>1 reduces their probability).
        """
        if penalty == 1.0:
            return logits
        for i, history in enumerate(token_histories):
            if not history:
                continue
            ids = torch.tensor(history, dtype=torch.long, device=logits.device)
            unique_ids = ids.unique()
            # Divide positive logits (reduce prob), multiply negative (increase prob)
            logits[i, unique_ids] = torch.where(
                logits[i, unique_ids] > 0,
                logits[i, unique_ids] / penalty,
                logits[i, unique_ids] * penalty,
            )
        return logits

    @staticmethod
    def _top_k_filter(logit: torch.Tensor, k: int) -> torch.Tensor:
        """Zero out all logits except the top-k, returning -inf for excluded."""
        k = min(k, logit.shape[-1])
        threshold = logit.topk(k).values[-1]
        return logit.masked_fill(logit < threshold, float("-inf"))

    @staticmethod
    def _top_p_filter(logit: torch.Tensor, p: float) -> torch.Tensor:
        """Top-P (nucleus) filter: retain tokens whose cumulative prob ≥ p."""
        sorted_logits, sorted_indices = logit.sort(descending=True)
        cumulative_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

        # Remove tokens that push cumulative probability above p
        # (shift by 1 to keep the token that crosses the threshold)
        sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > p
        sorted_logits = sorted_logits.masked_fill(sorted_indices_to_remove, float("-inf"))

        # Scatter back to original ordering
        logit = logit.scatter(0, sorted_indices, sorted_logits)
        return logit
