"""
Continuous batching scheduler — iteration-level scheduling.

Inspired by:
  - vLLM (Kwon et al., 2023): Efficient Memory Management for LLM Serving
    with PagedAttention (https://arxiv.org/abs/2309.06180)
  - Orca (Yu et al., 2022): A Distributed Serving System for Transformer-Based
    Generative Models (https://www.usenix.org/conference/osdi22/presentation/yu)

Core insight:
  Traditional batching sends all sequences through the same iteration,
  meaning the batch is limited by the slowest (longest) sequence.
  Continuous batching treats the forward pass as the granularity unit:
  finished sequences are immediately replaced with new ones, maximizing GPU
  utilization without waiting for all sequences in a batch to finish.

Three-queue design:
  WAITING   — submitted, not yet scheduled (no KV blocks allocated)
  RUNNING   — in active decode loop (KV blocks allocated)
  PREEMPTED — evicted to free blocks; will be re-prefilled when memory allows

Scheduling algorithm (one iteration):
  1. Try to admit WAITING sequences (FCFS) up to token budget
  2. Continue RUNNING sequences for decode
  3. If free blocks < threshold, preempt lowest-priority RUNNING sequence
  4. Return (prefill_batch, decode_batch) for the model forward pass

Tradeoffs exposed by this scheduler:
  - max_num_tokens:     higher → better GPU utilization, higher tail latency
  - max_prefill_tokens: lower → prefill doesn't starve decode iterations
  - max_num_seqs:       higher → more concurrent users, more KV memory
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Deque, Dict, List, Optional, Tuple

from flashgen.core.config import SamplingParams, SchedulerConfig
from flashgen.memory.paged_kv_cache import PagedKVCache


class RequestStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    PREEMPTED = auto()
    FINISHED = auto()
    ABORTED = auto()


@dataclass
class Request:
    """A single inference request tracked by the scheduler."""
    request_id: str
    prompt_token_ids: List[int]
    sampling: SamplingParams
    priority: float = 0.0
    arrival_time: float = field(default_factory=time.monotonic)

    # Mutable state
    output_token_ids: List[int] = field(default_factory=list)
    status: RequestStatus = RequestStatus.WAITING
    num_computed_tokens: int = 0  # tokens whose KV is already in cache

    # Streaming callback (called with each new token)
    stream_callback: Optional[Callable[[int, bool], None]] = None

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def output_len(self) -> int:
        return len(self.output_token_ids)

    @property
    def total_len(self) -> int:
        return self.prompt_len + self.output_len

    @property
    def remaining_prefill(self) -> int:
        """Tokens not yet computed (for chunked prefill)."""
        return self.prompt_len - self.num_computed_tokens

    def is_prefill(self) -> bool:
        return self.num_computed_tokens < self.prompt_len

    def blocks_needed(self, block_size: int) -> int:
        return (self.total_len + block_size - 1) // block_size

    def is_done(self) -> bool:
        return self.status in (RequestStatus.FINISHED, RequestStatus.ABORTED)


@dataclass
class SchedulerOutput:
    """Batch produced by one scheduling iteration."""
    prefill_requests: List[Request] = field(default_factory=list)
    decode_requests: List[Request] = field(default_factory=list)
    preempted: List[Request] = field(default_factory=list)
    finished: List[Request] = field(default_factory=list)

    # Tokens scheduled per request in this iteration (set by scheduler, not derived)
    _prefill_token_counts: dict = field(default_factory=dict)  # request_id → token count

    @property
    def num_prefill_tokens(self) -> int:
        return sum(self._prefill_token_counts.values())

    @property
    def num_decode_tokens(self) -> int:
        return len(self.decode_requests)

    @property
    def total_tokens(self) -> int:
        return self.num_prefill_tokens + self.num_decode_tokens

    @property
    def is_empty(self) -> bool:
        return not self.prefill_requests and not self.decode_requests


class ContinuousBatcher:
    """Iteration-level continuous batching scheduler.

    Example
    -------
    >>> batcher = ContinuousBatcher(config, kv_cache)
    >>> batcher.add_request(Request(request_id="0", prompt_token_ids=[...], sampling=params))
    >>> while batcher.has_work():
    ...     batch = batcher.schedule()
    ...     logits = model.forward(batch)
    ...     new_tokens = sampler.sample(logits, batch)
    ...     finished = batcher.update(batch, new_tokens)
    """

    def __init__(self, config: SchedulerConfig, kv_cache: PagedKVCache):
        self.config = config
        self.kv_cache = kv_cache

        self._waiting: Deque[Request] = deque()
        self._running: Deque[Request] = deque()
        self._preempted: Deque[Request] = deque()

    # ── Request management ────────────────────────────────────────────────────

    def add_request(self, request: Request) -> None:
        """Add a new request to the waiting queue (thread-safe for single producer)."""
        if request.request_id is None:
            request.request_id = str(uuid.uuid4())
        self._waiting.append(request)

    def abort_request(self, request_id: str) -> bool:
        """Cancel a request by ID. Returns True if found and aborted."""
        for queue in (self._waiting, self._running, self._preempted):
            for req in queue:
                if req.request_id == request_id:
                    req.status = RequestStatus.ABORTED
                    if request_id in self.kv_cache:
                        self.kv_cache.free(request_id)
                    queue.remove(req)
                    return True
        return False

    # ── Core scheduling ───────────────────────────────────────────────────────

    def schedule(self) -> SchedulerOutput:
        """Run one scheduling iteration. Returns batch for the model forward pass.

        Decision order:
          1. Admit WAITING requests (FCFS) up to token/seq budget
          2. Continue RUNNING decode sequences
          3. Preempt if free blocks are insufficient for decode step
        """
        output = SchedulerOutput()
        token_budget = self.config.max_num_tokens
        block_size = self.kv_cache.block_size

        # ── Step 1: Admit waiting requests for prefill ──────────────────────
        prefill_tokens_used = 0
        while self._waiting and token_budget > 0:
            req = self._waiting[0]

            # Check sequence limit
            total_running = len(self._running) + len(output.prefill_requests)
            if total_running >= self.config.max_num_seqs:
                break

            # Check prefill token budget
            tokens_to_prefill = min(req.remaining_prefill, self.config.max_prefill_tokens)
            if not self.config.enable_chunked_prefill:
                tokens_to_prefill = req.remaining_prefill
            if prefill_tokens_used + tokens_to_prefill > token_budget:
                break

            # Check memory: can we allocate blocks for this request?
            needed = req.blocks_needed(block_size)
            if self.kv_cache.num_free_blocks() < needed:
                # Try to preempt a running sequence
                if not self._try_preempt(output):
                    break
                if self.kv_cache.num_free_blocks() < needed:
                    break

            # Allocate KV blocks
            self._waiting.popleft()
            if req.request_id not in self.kv_cache:
                self.kv_cache.allocate(req.request_id, tokens_to_prefill)
            req.num_computed_tokens = tokens_to_prefill
            req.status = RequestStatus.RUNNING

            output.prefill_requests.append(req)
            output._prefill_token_counts[req.request_id] = tokens_to_prefill
            prefill_tokens_used += tokens_to_prefill
            token_budget -= tokens_to_prefill

        # ── Step 2: Schedule running decode sequences ───────────────────────
        decode_candidates = list(self._running)
        for req in decode_candidates:
            if token_budget <= 0:
                break
            if len(output.decode_requests) + len(output.prefill_requests) >= self.config.max_num_seqs:
                break

            # Ensure one free block available for the new decode token
            if self.kv_cache.num_free_blocks() < 1:
                if not self._try_preempt(output):
                    break

            output.decode_requests.append(req)
            token_budget -= 1

        return output

    def update(
        self,
        batch: SchedulerOutput,
        new_tokens: Dict[str, int],    # request_id → new token
    ) -> List[Request]:
        """Update state after a forward pass. Returns list of finished requests."""
        block_size = self.kv_cache.block_size
        finished = []

        all_requests = batch.prefill_requests + batch.decode_requests
        for req in all_requests:
            token = new_tokens.get(req.request_id)
            if token is None:
                continue

            # Extend KV cache for the new decode token
            if req.status == RequestStatus.RUNNING and not req.is_prefill():
                self.kv_cache.extend(req.request_id)

            req.output_token_ids.append(token)

            # Invoke streaming callback
            is_final = self._is_done(req)
            if req.stream_callback is not None:
                req.stream_callback(token, is_final)

            if is_final:
                req.status = RequestStatus.FINISHED
                self.kv_cache.free(req.request_id)
                if req in self._running:
                    self._running.remove(req)
                finished.append(req)
            elif req not in self._running:
                self._running.append(req)

        return finished

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _is_done(self, req: Request) -> bool:
        if req.output_len >= req.sampling.max_tokens:
            return True
        if req.output_token_ids and req.output_token_ids[-1] in req.sampling.stop_token_ids:
            return True
        return False

    def _try_preempt(self, output: SchedulerOutput) -> bool:
        """Preempt the lowest-priority running sequence. Returns True on success."""
        if not self._running:
            return False

        # Preempt the most recently added (LIFO) running sequence
        victim = self._running.pop()
        victim.status = RequestStatus.PREEMPTED
        self.kv_cache.free(victim.request_id)
        victim.num_computed_tokens = 0  # will re-prefill from scratch
        victim.output_token_ids.clear()

        output.preempted.append(victim)
        self._preempted.appendleft(victim)
        return True

    # ── Status queries ────────────────────────────────────────────────────────

    def has_work(self) -> bool:
        return bool(self._waiting or self._running or self._preempted)

    @property
    def num_waiting(self) -> int:
        return len(self._waiting)

    @property
    def num_running(self) -> int:
        return len(self._running)

    @property
    def num_preempted(self) -> int:
        return len(self._preempted)
