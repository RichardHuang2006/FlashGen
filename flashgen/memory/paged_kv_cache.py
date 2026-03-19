"""
Paged KV cache — vLLM-style block-table indirection.

Why paged allocation?
─────────────────────
Traditional KV caches allocate a contiguous memory region per sequence
(max_seq_len × n_heads × head_dim × 2 × elem_size). This wastes GPU memory
because most sequences end before max_seq_len, and the full allocation is
reserved upfront (internal fragmentation).

vLLM's insight: treat KV cache like virtual memory.
  - Divide cache into fixed-size "pages" (blocks) of block_size tokens.
  - Each sequence maintains a block table: logical_block_idx → physical_block_id.
  - Allocate one block at a time as the sequence grows.
  - Blocks are only freed when the sequence finishes.
  - Prefix sharing: multiple sequences can share prefix blocks (CoW semantics).

Memory efficiency gain:
  - Near-zero internal fragmentation (waste ≤ block_size tokens per sequence).
  - Up to 55% more sequences fit in the same GPU memory vs monolithic allocation.

Attention computation with block tables:
  Instead of reading KV from a contiguous buffer, the attention kernel
  follows the block table indirection:
    for logical_block in range(num_blocks):
      physical_block = block_table[seq_id][logical_block]
      K = allocator.get_key(physical_block, layer)
      V = allocator.get_value(physical_block, layer)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from flashgen.memory.block_allocator import BlockAllocator


@dataclass
class SequenceKVState:
    """KV cache state for one sequence."""
    seq_id: int
    block_table: List[int] = field(default_factory=list)  # physical block IDs
    num_cached_tokens: int = 0  # tokens whose KV is in cache


class PagedKVCache:
    """Manages paged KV cache for a batch of concurrent sequences.

    The block_table maps [seq_id][logical_block] → physical_block_id.
    The attention kernel uses this indirection to locate K,V tensors.

    Example
    -------
    >>> cache = PagedKVCache(allocator, block_size=16)
    >>> cache.allocate("seq_0", prompt_len=64)    # reserves 4 blocks
    >>> cache.extend("seq_0")                      # adds 1 token's worth
    >>> block_table = cache.get_block_table("seq_0")
    >>> cache.free("seq_0")
    """

    def __init__(self, allocator: BlockAllocator, block_size: int):
        self.allocator = allocator
        self.block_size = block_size
        self._sequences: Dict[str, SequenceKVState] = {}

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def allocate(self, seq_id: str, num_tokens: int) -> None:
        """Allocate blocks for `num_tokens` of K/V storage (prefill)."""
        if seq_id in self._sequences:
            raise ValueError(f"Sequence '{seq_id}' already exists in KV cache.")

        num_blocks = self._blocks_needed(num_tokens)
        blocks = [self.allocator.allocate() for _ in range(num_blocks)]

        self._sequences[seq_id] = SequenceKVState(
            seq_id=seq_id,
            block_table=blocks,
            num_cached_tokens=num_tokens,
        )

    def extend(self, seq_id: str) -> bool:
        """Extend by one token. Allocate a new block if current block is full.

        Returns True if a new block was allocated, False if an existing block
        had a free slot.
        """
        state = self._sequences[seq_id]
        current_block = state.num_cached_tokens // self.block_size
        current_slot = state.num_cached_tokens % self.block_size

        new_block_allocated = False
        if current_slot == 0:
            # Need a new block
            new_block = self.allocator.allocate()
            state.block_table.append(new_block)
            new_block_allocated = True

        state.num_cached_tokens += 1
        return new_block_allocated

    def free(self, seq_id: str) -> None:
        """Release all blocks held by a sequence."""
        state = self._sequences.pop(seq_id, None)
        if state is None:
            return
        for block_id in state.block_table:
            self.allocator.free(block_id)

    def fork(self, src_seq_id: str, new_seq_id: str) -> None:
        """Fork a sequence (for beam search). Shares blocks via refcounting."""
        src = self._sequences[src_seq_id]
        # Increment refcount on all shared blocks
        for block_id in src.block_table:
            self.allocator.incref(block_id)
        self._sequences[new_seq_id] = SequenceKVState(
            seq_id=new_seq_id,
            block_table=list(src.block_table),
            num_cached_tokens=src.num_cached_tokens,
        )

    # ── Block table access ───────────────────────────────────────────────────

    def get_block_table(self, seq_id: str) -> List[int]:
        """Return the physical block IDs for a sequence."""
        return self._sequences[seq_id].block_table

    def get_block_table_tensor(self, seq_id: str, device: str = "cuda") -> torch.Tensor:
        """Return block table as a GPU tensor for use in attention kernels."""
        return torch.tensor(
            self._sequences[seq_id].block_table,
            dtype=torch.int32,
            device=device,
        )

    def get_num_cached_tokens(self, seq_id: str) -> int:
        return self._sequences[seq_id].num_cached_tokens

    def get_current_slot(self, seq_id: str) -> Tuple[int, int]:
        """Return (block_idx, slot_idx) for the most recent token."""
        state = self._sequences[seq_id]
        n = state.num_cached_tokens - 1
        return n // self.block_size, n % self.block_size

    def write_kv(
        self,
        seq_id: str,
        layer: int,
        token_pos: int,
        key: torch.Tensor,    # [n_kv_heads, head_dim]
        value: torch.Tensor,
    ) -> None:
        """Write K,V for a specific token position to the appropriate block."""
        state = self._sequences[seq_id]
        block_idx = token_pos // self.block_size
        slot_idx = token_pos % self.block_size
        block_id = state.block_table[block_idx]
        self.allocator.write_kv(block_id, layer, slot_idx, key, value)

    def gather_kv(
        self,
        seq_id: str,
        layer: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather all cached K,V tensors for a sequence (for attention).

        Returns:
            K: [n_kv_heads, num_cached_tokens, head_dim]
            V: [n_kv_heads, num_cached_tokens, head_dim]
        """
        state = self._sequences[seq_id]
        n = state.num_cached_tokens
        H = self.allocator.n_kv_heads
        D = self.allocator.head_dim
        bs = self.block_size

        K_out = torch.empty(H, n, D, dtype=self.allocator.dtype, device="cuda")
        V_out = torch.empty(H, n, D, dtype=self.allocator.dtype, device="cuda")

        for logical_block, block_id in enumerate(state.block_table):
            start = logical_block * bs
            end = min(start + bs, n)
            count = end - start
            K_out[:, start:end, :] = self.allocator.get_key(block_id, layer)[:, :count, :]
            V_out[:, start:end, :] = self.allocator.get_value(block_id, layer)[:, :count, :]

        return K_out, V_out

    # ── Batch helpers ─────────────────────────────────────────────────────────

    def get_batch_block_tables(
        self, seq_ids: List[str], max_blocks: int, device: str = "cuda"
    ) -> torch.Tensor:
        """Build a padded block table tensor for a batch.

        Returns: [num_seqs, max_blocks] int32 tensor (padded with -1).
        """
        tables = []
        for seq_id in seq_ids:
            table = self.get_block_table(seq_id)
            padded = table + [-1] * (max_blocks - len(table))
            tables.append(padded[:max_blocks])
        return torch.tensor(tables, dtype=torch.int32, device=device)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _blocks_needed(self, num_tokens: int) -> int:
        return (num_tokens + self.block_size - 1) // self.block_size

    def num_free_blocks(self) -> int:
        return self.allocator.num_free

    def __contains__(self, seq_id: str) -> bool:
        return seq_id in self._sequences
