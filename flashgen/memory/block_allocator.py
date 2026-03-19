"""
Python-side paged KV cache block allocator.

This mirrors the C++ BlockAllocator in csrc/runtime/block_allocator.cpp
for use when the pure-Python engine path (TRT backend) is active.
When the C++ CUDA backend is used, the C++ BlockAllocator is called
directly through the flashgen._C PyBind11 extension.

Design: free-list of fixed-size GPU memory blocks.
  - O(1) allocate and free
  - Atomic reference counting for prefix sharing (copy-on-write)
  - All blocks are pre-allocated at construction time to avoid runtime cudaMalloc

Block memory layout (per block):
  [layer_0_K | layer_0_V | layer_1_K | layer_1_V | ... ]
  each K/V slab: [block_size × head_dim × elem_bytes]
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional

import torch


class Block:
    """Metadata for one physical KV cache block."""

    __slots__ = ("block_id", "ref_count")

    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_count = 0


class BlockAllocator:
    """Manages a pre-allocated pool of GPU KV cache blocks.

    Parameters
    ----------
    num_blocks    : Total blocks in the pool. Each block stores KV for
                    `block_size` tokens across all layers and KV heads.
    block_size    : Number of tokens per block (16 is typical).
    n_layers      : Number of transformer layers.
    n_kv_heads    : Number of KV attention heads (after GQA reduction).
    head_dim      : Size of each attention head.
    dtype         : Storage dtype (float16 recommended for memory efficiency).
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype

        # Pre-allocate all blocks in one contiguous GPU tensor.
        # Shape: [num_blocks, n_layers, 2, n_kv_heads, block_size, head_dim]
        # dim 2: 0 = K, 1 = V
        self._storage = torch.zeros(
            num_blocks, n_layers, 2, n_kv_heads, block_size, head_dim,
            dtype=dtype, device="cuda"
        )

        # Block metadata
        self._blocks: List[Block] = [Block(i) for i in range(num_blocks)]

        # Free list (stack for O(1) ops)
        self._free: List[int] = list(range(num_blocks))
        self._lock = threading.Lock()

    # ── Allocation ────────────────────────────────────────────────────────────

    def allocate(self) -> int:
        """Return block_id of a free block. Raises RuntimeError if OOM."""
        with self._lock:
            if not self._free:
                raise RuntimeError(
                    f"BlockAllocator OOM: all {self.num_blocks} blocks in use. "
                    "Increase num_gpu_blocks or lower concurrent requests."
                )
            block_id = self._free.pop()
            self._blocks[block_id].ref_count = 1
            return block_id

    def free(self, block_id: int) -> None:
        """Decrement ref_count; return block to free list when count hits 0."""
        with self._lock:
            blk = self._blocks[block_id]
            blk.ref_count -= 1
            if blk.ref_count <= 0:
                blk.ref_count = 0
                self._free.append(block_id)

    def incref(self, block_id: int) -> None:
        """Increment reference count (for prefix sharing)."""
        with self._lock:
            self._blocks[block_id].ref_count += 1

    # ── Storage access ────────────────────────────────────────────────────────

    def get_key(self, block_id: int, layer: int) -> torch.Tensor:
        """Return K storage slice: [n_kv_heads, block_size, head_dim]"""
        return self._storage[block_id, layer, 0]

    def get_value(self, block_id: int, layer: int) -> torch.Tensor:
        """Return V storage slice: [n_kv_heads, block_size, head_dim]"""
        return self._storage[block_id, layer, 1]

    def write_kv(
        self,
        block_id: int,
        layer: int,
        slot: int,                        # position within block [0, block_size)
        key: torch.Tensor,                # [n_kv_heads, head_dim]
        value: torch.Tensor,
    ) -> None:
        """Write one token's K and V into the specified slot."""
        self._storage[block_id, layer, 0, :, slot, :] = key
        self._storage[block_id, layer, 1, :, slot, :] = value

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def num_free(self) -> int:
        with self._lock:
            return len(self._free)

    @property
    def num_used(self) -> int:
        return self.num_blocks - self.num_free

    def memory_used_gb(self) -> float:
        elem_bytes = self._storage.element_size()
        total = self.num_used * self.block_size * self.n_layers * 2 * self.n_kv_heads * self.head_dim
        return total * elem_bytes / 1024**3

    @classmethod
    def from_free_memory(
        cls,
        free_bytes: int,
        utilization: float,
        block_size: int,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
    ) -> "BlockAllocator":
        """Create an allocator sized to consume `utilization` fraction of free VRAM."""
        elem = torch.tensor([], dtype=dtype).element_size()
        bytes_per_block = block_size * n_layers * 2 * n_kv_heads * head_dim * elem
        usable = int(free_bytes * utilization)
        num_blocks = max(1, usable // bytes_per_block)
        return cls(num_blocks, block_size, n_layers, n_kv_heads, head_dim, dtype)
