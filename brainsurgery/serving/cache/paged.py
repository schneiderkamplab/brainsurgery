from __future__ import annotations

import logging
from typing import Any

import torch

from .base import KVCache, KVCacheState

logger = logging.getLogger("brainsurgery.serving.cache")


class _PagedEntry:
    def __init__(self, seq_id: int):
        self.seq_id = seq_id
        self.block_table: list[int] = []
        self.num_tokens: int = 0


class TorchPagedKVCache(KVCache):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        *,
        block_size: int = 16,
        max_blocks: int = 1024,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ):
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._head_dim = head_dim
        self._block_size = block_size
        self._dtype = dtype
        self._device = device

        self._entries: dict[int, _PagedEntry] = {}
        self._free_blocks: list[int] = list(range(max_blocks))

        shape = (max_blocks, num_layers, num_heads, block_size, head_dim)
        self._k_pool = torch.zeros(shape, dtype=dtype, device=device)
        self._v_pool = torch.zeros(shape, dtype=dtype, device=device)

        # Prefix caching state
        self._hash_to_block: dict[tuple[int, ...], int] = {}
        self._block_refcount: dict[int, int] = {}

        mb = shape[0] * shape[1] * shape[2] * shape[3] * shape[4] * 2 * 4 / (1024 * 1024)
        logger.info(
            "PagedKVCache: %d blocks x %d tokens = %d max tokens, %.1f MB",
            max_blocks, block_size, max_blocks * block_size, mb,
        )

    def _block_token_hash(self, tokens: tuple[int, ...]) -> tuple[int, ...]:
        return hash(tokens)

    def init_entry(self, seq_id: int, prompt_tokens: list[int] | None = None) -> int:
        self._entries[seq_id] = _PagedEntry(seq_id=seq_id)
        if prompt_tokens is None:
            return 0

        entry = self._entries[seq_id]
        cached = 0
        block_hashes: list[int] = []
        all_previous_cached = True
        for i in range(0, len(prompt_tokens), self._block_size):
            blk_tokens = tuple(prompt_tokens[i:i + self._block_size])
            if len(blk_tokens) < self._block_size:
                break
            blk_hash = self._block_token_hash(blk_tokens)
            block_hashes.append(blk_hash)
            if all_previous_cached:
                cached_id = self._hash_to_block.get(blk_hash)
                if cached_id is not None:
                    entry.block_table.append(cached_id)
                    self._block_refcount[cached_id] = self._block_refcount.get(cached_id, 0) + 1
                    cached += self._block_size
                else:
                    all_previous_cached = False

        entry.num_tokens = cached
        entry._block_hashes = block_hashes
        entry._next_block_to_register = cached // self._block_size
        return cached

    def register_blocks(self, seq_id: int, tokens: list[int]) -> None:
        entry = self._entries.get(seq_id)
        if entry is None or not hasattr(entry, '_block_hashes'):
            return
        while entry._next_block_to_register < len(entry._block_hashes):
            blk_idx = entry._next_block_to_register
            if blk_idx >= len(entry.block_table):
                break
            if (blk_idx + 1) * self._block_size > entry.num_tokens:
                break
            blk_hash = entry._block_hashes[blk_idx]
            self._hash_to_block[blk_hash] = entry.block_table[blk_idx]
            entry._next_block_to_register += 1
        if entry._next_block_to_register >= len(entry._block_hashes):
            del entry._block_hashes
            del entry._next_block_to_register

    def _alloc_block(self) -> int | None:
        if not self._free_blocks:
            logger.warning("Out of cache blocks!")
            return None
        block_id = self._free_blocks.pop()
        self._k_pool[block_id].zero_()
        self._v_pool[block_id].zero_()
        self._block_refcount[block_id] = 1
        return block_id

    def _cow_block(self, blk_id: int) -> int:
        """Copy-on-write: if blk_id is shared, allocate a fresh copy and return it."""
        if self._block_refcount.get(blk_id, 1) <= 1:
            return blk_id
        new_id = self._alloc_block()
        if new_id is None:
            raise RuntimeError("Cache full during copy-on-write")
        self._k_pool[new_id] = self._k_pool[blk_id].clone()
        self._v_pool[new_id] = self._v_pool[blk_id].clone()
        self._block_refcount[blk_id] -= 1
        return new_id

    def append_layer_tokens(
        self,
        seq_id: int,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        entry = self._entries.get(seq_id)
        if entry is None:
            raise KeyError(f"Unknown seq_id: {seq_id}")
        assert k.shape == v.shape
        assert k.dim() == 3, f"expected [H, T, D], got {k.shape}"
        n_heads, num_new, head_dim = k.shape
        if num_new == 0:
            return
        pos_start = entry.num_tokens
        blocks_needed = (pos_start + num_new + self._block_size - 1) // self._block_size
        while len(entry.block_table) < blocks_needed:
            blk = self._alloc_block()
            if blk is None:
                raise RuntimeError("Cache full")
            entry.block_table.append(blk)

        positions = torch.arange(pos_start, pos_start + num_new, device=self._device)
        block_indices = positions // self._block_size
        offsets = positions % self._block_size
        blk_table = torch.tensor(entry.block_table, device=self._device, dtype=torch.long)
        blk_ids = blk_table[block_indices]

        # Copy-on-write for any shared blocks we're about to write to
        unique_blks = torch.unique(blk_ids).tolist()
        for old_id in unique_blks:
            new_id = self._cow_block(old_id)
            if new_id != old_id:
                idx_in_table = entry.block_table.index(old_id)
                entry.block_table[idx_in_table] = new_id

        blk_table = torch.tensor(entry.block_table, device=self._device, dtype=torch.long)
        blk_ids = blk_table[block_indices]
        self._k_pool[blk_ids, layer_idx, :n_heads, offsets, :] = k.permute(1, 0, 2)
        self._v_pool[blk_ids, layer_idx, :n_heads, offsets, :] = v.permute(1, 0, 2)

    def advance_tokens(self, seq_id: int, num_new: int) -> None:
        entry = self._entries.get(seq_id)
        if entry is not None:
            entry.num_tokens += num_new

    def append(self, seq_id: int, k: torch.Tensor, v: torch.Tensor) -> None:
        entry = self._entries.get(seq_id)
        if entry is None:
            raise KeyError(f"Unknown seq_id: {seq_id}")
        assert k.shape == v.shape
        assert k.dim() == 5, f"expected [L, B, H, T, D], got {k.shape}"
        n_layers, batch, n_heads, num_new, head_dim = k.shape
        assert batch == 1
        assert n_layers == self._num_layers
        if num_new == 0:
            return
        pos_start = entry.num_tokens
        blocks_needed = (pos_start + num_new + self._block_size - 1) // self._block_size
        while len(entry.block_table) < blocks_needed:
            blk = self._alloc_block()
            if blk is None:
                raise RuntimeError("Cache full")
            entry.block_table.append(blk)
        positions = torch.arange(pos_start, pos_start + num_new, device=self._device)
        block_indices = positions // self._block_size
        offsets = positions % self._block_size
        blk_table = torch.tensor(entry.block_table, device=self._device, dtype=torch.long)
        blk_ids = blk_table[block_indices]

        unique_blks = torch.unique(blk_ids).tolist()
        for old_id in unique_blks:
            new_id = self._cow_block(old_id)
            if new_id != old_id:
                idx_in_table = entry.block_table.index(old_id)
                entry.block_table[idx_in_table] = new_id

        blk_table = torch.tensor(entry.block_table, device=self._device, dtype=torch.long)
        blk_ids = blk_table[block_indices]
        self._k_pool[blk_ids, :, :n_heads, offsets, :] = k[:, 0, :, :, :].permute(3, 0, 1, 2)
        self._v_pool[blk_ids, :, :n_heads, offsets, :] = v[:, 0, :, :, :].permute(3, 0, 1, 2)
        entry.num_tokens += num_new

    def gather(self, seq_id: int) -> KVCacheState | None:
        entry = self._entries.get(seq_id)
        if entry is None or entry.num_tokens == 0:
            return None
        num_tokens = entry.num_tokens
        L, H, D = self._num_layers, self._num_heads, self._head_dim
        positions = torch.arange(num_tokens, device=self._device)
        block_indices = positions // self._block_size
        offsets = positions % self._block_size
        blk_table = torch.tensor(entry.block_table, device=self._device, dtype=torch.long)
        blk_ids = blk_table[block_indices]
        k_full = self._k_pool[blk_ids, :, :H, offsets, :].permute(1, 2, 0, 3)
        v_full = self._v_pool[blk_ids, :, :H, offsets, :].permute(1, 2, 0, 3)
        return tuple(
            (k_full[layer].unsqueeze(0), v_full[layer].unsqueeze(0))
            for layer in range(L)
        )

    def free(self, seq_id: int) -> None:
        entry = self._entries.pop(seq_id, None)
        if entry is None:
            return
        for blk in entry.block_table:
            refs = self._block_refcount.get(blk, 0)
            if refs <= 1:
                self._block_refcount.pop(blk, None)
                self._free_blocks.append(blk)
                # Also remove from hash cache
                to_remove = [k for k, v in self._hash_to_block.items() if v == blk]
                for k in to_remove:
                    self._hash_to_block.pop(k, None)
            else:
                self._block_refcount[blk] = refs - 1

    def release(self, seq_id: int) -> None:
        self.free(seq_id)

    @property
    def k_blocks(self) -> torch.Tensor:
        return self._k_pool.permute(1, 0, 2, 3, 4)

    @property
    def v_blocks(self) -> torch.Tensor:
        return self._v_pool.permute(1, 0, 2, 3, 4)

    @property
    def block_size(self) -> int:
        return self._block_size

    def get_block_table(self, seq_id: int) -> list[int] | None:
        entry = self._entries.get(seq_id)
        return entry.block_table if entry is not None else None

    def get_position(self, seq_id: int) -> int:
        entry = self._entries.get(seq_id)
        return entry.num_tokens if entry is not None else 0


PagedKVCache = TorchPagedKVCache
