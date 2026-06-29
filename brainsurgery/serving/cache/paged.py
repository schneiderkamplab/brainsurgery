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

        mb = shape[0] * shape[1] * shape[2] * shape[3] * shape[4] * 2 * 4 / (1024 * 1024)
        logger.info(
            "PagedKVCache: %d blocks x %d tokens = %d max tokens, %.1f MB",
            max_blocks, block_size, max_blocks * block_size, mb,
        )

    def init_entry(self, seq_id: int) -> int:
        self._entries[seq_id] = _PagedEntry(seq_id=seq_id)
        return 0

    def _alloc_block(self) -> int | None:
        if not self._free_blocks:
            logger.warning("Out of cache blocks!")
            return None
        block_id = self._free_blocks.pop()
        self._k_pool[block_id].zero_()
        self._v_pool[block_id].zero_()
        return block_id

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
        for i in range(num_new):
            pos = entry.num_tokens + i
            block_idx = pos // self._block_size
            offset = pos % self._block_size
            while block_idx >= len(entry.block_table):
                blk = self._alloc_block()
                if blk is None:
                    raise RuntimeError("Cache full")
                entry.block_table.append(blk)
            blk = entry.block_table[block_idx]
            self._k_pool[blk, layer_idx, :n_heads, offset, :] = k[:, i, :]
            self._v_pool[blk, layer_idx, :n_heads, offset, :] = v[:, i, :]

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
        for i in range(num_new):
            pos = entry.num_tokens + i
            block_idx = pos // self._block_size
            offset = pos % self._block_size
            while block_idx >= len(entry.block_table):
                blk = self._alloc_block()
                if blk is None:
                    raise RuntimeError("Cache full")
                entry.block_table.append(blk)
            blk = entry.block_table[block_idx]
            self._k_pool[blk, :, :n_heads, offset, :] = k[:, 0, :, i, :]
            self._v_pool[blk, :, :n_heads, offset, :] = v[:, 0, :, i, :]
        entry.num_tokens += num_new

    def gather(self, seq_id: int) -> KVCacheState | None:
        entry = self._entries.get(seq_id)
        if entry is None or entry.num_tokens == 0:
            return None
        num_tokens = entry.num_tokens
        L, H, D = self._num_layers, self._num_heads, self._head_dim
        k_full = torch.zeros(L, H, num_tokens, D, dtype=self._dtype, device=self._device)
        v_full = torch.zeros(L, H, num_tokens, D, dtype=self._dtype, device=self._device)
        for pos in range(num_tokens):
            block_idx = pos // self._block_size
            offset = pos % self._block_size
            blk = entry.block_table[block_idx]
            k_full[:, :, pos, :] = self._k_pool[blk, :, :H, offset, :]
            v_full[:, :, pos, :] = self._v_pool[blk, :, :H, offset, :]
        return tuple(
            (k_full[layer].unsqueeze(0), v_full[layer].unsqueeze(0))
            for layer in range(L)
        )

    def free(self, seq_id: int) -> None:
        entry = self._entries.pop(seq_id, None)
        if entry is None:
            return
        for blk in entry.block_table:
            self._free_blocks.append(blk)

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
