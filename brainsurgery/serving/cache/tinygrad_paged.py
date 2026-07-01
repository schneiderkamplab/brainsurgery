from __future__ import annotations

import logging
from typing import Any

from tinygrad import Tensor, dtypes

from .base import KVCache, KVCacheState

logger = logging.getLogger("brainsurgery.serving.cache")


class _PagedEntry:
    def __init__(self, seq_id: int):
        self.seq_id = seq_id
        self.block_table: list[int] = []
        self.num_tokens: int = 0


_DTYPE_MAP: dict[str, dtypes.DType] = {
    "float32": dtypes.float32,
    "float16": dtypes.float16,
    "bfloat16": dtypes.bfloat16,
}


def _d(device: str | None) -> dict[str, str]:
    return {"device": device} if device else {}


class TinygradPagedKVCache(KVCache):
    """Paged KV cache using a single pre-allocated pool tensor.

    Allocates one big tensor for all K (and one for V) and uses
    ``__setitem__`` to write individual blocks — avoids the O(num_blocks)
    stack overhead of the per-block-list approach.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        *,
        block_size: int = 16,
        max_blocks: int = 1024,
        dtype: str = "float32",
        device: str | None = None,
    ):
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._head_dim = head_dim
        self._block_size = block_size
        self._device = device
        tdtype = _DTYPE_MAP.get(dtype, dtypes.float32)

        self._entries: dict[int, _PagedEntry] = {}
        self._free_blocks: list[int] = list(range(max_blocks))

        pool_shape = (num_layers, max_blocks, num_heads, block_size, head_dim)
        devkw = _d(device)
        self._k_pool = Tensor.zeros(*pool_shape, dtype=tdtype, **devkw).contiguous().realize()
        self._v_pool = Tensor.zeros(*pool_shape, dtype=tdtype, **devkw).contiguous().realize()

        mb = max_blocks * num_layers * num_heads * block_size * head_dim * 2 * 4 / (1024 * 1024)
        logger.info(
            "TinygradPagedKVCache: %d blocks x %d tokens = %d max tokens, %.1f MB",
            max_blocks, block_size, max_blocks * block_size, mb,
        )

    def init_entry(self, seq_id: int, prompt_tokens: list[int] | None = None) -> int:
        self._entries[seq_id] = _PagedEntry(seq_id=seq_id)
        return 0

    def register_blocks(self, seq_id: int, tokens: list[int]) -> None:
        pass

    def _alloc_block(self) -> int | None:
        if not self._free_blocks:
            logger.warning("Out of cache blocks!")
            return None
        block_id = self._free_blocks.pop()
        self._k_pool[:, block_id].assign(Tensor.zeros(*self._k_pool[:, block_id].shape, dtype=self._k_pool.dtype, **(
            _d(self._device) if self._device else {}
        )).realize())
        self._v_pool[:, block_id].assign(Tensor.zeros(*self._v_pool[:, block_id].shape, dtype=self._v_pool.dtype, **(
            _d(self._device) if self._device else {}
        )).realize())
        return block_id

    def append_layer_tokens(
        self,
        seq_id: int,
        layer_idx: int,
        k: Any,
        v: Any,
    ) -> None:
        entry = self._entries.get(seq_id)
        if entry is None:
            raise KeyError(f"Unknown seq_id: {seq_id}")
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
        blk_first = pos_start // self._block_size
        blk_last = (pos_start + num_new - 1) // self._block_size
        for blk_idx in range(blk_first, blk_last + 1):
            blk_id = entry.block_table[blk_idx]
            blk_off_start = max(0, pos_start - blk_idx * self._block_size)
            blk_off_end = min(self._block_size, pos_start + num_new - blk_idx * self._block_size)
            k_start = max(0, blk_idx * self._block_size - pos_start)
            k_end = k_start + (blk_off_end - blk_off_start)
            self._k_pool[layer_idx, blk_id, :n_heads, blk_off_start:blk_off_end, :].assign(k[:, k_start:k_end, :])
            self._v_pool[layer_idx, blk_id, :n_heads, blk_off_start:blk_off_end, :].assign(v[:, k_start:k_end, :])

    def advance_tokens(self, seq_id: int, num_new: int) -> None:
        entry = self._entries.get(seq_id)
        if entry is not None:
            entry.num_tokens += num_new

    def append(self, seq_id: int, k: Any, v: Any) -> None:
        entry = self._entries.get(seq_id)
        if entry is None:
            raise KeyError(f"Unknown seq_id: {seq_id}")
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
        blk_first = pos_start // self._block_size
        blk_last = (pos_start + num_new - 1) // self._block_size
        for blk_idx in range(blk_first, blk_last + 1):
            blk_id = entry.block_table[blk_idx]
            blk_off_start = max(0, pos_start - blk_idx * self._block_size)
            blk_off_end = min(self._block_size, pos_start + num_new - blk_idx * self._block_size)
            k_start = max(0, blk_idx * self._block_size - pos_start)
            k_end = k_start + (blk_off_end - blk_off_start)
            self._k_pool[:, blk_id, :n_heads, blk_off_start:blk_off_end, :].assign(k[:, 0, :, k_start:k_end, :])
            self._v_pool[:, blk_id, :n_heads, blk_off_start:blk_off_end, :].assign(v[:, 0, :, k_start:k_end, :])
        entry.num_tokens += num_new

    def gather(self, seq_id: int) -> KVCacheState | None:
        entry = self._entries.get(seq_id)
        if entry is None or entry.num_tokens == 0:
            return None
        num_tokens = entry.num_tokens
        L, H = self._num_layers, self._num_heads
        table = entry.block_table
        devkw = _d(self._device)
        blocks_k = self._k_pool[:, table]  # (L, nb, H, B, D) — no stack needed!
        blocks_v = self._v_pool[:, table]
        positions = Tensor.arange(num_tokens, **devkw)
        block_indices = positions // self._block_size
        offsets = positions % self._block_size
        k_full = blocks_k[:, block_indices, :, offsets, :].permute(1, 2, 0, 3).realize()
        v_full = blocks_v[:, block_indices, :, offsets, :].permute(1, 2, 0, 3).realize()
        return tuple(
            (k_full[layer:layer+1], v_full[layer:layer+1])
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
    def k_blocks(self) -> Any:
        return self._k_pool

    @property
    def v_blocks(self) -> Any:
        return self._v_pool

    @property
    def block_size(self) -> int:
        return self._block_size

    def get_block_table(self, seq_id: int) -> list[int] | None:
        entry = self._entries.get(seq_id)
        return entry.block_table if entry is not None else None

    def get_position(self, seq_id: int) -> int:
        entry = self._entries.get(seq_id)
        return entry.num_tokens if entry is not None else 0
