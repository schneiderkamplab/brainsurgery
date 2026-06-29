from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CacheEntry:
    seq_id: int
    num_tokens: int
    block_table: list[int]
    k_blocks: list[Any] | None = None
    v_blocks: list[Any] | None = None


KVCacheState = Any


class KVCache(ABC):
    @abstractmethod
    def init_entry(self, seq_id: int, prompt_tokens: list[int] | None = None) -> int:
        ...

    @abstractmethod
    def register_blocks(self, seq_id: int, tokens: list[int]) -> None:
        ...

    @abstractmethod
    def append(self, seq_id: int, k: Any, v: Any) -> None:
        ...

    @abstractmethod
    def gather(self, seq_id: int) -> Any:
        ...

    @abstractmethod
    def free(self, seq_id: int) -> None:
        ...

    @abstractmethod
    def release(self, seq_id: int) -> None:
        ...
