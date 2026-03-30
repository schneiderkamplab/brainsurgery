import os
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass

import torch

from ..core import StateDictLike
from .arena import ProviderError, _SegmentedFileBackedArena, _TensorSlot
from .flags import get_runtime_flags


@dataclass
class TensorAccessCounts:
    reads: int = 0
    writes: int = 0


class SlotBackedStateDict(StateDictLike):
    def __init__(self) -> None:
        self._slots: dict[str, torch.Tensor | _TensorSlot] = {}
        self._access_counts: dict[str, TensorAccessCounts] = {}
        self._dry_run_slots: dict[str, torch.Tensor] = {}
        self._dry_run_deleted: set[str] = set()

    def __delitem__(self, key: str) -> None:
        if self._is_dry_run():
            self._dry_run_slots.pop(key, None)
            self._dry_run_deleted.add(key)
            return
        del self._slots[key]
        self._access_counts.pop(key, None)

    def __iter__(self) -> Iterator[str]:
        return iter(self._effective_keys())

    def __len__(self) -> int:
        return len(self._effective_keys())

    def keys(self):
        return self._effective_keys()

    def items(self):
        for key in self._effective_keys():
            yield key, self[key]

    def values(self):
        for key in self._effective_keys():
            yield self[key]

    def access_counts(self, key: str) -> dict[str, int]:
        counts = self._access_counts.get(key)
        if counts is None:
            return {"reads": 0, "writes": 0}
        return {"reads": counts.reads, "writes": counts.writes}

    def mark_write(self, key: str, count: int = 1) -> None:
        if self._is_dry_run():
            return
        if count < 0:
            raise ProviderError("write count increment must be non-negative")
        self._ensure_access_counts(key).writes += count

    def _mark_read(self, key: str, count: int = 1) -> None:
        if self._is_dry_run():
            return
        if count < 0:
            raise ProviderError("read count increment must be non-negative")
        self._ensure_access_counts(key).reads += count

    def _ensure_access_counts(self, key: str) -> TensorAccessCounts:
        if key not in self._slots:
            raise KeyError(key)
        counts = self._access_counts.get(key)
        if counts is None:
            counts = TensorAccessCounts()
            self._access_counts[key] = counts
        return counts

    def _is_dry_run(self) -> bool:
        dry_run = get_runtime_flags().dry_run
        if not dry_run and (self._dry_run_slots or self._dry_run_deleted):
            self._dry_run_slots.clear()
            self._dry_run_deleted.clear()
        return dry_run

    def _effective_keys(self) -> list[str]:
        if not self._is_dry_run():
            return list(self._slots.keys())
        keys = [key for key in self._slots if key not in self._dry_run_deleted]
        for key in self._dry_run_slots:
            if key not in self._dry_run_deleted and key not in self._slots:
                keys.append(key)
        return keys


class _InMemoryStateDict(SlotBackedStateDict):
    def __getitem__(self, key: str) -> torch.Tensor:
        if self._is_dry_run():
            if key in self._dry_run_deleted:
                raise KeyError(key)
            if key not in self._dry_run_slots:
                slot_value = self._slots[key]
                assert isinstance(slot_value, torch.Tensor)
                value = slot_value.clone()
                self._dry_run_slots[key] = value
            value = self._dry_run_slots[key]
            assert isinstance(value, torch.Tensor)
            return value

        slot_value = self._slots[key]
        assert isinstance(slot_value, torch.Tensor)
        self._mark_read(key)
        return slot_value

    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        if not torch.is_tensor(value):
            raise ProviderError(f"value for key {key!r} is not a tensor")
        if self._is_dry_run():
            self._dry_run_slots[key] = value.clone()
            self._dry_run_deleted.discard(key)
            return
        self._slots[key] = value
        self.mark_write(key)

    def slot(self, key: str) -> torch.Tensor:
        if self._is_dry_run():
            return self[key]

        value = self._slots[key]
        assert isinstance(value, torch.Tensor)
        return value

    def bind_slot(self, key: str, slot: torch.Tensor) -> None:
        if not torch.is_tensor(slot):
            raise ProviderError(f"slot for key {key!r} is not a tensor")
        if self._is_dry_run():
            self._dry_run_slots[key] = slot.clone()
            self._dry_run_deleted.discard(key)
            return
        self._slots[key] = slot
        self.mark_write(key)


class _ArenaStateDict(SlotBackedStateDict):
    def __init__(self, arena: _SegmentedFileBackedArena):
        super().__init__()
        self._arena = arena

    def __getitem__(self, key: str) -> torch.Tensor:
        if self._is_dry_run():
            if key in self._dry_run_deleted:
                raise KeyError(key)
            if key not in self._dry_run_slots:
                slot = self._slots[key]
                assert isinstance(slot, _TensorSlot)
                self._dry_run_slots[key] = self._arena.tensor_from_slot(slot).clone()
            value = self._dry_run_slots[key]
            assert isinstance(value, torch.Tensor)
            return value

        try:
            slot = self._slots[key]
        except KeyError as exc:
            raise KeyError(key) from exc
        assert isinstance(slot, _TensorSlot)
        self._mark_read(key)
        return self._arena.tensor_from_slot(slot)

    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        if not torch.is_tensor(value):
            raise ProviderError(f"value for key {key!r} is not a tensor")
        if self._is_dry_run():
            self._dry_run_slots[key] = value.clone()
            self._dry_run_deleted.discard(key)
            return
        self._slots[key] = self._arena.store_tensor(value)
        self.mark_write(key)

    def slot(self, key: str) -> _TensorSlot:
        try:
            slot = self._slots[key]
            assert isinstance(slot, _TensorSlot)
            return slot
        except KeyError as exc:
            raise KeyError(key) from exc

    def bind_slot(self, key: str, slot: _TensorSlot) -> None:
        if not isinstance(slot, _TensorSlot):
            raise ProviderError(f"slot for key {key!r} is not a _TensorSlot")
        if self._is_dry_run():
            self._dry_run_slots[key] = self._arena.tensor_from_slot(slot).clone()
            self._dry_run_deleted.discard(key)
            return
        self._slots[key] = slot
        self.mark_write(key)


@dataclass(frozen=True)
class GpuCacheConfig:
    device: str = "cuda"
    max_cache_bytes: int | None = None
    memory_fraction: float = 0.8
    non_blocking: bool = False


class GpuCachedStateDict(StateDictLike):
    def __init__(
        self,
        backing_state_dict: StateDictLike,
        *,
        config: GpuCacheConfig,
    ) -> None:
        if config.max_cache_bytes is not None and config.max_cache_bytes < 0:
            raise ProviderError("gpu cache max_cache_bytes must be non-negative")
        if config.memory_fraction <= 0 or config.memory_fraction > 1:
            raise ProviderError("gpu cache memory_fraction must be in the interval (0, 1]")
        self._backing_state_dict = backing_state_dict
        self._config = config
        self._device = torch.device(config.device)
        self._validate_backend_available(self._device)
        self._max_cache_bytes = self._resolve_max_cache_bytes()
        self._cache: dict[str, torch.Tensor] = {}
        self._lru: OrderedDict[str, None] = OrderedDict()
        self._dirty: set[str] = set()
        self._cache_bytes = 0
        self._access_counts: dict[str, TensorAccessCounts] = {}

    @property
    def backing_state_dict(self) -> StateDictLike:
        return self._backing_state_dict

    def __getitem__(self, key: str) -> torch.Tensor:
        cached = self._cache.get(key)
        if cached is not None:
            self._touch_lru(key)
            self._mark_read(key)
            return cached

        source = self._backing_state_dict[key]
        cached_tensor = self._to_cache_tensor(source)
        self._insert_cache_entry(key, cached_tensor, dirty=False)
        self._mark_read(key)
        return cached_tensor

    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        if not torch.is_tensor(value):
            raise ProviderError(f"value for key {key!r} is not a tensor")
        self._backing_state_dict[key] = value
        cached_tensor = self._to_cache_tensor(value)
        self._insert_cache_entry(key, cached_tensor, dirty=False)
        self._mark_write(key, count=1)

    def __delitem__(self, key: str) -> None:
        self._evict_cache_key(key)
        self._dirty.discard(key)
        del self._backing_state_dict[key]
        self._access_counts.pop(key, None)

    def __iter__(self) -> Iterator[str]:
        return iter(self._backing_state_dict)

    def __len__(self) -> int:
        return len(self._backing_state_dict)

    def keys(self):
        return self._backing_state_dict.keys()

    def items(self):
        for key in self.keys():
            yield key, self[key]

    def values(self):
        for key in self.keys():
            yield self[key]

    def slot(self, key: str):
        return self._backing_state_dict.slot(key)

    def bind_slot(self, key: str, slot) -> None:
        self._backing_state_dict.bind_slot(key, slot)
        self._evict_cache_key(key)
        self._dirty.discard(key)
        self._mark_write(key, count=1)

    def access_counts(self, key: str) -> dict[str, int]:
        counts = self._access_counts.get(key)
        if counts is None:
            return {"reads": 0, "writes": 0}
        return {"reads": counts.reads, "writes": counts.writes}

    def mark_write(self, key: str, count: int = 1) -> None:
        if count < 0:
            raise ProviderError("write count increment must be non-negative")
        self._mark_write(key, count=count)
        self._dirty.add(key)
        self._write_back_key(key)

    def flush(self) -> None:
        dirty_keys = list(self._dirty)
        for key in dirty_keys:
            self._write_back_key(key)

    def cache_bytes(self) -> int:
        return self._cache_bytes

    def _mark_read(self, key: str) -> None:
        self._ensure_counts(key).reads += 1

    def _mark_write(self, key: str, *, count: int) -> None:
        self._ensure_counts(key).writes += count
        self._backing_state_dict.mark_write(key, count=count)

    def _ensure_counts(self, key: str) -> TensorAccessCounts:
        if key not in self._backing_state_dict:
            raise KeyError(key)
        counts = self._access_counts.get(key)
        if counts is None:
            counts = TensorAccessCounts()
            self._access_counts[key] = counts
        return counts

    def _to_cache_tensor(self, source: torch.Tensor) -> torch.Tensor:
        if source.device == self._device:
            return source.clone()
        return source.to(device=self._device, non_blocking=self._config.non_blocking)

    def _insert_cache_entry(self, key: str, tensor: torch.Tensor, *, dirty: bool) -> None:
        self._evict_cache_key(key)
        self._cache[key] = tensor
        self._cache_bytes += self._tensor_nbytes(tensor)
        self._touch_lru(key)
        if dirty:
            self._dirty.add(key)
        else:
            self._dirty.discard(key)
        self._enforce_cache_budget(skip_key=key)

    def _touch_lru(self, key: str) -> None:
        self._lru.pop(key, None)
        self._lru[key] = None

    def _evict_cache_key(self, key: str) -> None:
        cached = self._cache.pop(key, None)
        self._lru.pop(key, None)
        if cached is None:
            return
        self._cache_bytes -= self._tensor_nbytes(cached)
        if key in self._dirty:
            self._write_back_tensor(key, cached)
            self._dirty.discard(key)

    def _enforce_cache_budget(self, *, skip_key: str) -> None:
        if self._max_cache_bytes == 0:
            for key in list(self._lru):
                if key == skip_key:
                    continue
                self._evict_cache_key(key)
            return

        while self._cache_bytes > self._max_cache_bytes:
            oldest_key = next(iter(self._lru), None)
            if oldest_key is None:
                return
            if oldest_key == skip_key and len(self._lru) == 1:
                return
            if oldest_key == skip_key:
                self._touch_lru(oldest_key)
                continue
            self._evict_cache_key(oldest_key)

    def _write_back_key(self, key: str) -> None:
        cached = self._cache.get(key)
        if cached is None:
            return
        self._write_back_tensor(key, cached)
        self._dirty.discard(key)

    def _write_back_tensor(self, key: str, tensor: torch.Tensor) -> None:
        try:
            backing_tensor = self._backing_state_dict[key]
        except KeyError as exc:
            raise KeyError(key) from exc
        if (
            backing_tensor.shape == tensor.shape
            and backing_tensor.dtype == tensor.dtype
            and backing_tensor.layout == tensor.layout
        ):
            source = tensor
            if tensor.device != backing_tensor.device:
                source = tensor.to(
                    device=backing_tensor.device,
                    non_blocking=self._config.non_blocking,
                )
            backing_tensor.copy_(source)
            return

        self._backing_state_dict[key] = tensor.to(device="cpu")

    @staticmethod
    def _tensor_nbytes(tensor: torch.Tensor) -> int:
        return tensor.numel() * tensor.element_size()

    def _resolve_max_cache_bytes(self) -> int:
        if self._config.max_cache_bytes is not None:
            return self._config.max_cache_bytes
        total_memory = _detect_total_memory_bytes_for_device(self._device)
        budget = int(total_memory * self._config.memory_fraction)
        if budget < 0:
            raise ProviderError("gpu cache budget must not be negative")
        return budget

    @staticmethod
    def _validate_backend_available(device: torch.device) -> None:
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise ProviderError("gpu cache device 'cuda' is not available in this runtime")
            return
        if device.type == "mps":
            mps_backend = getattr(torch.backends, "mps", None)
            if mps_backend is None or not mps_backend.is_available():
                raise ProviderError("gpu cache device 'mps' is not available in this runtime")
            return
        if device.type == "cpu":
            return
        raise ProviderError(
            f"unsupported gpu cache device type: {device.type!r}; expected one of 'cuda', 'mps', 'cpu'"
        )


def _detect_total_memory_bytes_for_device(device: torch.device) -> int:
    if device.type == "cuda":
        if device.index is None:
            device_index = torch.cuda.current_device()
        else:
            device_index = device.index
        return int(torch.cuda.get_device_properties(device_index).total_memory)

    if device.type == "mps":
        mps_module = getattr(torch, "mps", None)
        if mps_module is None:
            raise ProviderError("torch.mps is unavailable; cannot derive mps memory budget")
        recommended = getattr(mps_module, "recommended_max_memory", None)
        if callable(recommended):
            value = int(recommended())
            if value > 0:
                return value
        return _detect_system_memory_bytes()

    if device.type == "cpu":
        return _detect_system_memory_bytes()

    raise ProviderError(f"unsupported device type for memory detection: {device.type!r}")


def _detect_system_memory_bytes() -> int:
    if hasattr(os, "sysconf"):
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        if isinstance(page_size, int) and isinstance(pages, int) and page_size > 0 and pages > 0:
            return page_size * pages
    raise ProviderError("unable to detect system memory for cache budget derivation")


assert issubclass(GpuCachedStateDict, StateDictLike)


__all__ = [
    "TensorAccessCounts",
    "SlotBackedStateDict",
    "_InMemoryStateDict",
    "_ArenaStateDict",
    "GpuCacheConfig",
    "GpuCachedStateDict",
]
