import logging
import os
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
from time import perf_counter

import torch

from ..core import StateDictLike
from .arena import ProviderError, _SegmentedFileBackedArena, _TensorSlot
from .flags import get_runtime_flags

logger = logging.getLogger("brainsurgery")


@dataclass
class TensorAccessCounts:
    reads: int = 0
    writes: int = 0


@dataclass
class _GpuCacheStats:
    cache_hits: int = 0
    cache_misses: int = 0
    cache_inserts: int = 0
    cache_evictions: int = 0
    cache_evictions_dirty: int = 0
    cache_delete_ops: int = 0
    cache_bind_slot_ops: int = 0
    mark_write_calls: int = 0
    mark_write_missing_cache: int = 0
    write_back_calls: int = 0
    write_back_transfers: int = 0
    write_back_copies: int = 0
    write_back_replaces: int = 0
    transfer_to_device_ops: int = 0
    clone_local_device_ops: int = 0
    flush_calls: int = 0
    bytes_inserted: int = 0
    bytes_evicted: int = 0
    bytes_transfer_to_device: int = 0
    bytes_write_back_transfer: int = 0
    bytes_write_back_copy: int = 0
    bytes_write_back_replace: int = 0

    def as_fields(self) -> dict[str, int]:
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "inserts": self.cache_inserts,
            "evictions": self.cache_evictions,
            "evictions_dirty": self.cache_evictions_dirty,
            "delete_ops": self.cache_delete_ops,
            "bind_slot_ops": self.cache_bind_slot_ops,
            "mark_writes": self.mark_write_calls,
            "mark_write_missing_cache": self.mark_write_missing_cache,
            "write_backs": self.write_back_calls,
            "write_back_transfers": self.write_back_transfers,
            "write_back_copies": self.write_back_copies,
            "write_back_replaces": self.write_back_replaces,
            "to_device_ops": self.transfer_to_device_ops,
            "clone_local_ops": self.clone_local_device_ops,
            "flush_calls": self.flush_calls,
            "bytes_inserted": self.bytes_inserted,
            "bytes_evicted": self.bytes_evicted,
            "bytes_to_device": self.bytes_transfer_to_device,
            "bytes_write_back_transfer": self.bytes_write_back_transfer,
            "bytes_write_back_copy": self.bytes_write_back_copy,
            "bytes_write_back_replace": self.bytes_write_back_replace,
        }


@dataclass
class _TimedAccumulator:
    calls: int = 0
    seconds: float = 0.0
    bytes: int = 0


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
    debug: bool = False


class GpuCachedStateDict(StateDictLike):
    def __init__(
        self,
        backing_state_dict: StateDictLike,
        *,
        config: GpuCacheConfig,
        cache_name: str | None = None,
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
        self._cache_name = cache_name or "unnamed"
        self._debug = config.debug
        self._cache: dict[str, torch.Tensor] = {}
        self._lru: OrderedDict[str, None] = OrderedDict()
        self._dirty: set[str] = set()
        self._cache_bytes = 0
        self._access_counts: dict[str, TensorAccessCounts] = {}
        self._stats = _GpuCacheStats()
        self._timings_by_event_and_key: dict[tuple[str, str], _TimedAccumulator] = {}
        self._timings_by_event_and_part: dict[tuple[str, str], _TimedAccumulator] = {}

    @property
    def backing_state_dict(self) -> StateDictLike:
        return self._backing_state_dict

    @property
    def debug_enabled(self) -> bool:
        return self._debug

    def __getitem__(self, key: str) -> torch.Tensor:
        cached = self._cache.get(key)
        if cached is not None:
            self._touch_lru(key)
            self._mark_read(key)
            self._stats_inc("cache_hits")
            self._debug_log(
                "cache-hit",
                key=key,
                bytes=self._tensor_nbytes(cached),
                cache_bytes=self._cache_bytes,
            )
            return cached

        source = self._backing_state_dict[key]
        self._stats_inc("cache_misses")
        self._debug_log(
            "cache-miss",
            key=key,
            source_device=str(source.device),
            source_bytes=self._tensor_nbytes(source),
        )
        cached_tensor = self._to_cache_tensor(key=key, source=source)
        self._insert_cache_entry(key, cached_tensor, dirty=False)
        self._mark_read(key)
        return cached_tensor

    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        if not torch.is_tensor(value):
            raise ProviderError(f"value for key {key!r} is not a tensor")
        self._backing_state_dict[key] = value
        self._debug_log(
            "setitem",
            key=key,
            value_device=str(value.device),
            value_bytes=self._tensor_nbytes(value),
        )
        cached_tensor = self._to_cache_tensor(key=key, source=value)
        self._insert_cache_entry(key, cached_tensor, dirty=False)
        self._mark_write(key, count=1)

    def __delitem__(self, key: str) -> None:
        self._stats_inc("cache_delete_ops")
        self._debug_log("delete", key=key)
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
        self._stats_inc("cache_bind_slot_ops")
        self._debug_log("bind-slot", key=key)

    def access_counts(self, key: str) -> dict[str, int]:
        counts = self._access_counts.get(key)
        if counts is None:
            return {"reads": 0, "writes": 0}
        return {"reads": counts.reads, "writes": counts.writes}

    def mark_write(self, key: str, count: int = 1) -> None:
        if count < 0:
            raise ProviderError("write count increment must be non-negative")
        self._mark_write(key, count=count)
        self._stats_inc("mark_write_calls")
        self._dirty.add(key)
        self._debug_log("mark-write", key=key, count=count, dirty_count=len(self._dirty))
        self._write_back_key(key)

    def flush(self) -> None:
        dirty_keys = list(self._dirty)
        self._stats_inc("flush_calls")
        self._debug_log("flush-begin", dirty_count=len(dirty_keys))
        for key in dirty_keys:
            self._write_back_key(key)
        self._debug_log("flush-end", dirty_count=len(self._dirty))
        self._debug_log("stats", **self._stats.as_fields(), resident_cache_bytes=self._cache_bytes)
        self._debug_log_timing_summary()

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

    def _to_cache_tensor(self, *, key: str, source: torch.Tensor) -> torch.Tensor:
        if source.device == self._device:
            source_bytes = self._tensor_nbytes(source)
            self._stats_inc("clone_local_device_ops")
            start = perf_counter()
            cloned = source.clone()
            duration = perf_counter() - start
            self._record_timing(
                event="clone_local_device",
                key=key,
                duration_seconds=duration,
                byte_count=source_bytes,
            )
            self._debug_log(
                "cache-clone-local-device",
                key=key,
                source_device=str(source.device),
                target_device=str(self._device),
                bytes=source_bytes,
            )
            return cloned
        source_bytes = self._tensor_nbytes(source)
        self._stats_inc("transfer_to_device_ops")
        self._stats_add_bytes("bytes_transfer_to_device", source_bytes)
        start = perf_counter()
        moved = source.to(device=self._device, non_blocking=self._config.non_blocking)
        duration = perf_counter() - start
        self._record_timing(
            event="transfer_to_device",
            key=key,
            duration_seconds=duration,
            byte_count=source_bytes,
        )
        self._debug_log(
            "cache-transfer-to-device",
            key=key,
            source_device=str(source.device),
            target_device=str(self._device),
            bytes=source_bytes,
            non_blocking=self._config.non_blocking,
        )
        return moved

    def _insert_cache_entry(self, key: str, tensor: torch.Tensor, *, dirty: bool) -> None:
        self._evict_cache_key(key)
        self._cache[key] = tensor
        tensor_bytes = self._tensor_nbytes(tensor)
        self._cache_bytes += tensor_bytes
        self._touch_lru(key)
        if dirty:
            self._dirty.add(key)
        else:
            self._dirty.discard(key)
        self._stats_inc("cache_inserts")
        self._stats_add_bytes("bytes_inserted", tensor_bytes)
        self._enforce_cache_budget(skip_key=key)
        self._debug_log(
            "cache-insert",
            key=key,
            bytes=tensor_bytes,
            dirty=dirty,
            cache_bytes=self._cache_bytes,
            max_cache_bytes=self._max_cache_bytes,
        )

    def _touch_lru(self, key: str) -> None:
        self._lru.pop(key, None)
        self._lru[key] = None

    def _evict_cache_key(self, key: str) -> None:
        cached = self._cache.pop(key, None)
        self._lru.pop(key, None)
        if cached is None:
            return
        cached_bytes = self._tensor_nbytes(cached)
        self._cache_bytes -= cached_bytes
        self._stats_inc("cache_evictions")
        self._stats_add_bytes("bytes_evicted", cached_bytes)
        self._debug_log(
            "cache-evict",
            key=key,
            bytes=cached_bytes,
            dirty=key in self._dirty,
            cache_bytes=self._cache_bytes,
            max_cache_bytes=self._max_cache_bytes,
        )
        if key in self._dirty:
            self._stats_inc("cache_evictions_dirty")
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
            self._stats_inc("mark_write_missing_cache")
            return
        self._stats_inc("write_back_calls")
        self._write_back_tensor(key, cached)
        self._dirty.discard(key)
        self._debug_log("write-back", key=key)

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
                tensor_bytes = self._tensor_nbytes(tensor)
                self._stats_inc("write_back_transfers")
                self._stats_add_bytes("bytes_write_back_transfer", tensor_bytes)
                self._debug_log(
                    "write-back-transfer",
                    key=key,
                    source_device=str(tensor.device),
                    target_device=str(backing_tensor.device),
                    bytes=tensor_bytes,
                    non_blocking=self._config.non_blocking,
                )
                start = perf_counter()
                source = tensor.to(
                    device=backing_tensor.device,
                    non_blocking=self._config.non_blocking,
                )
                duration = perf_counter() - start
                self._record_timing(
                    event="write_back_transfer",
                    key=key,
                    duration_seconds=duration,
                    byte_count=tensor_bytes,
                )
            start = perf_counter()
            backing_tensor.copy_(source)
            duration = perf_counter() - start
            copied_bytes = self._tensor_nbytes(backing_tensor)
            self._stats_inc("write_back_copies")
            self._stats_add_bytes("bytes_write_back_copy", copied_bytes)
            self._record_timing(
                event="write_back_copy",
                key=key,
                duration_seconds=duration,
                byte_count=copied_bytes,
            )
            self._debug_log(
                "write-back-copy",
                key=key,
                target_device=str(backing_tensor.device),
                bytes=copied_bytes,
            )
            return

        replaced_bytes = self._tensor_nbytes(tensor)
        self._stats_inc("write_back_replaces")
        self._stats_add_bytes("bytes_write_back_replace", replaced_bytes)
        self._debug_log(
            "write-back-replace",
            key=key,
            source_device=str(tensor.device),
            target_device="cpu",
            bytes=replaced_bytes,
        )
        start = perf_counter()
        self._backing_state_dict[key] = tensor.to(device="cpu")
        duration = perf_counter() - start
        self._record_timing(
            event="write_back_replace",
            key=key,
            duration_seconds=duration,
            byte_count=replaced_bytes,
        )

    @staticmethod
    def _tensor_nbytes(tensor: torch.Tensor) -> int:
        return tensor.numel() * tensor.element_size()

    def _resolve_max_cache_bytes(self) -> int:
        if self._config.max_cache_bytes is not None:
            if self._config.debug:
                logger.info(
                    "[gpu-cache:%s] explicit budget bytes=%d",
                    self._config.device,
                    self._config.max_cache_bytes,
                )
            return self._config.max_cache_bytes
        total_memory = _detect_total_memory_bytes_for_device(self._device)
        budget = int(total_memory * self._config.memory_fraction)
        if budget < 0:
            raise ProviderError("gpu cache budget must not be negative")
        if self._config.debug:
            logger.info(
                "[gpu-cache:%s] auto budget bytes=%d total=%d fraction=%.3f",
                self._config.device,
                budget,
                total_memory,
                self._config.memory_fraction,
            )
        return budget

    def _debug_log(self, event: str, **fields: object) -> None:
        if not self._debug:
            return
        details = " ".join(f"{name}={value}" for name, value in fields.items())
        if details:
            logger.info("[gpu-cache:%s] %s %s", self._cache_name, event, details)
        else:
            logger.info("[gpu-cache:%s] %s", self._cache_name, event)

    def _stats_inc(self, field: str, count: int = 1) -> None:
        if not self._debug:
            return
        setattr(self._stats, field, getattr(self._stats, field) + count)

    def _stats_add_bytes(self, field: str, value: int) -> None:
        if not self._debug:
            return
        setattr(self._stats, field, getattr(self._stats, field) + value)

    def _record_timing(
        self,
        *,
        event: str,
        key: str,
        duration_seconds: float,
        byte_count: int,
    ) -> None:
        if not self._debug:
            return
        event_key = (event, key)
        by_key = self._timings_by_event_and_key.get(event_key)
        if by_key is None:
            by_key = _TimedAccumulator()
            self._timings_by_event_and_key[event_key] = by_key
        by_key.calls += 1
        by_key.seconds += duration_seconds
        by_key.bytes += byte_count

        part = self._derive_model_part(key)
        event_part = (event, part)
        by_part = self._timings_by_event_and_part.get(event_part)
        if by_part is None:
            by_part = _TimedAccumulator()
            self._timings_by_event_and_part[event_part] = by_part
        by_part.calls += 1
        by_part.seconds += duration_seconds
        by_part.bytes += byte_count

    def _debug_log_timing_summary(self) -> None:
        if not self._debug:
            return
        events = sorted({event for event, _ in self._timings_by_event_and_key})
        for event in events:
            key_rows: list[tuple[str, _TimedAccumulator]] = []
            part_rows: list[tuple[str, _TimedAccumulator]] = []
            for (row_event, key), row in self._timings_by_event_and_key.items():
                if row_event == event:
                    key_rows.append((key, row))
            for (row_event, part), row in self._timings_by_event_and_part.items():
                if row_event == event:
                    part_rows.append((part, row))
            if not key_rows:
                continue

            total_calls = sum(row.calls for _, row in key_rows)
            total_seconds = sum(row.seconds for _, row in key_rows)
            total_bytes = sum(row.bytes for _, row in key_rows)
            self._debug_log(
                "timing-total",
                timing_event=event,
                calls=total_calls,
                seconds=f"{total_seconds:.6f}",
                bytes=total_bytes,
            )
            for rank, (key, row) in enumerate(
                sorted(key_rows, key=lambda item: item[1].seconds, reverse=True)[:10],
                start=1,
            ):
                self._debug_log(
                    "timing-top-keys",
                    timing_event=event,
                    rank=rank,
                    key=key,
                    calls=row.calls,
                    seconds=f"{row.seconds:.6f}",
                    bytes=row.bytes,
                )
            for rank, (part, row) in enumerate(
                sorted(part_rows, key=lambda item: item[1].seconds, reverse=True)[:10],
                start=1,
            ):
                self._debug_log(
                    "timing-top-parts",
                    timing_event=event,
                    rank=rank,
                    part=part,
                    calls=row.calls,
                    seconds=f"{row.seconds:.6f}",
                    bytes=row.bytes,
                )

    @staticmethod
    def _derive_model_part(key: str) -> str:
        parts = key.split(".")
        if len(parts) >= 3 and parts[0] == "model" and parts[1] == "layers" and parts[2].isdigit():
            layer = f"model.layers.{parts[2]}"
            if (
                len(parts) >= 6
                and parts[3] == "mlp"
                and parts[4] == "experts"
                and parts[5].isdigit()
            ):
                return f"{layer}.mlp.experts"
            return layer
        if len(parts) >= 2:
            return f"{parts[0]}.{parts[1]}"
        return key

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
