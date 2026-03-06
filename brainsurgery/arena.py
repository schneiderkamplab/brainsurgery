from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

from .model import parse_shard_size


class ArenaError(RuntimeError):
    pass


@dataclass(frozen=True)
class TensorSlot:
    segment_id: int
    offset: int
    nbytes: int
    dtype: torch.dtype
    shape: tuple[int, ...]


class ArenaSegment:
    def __init__(self, path: Path, size_bytes: int):
        if size_bytes <= 0:
            raise ArenaError("segment size must be positive")

        self.path = path
        self.size_bytes = size_bytes
        self.path.parent.mkdir(parents=True, exist_ok=True)

        with self.path.open("wb") as f:
            f.truncate(size_bytes)

        self.mem = np.memmap(
            self.path,
            mode="r+",
            dtype=np.uint8,
            shape=(size_bytes,),
        )

    def close(self) -> None:
        self.mem.flush()
        del self.mem

    def flush(self) -> None:
        self.mem.flush()


class SegmentedFileBackedArena:
    def __init__(
        self,
        root: Path,
        *,
        segment_size_bytes: int = 1024**3,
        alignment: int = 64,
    ):
        if segment_size_bytes <= 0:
            raise ArenaError("segment_size_bytes must be positive")
        if alignment <= 0:
            raise ArenaError("alignment must be positive")

        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

        self.segment_size_bytes = segment_size_bytes
        self.alignment = alignment

        self._segments: list[ArenaSegment] = []
        self._write_segment_id = -1
        self._write_offset = 0

        self._ensure_segment()

    def __enter__(self) -> "SegmentedFileBackedArena":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        for segment in self._segments:
            segment.close()

    def flush(self) -> None:
        for segment in self._segments:
            segment.flush()

    def allocate(self, nbytes: int) -> tuple[int, int]:
        if nbytes < 0:
            raise ArenaError("cannot allocate negative bytes")
        if nbytes == 0:
            raise ArenaError("zero-byte tensors are not supported in arena v1")
        if nbytes > self.segment_size_bytes:
            raise ArenaError(
                f"tensor of {nbytes} bytes exceeds segment size {self.segment_size_bytes}; "
                "arena v1 requires every tensor to fit within one segment"
            )

        offset = self._align(self._write_offset)

        if offset + nbytes > self.segment_size_bytes:
            self._ensure_segment()
            offset = self._align(self._write_offset)

        segment_id = self._write_segment_id
        self._write_offset = offset + nbytes
        return segment_id, offset

    def store_tensor(self, tensor: torch.Tensor) -> TensorSlot:
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()

        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        ensure_supported_dtype(tensor.dtype)

        nbytes = tensor.numel() * tensor.element_size()
        segment_id, offset = self.allocate(nbytes)

        view = self.tensor_view(
            segment_id=segment_id,
            offset=offset,
            dtype=tensor.dtype,
            shape=tuple(tensor.shape),
        )
        view.copy_(tensor)

        return TensorSlot(
            segment_id=segment_id,
            offset=offset,
            nbytes=nbytes,
            dtype=tensor.dtype,
            shape=tuple(tensor.shape),
        )

    def tensor_from_slot(self, slot: TensorSlot) -> torch.Tensor:
        return self.tensor_view(
            segment_id=slot.segment_id,
            offset=slot.offset,
            dtype=slot.dtype,
            shape=slot.shape,
        )

    def tensor_view(
        self,
        *,
        segment_id: int,
        offset: int,
        dtype: torch.dtype,
        shape: tuple[int, ...],
    ) -> torch.Tensor:
        ensure_supported_dtype(dtype)

        numel = prod(shape)
        element_size = torch_element_size(dtype)
        nbytes = numel * element_size

        segment = self._segments[segment_id]
        end = offset + nbytes
        if end > segment.size_bytes:
            raise ArenaError(
                f"tensor view exceeds segment bounds: segment={segment_id}, offset={offset}, nbytes={nbytes}"
            )

        try:
            tensor = torch.frombuffer(
                segment.mem,
                dtype=dtype,
                count=numel,
                offset=offset,
            ).reshape(shape)
        except Exception as exc:
            raise ArenaError(
                f"failed to create tensor view: segment={segment_id}, offset={offset}, "
                f"dtype={dtype}, shape={shape}"
            ) from exc

        return tensor

    def _ensure_segment(self) -> None:
        segment_id = len(self._segments)
        segment_path = self.root / f"segment-{segment_id:05d}.bin"
        segment = ArenaSegment(segment_path, self.segment_size_bytes)
        self._segments.append(segment)
        self._write_segment_id = segment_id
        self._write_offset = 0

    def _align(self, offset: int) -> int:
        rem = offset % self.alignment
        if rem == 0:
            return offset
        return offset + (self.alignment - rem)


def ensure_supported_dtype(dtype: torch.dtype) -> None:
    supported = {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.bool,
    }
    if dtype not in supported:
        raise ArenaError(f"unsupported dtype for arena storage: {dtype}")


def torch_element_size(dtype: torch.dtype) -> int:
    sizes = {
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.float32: 4,
        torch.float64: 8,
        torch.uint8: 1,
        torch.int8: 1,
        torch.int16: 2,
        torch.int32: 4,
        torch.int64: 8,
        torch.bool: 1,
    }
    try:
        return sizes[dtype]
    except KeyError as exc:
        raise ArenaError(f"unsupported dtype for arena storage: {dtype}") from exc


def prod(shape: Tuple[int, ...]) -> int:
    out = 1
    for x in shape:
        out *= x
    return out
