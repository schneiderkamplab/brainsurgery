from __future__ import annotations

import contextlib
import gc
import io
import os
import time
from pathlib import Path
from typing import Any


class TeeWriter:
    def __init__(self, *streams: Any) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        for stream in self._streams:
            isatty = getattr(stream, "isatty", None)
            if callable(isatty):
                try:
                    if bool(isatty()):
                        return True
                except Exception:
                    continue
        return False


class LogFileWriter:
    def __init__(self, stream: Any) -> None:
        self._stream = stream

    def write(self, data: str) -> int:
        normalized = data.replace("\r", "\n")
        self._stream.write(normalized)
        return len(data)

    def flush(self) -> None:
        self._stream.flush()


class ParentLogger:
    def __init__(self, log_dir: Path | None) -> None:
        self._path = (log_dir / f"parent-{os.getpid()}.txt") if log_dir is not None else None
        self._fh: io.TextIOWrapper | None = None
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self._path.open("w", encoding="utf-8", buffering=1)

    @property
    def path(self) -> Path | None:
        return self._path

    def log(self, message: str) -> None:
        if self._fh is None:
            return
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self._fh.write(f"{timestamp} {message}\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def is_cuda_oom(exc: BaseException, *, device: str) -> bool:
    normalized = str(device).strip().lower()
    if not normalized.startswith("cuda"):
        return False
    message = str(exc).lower()
    if "out of memory" in message or "cuda error: out of memory" in message:
        return True
    try:
        import torch
    except Exception:
        return False
    return isinstance(exc, torch.cuda.OutOfMemoryError)


def cleanup_cuda_after_oom(device: str) -> None:
    normalized = str(device).strip().lower()
    if not normalized.startswith("cuda"):
        return
    gc.collect()
    try:
        import torch
    except Exception:
        return
    if torch.cuda.is_available():
        with contextlib.suppress(Exception):
            torch.cuda.empty_cache()
        with contextlib.suppress(Exception):
            torch.cuda.ipc_collect()


def resolve_worker_devices(device: str, processes: int) -> list[str]:
    if processes <= 0:
        raise ValueError("processes must be >= 1")
    normalized = str(device).strip().lower()
    if not normalized:
        raise ValueError("device must not be empty")
    if processes == 1:
        return [device]

    try:
        import torch
    except Exception:
        if normalized.startswith("cuda") or normalized == "auto":
            raise ValueError("torch is required to resolve multi-process CUDA worker devices")
        return [device for _ in range(processes)]

    def _cpu_or_fallback() -> list[str]:
        if normalized == "auto":
            if torch.cuda.is_available():
                return _cuda_devices(start_index=0)
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return ["mps" for _ in range(processes)]
            return ["cpu" for _ in range(processes)]
        if normalized == "mps":
            return ["mps" for _ in range(processes)]
        return [device for _ in range(processes)]

    def _cuda_devices(*, start_index: int) -> list[str]:
        device_count = int(torch.cuda.device_count())
        if device_count <= 0:
            raise ValueError("requested CUDA multiprocessing but no CUDA devices are available")
        end_index = start_index + processes
        if end_index > device_count:
            raise ValueError(
                f"requested {processes} processes starting at cuda:{start_index}, "
                f"but only {device_count} CUDA devices are available"
            )
        return [f"cuda:{idx}" for idx in range(start_index, end_index)]

    if normalized == "cuda":
        return _cuda_devices(start_index=0)
    if normalized.startswith("cuda:"):
        index_text = normalized.split(":", 1)[1].strip()
        if not index_text.isdigit():
            raise ValueError(f"invalid CUDA device specifier for multiprocessing: {device!r}")
        return _cuda_devices(start_index=int(index_text))
    return _cpu_or_fallback()


def worker_log_path(
    log_dir: Path | None,
    *,
    axon_name: str,
    model_name: str,
    pid: int | None,
) -> Path | None:
    if log_dir is None:
        return None
    if pid is None:
        return log_dir / f"log-pending-{axon_name}-{model_name}.txt"
    return log_dir / f"log-{pid}-{axon_name}-{model_name}.txt"


def worker_log_display_path(
    log_dir: Path | None,
    *,
    axon_name: str,
    model_name: str,
    pid: int | None,
) -> str | None:
    path = worker_log_path(log_dir, axon_name=axon_name, model_name=model_name, pid=pid)
    if path is None:
        return None
    return path.name
