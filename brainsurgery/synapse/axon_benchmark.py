from __future__ import annotations

import contextlib
import csv
import gc
import io
import multiprocessing as mp
import os
import queue
import signal
import traceback
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from tqdm import tqdm

from .axon_runner_common import (
    LogFileWriter,
    ParentLogger,
    TeeWriter,
    resolve_worker_devices,
    worker_log_display_path,
    worker_log_path,
)
from .axon_test import (
    _declared_checkpoints_from_axon,
    _ensure_checkpoint_model_dir,
    _format_checkpoint_summary_table,
    _format_metric_value,
    _repo_root,
    _run_axon_test_single,
)


@dataclass(frozen=True)
class _BenchmarkPair:
    axon_file: Path
    checkpoint_id: str
    model_dir: Path


@dataclass(frozen=True)
class _BenchmarkWorkerError:
    exc_type: str
    message: str
    traceback_text: str
    captured_output: str


_SUMMARY_FIELDNAMES = [
    "axon",
    "checkpoint",
    "model_dir",
    "masked_top1_eq",
    "masked_max_abs_diff",
    "masked_max_rel_diff",
]

_MAX_BENCHMARK_WORKER_RETRIES = 1


def _summary_sort_key(row: dict[str, Any]) -> tuple[str, int, float, float]:
    masked_top1_eq = row.get("masked_top1_eq")
    if masked_top1_eq is True:
        top1_rank = 0
    elif masked_top1_eq is False:
        top1_rank = 1
    elif masked_top1_eq == "True":
        top1_rank = 0
    elif masked_top1_eq == "False":
        top1_rank = 1
    else:
        top1_rank = 2

    def _num(value: object) -> float:
        if value is None:
            return float("inf")
        try:
            return float(cast(Any, value))
        except Exception:
            return float("inf")

    return (
        str(row.get("checkpoint_id", row.get("checkpoint", ""))),
        top1_rank,
        _num(row.get("masked_max_diff", row.get("masked_max_abs_diff"))),
        _num(row.get("masked_max_rel_diff")),
    )


def _summary_row_from_result(row: dict[str, Any]) -> dict[str, object]:
    masked_top1_eq = row.get("masked_top1_eq")
    if masked_top1_eq is None:
        masked_top1_eq_text = "N/A"
    elif isinstance(masked_top1_eq, bool):
        masked_top1_eq_text = str(masked_top1_eq)
    else:
        masked_top1_eq_text = str(masked_top1_eq)
    return {
        "axon": str(row["axon_file"]),
        "checkpoint": str(row["checkpoint_id"]),
        "model_dir": str(row["weights"]),
        "masked_top1_eq": masked_top1_eq_text,
        "masked_max_abs_diff": _format_metric_value(row.get("masked_max_diff")),
        "masked_max_rel_diff": _format_metric_value(row.get("masked_max_rel_diff")),
    }


def _sanitize_benchmark_result(row: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {
        "axon_file": row["axon_file"],
        "checkpoint_id": row["checkpoint_id"],
        "weights": row["weights"],
        "hf_model_dir": row["hf_model_dir"],
    }
    for key in (
        "hf_time",
        "axon_time",
        "speed_ratio_axon_over_hf",
        "mean_diff",
        "max_diff",
        "last_max_diff",
        "mean_rel_diff",
        "max_rel_diff",
        "top1_eq",
        "masked_mean_diff",
        "masked_max_diff",
        "masked_last_max_diff",
        "masked_mean_rel_diff",
        "masked_max_rel_diff",
        "masked_top1_eq",
        "debug_max_logit_diff",
        "debug_max_rel_diff",
        "debug_top1_eq",
    ):
        if key in row:
            sanitized[key] = row[key]
    return sanitized


def _error_result_for_pair(
    pair: _BenchmarkPair,
    err: BaseException | _BenchmarkWorkerError,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    model_dir = repo_root / "models" / pair.checkpoint_id
    if isinstance(err, _BenchmarkWorkerError):
        error_text = f"ERROR: {err.exc_type}: {err.message}"
    else:
        error_text = f"ERROR: {type(err).__name__}: {err}"
    return {
        "axon_file": pair.axon_file,
        "checkpoint_id": pair.checkpoint_id,
        "weights": model_dir,
        "hf_model_dir": model_dir,
        "masked_top1_eq": "ERROR",
        "masked_max_diff": "ERROR",
        "masked_max_rel_diff": "ERROR",
        "error": error_text,
    }


def _signal_name(exitcode: int) -> str:
    signum = abs(int(exitcode))
    try:
        return signal.Signals(signum).name
    except Exception:
        return f"signal {signum}"


def _initialize_stream_csv(csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_SUMMARY_FIELDNAMES)
        writer.writeheader()


def _append_stream_csv_row(csv_path: Path, row: dict[str, object]) -> None:
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_SUMMARY_FIELDNAMES)
        writer.writerow({key: "" if value is None else str(value) for key, value in row.items()})


def render_axon_benchmark_csv(*, csv_path: Path, table_format: str = "markdown") -> str:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    sorted_rows = sorted(rows, key=_summary_sort_key)
    return _format_checkpoint_summary_table(
        cast(Sequence[dict[str, object]], sorted_rows),
        table_format=table_format,
    )


def _expand_axon_inputs(axon_inputs: Sequence[Path]) -> list[Path]:
    discovered: list[Path] = []
    seen: set[Path] = set()
    for axon_input in axon_inputs:
        resolved_input = Path(axon_input).resolve()
        candidates: list[Path]
        if resolved_input.is_dir():
            candidates = sorted(path.resolve() for path in resolved_input.rglob("*.axon"))
        elif resolved_input.is_file():
            if resolved_input.suffix != ".axon":
                raise ValueError(f"Expected .axon file, got: {resolved_input}")
            candidates = [resolved_input]
        else:
            raise ValueError(f"Path does not exist: {resolved_input}")
        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                discovered.append(candidate)
    if not discovered:
        raise ValueError("No .axon files found in the provided inputs")
    return discovered


def _run_benchmark_pair(
    pair: _BenchmarkPair,
    *,
    repo_root: Path,
    device: str,
    text: str | Sequence[str],
    max_len: int,
    tokenizer: str | None,
    class_name: str,
    main_module: str | None,
    dtype: str,
    model_task: str,
    trace_layers: bool,
    hf_align_bf16_profile: bool,
    hf_align_mask_contract: bool,
    hf_align_position_ids: bool,
    hf_align_add_fp32_accum: bool,
    hf_align_linear_fp32_accum: bool,
    hf_align_norm_fp32: bool,
    compile_hf: bool,
    compile_axon: bool,
    compile_backend: str | None,
    compile_mode: str | None,
    compile_fullgraph: bool,
    compile_dynamic: bool,
    trust_remote_code: bool,
) -> dict[str, Any]:
    model_dir = _ensure_checkpoint_model_dir(repo_root=repo_root, checkpoint_id=pair.checkpoint_id)
    print()
    print("=" * 80)
    print(f"Axon file:      {pair.axon_file}")
    print(f"Checkpoint:     {pair.checkpoint_id}")
    print(f"Model dir:      {model_dir}")
    print("=" * 80)
    print()
    result = _run_axon_test_single(
        axon_file=pair.axon_file,
        weights=model_dir,
        device=device,
        text=text,
        max_len=max_len,
        hf_model_dir=model_dir,
        tokenizer=tokenizer,
        class_name=class_name,
        main_module=main_module,
        dtype=dtype,
        model_task=model_task,
        trace_layers=trace_layers,
        hf_align_bf16_profile=hf_align_bf16_profile,
        hf_align_mask_contract=hf_align_mask_contract,
        hf_align_position_ids=hf_align_position_ids,
        hf_align_add_fp32_accum=hf_align_add_fp32_accum,
        hf_align_linear_fp32_accum=hf_align_linear_fp32_accum,
        hf_align_norm_fp32=hf_align_norm_fp32,
        compile_hf=compile_hf,
        compile_axon=compile_axon,
        compile_backend=compile_backend,
        compile_mode=compile_mode,
        compile_fullgraph=compile_fullgraph,
        compile_dynamic=compile_dynamic,
        trust_remote_code=trust_remote_code,
    )
    enriched = dict(result)
    enriched["axon_file"] = pair.axon_file
    enriched["checkpoint_id"] = pair.checkpoint_id
    enriched["weights"] = model_dir
    enriched["hf_model_dir"] = model_dir
    return _sanitize_benchmark_result(enriched)


def _run_benchmark_worker_loop(
    pair_index: int,
    pair: _BenchmarkPair,
    worker_device: str,
    result_queue: Any,
    common_kwargs: dict[str, Any],
    log_dir: str | None,
) -> None:
    log_path = (
        worker_log_path(
            Path(log_dir),
            axon_name=pair.axon_file.stem.replace(" ", "_"),
            model_name=pair.model_dir.name.replace(" ", "_"),
            pid=os.getpid(),
        )
        if log_dir is not None
        else None
    )
    capture = io.StringIO()
    file_handle: io.TextIOWrapper | None = None
    tee: Any = capture
    try:
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handle = log_path.open("w", encoding="utf-8", buffering=1)
            print(f"log_path={log_path.name}", file=file_handle)
            tee = TeeWriter(capture, LogFileWriter(file_handle))
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            try:
                result = _run_benchmark_pair(pair, device=worker_device, **common_kwargs)
                result_queue.put(
                    (
                        pair_index,
                        result,
                        None,
                        capture.getvalue(),
                        log_path.name if log_path else None,
                    )
                )
            except Exception as exc:
                result_queue.put(
                    (
                        pair_index,
                        None,
                        _BenchmarkWorkerError(
                            exc_type=type(exc).__name__,
                            message=str(exc),
                            traceback_text=traceback.format_exc(),
                            captured_output=capture.getvalue(),
                        ),
                        capture.getvalue(),
                        log_path.name if log_path else None,
                    )
                )
    finally:
        if file_handle is not None:
            file_handle.close()
        gc.collect()


def _run_benchmark_jobs_parallel(
    pairs: list[_BenchmarkPair],
    *,
    processes: int,
    device: str,
    log_dir: Path | None,
    stream_csv: Path | None,
    common_kwargs: dict[str, Any],
) -> list[dict[str, Any]]:
    worker_devices = resolve_worker_devices(device, processes)
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    active_processes: dict[int, Any] = {}
    retry_counts: dict[int, int] = {}
    results_by_index: list[dict[str, Any] | None] = [None] * len(pairs)
    progress = tqdm(total=len(pairs), desc="synapse axon-benchmark", unit="pair")
    parent_logger = ParentLogger(log_dir)
    pending_indices = list(range(len(pairs)))
    next_device_index = 0
    if parent_logger.path is not None:
        print(f"Parent log: {parent_logger.path}")
    parent_logger.log(
        f"run_start total_pairs={len(pairs)} devices={worker_devices} max_concurrent={len(worker_devices)}"
    )

    def _spawn_next_pair(pair_index: int) -> None:
        nonlocal next_device_index
        pair = pairs[pair_index]
        worker_device = worker_devices[next_device_index % len(worker_devices)]
        next_device_index += 1
        process = ctx.Process(
            target=_run_benchmark_worker_loop,
            args=(
                pair_index,
                pair,
                worker_device,
                result_queue,
                common_kwargs,
                str(log_dir) if log_dir else None,
            ),
            daemon=False,
        )
        process.start()
        active_processes[pair_index] = process
        parent_logger.log(
            "child_start "
            f"pair_index={pair_index} pid={process.pid} device={worker_device} "
            f"axon={pair.axon_file.name} checkpoint={pair.checkpoint_id} model_dir={pair.model_dir} "
            f"log_path={worker_log_display_path(log_dir, axon_name=pair.axon_file.stem.replace(' ', '_'), model_name=pair.model_dir.name.replace(' ', '_'), pid=process.pid)}"
        )

    def _requeue_pair(pair_index: int, *, reason: str) -> bool:
        retry_count = retry_counts.get(pair_index, 0)
        if retry_count >= _MAX_BENCHMARK_WORKER_RETRIES:
            return False
        retry_counts[pair_index] = retry_count + 1
        pending_indices.insert(0, pair_index)
        pair = pairs[pair_index]
        parent_logger.log(
            "child_retry "
            f"pair_index={pair_index} retry={retry_counts[pair_index]} "
            f"axon={pair.axon_file.name} checkpoint={pair.checkpoint_id} model_dir={pair.model_dir} "
            f"reason={reason}"
        )
        return True

    try:
        while pending_indices or active_processes:
            while pending_indices and len(active_processes) < len(worker_devices):
                _spawn_next_pair(pending_indices.pop(0))

            try:
                pair_index, result, error, captured_output, log_path = result_queue.get(timeout=1.0)
            except queue.Empty:
                for active_pair_index, process in list(active_processes.items()):
                    if process.is_alive():
                        continue
                    pair = pairs[int(active_pair_index)]
                    process.join(timeout=0.1)
                    dead_exitcode = int(process.exitcode or 0)
                    log_path = worker_log_display_path(
                        log_dir,
                        axon_name=pair.axon_file.stem.replace(" ", "_"),
                        model_name=pair.model_dir.name.replace(" ", "_"),
                        pid=process.pid,
                    )
                    status = "abnormal" if dead_exitcode != 0 else "missing_result"
                    parent_logger.log(
                        "child_finish "
                        f"pair_index={active_pair_index} pid={process.pid} status={status} exitcode={dead_exitcode} "
                        f"axon={pair.axon_file.name} checkpoint={pair.checkpoint_id} "
                        f"model_dir={pair.model_dir} log_path={log_path}"
                    )
                    active_processes.pop(active_pair_index, None)
                    reason = (
                        (
                            f"worker exited abnormally with exit code {dead_exitcode}"
                            + (f" ({_signal_name(dead_exitcode)})" if dead_exitcode < 0 else "")
                        )
                        if dead_exitcode != 0
                        else "worker exited without publishing a result"
                    )
                    if _requeue_pair(int(active_pair_index), reason=reason):
                        continue
                    error_result = _error_result_for_pair(
                        pair,
                        RuntimeError(reason),
                        repo_root=cast(Path, common_kwargs["repo_root"]),
                    )
                    results_by_index[int(active_pair_index)] = error_result
                    if stream_csv is not None:
                        _append_stream_csv_row(stream_csv, _summary_row_from_result(error_result))
                    progress.update(1)
                    parent_logger.log(
                        "child_result "
                        f"pair_index={active_pair_index} status=error_result "
                        f"axon={pair.axon_file.name} checkpoint={pair.checkpoint_id}"
                    )
                continue

            pair = pairs[int(pair_index)]
            process = active_processes.pop(int(pair_index), None)
            exitcode: int | None = None
            if process is not None:
                process.join(timeout=5.0)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5.0)
                exitcode = int(process.exitcode or 0)
            if error is not None:
                assert isinstance(error, _BenchmarkWorkerError)
                if captured_output:
                    print(captured_output.rstrip())
                if log_path is not None:
                    print(f"log file: {log_path}")
                parent_logger.log(
                    "child_finish "
                    f"pair_index={pair_index} pid={getattr(process, 'pid', 'unknown')} status=error "
                    f"exitcode={exitcode} axon={pair.axon_file.name} checkpoint={pair.checkpoint_id} "
                    f"model_dir={pair.model_dir} log_path={log_path} error={error.exc_type}:{error.message}"
                )
                if _requeue_pair(
                    int(pair_index),
                    reason=f"{error.exc_type}: {error.message}",
                ):
                    continue
                error_result = _error_result_for_pair(
                    pair,
                    error,
                    repo_root=cast(Path, common_kwargs["repo_root"]),
                )
                results_by_index[int(pair_index)] = error_result
                if stream_csv is not None:
                    _append_stream_csv_row(stream_csv, _summary_row_from_result(error_result))
                progress.update(1)
                parent_logger.log(
                    "child_result "
                    f"pair_index={pair_index} status=error_result "
                    f"axon={pair.axon_file.name} checkpoint={pair.checkpoint_id}"
                )
                continue
            assert isinstance(result, dict)
            results_by_index[int(pair_index)] = result
            if stream_csv is not None:
                _append_stream_csv_row(stream_csv, _summary_row_from_result(result))
            progress.update(1)
            parent_logger.log(
                "child_finish "
                f"pair_index={pair_index} pid={getattr(process, 'pid', 'unknown')} status=success "
                f"exitcode={exitcode} axon={pair.axon_file.name} checkpoint={pair.checkpoint_id} "
                f"model_dir={pair.model_dir} log_path={log_path}"
            )
    finally:
        progress.close()
        for process in active_processes.values():
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
        result_queue.close()
        result_queue.join_thread()
        parent_logger.log(
            f"run_finish total_rows={len([r for r in results_by_index if r is not None])}"
        )
        parent_logger.close()

    return [result for result in results_by_index if result is not None]


def run_axon_benchmark(
    *,
    axon_files: Sequence[Path],
    device: str = "cpu",
    processes: int = 1,
    text: str | Sequence[str] = ("The future of AI is", "Hello World"),
    max_len: int = 32,
    tokenizer: str | None = None,
    class_name: str = "AxonGeneratedModel",
    main_module: str | None = None,
    dtype: str = "float32",
    model_task: str = "auto",
    trace_layers: bool = False,
    hf_align_bf16_profile: bool = False,
    hf_align_mask_contract: bool = False,
    hf_align_position_ids: bool = False,
    hf_align_add_fp32_accum: bool = False,
    hf_align_linear_fp32_accum: bool = False,
    hf_align_norm_fp32: bool = False,
    compile_hf: bool = False,
    compile_axon: bool = False,
    compile_backend: str | None = None,
    compile_mode: str | None = None,
    compile_fullgraph: bool = False,
    compile_dynamic: bool = False,
    trust_remote_code: bool = False,
    table_format: str = "markdown",
    log_dir: Path | None = None,
    stream_csv: Path | None = None,
) -> dict[str, Any]:
    repo_root = _repo_root()
    pairs: list[_BenchmarkPair] = []
    for resolved_axon_file in _expand_axon_inputs(axon_files):
        checkpoints = _declared_checkpoints_from_axon(
            axon_file=resolved_axon_file,
            main_module=main_module,
        )
        for checkpoint_id in checkpoints:
            pairs.append(
                _BenchmarkPair(
                    axon_file=resolved_axon_file,
                    checkpoint_id=checkpoint_id,
                    model_dir=repo_root / "models" / checkpoint_id,
                )
            )

    common_kwargs = {
        "repo_root": repo_root,
        "text": text,
        "max_len": max_len,
        "tokenizer": tokenizer,
        "class_name": class_name,
        "main_module": main_module,
        "dtype": dtype,
        "model_task": model_task,
        "trace_layers": trace_layers,
        "hf_align_bf16_profile": hf_align_bf16_profile,
        "hf_align_mask_contract": hf_align_mask_contract,
        "hf_align_position_ids": hf_align_position_ids,
        "hf_align_add_fp32_accum": hf_align_add_fp32_accum,
        "hf_align_linear_fp32_accum": hf_align_linear_fp32_accum,
        "hf_align_norm_fp32": hf_align_norm_fp32,
        "compile_hf": compile_hf,
        "compile_axon": compile_axon,
        "compile_backend": compile_backend,
        "compile_mode": compile_mode,
        "compile_fullgraph": compile_fullgraph,
        "compile_dynamic": compile_dynamic,
        "trust_remote_code": trust_remote_code,
    }

    if stream_csv is not None:
        _initialize_stream_csv(stream_csv)

    if processes <= 1:
        results = []
        for pair in pairs:
            try:
                result = _run_benchmark_pair(
                    pair,
                    repo_root=repo_root,
                    device=device,
                    text=text,
                    max_len=max_len,
                    tokenizer=tokenizer,
                    class_name=class_name,
                    main_module=main_module,
                    dtype=dtype,
                    model_task=model_task,
                    trace_layers=trace_layers,
                    hf_align_bf16_profile=hf_align_bf16_profile,
                    hf_align_mask_contract=hf_align_mask_contract,
                    hf_align_position_ids=hf_align_position_ids,
                    hf_align_add_fp32_accum=hf_align_add_fp32_accum,
                    hf_align_linear_fp32_accum=hf_align_linear_fp32_accum,
                    hf_align_norm_fp32=hf_align_norm_fp32,
                    compile_hf=compile_hf,
                    compile_axon=compile_axon,
                    compile_backend=compile_backend,
                    compile_mode=compile_mode,
                    compile_fullgraph=compile_fullgraph,
                    compile_dynamic=compile_dynamic,
                    trust_remote_code=trust_remote_code,
                )
            except Exception as exc:
                result = _error_result_for_pair(pair, exc, repo_root=repo_root)
            results.append(result)
            if stream_csv is not None:
                _append_stream_csv_row(stream_csv, _summary_row_from_result(result))
    else:
        results = _run_benchmark_jobs_parallel(
            pairs,
            processes=processes,
            device=device,
            log_dir=log_dir,
            stream_csv=stream_csv,
            common_kwargs=common_kwargs,
        )

    sorted_results = sorted(results, key=_summary_sort_key)

    summary_rows = [_summary_row_from_result(row) for row in sorted_results]
    print()
    print(_format_checkpoint_summary_table(summary_rows, table_format=table_format))
    return {"results": results}


__all__ = ["run_axon_benchmark", "render_axon_benchmark_csv"]
