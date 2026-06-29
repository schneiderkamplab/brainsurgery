from __future__ import annotations

import contextlib
import csv
import gc
import io
import json
import math
import multiprocessing as mp
import os
import queue
import signal
import sys
import traceback
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import safetensors
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
    _resolve_benchmark_mode,
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


@dataclass(frozen=True)
class _ParamCountSkip:
    pair: _BenchmarkPair
    param_count: int
    is_exact: bool


@dataclass(frozen=True)
class _WorkerSpec:
    run_device: str
    cuda_visible_devices: str | None = None
    label: str | None = None


_SUMMARY_FIELDNAMES = [
    "axon",
    "checkpoint",
    "model_dir",
    "fallback",
    "masked_top1_eq",
    "masked_max_abs_diff",
    "masked_max_rel_diff",
]

_MAX_BENCHMARK_WORKER_RETRIES = 1


def _cuda_visible_tokens_for_indices(indices: Sequence[int]) -> list[str]:
    parent_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if parent_visible:
        tokens = [token.strip() for token in parent_visible.split(",") if token.strip()]
        if tokens:
            resolved: list[str] = []
            for index in indices:
                if index < 0 or index >= len(tokens):
                    raise ValueError(
                        f"visible CUDA index {index} is outside CUDA_VISIBLE_DEVICES={parent_visible!r}"
                    )
                resolved.append(tokens[index])
            return resolved
    return [str(index) for index in indices]


def _resolve_pipeline_worker_specs(
    *,
    backend: str,
    device: str,
    processes: int,
    pipeline_parallel_size: int | None,
) -> list[_WorkerSpec]:
    normalized = str(device).strip().lower()
    if not (normalized == "cuda" or normalized.startswith("cuda:") or normalized == "auto"):
        raise ValueError(
            f"axon_backend={backend!r} requires a CUDA device target "
            "(use --device cuda or --device cuda:<index>)"
        )
    if processes <= 0:
        raise ValueError("processes must be >= 1")
    if pipeline_parallel_size is not None and pipeline_parallel_size <= 0:
        raise ValueError("pipeline_parallel_size must be >= 1 when provided")

    try:
        import torch
    except Exception as exc:
        raise ValueError("torch is required for pipeline worker device resolution") from exc
    if not torch.cuda.is_available():
        raise ValueError(f"axon_backend={backend!r} requires CUDA, but CUDA is unavailable")
    device_count = int(torch.cuda.device_count())
    if device_count <= 0:
        raise ValueError(f"axon_backend={backend!r} requires at least one CUDA device")

    if normalized == "auto":
        start_index = 0
    elif normalized.startswith("cuda:"):
        index_text = normalized.split(":", 1)[1].strip()
        if not index_text.isdigit():
            raise ValueError(f"invalid CUDA device specifier: {device!r}")
        start_index = int(index_text)
    else:
        start_index = 0

    if start_index < 0 or start_index >= device_count:
        raise ValueError(
            f"requested start device cuda:{start_index}, but visible CUDA range is "
            f"0..{max(0, device_count - 1)}"
        )

    available_from_start = device_count - start_index
    if pipeline_parallel_size is None:
        if available_from_start % processes != 0:
            raise ValueError(
                "cannot infer pipeline parallel size: visible CUDA count is not divisible by "
                f"processes ({available_from_start} GPUs for {processes} processes). "
                "Pass --pipeline-parallel-size/--pp explicitly."
            )
        pp_size = available_from_start // processes
    else:
        pp_size = pipeline_parallel_size

    required = processes * pp_size
    if required > available_from_start:
        raise ValueError(
            f"requested processes={processes} with pp={pp_size} requires {required} GPUs "
            f"starting at cuda:{start_index}, but only {available_from_start} are available"
        )

    worker_specs: list[_WorkerSpec] = []
    for worker_index in range(processes):
        group_start = start_index + worker_index * pp_size
        group_indices = list(range(group_start, group_start + pp_size))
        group = _cuda_visible_tokens_for_indices(group_indices)
        visible = ",".join(group)
        label = ",".join(f"cuda:{idx}" for idx in group_indices)
        worker_specs.append(
            _WorkerSpec(
                run_device="cuda",
                cuda_visible_devices=visible,
                label=label,
            )
        )
    return worker_specs


def _resolve_tinygrad_worker_specs(*, device: str, processes: int) -> list[_WorkerSpec] | None:
    normalized = str(device).strip().lower()
    if processes <= 1 or not (normalized == "cuda" or normalized.startswith("cuda:") or normalized == "auto"):
        return None
    return _resolve_pipeline_worker_specs(
        backend="codegen2-tinygrad",
        device=device,
        processes=processes,
        pipeline_parallel_size=1,
    )


def _resolve_mlx_worker_specs(*, device: str, processes: int) -> list[_WorkerSpec] | None:
    if processes <= 1:
        return None
    return None


def _estimate_model_param_count(model_dir: Path) -> tuple[int, bool] | None:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict):
            return None
        shard_names = sorted({str(name) for name in weight_map.values()})
        shard_paths = [model_dir / shard_name for shard_name in shard_names]
        if not all(path.exists() for path in shard_paths):
            return None
    else:
        single_path = model_dir / "model.safetensors"
        if single_path.exists():
            shard_paths = [single_path]
        else:
            shard_paths = sorted(model_dir.glob("*.safetensors"))
            if not shard_paths:
                return None

    total_params = 0
    for shard_path in shard_paths:
        handle = safetensors.safe_open(str(shard_path), framework="pt")
        for key in handle.keys():
            total_params += int(math.prod(handle.get_slice(key).get_shape()))
    return total_params, True


def _param_count_cache_path(model_dir: Path) -> Path:
    return model_dir / "._param_count.json"


def _read_param_count_cache(model_dir: Path) -> tuple[int, bool] | None:
    cache_path = _param_count_cache_path(model_dir)
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(payload, int) and payload > 0:
        return payload, True
    if not isinstance(payload, dict):
        return None
    param_count = payload.get("param_count")
    if not isinstance(param_count, int) or param_count <= 0:
        return None
    is_exact = payload.get("is_exact", True)
    return param_count, bool(is_exact)


def _write_param_count_cache(
    model_dir: Path,
    *,
    param_count: int,
    is_exact: bool,
    method: str,
) -> None:
    if param_count <= 0:
        return
    payload = {
        "param_count": int(param_count),
        "is_exact": bool(is_exact),
        "method": str(method),
    }
    cache_path = _param_count_cache_path(model_dir)
    try:
        cache_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    except Exception:
        return


def _estimate_model_param_count_via_cpu_load(model_dir: Path) -> tuple[int, bool] | None:
    try:
        from transformers import (
            AutoModel,
            AutoModelForCausalLM,
            AutoModelForMaskedLM,
            AutoModelForSeq2SeqLM,
        )
    except Exception:
        return None

    constructors: tuple[Any, ...] = (
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        AutoModelForMaskedLM,
        AutoModel,
    )
    for constructor in constructors:
        model: Any | None = None
        try:
            model = constructor.from_pretrained(
                str(model_dir),
                local_files_only=True,
                trust_remote_code=True,
            )
            param_count = int(sum(int(param.numel()) for param in model.parameters()))
            if param_count > 0:
                return param_count, True
        except Exception:
            continue
        finally:
            if model is not None:
                with contextlib.suppress(Exception):
                    del model
            gc.collect()
    return None


def _estimate_model_param_count_lower_bound(model_dir: Path) -> tuple[int, bool] | None:
    cached = _read_param_count_cache(model_dir)
    if cached is not None:
        return cached

    exact_from_safetensors = _estimate_model_param_count(model_dir)
    if exact_from_safetensors is not None:
        _write_param_count_cache(
            model_dir,
            param_count=exact_from_safetensors[0],
            is_exact=exact_from_safetensors[1],
            method="safetensors_shape",
        )
        return exact_from_safetensors

    exact_from_cpu_load = _estimate_model_param_count_via_cpu_load(model_dir)
    if exact_from_cpu_load is not None:
        _write_param_count_cache(
            model_dir,
            param_count=exact_from_cpu_load[0],
            is_exact=exact_from_cpu_load[1],
            method="cpu_model_load",
        )
        return exact_from_cpu_load
    return None


def _apply_billions_params_filter(
    pairs: list[_BenchmarkPair],
    *,
    min_billions_params: float | None,
    max_billions_params: float | None,
) -> tuple[list[_BenchmarkPair], list[_ParamCountSkip]]:
    if min_billions_params is None and max_billions_params is None:
        return pairs, []
    if min_billions_params is not None and min_billions_params <= 0:
        raise ValueError("min_billions_params must be > 0 when provided")
    if max_billions_params is not None and max_billions_params <= 0:
        raise ValueError("max_billions_params must be > 0 when provided")
    if (
        min_billions_params is not None
        and max_billions_params is not None
        and min_billions_params > max_billions_params
    ):
        raise ValueError("min_billions_params must be <= max_billions_params when both provided")

    min_params = (
        int(min_billions_params * 1_000_000_000) if min_billions_params is not None else None
    )
    max_params = (
        int(max_billions_params * 1_000_000_000) if max_billions_params is not None else None
    )
    kept: list[_BenchmarkPair] = []
    skipped: list[_ParamCountSkip] = []
    for pair in pairs:
        estimate = _estimate_model_param_count_lower_bound(pair.model_dir)
        if estimate is None:
            # If no local estimate is available, keep the pair rather than incorrectly dropping it.
            kept.append(pair)
            continue
        param_count, is_exact = estimate
        # Fast exact-safe upper-bound cull using index/bytes lower-bound estimate.
        if max_params is not None and param_count > max_params:
            skipped.append(_ParamCountSkip(pair=pair, param_count=param_count, is_exact=is_exact))
            continue
        # If estimate is not exact and min-bound might still be satisfiable, refine to exact count.
        if not is_exact and min_params is not None and param_count < min_params:
            exact = _estimate_model_param_count(pair.model_dir)
            if exact is not None:
                param_count, is_exact = exact
        if min_params is not None and param_count < min_params:
            skipped.append(_ParamCountSkip(pair=pair, param_count=param_count, is_exact=is_exact))
            continue
        kept.append(pair)
    return kept, skipped


def _format_billions_params(param_count: int, *, is_exact: bool) -> str:
    prefix = "" if is_exact else ">="
    return f"{prefix}{param_count / 1_000_000_000:.3f}B"


def _print_param_count_skips(
    skipped_pairs: list[_ParamCountSkip],
    *,
    min_billions_params: float | None,
    max_billions_params: float | None,
) -> None:
    if not skipped_pairs:
        return
    if min_billions_params is not None and max_billions_params is not None:
        bound_desc = (
            f"--min-billion-parameters/--max-billion-parameters "
            f"({min_billions_params:g}B..{max_billions_params:g}B)"
        )
    elif min_billions_params is not None:
        bound_desc = f"--min-billion-parameters ({min_billions_params:g}B)"
    else:
        bound_desc = f"--max-billion-parameters ({max_billions_params:g}B)"
    print(f"Skipped due to {bound_desc}: {len(skipped_pairs)}")
    for skipped in skipped_pairs:
        print(
            f"  - {skipped.pair.axon_file.name} | {skipped.pair.checkpoint_id} | "
            f"{_format_billions_params(skipped.param_count, is_exact=skipped.is_exact)}"
        )
    print()


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
        "fallback": str(row.get("fallback", "none")),
        "masked_top1_eq": masked_top1_eq_text,
        "masked_max_abs_diff": _format_metric_value(row.get("masked_max_diff")),
        "masked_max_rel_diff": _format_metric_value(row.get("masked_max_rel_diff")),
    }


def _summary_row_from_pair(pair: _BenchmarkPair, *, fallback: str) -> dict[str, object]:
    return {
        "axon": str(pair.axon_file),
        "checkpoint": pair.checkpoint_id,
        "model_dir": str(pair.model_dir),
        "fallback": fallback,
        "masked_top1_eq": fallback,
        "masked_max_abs_diff": fallback,
        "masked_max_rel_diff": fallback,
    }


def _sanitize_benchmark_result(row: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {
        "axon_file": row["axon_file"],
        "checkpoint_id": row["checkpoint_id"],
        "weights": row["weights"],
        "hf_model_dir": row["hf_model_dir"],
    }
    for key in (
        "fallback",
        "hf_device",
        "axon_device",
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
        "fallback": "none",
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


def _worker_result_path(log_path: Path | None) -> Path | None:
    if log_path is None:
        return None
    return log_path.with_suffix(".result.json")


def _write_worker_result_file(log_path: Path | None, result: dict[str, Any]) -> None:
    result_path = _worker_result_path(log_path)
    if result_path is None:
        return
    serializable = {key: str(value) if isinstance(value, Path) else value for key, value in result.items()}
    tmp_path = result_path.with_suffix(result_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, sort_keys=True)
    tmp_path.replace(result_path)


def _read_worker_result_file(log_path: Path | None) -> dict[str, Any] | None:
    result_path = _worker_result_path(log_path)
    if result_path is None or not result_path.exists():
        return None
    with result_path.open("r", encoding="utf-8") as handle:
        row = json.load(handle)
    for key in ("axon_file", "weights", "hf_model_dir"):
        if key in row:
            row[key] = Path(row[key])
    return cast(dict[str, Any], row)


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


def _matches_any_selector(path: Path, selectors: set[str]) -> bool:
    if not selectors:
        return False
    haystacks = {
        str(path),
        str(path.resolve()),
        path.name,
        path.stem,
    }
    for selector in selectors:
        if any(selector in haystack for haystack in haystacks):
            return True
    return False


def _run_benchmark_pair(
    pair: _BenchmarkPair,
    *,
    repo_root: Path,
    device: str,
    text: str | Sequence[str],
    max_len: int,
    tokenizer: str | None,
    class_name: str,
    dtype: str,
    model_task: str,
    benchmark_mode: str,
    trace_layers: bool,
    hf_align_bf16_profile: bool,
    hf_align_mask_contract: bool,
    hf_align_position_ids: bool,
    hf_align_add_fp32_accum: bool,
    hf_align_linear_fp32_accum: bool,
    hf_align_norm_fp32: bool,
    hf_attn_implementation: str | None,
    hf_experts_implementation: str | None,
    compile_hf: bool,
    compile_axon: bool,
    compile_backend: str | None,
    compile_mode: str | None,
    compile_fullgraph: bool,
    compile_dynamic: bool,
    trust_remote_code: bool,
    axon_backend: str,
    axon_typechecker: str,
    optimize_ast: bool,
    optimize_graph: bool,
    graph_backend_intrinsics: str | None,
    skip_hf: bool = False,
    hf_strict_dtype: bool = False,
    oom_cpu_fallback: bool = True,
    profile_axon: bool = False,
    profile_axon_top_n: int = 40,
    metal_capture: bool = False,
    forward_warmup: int = 0,
    forward_repeat: int = 1,
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
        dtype=dtype,
        model_task=model_task,
        benchmark_mode=benchmark_mode,
        trace_layers=trace_layers,
        hf_align_bf16_profile=hf_align_bf16_profile,
        hf_align_mask_contract=hf_align_mask_contract,
        hf_align_position_ids=hf_align_position_ids,
        hf_align_add_fp32_accum=hf_align_add_fp32_accum,
        hf_align_linear_fp32_accum=hf_align_linear_fp32_accum,
        hf_align_norm_fp32=hf_align_norm_fp32,
        hf_attn_implementation=hf_attn_implementation,
        hf_experts_implementation=hf_experts_implementation,
        compile_hf=compile_hf,
        compile_axon=compile_axon,
        compile_backend=compile_backend,
        compile_mode=compile_mode,
        compile_fullgraph=compile_fullgraph,
        compile_dynamic=compile_dynamic,
        trust_remote_code=trust_remote_code,
        axon_backend=axon_backend,
        axon_typechecker=axon_typechecker,
        optimize_ast=optimize_ast,
        optimize_graph=optimize_graph,
        graph_backend_intrinsics=graph_backend_intrinsics,
        skip_hf=skip_hf,
        hf_strict_dtype=hf_strict_dtype,
        oom_cpu_fallback=oom_cpu_fallback,
        profile_axon=profile_axon,
        profile_axon_top_n=profile_axon_top_n,
        metal_capture=metal_capture,
        forward_warmup=forward_warmup,
        forward_repeat=forward_repeat,
    )
    enriched = dict(result)
    enriched["axon_file"] = pair.axon_file
    enriched["checkpoint_id"] = pair.checkpoint_id
    enriched["weights"] = model_dir
    enriched["hf_model_dir"] = model_dir
    return _sanitize_benchmark_result(enriched)


def _run_benchmark_jobs_serial(
    pairs: list[_BenchmarkPair],
    *,
    device: str,
    log_dir: Path | None,
    stream_csv: Path | None,
    common_kwargs: dict[str, Any],
    worker_cuda_visible_devices: str | None = None,
    debug_errors: bool = False,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    parent_logger = ParentLogger(log_dir)
    if parent_logger.path is not None:
        print(f"Parent log: {parent_logger.path}")
    device_label = (
        f"{device} (CUDA_VISIBLE_DEVICES={worker_cuda_visible_devices})"
        if worker_cuda_visible_devices is not None
        else device
    )
    parent_logger.log(
        f"run_start total_pairs={len(pairs)} devices={[device_label]} max_concurrent=1"
    )
    progress = tqdm(total=len(pairs), desc="synapse axon-benchmark", unit="pair")
    previous_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if worker_cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = worker_cuda_visible_devices
    try:
        for pair_index, pair in enumerate(pairs):
            log_path = worker_log_path(
                log_dir,
                axon_name=pair.axon_file.stem.replace(" ", "_"),
                model_name=pair.model_dir.name.replace(" ", "_"),
                pid=os.getpid(),
            )
            parent_logger.log(
                "child_start "
                f"pair_index={pair_index} pid={os.getpid()} device={device} "
                f"axon={pair.axon_file.name} checkpoint={pair.checkpoint_id} model_dir={pair.model_dir} "
                f"log_path={worker_log_display_path(log_dir, axon_name=pair.axon_file.stem.replace(' ', '_'), model_name=pair.model_dir.name.replace(' ', '_'), pid=os.getpid())}"
            )
            file_handle: io.TextIOWrapper | None = None
            tee_stdout: Any = sys.stdout
            tee_stderr: Any = sys.stderr
            try:
                if log_path is not None:
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    file_handle = log_path.open("w", encoding="utf-8", buffering=1)
                    print(f"log_path={log_path.name}", file=file_handle)
                    tee_stdout = TeeWriter(sys.stdout, LogFileWriter(file_handle))
                    tee_stderr = TeeWriter(sys.stderr, LogFileWriter(file_handle))
                with contextlib.redirect_stdout(tee_stdout), contextlib.redirect_stderr(tee_stderr):
                    if worker_cuda_visible_devices is not None:
                        print(f"worker.CUDA_VISIBLE_DEVICES={worker_cuda_visible_devices}")
                    try:
                        result = _run_benchmark_pair(pair, device=device, **common_kwargs)
                    except Exception as exc:
                        if debug_errors:
                            print(
                                "Benchmark pair failed:",
                                f"axon={pair.axon_file}",
                                f"checkpoint={pair.checkpoint_id}",
                            )
                            print(traceback.format_exc())
                        result = _error_result_for_pair(
                            pair, exc, repo_root=cast(Path, common_kwargs["repo_root"])
                        )
                    print(f"result.axon={pair.axon_file.name}")
                    print(f"result.checkpoint={pair.checkpoint_id}")
                    print(f"result.model_dir={result['weights']}")
                    print(f"result.fallback={result.get('fallback', 'none')}")
                    print(f"result.masked_top1_eq={result.get('masked_top1_eq')}")
                    print(f"result.masked_max_abs_diff={result.get('masked_max_diff')}")
                    print(f"result.masked_max_rel_diff={result.get('masked_max_rel_diff')}")
                results.append(result)
                if stream_csv is not None:
                    _append_stream_csv_row(stream_csv, _summary_row_from_result(result))
                parent_logger.log(
                    "child_finish "
                    f"pair_index={pair_index} pid={os.getpid()} status=ok "
                    f"masked_top1_eq={result.get('masked_top1_eq')} "
                    f"masked_max_abs_diff={result.get('masked_max_diff')} "
                    f"masked_max_rel_diff={result.get('masked_max_rel_diff')}"
                )
            finally:
                if file_handle is not None:
                    file_handle.close()
            progress.update(1)
        parent_logger.log(f"run_finish total_rows={len(results)}")
    finally:
        if worker_cuda_visible_devices is not None:
            if previous_cuda_visible_devices is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = previous_cuda_visible_devices
        progress.close()
        parent_logger.close()
    return results


def _run_benchmark_worker_loop(
    pair_index: int,
    pair: _BenchmarkPair,
    worker_device: str,
    worker_cuda_visible_devices: str | None,
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
    previous_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        if worker_cuda_visible_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = worker_cuda_visible_devices
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handle = log_path.open("w", encoding="utf-8", buffering=1)
            print(f"log_path={log_path.name}", file=file_handle)
            tee = TeeWriter(capture, LogFileWriter(file_handle))
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            if worker_cuda_visible_devices is not None:
                print(f"worker.CUDA_VISIBLE_DEVICES={worker_cuda_visible_devices}")
            try:
                result = _run_benchmark_pair(pair, device=worker_device, **common_kwargs)
                print(f"result.axon={pair.axon_file.name}")
                print(f"result.checkpoint={pair.checkpoint_id}")
                print(f"result.model_dir={result['weights']}")
                print(f"result.fallback={result.get('fallback', 'none')}")
                print(f"result.masked_top1_eq={result.get('masked_top1_eq')}")
                print(f"result.masked_max_abs_diff={result.get('masked_max_diff')}")
                print(f"result.masked_max_rel_diff={result.get('masked_max_rel_diff')}")
                _write_worker_result_file(log_path, result)
                result_queue.put(
                    (
                        pair_index,
                        result,
                        None,
                        "",
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
        if worker_cuda_visible_devices is not None:
            if previous_cuda_visible_devices is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = previous_cuda_visible_devices
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
    worker_specs: list[_WorkerSpec] | None = None,
    debug_errors: bool = False,
) -> list[dict[str, Any]]:
    resolved_worker_specs = worker_specs
    if resolved_worker_specs is None:
        worker_devices = resolve_worker_devices(device, processes)
        resolved_worker_specs = [
            _WorkerSpec(run_device=item, label=item) for item in worker_devices
        ]
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    active_processes: dict[int, Any] = {}
    active_worker_specs: dict[int, int] = {}
    retry_counts: dict[int, int] = {}
    results_by_index: list[dict[str, Any] | None] = [None] * len(pairs)
    progress = tqdm(total=len(pairs), desc="synapse axon-benchmark", unit="pair")
    parent_logger = ParentLogger(log_dir)
    pending_indices = list(range(len(pairs)))
    next_device_index = 0
    if parent_logger.path is not None:
        print(f"Parent log: {parent_logger.path}")
    parent_logger.log(
        "run_start "
        f"total_pairs={len(pairs)} "
        f"devices={[spec.label or spec.run_device for spec in resolved_worker_specs]} "
        f"max_concurrent={len(resolved_worker_specs)}"
    )

    def _next_free_worker_spec_index() -> int | None:
        nonlocal next_device_index
        active_spec_indices = set(active_worker_specs.values())
        if len(active_spec_indices) >= len(resolved_worker_specs):
            return None
        for offset in range(len(resolved_worker_specs)):
            spec_index = (next_device_index + offset) % len(resolved_worker_specs)
            if spec_index not in active_spec_indices:
                next_device_index = (spec_index + 1) % len(resolved_worker_specs)
                return spec_index
        return None

    def _spawn_next_pair(pair_index: int) -> bool:
        pair = pairs[pair_index]
        worker_spec_index = _next_free_worker_spec_index()
        if worker_spec_index is None:
            return False
        worker_spec = resolved_worker_specs[worker_spec_index]
        process = ctx.Process(
            target=_run_benchmark_worker_loop,
            args=(
                pair_index,
                pair,
                worker_spec.run_device,
                worker_spec.cuda_visible_devices,
                result_queue,
                common_kwargs,
                str(log_dir) if log_dir else None,
            ),
            daemon=False,
        )
        process.start()
        active_processes[pair_index] = process
        active_worker_specs[pair_index] = worker_spec_index
        parent_logger.log(
            "child_start "
            f"pair_index={pair_index} pid={process.pid} device={worker_spec.label or worker_spec.run_device} "
            f"axon={pair.axon_file.name} checkpoint={pair.checkpoint_id} model_dir={pair.model_dir} "
            f"log_path={worker_log_display_path(log_dir, axon_name=pair.axon_file.stem.replace(' ', '_'), model_name=pair.model_dir.name.replace(' ', '_'), pid=process.pid)}"
        )
        return True

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
            while pending_indices and len(active_processes) < len(resolved_worker_specs):
                next_pair_index = pending_indices.pop(0)
                if not _spawn_next_pair(next_pair_index):
                    pending_indices.insert(0, next_pair_index)
                    break

            try:
                pair_index, result, error, captured_output, log_path = result_queue.get(timeout=1.0)
            except queue.Empty:
                for active_pair_index, process in list(active_processes.items()):
                    pair = pairs[int(active_pair_index)]
                    actual_log_path = worker_log_path(
                        log_dir,
                        axon_name=pair.axon_file.stem.replace(" ", "_"),
                        model_name=pair.model_dir.name.replace(" ", "_"),
                        pid=process.pid,
                    )
                    published_result = _read_worker_result_file(actual_log_path)
                    if published_result is not None:
                        active_processes.pop(active_pair_index, None)
                        active_worker_specs.pop(active_pair_index, None)
                        process.join(timeout=1.0)
                        if process.is_alive():
                            process.terminate()
                            process.join(timeout=5.0)
                        results_by_index[int(active_pair_index)] = published_result
                        if stream_csv is not None:
                            _append_stream_csv_row(stream_csv, _summary_row_from_result(published_result))
                        progress.update(1)
                        parent_logger.log(
                            "child_finish "
                            f"pair_index={active_pair_index} pid={process.pid} status=success_result_file "
                            f"exitcode={process.exitcode} axon={pair.axon_file.name} checkpoint={pair.checkpoint_id} "
                            f"model_dir={pair.model_dir} log_path={actual_log_path.name if actual_log_path else None}"
                        )
                        continue
                    if process.is_alive():
                        continue
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
                    active_worker_specs.pop(active_pair_index, None)
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
            active_worker_specs.pop(int(pair_index), None)
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
                if debug_errors and error.traceback_text:
                    print("Worker traceback:")
                    print(error.traceback_text.rstrip())
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
    checkpoints: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    device: str = "cpu",
    processes: int = 1,
    text: str | Sequence[str] = ("The future of AI is", "Hello World"),
    max_len: int = 32,
    tokenizer: str | None = None,
    class_name: str = "AxonGeneratedModel",
    dtype: str = "float32",
    model_task: str = "auto",
    benchmark_mode: str = "auto",
    trace_layers: bool = False,
    hf_align_bf16_profile: bool = False,
    hf_align_mask_contract: bool = False,
    hf_align_position_ids: bool = False,
    hf_align_add_fp32_accum: bool = False,
    hf_align_linear_fp32_accum: bool = False,
    hf_align_norm_fp32: bool = False,
    hf_attn_implementation: str | None = None,
    hf_experts_implementation: str | None = None,
    compile_hf: bool = False,
    compile_axon: bool = False,
    compile_backend: str | None = None,
    compile_mode: str | None = None,
    compile_fullgraph: bool = False,
    compile_dynamic: bool = False,
    trust_remote_code: bool = False,
    axon_backend: str = "codegen2-torch",
    axon_typechecker: str = "typecheck2",
    pipeline_parallel_size: int | None = None,
    optimize_ast: bool = False,
    optimize_graph: bool = False,
    graph_backend_intrinsics: str | None = None,
    skip_hf: bool = False,
    hf_strict_dtype: bool = False,
    oom_cpu_fallback: bool = True,
    profile_axon: bool = False,
    profile_axon_top_n: int = 40,
    metal_capture: bool = False,
    forward_warmup: int = 0,
    forward_repeat: int = 1,
    table_format: str = "markdown",
    log_dir: Path | None = None,
    stream_csv: Path | None = None,
    debug_errors: bool = False,
    min_billion_parameters: float | None = None,
    max_billion_parameters: float | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    backend_token = str(axon_backend).strip().lower()
    if backend_token == "single":
        backend_token = "codegen2-torch"
    valid_backends = {"codegen2-torch", "codegen2-tinygrad", "codegen2-mlx", "runtime2-torch", "pipeline2-torch"}
    if backend_token not in valid_backends:
        raise ValueError(
            "axon_backend must be 'codegen2-torch', 'codegen2-tinygrad', "
            "'codegen2-mlx', 'runtime2-torch', or 'pipeline2-torch'"
        )
    axon_backend = backend_token
    typechecker_token = str(axon_typechecker).strip().lower()
    if typechecker_token != "typecheck2":
        raise ValueError("axon_typechecker must be 'typecheck2'")
    axon_typechecker = typechecker_token
    benchmark_mode = _resolve_benchmark_mode(benchmark_mode)
    forward_warmup = max(0, int(forward_warmup))
    forward_repeat = max(1, int(forward_repeat))
    if axon_backend != "pipeline2-torch" and pipeline_parallel_size is not None:
        raise ValueError("--pipeline-parallel-size/--pp is only valid with --axon-backend pipeline2-torch")
    repo_root = _repo_root()
    checkpoint_filter = {str(item).strip() for item in (checkpoints or ()) if str(item).strip()}
    exclude_filter = {str(item).strip() for item in (exclude or ()) if str(item).strip()}
    pairs: list[_BenchmarkPair] = []
    for resolved_axon_file in _expand_axon_inputs(axon_files):
        if _matches_any_selector(resolved_axon_file, exclude_filter):
            continue
        declared_checkpoints = _declared_checkpoints_from_axon(
            axon_file=resolved_axon_file,
        )
        if checkpoint_filter:
            checkpoints_to_run = tuple(
                checkpoint_id
                for checkpoint_id in declared_checkpoints
                if checkpoint_id in checkpoint_filter
            )
        else:
            checkpoints_to_run = declared_checkpoints
        for checkpoint_id in checkpoints_to_run:
            pairs.append(
                _BenchmarkPair(
                    axon_file=resolved_axon_file,
                    checkpoint_id=checkpoint_id,
                    model_dir=repo_root / "models" / checkpoint_id,
                )
            )
    if not pairs:
        if checkpoint_filter:
            requested = ", ".join(sorted(checkpoint_filter))
            raise ValueError(f"No benchmark pairs found for explicit checkpoints: {requested}")
        raise ValueError("No benchmark pairs found")
    pairs, skipped_by_param_count = _apply_billions_params_filter(
        pairs,
        min_billions_params=min_billion_parameters,
        max_billions_params=max_billion_parameters,
    )
    _print_param_count_skips(
        skipped_by_param_count,
        min_billions_params=min_billion_parameters,
        max_billions_params=max_billion_parameters,
    )
    if not pairs:
        raise ValueError("No benchmark pairs remain after parameter-range filtering")

    if dry_run:
        summary_rows = [_summary_row_from_pair(pair, fallback="DRY-RUN") for pair in pairs]
        if stream_csv is not None:
            _initialize_stream_csv(stream_csv)
            for row in summary_rows:
                _append_stream_csv_row(stream_csv, row)
        print()
        print(_format_checkpoint_summary_table(summary_rows, table_format=table_format))
        return {
            "dry_run": True,
            "pairs": [
                {
                    "axon_file": pair.axon_file,
                    "checkpoint_id": pair.checkpoint_id,
                    "model_dir": pair.model_dir,
                }
                for pair in pairs
            ],
            "results": [],
        }

    common_kwargs = {
        "repo_root": repo_root,
        "text": text,
        "max_len": max_len,
        "tokenizer": tokenizer,
        "class_name": class_name,
        "dtype": dtype,
        "model_task": model_task,
        "benchmark_mode": benchmark_mode,
        "trace_layers": trace_layers,
        "hf_align_bf16_profile": hf_align_bf16_profile,
        "hf_align_mask_contract": hf_align_mask_contract,
        "hf_align_position_ids": hf_align_position_ids,
        "hf_align_add_fp32_accum": hf_align_add_fp32_accum,
        "hf_align_linear_fp32_accum": hf_align_linear_fp32_accum,
        "hf_align_norm_fp32": hf_align_norm_fp32,
        "hf_attn_implementation": hf_attn_implementation,
        "hf_experts_implementation": hf_experts_implementation,
        "compile_hf": compile_hf,
        "compile_axon": compile_axon,
        "compile_backend": compile_backend,
        "compile_mode": compile_mode,
        "compile_fullgraph": compile_fullgraph,
        "compile_dynamic": compile_dynamic,
        "trust_remote_code": trust_remote_code,
        "axon_backend": axon_backend,
        "axon_typechecker": axon_typechecker,
        "optimize_ast": optimize_ast,
        "optimize_graph": optimize_graph,
        "graph_backend_intrinsics": graph_backend_intrinsics,
        "skip_hf": skip_hf,
        "hf_strict_dtype": hf_strict_dtype,
        "oom_cpu_fallback": oom_cpu_fallback,
        "profile_axon": profile_axon,
        "profile_axon_top_n": profile_axon_top_n,
        "metal_capture": metal_capture,
        "forward_warmup": forward_warmup,
        "forward_repeat": forward_repeat,
    }

    if stream_csv is not None:
        _initialize_stream_csv(stream_csv)

    serial_cuda_visible_devices: str | None = None
    pipeline_worker_specs: list[_WorkerSpec] | None = None
    effective_device = device
    if axon_backend == "pipeline2-torch":
        pipeline_worker_specs = _resolve_pipeline_worker_specs(
            backend=axon_backend,
            device=device,
            processes=max(1, int(processes)),
            pipeline_parallel_size=pipeline_parallel_size,
        )
        if processes <= 1:
            serial_cuda_visible_devices = pipeline_worker_specs[0].cuda_visible_devices
            effective_device = pipeline_worker_specs[0].run_device
    elif axon_backend == "codegen2-tinygrad":
        pipeline_worker_specs = _resolve_tinygrad_worker_specs(
            device=device,
            processes=max(1, int(processes)),
        )
    elif axon_backend == "codegen2-mlx":
        pipeline_worker_specs = _resolve_mlx_worker_specs(
            device=device,
            processes=max(1, int(processes)),
        )

    if processes <= 1:
        results = _run_benchmark_jobs_serial(
            pairs,
            device=effective_device,
            log_dir=log_dir,
            stream_csv=stream_csv,
            common_kwargs=common_kwargs,
            worker_cuda_visible_devices=serial_cuda_visible_devices,
            debug_errors=debug_errors,
        )
    else:
        results = _run_benchmark_jobs_parallel(
            pairs,
            processes=processes,
            device=effective_device,
            log_dir=log_dir,
            stream_csv=stream_csv,
            common_kwargs=common_kwargs,
            worker_specs=pipeline_worker_specs,
            debug_errors=debug_errors,
        )

    sorted_results = sorted(results, key=_summary_sort_key)

    summary_rows = [_summary_row_from_result(row) for row in sorted_results]
    print()
    print(_format_checkpoint_summary_table(summary_rows, table_format=table_format))
    return {"results": results}


__all__ = ["run_axon_benchmark", "render_axon_benchmark_csv"]
