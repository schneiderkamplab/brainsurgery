from __future__ import annotations

import argparse
import contextlib
import gc
import html
import io
import json
import math
import multiprocessing as mp
import os
import queue
import re
import signal
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import safetensors
from tqdm import tqdm

from .axon_runner_common import (
    LogFileWriter as _LogFileWriter,
)
from .axon_runner_common import (
    ParentLogger as _ParentLogger,
)
from .axon_runner_common import (
    TeeWriter as _TeeWriter,
)
from .axon_runner_common import (
    cleanup_cuda_after_oom as _cleanup_cuda_after_oom,
)
from .axon_runner_common import (
    is_cuda_oom as _is_cuda_oom,
)
from .axon_runner_common import (
    resolve_worker_devices as _resolve_worker_devices,
)
from .axon_runner_common import (
    worker_log_display_path as _common_worker_log_display_path,
)
from .axon_runner_common import (
    worker_log_path as _common_worker_log_path,
)
from .axon_test import run_axon_test
from .matrix_models import (
    MATRIX_AXON_MODEL_DIRS,
    MODEL_SPECS,
    ensure_model_downloaded,
    estimate_remote_param_count_lower_bound,
)


@dataclass(frozen=True)
class _Pair:
    axon_path: Path
    model_dir: Path


@dataclass(frozen=True)
class _SummaryRow:
    axon_file: str
    model_dir: str
    hf_runtime_s: str
    axon_runtime_s: str
    runtime_ratio: str
    eval_max_abs_diff: str
    eval_max_rel_diff: str
    debug_max_logit_diff: str
    debug_max_rel_diff: str
    mean_rel_diff: str
    masked_max_diff: str
    masked_last_max_diff: str
    masked_mean_rel_diff: str
    masked_max_rel_diff: str
    eval_top1_eq: str
    debug_top1_eq: str
    masked_top1_eq: str


@dataclass(frozen=True)
class _ParamCountSkip:
    pair: _Pair
    param_count: int
    is_exact: bool


@dataclass(frozen=True)
class _WorkerError:
    exc_type: str
    message: str
    traceback_text: str
    captured_output: str


def _run_pair_with_fallback(
    pair: _Pair,
    *,
    model_task: str,
    device: str,
    dtype: str,
    max_len: int,
    text: list[str],
    verbose: bool,
    no_capture_output: bool,
    compile_hf: bool,
    compile_axon: bool,
    compile_backend: str | None,
    compile_mode: str | None,
    compile_fullgraph: bool,
    compile_dynamic: bool,
) -> tuple[dict[str, Any], str | None]:
    try:
        return (
            _run_pair(
                pair,
                model_task=model_task,
                device=device,
                dtype=dtype,
                max_len=max_len,
                text=text,
                verbose=verbose,
                no_capture_output=no_capture_output,
                compile_hf=compile_hf,
                compile_axon=compile_axon,
                compile_backend=compile_backend,
                compile_mode=compile_mode,
                compile_fullgraph=compile_fullgraph,
                compile_dynamic=compile_dynamic,
            ),
            None,
        )
    except Exception as exc:
        if not _is_cuda_oom(exc, device=device):
            raise
        _cleanup_cuda_after_oom(device)
        retry_message = (
            f"CUDA OOM on {device}; retrying on cpu for {pair.axon_path.name} | {pair.model_dir}"
        )
        return (
            _run_pair(
                pair,
                model_task=model_task,
                device="cpu",
                dtype=dtype,
                max_len=max_len,
                text=text,
                verbose=verbose,
                no_capture_output=no_capture_output,
                compile_hf=compile_hf,
                compile_axon=compile_axon,
                compile_backend=compile_backend,
                compile_mode=compile_mode,
                compile_fullgraph=compile_fullgraph,
                compile_dynamic=compile_dynamic,
            ),
            retry_message,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run brainsurgery synapse axon-test across matching examples/*.axon and models/* dirs."
        )
    )
    parser.add_argument(
        "--examples-dir",
        type=Path,
        default=Path("examples"),
        help="Directory with Axon files (default: examples).",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Directory with model directories (default: models).",
    )
    parser.add_argument("--device", default="cpu", help="Device passed to axon-test.")
    parser.add_argument(
        "--processes",
        type=int,
        default=1,
        help=(
            "Number of model-evaluation worker processes to run simultaneously. "
            "When using CUDA with multiple processes, workers are assigned round-robin "
            "to cuda:0, cuda:1, ... up to the requested worker count."
        ),
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "bfloat16", "float16"],
        help="Floating point dtype passed to axon-test.",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=32,
        help="Total sequence length target for generation.",
    )
    parser.add_argument(
        "--text",
        action="append",
        default=None,
        help="Prompt text. Repeat to pass multiple prompts.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show per-run output from synapse axon-test.",
    )
    parser.add_argument(
        "--no-capture-output",
        action="store_true",
        help="Do not capture per-run output; stream run_axon_test output directly.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help=(
            "If set, each model worker writes stdout/stderr to a separate log file "
            "under this directory using log-<pid>-<axon>-<model>.txt."
        ),
    )
    parser.add_argument(
        "--include",
        action="append",
        default=None,
        help=(
            "Only run pairs matching these selectors (repeatable). "
            "Selectors are model directory names or .axon file names."
        ),
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=None,
        help=(
            "Exclude pairs matching these selectors (repeatable). "
            "Selectors are model directory names or .axon file names."
        ),
    )
    parser.add_argument(
        "--min-billions-params",
        type=float,
        default=None,
        help=(
            "Only run models with estimated parameter count at or above this many "
            "billions of parameters."
        ),
    )
    parser.add_argument(
        "--max-billions-params",
        type=float,
        default=None,
        help=(
            "Only run models with estimated parameter count at or below this many "
            "billions of parameters."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only resolve and print matching pairs; do not run tests.",
    )
    parser.add_argument(
        "--table-format",
        default="plain",
        choices=["plain", "markdown", "html"],
        help="Summary table format (plain, markdown, or html).",
    )
    parser.add_argument(
        "--compile-hf",
        action="store_true",
        help="Compile the HF reference model with torch.compile.",
    )
    parser.add_argument(
        "--compile-axon",
        action="store_true",
        help="Compile the Axon-derived model with torch.compile.",
    )
    parser.add_argument(
        "--compile-backend",
        default=None,
        help="Optional torch.compile backend (e.g. inductor).",
    )
    parser.add_argument(
        "--compile-mode",
        default=None,
        help="Optional torch.compile mode (e.g. default/reduce-overhead/max-autotune).",
    )
    parser.add_argument(
        "--compile-fullgraph",
        action="store_true",
        help="Set torch.compile(fullgraph=True).",
    )
    parser.add_argument(
        "--compile-dynamic",
        action="store_true",
        help="Set torch.compile(dynamic=True).",
    )
    parser.add_argument(
        "--model-task",
        default="auto",
        choices=["auto", "causal_lm", "masked_lm", "seq2seq_lm"],
        help=(
            "Execution task for run_axon_test. "
            "'auto' picks per model (encoder-only masked LM models -> masked_lm, "
            "t5_small=seq2seq_lm, others=causal_lm)."
        ),
    )
    return parser.parse_args()


def _normalize_selectors(values: list[str] | None) -> set[str]:
    out: set[str] = set()
    if not values:
        return out
    for value in values:
        raw = str(value).strip().lower()
        if not raw:
            continue
        out.add(Path(raw).name)
    return out


def _pair_matches_selector(pair: _Pair, selectors: set[str]) -> bool:
    if not selectors:
        return False
    axon_name = pair.axon_path.name.lower()
    axon_stem = pair.axon_path.stem.lower()
    model_name = pair.model_dir.name.lower()
    for selector in selectors:
        if selector.endswith(".axon"):
            if selector == axon_name:
                return True
            continue
        if selector == axon_stem:
            return True
        if selector == model_name:
            return True
    return False


def _apply_pair_filters(
    pairs: list[_Pair],
    *,
    include: set[str],
    exclude: set[str],
) -> list[_Pair]:
    if include and exclude:
        raise ValueError("axon-test-matrix accepts either include or exclude selectors, not both")
    if include:
        return [pair for pair in pairs if _pair_matches_selector(pair, include)]
    if exclude:
        return [pair for pair in pairs if not _pair_matches_selector(pair, exclude)]
    return pairs


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


def _estimate_model_param_count_lower_bound(model_dir: Path) -> tuple[int, bool] | None:
    exact = _estimate_model_param_count(model_dir)
    if exact is not None:
        if isinstance(exact, int):
            return exact, True
        return exact

    bin_paths = sorted(path for path in model_dir.glob("*.bin") if path.is_file())
    if not bin_paths:
        return None

    total_bytes = sum(path.stat().st_size for path in bin_paths)
    if total_bytes <= 0:
        return None
    return (total_bytes + 3) // 4, False


def _apply_billions_params_filter(
    pairs: list[_Pair],
    *,
    min_billions_params: float | None,
    max_billions_params: float | None,
) -> tuple[list[_Pair], list[_ParamCountSkip]]:
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
    kept: list[_Pair] = []
    skipped: list[_ParamCountSkip] = []
    remote_param_count_cache: dict[str, int | None] = {}
    for pair in pairs:
        local_param_estimate = _estimate_model_param_count_lower_bound(pair.model_dir)
        if local_param_estimate is not None:
            local_param_count, is_exact = local_param_estimate
            if min_params is not None and local_param_count < min_params:
                skipped.append(
                    _ParamCountSkip(pair=pair, param_count=local_param_count, is_exact=is_exact)
                )
                continue
            if max_params is not None and local_param_count > max_params:
                skipped.append(
                    _ParamCountSkip(pair=pair, param_count=local_param_count, is_exact=is_exact)
                )
                continue
            kept.append(pair)
            continue

        if max_params is None:
            kept.append(pair)
            continue

        model_name = pair.model_dir.name
        remote_param_count = remote_param_count_cache.get(model_name)
        if model_name not in remote_param_count_cache:
            spec = MODEL_SPECS.get(model_name)
            if spec is None:
                remote_param_count = None
            else:
                remote_param_count = estimate_remote_param_count_lower_bound(
                    repo_root=pair.model_dir.parent.parent,
                    spec=spec,
                )
            remote_param_count_cache[model_name] = remote_param_count
        if remote_param_count is None:
            kept.append(pair)
            continue
        if remote_param_count > max_params:
            skipped.append(
                _ParamCountSkip(pair=pair, param_count=remote_param_count, is_exact=False)
            )
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
    if not skipped_pairs or (min_billions_params is None and max_billions_params is None):
        return
    if min_billions_params is not None and max_billions_params is not None:
        bound_desc = f"--min-billions-params/--max-billions-params ({min_billions_params:g}B..{max_billions_params:g}B)"
    elif min_billions_params is not None:
        bound_desc = f"--min-billions-params ({min_billions_params:g}B)"
    else:
        bound_desc = f"--max-billions-params ({max_billions_params:g}B)"
    print(f"Skipped due to {bound_desc}: {len(skipped_pairs)}")
    for skipped in skipped_pairs:
        print(
            f"  - {skipped.pair.axon_path.name} | {skipped.pair.model_dir} | "
            f"{_format_billions_params(skipped.param_count, is_exact=skipped.is_exact)}"
        )
    print()


def _skip_reason_for_pair(pair: _Pair) -> str | None:
    return None


def _resolve_model_task_for_pair(pair: _Pair) -> str:
    masked_lm_stems = {
        "albert",
        "bert",
        "deberta_v2",
        "distilbert",
        "electra",
        "longformer",
        "modernbert",
        "roberta",
    }
    masked_lm_model_dirs = {
        "albert",
        "bert",
        "camembert",
        "deberta_v2",
        "distilbert",
        "electra",
        "longformer",
        "modernbert",
        "roberta",
        "xlm_roberta",
    }
    seq2seq_lm_stems = {
        "t5",
        "t5_small",
        "mt5",
        "bart",
        "mbart",
        "marian",
        "t5gemma",
        "t5gemma2",
    }
    seq2seq_lm_model_dirs = {
        "t5_small",
        "t5_base",
        "t5_large",
        "t5_3b",
        "t5_11b",
        "mt5_small",
        "mt5_base",
        "mt5_large",
        "mt5_xl",
        "mt5_xxl",
        "bart_base",
        "mbart_large_50_m2m",
        "marian_en_de",
        "t5gemma_s_s_ul2",
        "t5gemma2_270m",
    }
    if pair.axon_path.stem in masked_lm_stems or pair.model_dir.name in masked_lm_model_dirs:
        return "masked_lm"
    if (
        pair.axon_path.stem in seq2seq_lm_stems
        or pair.model_dir.name in seq2seq_lm_model_dirs
        or pair.model_dir.name.startswith("t5gemma")
    ):
        return "seq2seq_lm"
    return "causal_lm"


def _resolve_pairs(
    examples_dir: Path,
    models_dir: Path,
) -> list[_Pair]:
    if not examples_dir.is_dir():
        raise FileNotFoundError(f"Examples directory not found: {examples_dir}")
    if not models_dir.is_dir():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    model_dirs = sorted(
        path for path in models_dir.iterdir() if path.is_dir() and not path.name.startswith(".")
    )
    model_by_name = {path.name: path for path in model_dirs}
    axon_paths = sorted(
        path for path in examples_dir.glob("*.axon") if not path.stem.endswith("_config")
    )
    axon_by_stem = {path.stem: path for path in axon_paths}

    pairs: list[_Pair] = []
    covered_model_names: set[str] = set()

    for axon_path in axon_paths:
        stem = axon_path.stem
        model_dir_names = MATRIX_AXON_MODEL_DIRS.get(stem)
        if model_dir_names is None:
            print(f"Ignoring {axon_path} as it is not registered in the matrix model map")
            continue
        matched = False
        for model_dir_name in model_dir_names:
            model_dir = model_by_name.get(model_dir_name)
            if model_dir is None:
                spec = MODEL_SPECS.get(model_dir_name)
                if spec is None:
                    continue
                model_dir = models_dir / spec.local_dir
            pairs.append(_Pair(axon_path=axon_path, model_dir=model_dir))
            covered_model_names.add(model_dir_name)
            matched = True
        if not matched:
            print(
                f"Ignoring {axon_path} as I did not locate any configured model dir or spec for {stem}"
            )

    reverse_axon_by_model: dict[str, set[str]] = {}
    for axon_stem, model_dir_names in MATRIX_AXON_MODEL_DIRS.items():
        for model_dir_name in model_dir_names:
            reverse_axon_by_model.setdefault(model_dir_name, set()).add(axon_stem)

    for model_dir in model_dirs:
        model_name = model_dir.name
        if model_name in covered_model_names:
            continue
        candidate_axons = sorted(reverse_axon_by_model.get(model_name, set()))
        if not candidate_axons:
            print(f"Ignoring model dir {model_dir} as it is not registered in the matrix model map")
            continue
        added = False
        for axon_stem in candidate_axons:
            maybe_axon_path = axon_by_stem.get(axon_stem)
            if maybe_axon_path is None:
                continue
            pairs.append(_Pair(axon_path=maybe_axon_path, model_dir=model_dir))
            added = True
        if not added:
            print(
                f"Ignoring model dir {model_dir} as I did not locate matching registered .axon file"
            )

    return pairs


def _maybe_ensure_pair_model_ready(pair: _Pair) -> None:
    spec = MODEL_SPECS.get(pair.model_dir.name)
    if spec is None:
        return
    repo_root = pair.model_dir.parent.parent
    expected_dir = (repo_root / "models" / spec.local_dir).resolve()
    if pair.model_dir.resolve() != expected_dir:
        return
    ensure_model_downloaded(
        repo_root=repo_root,
        spec=spec,
        status_cb=lambda message: print(f"[model-download] {message}", flush=True),
    )


def _selectors_match_axon(selector: str, axon_stem: str) -> bool:
    if selector.endswith(".axon"):
        return selector == f"{axon_stem}.axon"
    return selector == axon_stem


def _format_table(rows: list[_SummaryRow]) -> str:
    headers = [
        "axon_file",
        "model_dir",
        "HF runtime (s)",
        "AxonDerived runtime (s)",
        "AxonDerived runtime/HF runtime",
        "eval max abs diff",
        "eval max rel diff",
        "eval top1_eq",
        "masked max abs diff",
        "masked last max abs diff",
        "masked max rel diff",
        "masked_top1_eq",
        "debug max abs diff",
        "debug max rel diff",
        "debug top1_eq",
        "mean rel diff",
        "masked mean rel diff",
    ]

    body = [
        [
            row.axon_file,
            row.model_dir,
            row.hf_runtime_s,
            row.axon_runtime_s,
            row.runtime_ratio,
            row.eval_max_abs_diff,
            row.eval_max_rel_diff,
            row.eval_top1_eq,
            row.masked_max_diff,
            row.masked_last_max_diff,
            row.masked_max_rel_diff,
            row.masked_top1_eq,
            row.debug_max_logit_diff,
            row.debug_max_rel_diff,
            row.debug_top1_eq,
            row.mean_rel_diff,
            row.masked_mean_rel_diff,
        ]
        for row in rows
    ]

    widths = [len(header) for header in headers]
    for line in body:
        for idx, cell in enumerate(line):
            widths[idx] = max(widths[idx], len(cell))

    def _fmt(line: list[str]) -> str:
        return " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(line))

    divider = "-+-".join("-" * width for width in widths)
    out_lines = [_fmt(headers), divider]
    out_lines.extend(_fmt(line) for line in body)
    return "\n".join(out_lines)


def _format_table_markdown(rows: list[_SummaryRow]) -> str:
    headers = [
        "axon_file",
        "model_dir",
        "HF runtime (s)",
        "AxonDerived runtime (s)",
        "AxonDerived runtime/HF runtime",
        "eval max abs diff",
        "eval max rel diff",
        "eval top1_eq",
        "masked max abs diff",
        "masked last max abs diff",
        "masked max rel diff",
        "masked_top1_eq",
        "debug max abs diff",
        "debug max rel diff",
        "debug top1_eq",
        "mean rel diff",
        "masked mean rel diff",
    ]

    body = [
        [
            row.axon_file,
            row.model_dir,
            row.hf_runtime_s,
            row.axon_runtime_s,
            row.runtime_ratio,
            row.eval_max_abs_diff,
            row.eval_max_rel_diff,
            row.eval_top1_eq,
            row.masked_max_diff,
            row.masked_last_max_diff,
            row.masked_max_rel_diff,
            row.masked_top1_eq,
            row.debug_max_logit_diff,
            row.debug_max_rel_diff,
            row.debug_top1_eq,
            row.mean_rel_diff,
            row.masked_mean_rel_diff,
        ]
        for row in rows
    ]

    def _esc(cell: str) -> str:
        return cell.replace("|", r"\|")

    header_row = "| " + " | ".join(_esc(h) for h in headers) + " |"
    divider = "| " + " | ".join("---" for _ in headers) + " |"
    data_rows = ["| " + " | ".join(_esc(cell) for cell in line) + " |" for line in body]
    return "\n".join([header_row, divider, *data_rows])


def _format_table_html(rows: list[_SummaryRow]) -> str:
    headers = [
        "axon_file",
        "model_dir",
        "HF runtime (s)",
        "AxonDerived runtime (s)",
        "AxonDerived runtime/HF runtime",
        "eval max abs diff",
        "eval max rel diff",
        "eval top1_eq",
        "masked max abs diff",
        "masked last max abs diff",
        "masked max rel diff",
        "masked_top1_eq",
        "debug max abs diff",
        "debug max rel diff",
        "debug top1_eq",
        "mean rel diff",
        "masked mean rel diff",
    ]

    body = [
        [
            row.axon_file,
            row.model_dir,
            row.hf_runtime_s,
            row.axon_runtime_s,
            row.runtime_ratio,
            row.eval_max_abs_diff,
            row.eval_max_rel_diff,
            row.eval_top1_eq,
            row.masked_max_diff,
            row.masked_last_max_diff,
            row.masked_max_rel_diff,
            row.masked_top1_eq,
            row.debug_max_logit_diff,
            row.debug_max_rel_diff,
            row.debug_top1_eq,
            row.mean_rel_diff,
            row.masked_mean_rel_diff,
        ]
        for row in rows
    ]

    def _numeric(cell: str) -> float | None:
        try:
            return float(cell)
        except Exception:
            return None

    def _cell_style(header: str, cell: str) -> str:
        if header == "masked_top1_eq" and cell != "True":
            return "background-color: #f8d7da;"
        if header == "masked max abs diff":
            numeric = _numeric(cell)
            if numeric is not None:
                if numeric > 1e-2:
                    return "background-color: #ffe5b4;"
                if numeric > 1e-3:
                    return "background-color: #fff3cd;"
        if "max abs diff" in header and header != "masked max abs diff":
            numeric = _numeric(cell)
            if numeric is not None and numeric > 1.0:
                return "background-color: #fff3cd;"
        return ""

    out = [
        "<table>",
        "  <thead>",
        "    <tr>",
        *[f"      <th>{html.escape(header)}</th>" for header in headers],
        "    </tr>",
        "  </thead>",
        "  <tbody>",
    ]
    for line in body:
        out.append("    <tr>")
        for header, cell in zip(headers, line, strict=False):
            style = _cell_style(header, cell)
            style_attr = f' style="{style}"' if style else ""
            out.append(f"      <td{style_attr}>{html.escape(cell)}</td>")
        out.append("    </tr>")
    out.extend(["  </tbody>", "</table>"])
    return "\n".join(out)


def _run_pair(
    pair: _Pair,
    *,
    model_task: str,
    device: str,
    dtype: str,
    max_len: int,
    text: list[str],
    verbose: bool,
    no_capture_output: bool,
    compile_hf: bool,
    compile_axon: bool,
    compile_backend: str | None,
    compile_mode: str | None,
    compile_fullgraph: bool,
    compile_dynamic: bool,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "axon_file": pair.axon_path,
        "weights": pair.model_dir,
        "hf_model_dir": pair.model_dir,
        "device": device,
        "dtype": dtype,
        "model_task": model_task,
        "max_len": max_len,
        "text": text,
        "compile_hf": compile_hf,
        "compile_axon": compile_axon,
        "compile_backend": compile_backend,
        "compile_mode": compile_mode,
        "compile_fullgraph": compile_fullgraph,
        "compile_dynamic": compile_dynamic,
    }
    if verbose:
        print(f"Running: {kwargs}")
    if no_capture_output:
        return run_axon_test(**kwargs)

    with contextlib.redirect_stdout(io.StringIO()):
        return run_axon_test(**kwargs)


def _summary_row_from_result(pair: _Pair, result: dict[str, Any]) -> _SummaryRow:
    masked_max_diff_value = result.get("masked_max_diff")
    masked_max_rel_diff_value = result.get("masked_max_rel_diff")
    masked_top1_eq_value = result.get("masked_top1_eq")
    eval_max_abs_diff_value = (
        masked_max_diff_value if masked_max_diff_value is not None else result["max_diff"]
    )
    eval_max_rel_diff_value = (
        masked_max_rel_diff_value
        if masked_max_rel_diff_value is not None
        else result["max_rel_diff"]
    )
    eval_top1_eq_value = (
        bool(masked_top1_eq_value) if masked_top1_eq_value is not None else bool(result["top1_eq"])
    )
    return _SummaryRow(
        axon_file=pair.axon_path.name,
        model_dir=str(pair.model_dir),
        hf_runtime_s=f"{result['hf_time']:.6g}",
        axon_runtime_s=f"{result['axon_time']:.6g}",
        runtime_ratio=f"{result['speed_ratio_axon_over_hf']:.3f}",
        eval_max_abs_diff=f"{float(eval_max_abs_diff_value):.6g}",
        eval_max_rel_diff=f"{float(eval_max_rel_diff_value):.6g}",
        debug_max_logit_diff=f"{result['max_diff']:.6g}",
        debug_max_rel_diff=f"{float(result['max_rel_diff']):.6g}",
        mean_rel_diff=f"{float(result['mean_rel_diff']):.6g}",
        masked_max_diff=(
            "N/A"
            if result.get("masked_max_diff") is None
            else f"{float(result['masked_max_diff']):.6g}"
        ),
        masked_last_max_diff=(
            "N/A"
            if result.get("masked_last_max_diff") is None
            else f"{float(result['masked_last_max_diff']):.6g}"
        ),
        masked_mean_rel_diff=(
            "N/A"
            if result.get("masked_mean_rel_diff") is None
            else f"{float(result['masked_mean_rel_diff']):.6g}"
        ),
        masked_max_rel_diff=(
            "N/A"
            if result.get("masked_max_rel_diff") is None
            else f"{float(result['masked_max_rel_diff']):.6g}"
        ),
        eval_top1_eq=str(eval_top1_eq_value),
        debug_top1_eq=str(bool(result["top1_eq"])),
        masked_top1_eq=(
            "N/A" if result.get("masked_top1_eq") is None else str(bool(result["masked_top1_eq"]))
        ),
    )


def _summary_row_from_error(pair: _Pair, err: BaseException | _WorkerError) -> _SummaryRow:
    if isinstance(err, _WorkerError):
        error_text = f"ERROR: {err.exc_type}: {err.message}"
    else:
        error_text = f"ERROR: {type(err).__name__}: {err}"
    return _SummaryRow(
        axon_file=pair.axon_path.name,
        model_dir=str(pair.model_dir),
        hf_runtime_s="ERROR",
        axon_runtime_s="ERROR",
        runtime_ratio="ERROR",
        eval_max_abs_diff="ERROR",
        eval_max_rel_diff="ERROR",
        debug_max_logit_diff="ERROR",
        debug_max_rel_diff="ERROR",
        mean_rel_diff="ERROR",
        masked_max_diff="ERROR",
        masked_last_max_diff="ERROR",
        masked_mean_rel_diff="ERROR",
        masked_max_rel_diff="ERROR",
        eval_top1_eq=error_text,
        debug_top1_eq="ERROR",
        masked_top1_eq="ERROR",
    )


def _emit_worker_result_summary(pair: _Pair, result: dict[str, Any]) -> None:
    print(f"result.axon={pair.axon_path.name}")
    print(f"result.model_dir={pair.model_dir}")
    print(f"result.masked_top1_eq={result.get('masked_top1_eq')}")
    print(f"result.masked_max_abs_diff={result.get('masked_max_diff')}")
    print(f"result.masked_max_rel_diff={result.get('masked_max_rel_diff')}")


def _slugify_log_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    text = text.strip("-._")
    return text or "unknown"


def _worker_log_path(log_dir: Path | None, pair: _Pair, pid: int | None) -> Path | None:
    return _common_worker_log_path(
        log_dir,
        axon_name=_slugify_log_name(pair.axon_path.stem),
        model_name=_slugify_log_name(pair.model_dir.name),
        pid=pid,
    )


def _worker_log_display_path(log_dir: Path | None, pair: _Pair, pid: int | None) -> str | None:
    return _common_worker_log_display_path(
        log_dir,
        axon_name=_slugify_log_name(pair.axon_path.stem),
        model_name=_slugify_log_name(pair.model_dir.name),
        pid=pid,
    )


def _signal_name(exitcode: int) -> str:
    signum = abs(int(exitcode))
    with contextlib.suppress(Exception):
        return signal.Signals(signum).name
    return f"SIG{signum}"


def _run_worker_loop(
    pair_index: int,
    pair: _Pair,
    worker_device: str,
    result_queue: Any,
    common_kwargs: dict[str, Any],
    log_dir: str | None,
) -> None:
    model_task = str(common_kwargs["model_task_override"] or _resolve_model_task_for_pair(pair))
    captured = io.StringIO()
    file_handle: io.TextIOWrapper | None = None
    log_path_obj = (
        _worker_log_path(Path(log_dir), pair, os.getpid()) if log_dir is not None else None
    )
    log_path = str(log_path_obj) if log_path_obj is not None else None
    log_path_display = (
        _worker_log_display_path(Path(log_dir), pair, os.getpid()) if log_dir is not None else None
    )
    try:
        stdout_target: Any = captured
        stderr_target: Any = captured
        if log_path is not None:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            file_handle = open(log_path, "w", encoding="utf-8", buffering=1)
            file_writer = _LogFileWriter(file_handle)
            tee = _TeeWriter(captured, file_writer)
            stdout_target = tee
            stderr_target = tee
        with contextlib.redirect_stdout(stdout_target), contextlib.redirect_stderr(stderr_target):
            if log_path is not None:
                print(
                    f"worker_pid={os.getpid()} axon={pair.axon_path.name} model_dir={pair.model_dir}"
                )
                print(f"device={worker_device} log_path={log_path_display}")
            _maybe_ensure_pair_model_ready(pair)
            result, retry_message = _run_pair_with_fallback(
                pair,
                model_task=model_task,
                device=worker_device,
                dtype=str(common_kwargs["dtype"]),
                max_len=int(common_kwargs["max_len"]),
                text=list(common_kwargs["text"]),
                verbose=bool(common_kwargs["verbose"]),
                no_capture_output=bool(common_kwargs["no_capture_output"]),
                compile_hf=bool(common_kwargs["compile_hf"]),
                compile_axon=bool(common_kwargs["compile_axon"]),
                compile_backend=common_kwargs["compile_backend"],
                compile_mode=common_kwargs["compile_mode"],
                compile_fullgraph=bool(common_kwargs["compile_fullgraph"]),
                compile_dynamic=bool(common_kwargs["compile_dynamic"]),
            )
            if retry_message:
                print(retry_message, flush=True)
            _emit_worker_result_summary(pair, result)
        result_queue.put(
            (
                pair_index,
                _summary_row_from_result(pair, result),
                None,
                captured.getvalue(),
                log_path,
            )
        )
    except Exception as exc:
        result_queue.put(
            (
                pair_index,
                None,
                _WorkerError(
                    exc_type=type(exc).__name__,
                    message=str(exc),
                    traceback_text=traceback.format_exc(),
                    captured_output=captured.getvalue(),
                ),
                captured.getvalue(),
                log_path,
            )
        )
    finally:
        if file_handle is not None:
            file_handle.close()
        gc.collect()


def _run_runnable_pairs_serial(
    runnable_pairs: list[_Pair],
    *,
    model_task_override: str | None,
    device: str,
    dtype: str,
    max_len: int,
    prompts: list[str],
    verbose: bool,
    no_capture_output: bool,
    compile_hf: bool,
    compile_axon: bool,
    compile_backend: str | None,
    compile_mode: str | None,
    compile_fullgraph: bool,
    compile_dynamic: bool,
    log_dir: Path | None = None,
) -> tuple[list[_SummaryRow], int, int]:
    rows: list[_SummaryRow] = []
    passed = 0
    failed = 0
    progress = tqdm(total=len(runnable_pairs), desc="synapse axon-test", unit="pair")
    for pair in runnable_pairs:
        progress.set_postfix_str(pair.axon_path.name)
        try:
            _maybe_ensure_pair_model_ready(pair)
            model_task = model_task_override or _resolve_model_task_for_pair(pair)
            result, retry_message = _run_pair_with_fallback(
                pair,
                model_task=model_task,
                device=device,
                dtype=dtype,
                max_len=max_len,
                text=prompts,
                verbose=verbose,
                no_capture_output=no_capture_output,
                compile_hf=compile_hf,
                compile_axon=compile_axon,
                compile_backend=compile_backend,
                compile_mode=compile_mode,
                compile_fullgraph=compile_fullgraph,
                compile_dynamic=compile_dynamic,
            )
            if retry_message:
                tqdm.write(retry_message)
            rows.append(_summary_row_from_result(pair, result))
            passed += 1
        except Exception as exc:
            if verbose:
                tqdm.write(
                    f"ERROR in {pair.axon_path.name} ({pair.model_dir}): {type(exc).__name__}: {exc}"
                )
                for line in traceback.format_exc().rstrip().splitlines():
                    tqdm.write(line)
            rows.append(_summary_row_from_error(pair, exc))
            failed += 1
        finally:
            gc.collect()
            progress.update(1)
            progress.set_postfix_str(f"passed={passed} failed={failed}")
    progress.close()
    return rows, passed, failed


def _run_runnable_pairs_parallel(
    runnable_pairs: list[_Pair],
    *,
    model_task_override: str | None,
    processes: int,
    device: str,
    dtype: str,
    max_len: int,
    prompts: list[str],
    verbose: bool,
    no_capture_output: bool,
    compile_hf: bool,
    compile_axon: bool,
    compile_backend: str | None,
    compile_mode: str | None,
    compile_fullgraph: bool,
    compile_dynamic: bool,
    log_dir: Path | None = None,
) -> tuple[list[_SummaryRow], int, int]:
    worker_devices = _resolve_worker_devices(device, processes)
    parent_logger = _ParentLogger(log_dir)
    if verbose:
        print(
            "Starting per-model workers with devices: "
            + ", ".join(worker_devices)
            + f" (max concurrent={len(worker_devices)})"
        )
    if parent_logger.path is not None:
        print(f"Parent log: {parent_logger.path}")
    parent_logger.log(
        f"run_start total_pairs={len(runnable_pairs)} devices={worker_devices} max_concurrent={len(worker_devices)}"
    )

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    common_kwargs = {
        "dtype": dtype,
        "max_len": max_len,
        "text": list(prompts),
        "verbose": verbose,
        "no_capture_output": no_capture_output,
        "compile_hf": compile_hf,
        "compile_axon": compile_axon,
        "compile_backend": compile_backend,
        "compile_mode": compile_mode,
        "compile_fullgraph": compile_fullgraph,
        "compile_dynamic": compile_dynamic,
        "model_task_override": model_task_override,
    }

    rows_by_index: list[_SummaryRow | None] = [None] * len(runnable_pairs)
    passed = 0
    failed = 0
    progress = tqdm(total=len(runnable_pairs), desc="synapse axon-test", unit="pair")
    pending_indices: deque[int] = deque(range(len(runnable_pairs)))
    next_device_index = 0
    active_processes: dict[int, Any] = {}

    def _mark_pair_finished(pair_index: int, row: _SummaryRow) -> None:
        nonlocal passed, failed
        rows_by_index[pair_index] = row
        is_failed = row.masked_top1_eq == "ERROR"
        if is_failed:
            failed += 1
        else:
            passed += 1
        progress.update(1)
        progress.set_postfix_str(f"passed={passed} failed={failed}")

    def _spawn_next_pair(pair_index: int) -> None:
        nonlocal next_device_index
        pair = runnable_pairs[pair_index]
        worker_device = worker_devices[next_device_index % len(worker_devices)]
        next_device_index += 1
        progress.set_postfix_str(pair.axon_path.name)
        process = ctx.Process(
            target=_run_worker_loop,
            args=(
                pair_index,
                pair,
                worker_device,
                result_queue,
                common_kwargs,
                str(log_dir) if log_dir is not None else None,
            ),
            daemon=False,
        )
        process.start()
        active_processes[pair_index] = process
        log_path = _worker_log_display_path(log_dir, pair, process.pid)
        parent_logger.log(
            "child_start "
            f"pair_index={pair_index} pid={process.pid} device={worker_device} "
            f"axon={pair.axon_path.name} model_dir={pair.model_dir} log_path={log_path}"
        )
        if verbose and log_path is not None:
            tqdm.write(f"Logging {pair.axon_path.name} | {pair.model_dir} -> {log_path}")

    try:
        while pending_indices or active_processes:
            while pending_indices and len(active_processes) < len(worker_devices):
                _spawn_next_pair(pending_indices.popleft())

            try:
                pair_index, result, error, captured_output, log_path = result_queue.get(timeout=1.0)
            except queue.Empty:
                for active_pair_index, process in list(active_processes.items()):
                    if process.is_alive():
                        continue
                    pair = runnable_pairs[int(active_pair_index)]
                    process.join(timeout=0.1)
                    exitcode = int(process.exitcode or 0)
                    log_path = _worker_log_display_path(log_dir, pair, process.pid)
                    status = "abnormal" if exitcode != 0 else "missing_result"
                    parent_logger.log(
                        "child_finish "
                        f"pair_index={active_pair_index} pid={process.pid} status={status} exitcode={exitcode} "
                        f"axon={pair.axon_path.name} model_dir={pair.model_dir} log_path={log_path}"
                    )
                    active_processes.pop(active_pair_index, None)
                    row = _summary_row_from_error(
                        pair,
                        RuntimeError(
                            (
                                f"worker exited abnormally with exit code {exitcode}"
                                + (f" ({_signal_name(exitcode)})" if exitcode < 0 else "")
                            )
                            if exitcode != 0
                            else "worker exited without publishing a result"
                        ),
                    )
                    _mark_pair_finished(int(active_pair_index), row)
                continue

            pair = runnable_pairs[int(pair_index)]
            process = active_processes.pop(int(pair_index), None)
            result_exitcode: int | None = None
            if process is not None:
                process.join(timeout=5.0)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5.0)
                result_exitcode = int(process.exitcode or 0)
            output_text = str(captured_output).rstrip()
            if output_text and (no_capture_output or verbose or error is not None):
                tqdm.write(f"===== {pair.axon_path.name} | {pair.model_dir} =====")
                for line in output_text.splitlines():
                    tqdm.write(line)
            if log_path is not None and (verbose or error is not None):
                tqdm.write(f"log file: {log_path}")
            if error is None:
                assert isinstance(result, _SummaryRow)
                row = result
                parent_logger.log(
                    "child_finish "
                    f"pair_index={pair_index} pid={getattr(process, 'pid', 'unknown')} status=success "
                    f"exitcode={result_exitcode} axon={pair.axon_path.name} model_dir={pair.model_dir} log_path={log_path}"
                )
            else:
                assert isinstance(error, _WorkerError)
                if verbose:
                    tqdm.write(
                        f"ERROR in {pair.axon_path.name} ({pair.model_dir}): "
                        f"{error.exc_type}: {error.message}"
                    )
                    for line in error.traceback_text.rstrip().splitlines():
                        tqdm.write(line)
                row = _summary_row_from_error(pair, error)
                parent_logger.log(
                    "child_finish "
                    f"pair_index={pair_index} pid={getattr(process, 'pid', 'unknown')} status=error "
                    f"exitcode={result_exitcode} axon={pair.axon_path.name} model_dir={pair.model_dir} log_path={log_path} "
                    f"error={error.exc_type}:{error.message}"
                )
            _mark_pair_finished(int(pair_index), row)
    finally:
        progress.close()
        for process in active_processes.values():
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
        parent_logger.log(
            f"run_finish passed={passed} failed={failed} total_rows={len([r for r in rows_by_index if r is not None])}"
        )
        parent_logger.close()

    rows = [row for row in rows_by_index if row is not None]
    return rows, passed, failed


def run_axon_test_matrix(
    *,
    examples_dir: Path = Path("examples"),
    models_dir: Path = Path("models"),
    device: str = "cpu",
    processes: int = 1,
    dtype: str = "float32",
    min_billions_params: float | None = None,
    max_billions_params: float | None = None,
    max_len: int = 32,
    text: list[str] | None = None,
    verbose: bool = False,
    no_capture_output: bool = False,
    dry_run: bool = False,
    table_format: str = "plain",
    compile_hf: bool = False,
    compile_axon: bool = False,
    compile_backend: str | None = None,
    compile_mode: str | None = None,
    compile_fullgraph: bool = False,
    compile_dynamic: bool = False,
    model_task_override: str | None = None,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    log_dir: Path | None = None,
) -> int:
    if table_format not in {"plain", "markdown", "html"}:
        raise ValueError("table_format must be 'plain', 'markdown', or 'html'")
    if processes <= 0:
        raise ValueError("processes must be >= 1")

    prompts = text if text else ["The future of AI is"]
    include_selectors = _normalize_selectors(include)
    exclude_selectors = _normalize_selectors(exclude)
    if model_task_override is not None and model_task_override not in {
        "causal_lm",
        "masked_lm",
        "seq2seq_lm",
    }:
        raise ValueError("model_task_override must be one of: causal_lm, masked_lm, seq2seq_lm")
    if include_selectors and exclude_selectors:
        raise ValueError("axon-test-matrix accepts either include or exclude selectors, not both")
    pairs = _resolve_pairs(examples_dir.resolve(), models_dir.resolve())
    pairs = _apply_pair_filters(pairs, include=include_selectors, exclude=exclude_selectors)
    pairs, skipped_by_param_count = _apply_billions_params_filter(
        pairs,
        min_billions_params=min_billions_params,
        max_billions_params=max_billions_params,
    )
    _print_param_count_skips(
        skipped_by_param_count,
        min_billions_params=min_billions_params,
        max_billions_params=max_billions_params,
    )
    if not pairs:
        print("No matching .axon/model directory pairs found.")
        return 1

    if dry_run:
        dry_rows = [
            _SummaryRow(
                axon_file=pair.axon_path.name,
                model_dir=str(pair.model_dir),
                hf_runtime_s="DRY-RUN",
                axon_runtime_s="DRY-RUN",
                runtime_ratio="DRY-RUN",
                eval_max_abs_diff="DRY-RUN",
                eval_max_rel_diff="DRY-RUN",
                debug_max_logit_diff="DRY-RUN",
                debug_max_rel_diff="DRY-RUN",
                mean_rel_diff="DRY-RUN",
                masked_max_diff="DRY-RUN",
                masked_last_max_diff="DRY-RUN",
                masked_mean_rel_diff="DRY-RUN",
                masked_max_rel_diff="DRY-RUN",
                eval_top1_eq="DRY-RUN",
                debug_top1_eq="DRY-RUN",
                masked_top1_eq="DRY-RUN",
            )
            for pair in pairs
        ]
        if table_format == "markdown":
            print(_format_table_markdown(dry_rows))
        elif table_format == "html":
            print(_format_table_html(dry_rows))
        else:
            print(_format_table(dry_rows))
        return 0

    skipped = 0
    runnable_pairs: list[_Pair] = []
    skipped_pairs: list[tuple[_Pair, str]] = []
    for pair in pairs:
        reason = _skip_reason_for_pair(pair)
        if reason is None:
            runnable_pairs.append(pair)
        else:
            skipped_pairs.append((pair, reason))
    if skipped_pairs:
        skipped = len(skipped_pairs)
        print(f"Skipped due to unavailable tokenizer/model assets: {skipped}")
        for pair, reason in skipped_pairs:
            print(f"  - {pair.axon_path.name} | {pair.model_dir} | {reason}")
        print()

    if processes == 1:
        rows, passed, failed = _run_runnable_pairs_serial(
            runnable_pairs,
            model_task_override=model_task_override,
            device=device,
            dtype=dtype,
            max_len=max_len,
            prompts=prompts,
            verbose=verbose,
            no_capture_output=no_capture_output,
            compile_hf=compile_hf,
            compile_axon=compile_axon,
            compile_backend=compile_backend,
            compile_mode=compile_mode,
            compile_fullgraph=compile_fullgraph,
            compile_dynamic=compile_dynamic,
            log_dir=log_dir,
        )
    else:
        rows, passed, failed = _run_runnable_pairs_parallel(
            runnable_pairs,
            model_task_override=model_task_override,
            processes=processes,
            device=device,
            dtype=dtype,
            max_len=max_len,
            prompts=prompts,
            verbose=verbose,
            no_capture_output=no_capture_output,
            compile_hf=compile_hf,
            compile_axon=compile_axon,
            compile_backend=compile_backend,
            compile_mode=compile_mode,
            compile_fullgraph=compile_fullgraph,
            compile_dynamic=compile_dynamic,
            log_dir=log_dir,
        )

    print()
    if table_format == "markdown":
        print(_format_table_markdown(rows))
    elif table_format == "html":
        print(_format_table_html(rows))
    else:
        print(_format_table(rows))
    print()
    print(f"Total: {len(pairs)} | Passed: {passed} | Failed: {failed} | Skipped: {skipped}")
    return 0 if failed == 0 else 2


def main() -> int:
    args = _parse_args()
    model_task_override = None if args.model_task == "auto" else str(args.model_task)
    return run_axon_test_matrix(
        examples_dir=args.examples_dir,
        models_dir=args.models_dir,
        device=args.device,
        processes=args.processes,
        dtype=args.dtype,
        min_billions_params=args.min_billions_params,
        max_billions_params=args.max_billions_params,
        max_len=args.max_len,
        text=args.text,
        verbose=args.verbose,
        no_capture_output=bool(args.no_capture_output),
        dry_run=args.dry_run,
        table_format=args.table_format,
        compile_hf=bool(args.compile_hf),
        compile_axon=bool(args.compile_axon),
        compile_backend=(str(args.compile_backend) if args.compile_backend is not None else None),
        compile_mode=str(args.compile_mode) if args.compile_mode is not None else None,
        compile_fullgraph=bool(args.compile_fullgraph),
        compile_dynamic=bool(args.compile_dynamic),
        model_task_override=model_task_override,
        include=args.include,
        exclude=args.exclude,
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "run_axon_test_matrix"]
