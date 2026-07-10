#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median


def _load_rows(log_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    csv_paths = sorted(log_root.glob("**/stream.csv"))
    if not csv_paths:
        csv_paths = sorted(
            p
            for p in log_root.glob("**/*.csv")
            if p.name != "manifest.csv" and p.stat().st_size > 0
        )
    for csv_path in csv_paths:
        if not csv_path.exists() or csv_path.stat().st_size == 0:
            continue
        with csv_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row["_stream_path"] = str(csv_path)
                if not row.get("backend"):
                    inferred_backend = _infer_backend_from_path(csv_path)
                    if inferred_backend:
                        row["backend"] = inferred_backend
                rows.append(row)
    rows.extend(_load_result_json_rows(log_root))
    return rows


def _infer_backend_from_path(path: Path) -> str:
    for part in path.parts:
        if part.startswith("codegen2-") or part.startswith("runtime2-") or part.startswith("pipeline2-"):
            return part
    return ""


def _load_result_json_rows(log_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def value(data: dict[str, object], *names: str) -> str:
        for name in names:
            if name in data and data[name] is not None:
                return str(data[name])
        return ""

    def row_from_data(data: dict[str, object], path: Path) -> dict[str, str] | None:
        axon = data.get("axon_file") or data.get("axon") or data.get("axon_path")
        checkpoint = data.get("checkpoint_id") or data.get("checkpoint")
        if not axon or not checkpoint:
            return None
        return {
            "backend": value(data, "axon_backend", "backend"),
            "axon": str(axon),
            "checkpoint": str(checkpoint),
            "model_dir": value(data, "hf_model_dir", "model_dir"),
            "fallback": value(data, "fallback"),
            "benchmark_path": value(data, "benchmark_path"),
            "masked_top1_eq": value(data, "masked_top1_eq"),
            "masked_max_abs_diff": value(data, "masked_max_diff", "masked_max_abs_diff"),
            "masked_max_rel_diff": value(data, "masked_max_rel_diff"),
            "hf_time": value(data, "hf_time", "hf_time_s"),
            "axon_time": value(data, "axon_time", "axon_time_s"),
            "speed_ratio_axon_over_hf": value(data, "speed_ratio_axon_over_hf"),
            "_result_json_path": str(path),
        }

    for path in sorted(log_root.glob("**/*.result.json")):
        if not path.exists() or path.stat().st_size == 0:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    row = row_from_data(item, path)
                    if row is not None:
                        rows.append(row)
            continue
        if isinstance(data, dict):
            row = row_from_data(data, path)
            if row is not None:
                rows.append(row)
    return rows


def _planned_groups(log_root: Path) -> int:
    manifest = log_root / "manifest.csv"
    if manifest.exists() and manifest.stat().st_size > 0:
        with manifest.open(encoding="utf-8", newline="") as handle:
            manifest_rows = sum(1 for _ in csv.DictReader(handle))
        status = log_root / "paired-status.csv"
        if status.exists() and status.stat().st_size > 0:
            with status.open(encoding="utf-8", newline="") as handle:
                backends = {
                    row.get("backend", "").strip()
                    for row in csv.DictReader(handle)
                    if row.get("backend", "").strip()
                }
            if backends:
                return manifest_rows * len(backends)
        return manifest_rows
    parent_logs = sorted(log_root.glob("**/parent-*.txt"))
    run_start_re = re.compile(r"run_start\s+.*?(?:total_rows=(?P<rows>\d+)|total_pairs=(?P<pairs>\d+))")
    totals: list[int] = []
    for path in parent_logs:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for m in run_start_re.finditer(text):
            totals.append(int(m.group("rows") or m.group("pairs") or "0"))
    if totals:
        return max(totals)
    return 0


def _completed_groups(rows: list[dict[str, str]]) -> int:
    return len({_pair_key(row) for row in rows if _pair_key(row) is not None})


def _get(row: dict[str, str], *names: str) -> str:
    for name in names:
        value = row.get(name, "")
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return ""


def _pair_key(row: dict[str, str]) -> tuple[str, str, str] | None:
    backend = _get(row, "backend", "axon_backend")
    axon = _get(row, "axon")
    ckpt = _get(row, "checkpoint")
    if not axon or not ckpt:
        return None
    return backend, axon, ckpt


def _latest_rows_by_pair(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    latest: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        key = _pair_key(row)
        if key is None:
            continue
        latest[key] = row
    return list(latest.values())


def _drop_dry_run_rows_if_real_rows_exist(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    real_rows = [
        row
        for row in rows
        if _get(row, "fallback", "masked_top1_eq", "masked_max_abs_diff") != "DRY-RUN"
    ]
    return real_rows or rows


def _is_issue_row(row: dict[str, str], abs_threshold: float) -> bool:
    top1 = _get(row, "masked_top1_eq", "masked top-1 eq")
    abs_diff_raw = _get(row, "masked_max_abs_diff", "masked max abs diff")
    err = _get(row, "error", "exception", "traceback")

    if err:
        return True
    if top1 and top1 != "True":
        return True
    if abs_diff_raw:
        try:
            return float(abs_diff_raw) >= abs_threshold
        except ValueError:
            return True
    return False


def _is_generic_axon(axon_path: str) -> bool:
    return "/generic-" in axon_path or Path(axon_path).name.startswith("generic-")


def _fmt_pct(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "0.00%"
    return f"{(100.0 * numerator / denominator):.2f}%"


def _to_float(raw: str) -> float | None:
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    if value >= 0 and value < float("inf"):
        return value
    return None


def _timing_values(row: dict[str, str]) -> tuple[float, float, float] | None:
    ratio = _to_float(
        _get(
            row,
            "speed_ratio_axon_over_hf",
            "axon_over_hf",
            "Axon/HF",
        )
    )
    hf_time = _to_float(_get(row, "hf_time", "hf_time_s", "HF time", "hf"))
    axon_time = _to_float(_get(row, "axon_time", "axon_time_s", "Axon time", "axon_elapsed"))
    if ratio is None and hf_time is not None and axon_time is not None and hf_time > 0:
        ratio = axon_time / hf_time
    if ratio is None or hf_time is None or axon_time is None:
        return None
    return hf_time, axon_time, ratio


def _timed_rows(rows: list[dict[str, str]]) -> list[tuple[dict[str, str], float, float, float]]:
    out: list[tuple[dict[str, str], float, float, float]] = []
    for row in rows:
        values = _timing_values(row)
        if values is None:
            continue
        hf_time, axon_time, ratio = values
        out.append((row, hf_time, axon_time, ratio))
    return out


def _load_paired_status_rows(log_root: Path) -> list[dict[str, str]]:
    path = log_root / "paired-status.csv"
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    index = max(0, min(len(values) - 1, int(math.ceil(q * len(values))) - 1))
    return values[index]


def _paired_wall_time_summary(rows: list[dict[str, str]]) -> list[list[str]]:
    by_backend: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        backend = _get(row, "backend")
        seconds = _to_float(_get(row, "seconds"))
        if backend and seconds is not None:
            by_backend[backend].append(seconds)
    out: list[list[str]] = []
    for backend in sorted(by_backend):
        values = sorted(by_backend[backend])
        out.append(
            [
                backend,
                str(len(values)),
                f"{sum(values):.1f}s",
                f"{mean(values):.2f}s",
                f"{median(values):.2f}s",
                f"{_quantile(values, 0.9):.2f}s",
                f"{max(values):.2f}s",
            ]
        )
    return out


def _paired_backend_ratio_summary(rows: list[dict[str, str]]) -> list[str] | None:
    by_index: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        index = _get(row, "index")
        backend = _get(row, "backend")
        seconds = _to_float(_get(row, "seconds"))
        if index and backend and seconds is not None:
            by_index[index][backend] = seconds
    ratios: list[float] = []
    for per_backend in by_index.values():
        torch_time = per_backend.get("codegen2-torch")
        jax_time = per_backend.get("codegen2-jax")
        if torch_time and jax_time:
            ratios.append(jax_time / torch_time)
    if not ratios:
        return None
    ratios = sorted(ratios)
    return [
        str(len(ratios)),
        f"{mean(ratios):.2f}x",
        f"{median(ratios):.2f}x",
        f"{_quantile(ratios, 0.1):.2f}x",
        f"{_quantile(ratios, 0.9):.2f}x",
        f"{min(ratios):.2f}x",
        f"{max(ratios):.2f}x",
        str(sum(1 for ratio in ratios if ratio < 1.0)),
    ]


def _normalize_axon_path(raw: str) -> str:
    marker = "/brainsurgery/synapse/models/"
    if marker in raw:
        return raw.split(marker, 1)[1]
    return raw


def _detect_run_active(log_root: Path) -> str:
    try:
        proc = subprocess.run(
            ["ps", "-eo", "cmd"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    needle_abs = str(log_root)
    needle_name = log_root.name
    stream_name = f"{log_root.name}.csv"
    for line in proc.stdout.splitlines():
        if "axon-benchmark" not in line:
            continue
        if needle_abs in line or needle_name in line or stream_name in line:
            return "yes"
    return "no"


def _parse_run_start(log_root: Path) -> datetime | None:
    paired_log = log_root / "paired-runner.log"
    if paired_log.exists():
        paired_start_re = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+start\b")
        starts: list[datetime] = []
        for line in paired_log.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = paired_start_re.match(line.strip())
            if m is None:
                continue
            try:
                starts.append(datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
            except ValueError:
                continue
        if starts:
            return min(starts)
    parent_logs = sorted(log_root.glob("**/parent-*.txt"))
    run_start_re = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+run_start\b")
    starts: list[datetime] = []
    for path in parent_logs:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = run_start_re.match(line.strip())
            if m is None:
                continue
            try:
                starts.append(datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
            except ValueError:
                continue
    if not starts:
        return None
    return max(starts)


def _parse_run_end(log_root: Path) -> datetime | None:
    paired_log = log_root / "paired-runner.log"
    if paired_log.exists():
        end_re = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+(?:finish|paired run done)\b")
        ends: list[datetime] = []
        for line in paired_log.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = end_re.match(line.strip())
            if m is None:
                continue
            try:
                ends.append(datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
            except ValueError:
                continue
        if ends:
            return max(ends)
    return None


def _fmt_duration(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def _compute_elapsed_eta(
    *,
    run_start: datetime | None,
    run_end: datetime | None,
    completed: int,
    planned: int,
) -> tuple[str, str]:
    if run_start is None:
        return "n/a", "n/a"
    now = run_end if run_end is not None and planned <= completed else datetime.now()
    elapsed_s = (now - run_start).total_seconds()
    elapsed = _fmt_duration(elapsed_s)
    if completed <= 0 or planned <= completed:
        return elapsed, ("0m 00s" if planned <= completed else "n/a")
    rate = completed / max(elapsed_s, 1e-9)
    remaining = planned - completed
    eta_s = remaining / rate if rate > 0 else float("inf")
    if not (eta_s >= 0 and eta_s < float("inf")):
        return elapsed, "n/a"
    return elapsed, _fmt_duration(eta_s)


def _print_markdown_table(headers: list[str], rows: list[list[str]]) -> None:
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        print("| " + " | ".join(row) + " |")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render 4 markdown tables for axon-benchmark stream CSV/result logs."
    )
    parser.add_argument(
        "log_root",
        type=Path,
        help="Root directory containing per-run/per-family stream.csv files.",
    )
    parser.add_argument(
        "--abs-threshold",
        type=float,
        default=1e-3,
        help="masked_max_abs_diff threshold for issue rows (default: 1e-3).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=200,
        help="Maximum rows to print for issue/mismatch tables (default: 200).",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="",
        help="Optional progress label shown in the first table.",
    )
    args = parser.parse_args()

    log_root = args.log_root.resolve()
    rows_raw = _drop_dry_run_rows_if_real_rows_exist(_load_rows(log_root))
    rows = _latest_rows_by_pair(rows_raw)
    paired_status_rows = _load_paired_status_rows(log_root)
    planned = _planned_groups(log_root)
    completed = _completed_groups(rows)
    run_active = _detect_run_active(log_root)
    run_start = _parse_run_start(log_root)
    run_end = _parse_run_end(log_root)
    elapsed, eta = _compute_elapsed_eta(
        run_start=run_start,
        run_end=run_end,
        completed=completed,
        planned=planned,
    )
    progress_label = args.label.strip() or log_root.name

    issue_rows = [row for row in rows if _is_issue_row(row, args.abs_threshold)]
    error_rows = 0
    top1_bad = 0
    abs_bad = 0
    for row in rows:
        top1 = _get(row, "masked_top1_eq", "masked top-1 eq")
        abs_diff_raw = _get(row, "masked_max_abs_diff", "masked max abs diff")
        err = _get(row, "error", "exception", "traceback")
        has_error = bool(err) or top1.upper() == "ERROR" or abs_diff_raw.upper() == "ERROR"
        if has_error:
            error_rows += 1
            continue
        if top1 and top1 != "True":
            top1_bad += 1
        if abs_diff_raw:
            try:
                if float(abs_diff_raw) >= args.abs_threshold:
                    abs_bad += 1
            except ValueError:
                abs_bad += 1
    healthy = max(completed - len(issue_rows), 0)
    timed = _timed_rows(rows)
    timed_count = len(timed)
    axon_faster = sum(1 for _row, _hf, _axon, ratio in timed if ratio < 1.0)
    axon_slower = sum(1 for _row, _hf, _axon, ratio in timed if ratio >= 1.0)

    progress_rows = [
        [
            progress_label,
            str(completed),
            str(planned),
            _fmt_pct(completed, planned),
            elapsed,
            eta,
            str(healthy),
            str(error_rows),
            str(top1_bad),
            str(abs_bad),
            str(timed_count),
            str(axon_faster),
            str(axon_slower),
            run_active,
        ]
    ]
    _print_markdown_table(
        [
            "Progress",
            "Completed",
            "Planned",
            "Completion",
            "Elapsed",
            "ETA",
            "Healthy",
            "Error rows",
            "masked_top1_eq != True",
            "masked_max_abs_diff >= 1e-3",
            "Timed",
            "Axon faster",
            "Axon/HF >= 1.0",
            "Run active",
        ],
        progress_rows,
    )
    print()

    wall_time_rows = _paired_wall_time_summary(paired_status_rows)
    if wall_time_rows:
        _print_markdown_table(
            [
                "Backend",
                "Rows",
                "Total wall",
                "Mean",
                "Median",
                "P90",
                "Max",
            ],
            wall_time_rows,
        )
        ratio_row = _paired_backend_ratio_summary(paired_status_rows)
        if ratio_row is not None:
            print()
            _print_markdown_table(
                [
                    "Paired rows",
                    "Mean JAX/Torch",
                    "Median JAX/Torch",
                    "P10",
                    "P90",
                    "Min",
                    "Max",
                    "JAX faster",
                ],
                [ratio_row],
            )
        print()

    issue_table_rows: list[list[str]] = []
    for row in issue_rows[: args.max_rows]:
        axon = _normalize_axon_path(_get(row, "axon"))
        issue_table_rows.append(
            [
                axon,
                _get(row, "backend", "axon_backend"),
                _get(row, "checkpoint"),
                _get(row, "fallback"),
                _get(row, "masked_top1_eq", "masked top-1 eq"),
                _get(row, "masked_max_abs_diff", "masked max abs diff"),
            ]
        )
    if not issue_table_rows:
        issue_table_rows = [["(none)", "", "", "", ""]]
    _print_markdown_table(
        [
            "Axon",
            "Backend",
            "Checkpoint",
            "Fallback",
            "masked_top1_eq",
            "masked_max_abs_diff",
        ],
        issue_table_rows,
    )
    print()

    by_ckpt: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        backend = _get(row, "backend", "axon_backend")
        ckpt = _get(row, "checkpoint")
        axon = _get(row, "axon")
        if not ckpt or not axon:
            continue
        kind = "generic" if _is_generic_axon(axon) else "materialized"
        by_ckpt[(backend, ckpt)][kind] = row

    mismatch_rows: list[list[str]] = []
    for backend, ckpt in sorted(by_ckpt):
        pair = by_ckpt[(backend, ckpt)]
        generic = pair.get("generic")
        materialized = pair.get("materialized")
        if generic is None or materialized is None:
            continue
        g_top = _get(generic, "masked_top1_eq", "masked top-1 eq")
        m_top = _get(materialized, "masked_top1_eq", "masked top-1 eq")
        g_abs = _get(generic, "masked_max_abs_diff", "masked max abs diff")
        m_abs = _get(materialized, "masked_max_abs_diff", "masked max abs diff")
        if g_top == m_top and g_abs == m_abs:
            continue
        same_quality = "yes" if (g_top == m_top and g_abs == m_abs) else "no"
        mismatch_rows.append(
            [
                backend,
                ckpt,
                g_abs,
                m_abs,
                same_quality,
            ]
        )
        if len(mismatch_rows) >= args.max_rows:
            break
    if not mismatch_rows:
        mismatch_rows = [["(none)", "", "", "", ""]]
    _print_markdown_table(
        [
            "Backend",
            "Checkpoint",
            "Generic max abs",
            "Materialized max abs",
            "Same quality?",
        ],
        mismatch_rows,
    )
    print()

    slower_rows = sorted(
        (item for item in timed if item[3] >= 1.0),
        key=lambda item: item[3],
        reverse=True,
    )
    slower_table_rows: list[list[str]] = []
    for row, hf_time, axon_time, ratio in slower_rows[: args.max_rows]:
        slower_table_rows.append(
            [
                _normalize_axon_path(_get(row, "axon")),
                _get(row, "backend", "axon_backend"),
                _get(row, "checkpoint"),
                f"{hf_time:.4f}s",
                f"{axon_time:.4f}s",
                f"{ratio:.3f}",
                _get(row, "masked_top1_eq", "masked top-1 eq"),
                _get(row, "masked_max_abs_diff", "masked max abs diff"),
            ]
        )
    if not slower_table_rows:
        slower_table_rows = [["(none)", "", "", "", "", "", "", ""]]
    _print_markdown_table(
        [
            "Axon",
            "Backend",
            "Checkpoint",
            "HF time",
            "Axon time",
            "Axon/HF",
            "masked_top1_eq",
            "masked_max_abs_diff",
        ],
        slower_table_rows,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
