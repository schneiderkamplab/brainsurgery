from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

LOG_NAME_RE = re.compile(r"^log-(?P<pid>\d+)-(?P<axon>.+)-(?P<model>.+)\.txt$")
MASKED_TOP1_RE = re.compile(r"^Masked top1_eq:\s*(?P<value>.+?)\s*$")
MASKED_ABS_RE = re.compile(r"^Masked abs diff \(max\):\s*(?P<value>.+?)\s*$")
MASKED_REL_RE = re.compile(r"^Logits rel diff \(masked\) \| mean/max:\s+\S+\s+(?P<value>\S+)\s*$")
RESULT_AXON_RE = re.compile(r"^result\.axon=(?P<value>.+?)\s*$")
RESULT_MODEL_RE = re.compile(r"^result\.model_dir=(?P<value>.+?)\s*$")
RESULT_TOP1_RE = re.compile(r"^result\.masked_top1_eq=(?P<value>.+?)\s*$")
RESULT_ABS_RE = re.compile(r"^result\.masked_max_abs_diff=(?P<value>.+?)\s*$")
RESULT_REL_RE = re.compile(r"^result\.masked_max_rel_diff=(?P<value>.+?)\s*$")
PARENT_CHILD_START_RE = re.compile(r"child_start .*? log_path=(?P<log_path>\S+)\s*$")


@dataclass(frozen=True)
class AxonTestLogRow:
    axon: str
    model_dir: str
    masked_top1_eq: str
    masked_max_abs_diff: float
    masked_max_abs_diff_text: str
    masked_max_rel_diff_text: str


@dataclass(frozen=True)
class _RunRow:
    row: AxonTestLogRow
    run_mtime: float


def _parse_bool_key(value: str) -> tuple[int, str]:
    normalized = value.strip()
    if normalized == "True":
        return (0, normalized)
    if normalized == "False":
        return (1, normalized)
    return (2, normalized)


def _parse_float(text: str) -> float:
    value = float(text)
    if math.isnan(value):
        return math.inf
    return value


def _latest_parent_log(log_dir: Path) -> Path | None:
    parents = sorted(
        log_dir.glob("parent-*.txt"), key=lambda path: path.stat().st_mtime, reverse=True
    )
    if not parents:
        return None
    return parents[0]


def _all_parent_logs(log_dir: Path) -> list[Path]:
    return sorted(log_dir.glob("parent-*.txt"), key=lambda path: path.stat().st_mtime)


def _worker_logs_for_parent(parent_log: Path | None) -> list[Path]:
    if parent_log is None:
        return []
    base_dir = parent_log.parent
    worker_logs: list[Path] = []
    seen: set[Path] = set()
    with parent_log.open("r", encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            match = PARENT_CHILD_START_RE.search(raw_line.strip())
            if match is None:
                continue
            candidate = Path(match.group("log_path"))
            if not candidate.is_absolute():
                candidate = base_dir / candidate
            candidate = candidate.resolve()
            if candidate in seen:
                continue
            seen.add(candidate)
            worker_logs.append(candidate)
    return worker_logs


def _parse_row(path: Path) -> AxonTestLogRow | None:
    if not path.exists():
        return None

    match = LOG_NAME_RE.match(path.name)
    if match is None:
        return None

    axon_name = f"{match.group('axon')}.axon"
    model_dir = match.group("model")
    masked_top1_eq: str | None = None
    masked_max_abs_diff: str | None = None
    masked_max_rel_diff: str | None = None

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            axon_match = RESULT_AXON_RE.match(line)
            if axon_match is not None:
                axon_name = axon_match.group("value")
                continue
            model_match = RESULT_MODEL_RE.match(line)
            if model_match is not None:
                model_dir = Path(model_match.group("value")).name
                continue
            top1_result_match = RESULT_TOP1_RE.match(line)
            if top1_result_match is not None:
                masked_top1_eq = top1_result_match.group("value")
                continue
            abs_result_match = RESULT_ABS_RE.match(line)
            if abs_result_match is not None:
                masked_max_abs_diff = abs_result_match.group("value")
                continue
            rel_result_match = RESULT_REL_RE.match(line)
            if rel_result_match is not None:
                masked_max_rel_diff = rel_result_match.group("value")
                continue
            top1_match = MASKED_TOP1_RE.match(line)
            if top1_match is not None:
                masked_top1_eq = top1_match.group("value")
                continue
            abs_match = MASKED_ABS_RE.match(line)
            if abs_match is not None:
                masked_max_abs_diff = abs_match.group("value")
                continue
            rel_match = MASKED_REL_RE.match(line)
            if rel_match is not None:
                masked_max_rel_diff = rel_match.group("value")
                continue

    if masked_top1_eq is None or masked_max_abs_diff is None or masked_max_rel_diff is None:
        return None

    return AxonTestLogRow(
        axon=axon_name,
        model_dir=model_dir,
        masked_top1_eq=masked_top1_eq,
        masked_max_abs_diff=_parse_float(masked_max_abs_diff),
        masked_max_abs_diff_text=masked_max_abs_diff,
        masked_max_rel_diff_text=masked_max_rel_diff,
    )


def _rows_for_latest_parent(log_dir: Path) -> list[AxonTestLogRow]:
    latest_parent = _latest_parent_log(log_dir)
    files = _worker_logs_for_parent(latest_parent)
    return [row for row in (_parse_row(path) for path in files) if row is not None]


def _latest_rows_for_all_parents(log_dir: Path) -> list[AxonTestLogRow]:
    latest_by_pair: dict[tuple[str, str], _RunRow] = {}
    for parent in _all_parent_logs(log_dir):
        run_mtime = parent.stat().st_mtime
        for worker_log in _worker_logs_for_parent(parent):
            row = _parse_row(worker_log)
            if row is None:
                continue
            key = (row.axon, row.model_dir)
            prev = latest_by_pair.get(key)
            if prev is None or run_mtime >= prev.run_mtime:
                latest_by_pair[key] = _RunRow(row=row, run_mtime=run_mtime)
    return [item.row for item in latest_by_pair.values()]


def format_axon_test_log_markdown(rows: list[AxonTestLogRow]) -> str:
    header = [
        "| axon | model dir | masked top-1 eq | masked max abs diff | masked max rel diff |",
        "|---|---|---:|---:|---:|",
    ]
    body = [
        f"| `{row.axon}` | `{row.model_dir}` | `{row.masked_top1_eq}` | "
        f"`{row.masked_max_abs_diff_text}` | `{row.masked_max_rel_diff_text}` |"
        for row in rows
    ]
    return "\n".join(header + body)


def prune_axon_test_logs_to_latest_run(log_dir: Path) -> None:
    latest_parent = _latest_parent_log(log_dir)
    if latest_parent is None:
        return

    keep: set[Path] = {latest_parent.resolve()}
    keep.update(path.resolve() for path in _worker_logs_for_parent(latest_parent))

    for path in log_dir.glob("parent-*.txt"):
        if path.resolve() not in keep:
            path.unlink(missing_ok=True)

    for path in log_dir.glob("log-*.txt"):
        if path.resolve() not in keep:
            path.unlink(missing_ok=True)


def load_axon_test_log_rows(log_dir: Path, *, all_runs: bool = False) -> list[AxonTestLogRow]:
    rows = _latest_rows_for_all_parents(log_dir) if all_runs else _rows_for_latest_parent(log_dir)
    rows.sort(key=lambda row: (_parse_bool_key(row.masked_top1_eq), row.masked_max_abs_diff))
    return rows


def render_axon_test_log(log_dir: Path, *, all_runs: bool = False, prune: bool = False) -> str:
    if prune:
        prune_axon_test_logs_to_latest_run(log_dir)
    rows = load_axon_test_log_rows(log_dir, all_runs=all_runs)
    return format_axon_test_log_markdown(rows)
