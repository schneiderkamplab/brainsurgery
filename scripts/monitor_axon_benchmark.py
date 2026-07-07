#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Iterable

from rich import box
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table


_PAIRED_START_RE = re.compile(
    r"^(?P<ts>\S+ \S+) start index=(?P<index>\d+) gpu=(?P<gpu>\d+) "
    r"backend=(?P<backend>\S+) axon=(?P<axon>\S+) checkpoint=(?P<checkpoint>.+)$"
)
_PAIRED_FINISH_RE = re.compile(
    r"^(?P<ts>\S+ \S+) finish index=(?P<index>\d+) gpu=(?P<gpu>\d+) "
    r"backend=(?P<backend>\S+) rc=(?P<rc>-?\d+) seconds=(?P<seconds>[0-9.]+)"
)
_PARENT_START_RE = re.compile(
    r"child_start .*?pair_index=(?P<index>\d+)(?: backend=(?P<backend>\S+))?.*?"
    r"(?:device=(?P<device>\S+) )?axon=(?P<axon>\S+) checkpoint=(?P<checkpoint>\S+)"
)
_PARENT_FINISH_RE = re.compile(
    r"child_finish .*?pair_index=(?P<index>\d+)(?: backend=(?P<backend>\S+))?.*?"
    r"status=(?P<status>\S+)"
)
_RUN_START_ROWS_RE = re.compile(r"run_start .*?\btotal_rows=(?P<rows>\d+)")
_RUN_START_PAIRS_RE = re.compile(r"run_start .*?\btotal_pairs=(?P<pairs>\d+)")


@dataclass(frozen=True)
class RunningJob:
    lane: str
    task: str
    progress: str = ""
    eta: str = ""


@dataclass(frozen=True)
class MonitorState:
    planned: int
    completed: int
    errors: int
    running: list[RunningJob]
    failed_rows: list[dict[str, str]]
    active: bool
    elapsed_seconds: float | None
    median_min_repeat_times: dict[str, tuple[int, float]]
    median_min_warmup_times: dict[str, tuple[int, float]]
    fastest_counts: dict[str, dict[str, int]]


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except FileNotFoundError:
        return []


def _stream_rows(run_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(run_dir.rglob("stream.csv")):
        rows.extend(_read_csv_rows(path))
    return rows


def _to_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out >= 0 and out < float("inf"):
        return out
    return None


def _sample_min(row: dict[str, object], *, prefix: str, warmup: bool = False) -> float | None:
    suffix = "_warmup_samples" if warmup else "_samples"
    sample_keys = (f"{prefix}_forward{suffix}", f"{prefix}_generate{suffix}")
    for key in sample_keys:
        raw = row.get(key)
        if isinstance(raw, list):
            values = [value for value in (_to_float(item) for item in raw) if value is not None]
            if values:
                return min(values)
    if warmup:
        return None
    return _to_float(row.get(f"{prefix}_time"))


def _result_json_rows(run_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(run_dir.rglob("*.result.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, list):
            rows.extend(item for item in data if isinstance(item, dict))
        elif isinstance(data, dict):
            rows.append(data)
    return rows


def _median_min_times(run_dir: Path, *, warmup: bool) -> dict[str, tuple[int, float]]:
    paired: dict[tuple[str, str, str], dict[str, float]] = {}
    for row in _result_json_rows(run_dir):
        axon = str(row.get("axon_file") or row.get("axon") or "")
        checkpoint = str(row.get("checkpoint_id") or row.get("checkpoint") or "")
        benchmark_path = str(row.get("benchmark_path") or "")
        backend = str(row.get("axon_backend") or row.get("backend") or "")
        if not axon or not checkpoint or not benchmark_path:
            continue
        values = paired.setdefault((axon, checkpoint, benchmark_path), {})
        hf_min = _sample_min(row, prefix="hf", warmup=warmup)
        if hf_min is not None:
            values.setdefault("HF", hf_min)
        axon_min = _sample_min(row, prefix="axon", warmup=warmup)
        if axon_min is not None and backend == "codegen2-torch":
            values["Axon/torch"] = axon_min
        elif axon_min is not None and backend == "codegen2-jax":
            values["Axon/JAX"] = axon_min

    complete = [
        values
        for values in paired.values()
        if all(label in values for label in ("HF", "Axon/torch", "Axon/JAX"))
    ]
    if not complete:
        return {}
    return {
        label: (len(complete), float(median(values[label] for values in complete)))
        for label in ("HF", "Axon/torch", "Axon/JAX")
    }


def _fastest_counts(run_dir: Path, *, warmup: bool) -> dict[str, int]:
    paired: dict[tuple[str, str, str], dict[str, float]] = {}
    for row in _result_json_rows(run_dir):
        axon = str(row.get("axon_file") or row.get("axon") or "")
        checkpoint = str(row.get("checkpoint_id") or row.get("checkpoint") or "")
        benchmark_path = str(row.get("benchmark_path") or "")
        backend = str(row.get("axon_backend") or row.get("backend") or "")
        if not axon or not checkpoint or not benchmark_path:
            continue
        values = paired.setdefault((axon, checkpoint, benchmark_path), {})
        hf_min = _sample_min(row, prefix="hf", warmup=warmup)
        if hf_min is not None:
            values.setdefault("HF", hf_min)
        axon_min = _sample_min(row, prefix="axon", warmup=warmup)
        if axon_min is not None and backend == "codegen2-torch":
            values["Axon/torch"] = axon_min
        elif axon_min is not None and backend == "codegen2-jax":
            values["Axon/JAX"] = axon_min

    counts = {"complete": 0, "HF": 0, "Axon/torch": 0, "Axon/JAX": 0, "ties": 0}
    for values in paired.values():
        if not all(label in values for label in ("HF", "Axon/torch", "Axon/JAX")):
            continue
        counts["complete"] += 1
        fastest = min(values.values())
        winners = [label for label in ("HF", "Axon/torch", "Axon/JAX") if values[label] == fastest]
        if len(winners) != 1:
            counts["ties"] += 1
        else:
            counts[winners[0]] += 1
    return counts


def _paired_status_rows(run_dir: Path) -> list[dict[str, str]]:
    return _read_csv_rows(run_dir / "paired-status.csv")


def _manifest_count(run_dir: Path) -> int | None:
    path = run_dir / "manifest.csv"
    if not path.exists():
        return None
    rows = _read_csv_rows(path)
    return len(rows)


def _parent_log_paths(run_dir: Path) -> list[Path]:
    return (
        sorted(run_dir.rglob("parent-*.txt"))
        + sorted(run_dir.rglob("parent-*.log"))
        + sorted(run_dir.rglob("parent.log"))
    )


def _paired_log_path(run_dir: Path) -> Path | None:
    path = run_dir / "paired-runner.log"
    return path if path.exists() else None


def _iter_lines(paths: Iterable[Path]) -> Iterable[str]:
    for path in paths:
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                yield from handle
        except FileNotFoundError:
            continue


def _planned_from_logs(run_dir: Path) -> int | None:
    for line in _iter_lines(_parent_log_paths(run_dir)):
        rows_match = _RUN_START_ROWS_RE.search(line)
        if rows_match:
            return int(rows_match.group("rows"))
        pairs_match = _RUN_START_PAIRS_RE.search(line)
        if pairs_match:
            return int(pairs_match.group("pairs"))
    return None


def _parse_time(value: str) -> datetime | None:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            parsed = datetime.strptime(value, fmt)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=datetime.now().astimezone().tzinfo)
            return parsed
        except ValueError:
            pass
    return None


def _format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return ""
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _gpu_rows() -> list[tuple[str, str, str]]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
    except Exception:
        return []
    rows: list[tuple[str, str, str]] = []
    for index, line in enumerate(output.strip().splitlines()):
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        used_mib, total_mib, util = parts
        try:
            mem = f"{float(used_mib) / 1024:.1f}/{float(total_mib) / 1024:.1f} GiB"
        except ValueError:
            mem = f"{used_mib}/{total_mib} MiB"
        rows.append((f"GPU{index}", mem, f"{util}%"))
    return rows


def _display_lane(raw: str) -> str:
    if raw.startswith("cuda:") and raw.split(":", 1)[1].isdigit():
        return f"GPU{raw.split(':', 1)[1]}"
    return raw


def _paired_running(run_dir: Path) -> tuple[list[RunningJob], datetime | None]:
    log_path = _paired_log_path(run_dir)
    if log_path is None:
        return [], None
    active: dict[tuple[str, str], RunningJob] = {}
    first_ts: datetime | None = None
    for line in _iter_lines([log_path]):
        start = _PAIRED_START_RE.match(line.strip())
        if start:
            parsed_ts = _parse_time(start.group("ts"))
            first_ts = first_ts or parsed_ts
            key = (start.group("index"), start.group("backend"))
            active[key] = RunningJob(
                lane=f"GPU{start.group('gpu')}",
                task=f"{start.group('backend')} {Path(start.group('axon')).name} | {start.group('checkpoint')}",
            )
            continue
        finish = _PAIRED_FINISH_RE.match(line.strip())
        if finish:
            key = (finish.group("index"), finish.group("backend"))
            active.pop(key, None)
    return list(active.values()), first_ts


def _parent_running(run_dir: Path) -> tuple[list[RunningJob], datetime | None]:
    active: dict[tuple[str, str], RunningJob] = {}
    first_ts: datetime | None = None
    for line in _iter_lines(_parent_log_paths(run_dir)):
        start = _PARENT_START_RE.search(line)
        if start:
            key = (start.group("index"), start.group("backend") or "")
            backend = start.group("backend") or ""
            lane = _display_lane(start.group("device") or "worker")
            active[key] = RunningJob(
                lane=lane,
                task=f"{backend} {start.group('axon')} | {start.group('checkpoint')}".strip(),
            )
            continue
        finish = _PARENT_FINISH_RE.search(line)
        if finish:
            key = (finish.group("index"), finish.group("backend") or "")
            active.pop(key, None)
    return list(active.values()), first_ts


def _state(run_dir: Path) -> MonitorState:
    stream_rows = _stream_rows(run_dir)
    paired_rows = _paired_status_rows(run_dir)
    # Paired benchmark directories contain both per-row stream.csv files and an
    # aggregate paired-status.csv; use the aggregate to avoid double-counting.
    completed_rows = paired_rows if paired_rows else stream_rows
    completed = len(completed_rows)
    failed_rows = [
        row
        for row in completed_rows
        if row.get("masked_top1_eq") == "ERROR" or row.get("returncode") not in ("", None, "0")
    ]
    manifest_count = _manifest_count(run_dir)
    planned = _planned_from_logs(run_dir) or completed
    if paired_rows and manifest_count is not None:
        backend_count = len({row.get("backend", "") for row in paired_rows if row.get("backend")}) or 2
        planned = manifest_count * backend_count
    paired_running, paired_start = _paired_running(run_dir)
    parent_running, parent_start = _parent_running(run_dir)
    running = paired_running or parent_running
    started = paired_start or parent_start
    elapsed = None
    if started is not None:
        elapsed = (datetime.now(tz=started.tzinfo) - started).total_seconds()
    active = bool(running)
    return MonitorState(
        planned=max(planned, completed),
        completed=completed,
        errors=len(failed_rows),
        running=running,
        failed_rows=failed_rows[-20:],
        active=active,
        elapsed_seconds=elapsed,
        median_min_repeat_times=_median_min_times(run_dir, warmup=False),
        median_min_warmup_times=_median_min_times(run_dir, warmup=True),
        fastest_counts={
            "Warmup": _fastest_counts(run_dir, warmup=True),
            "Repeat": _fastest_counts(run_dir, warmup=False),
        },
    )


def _render(run_dir: Path) -> Group:
    state = _state(run_dir)
    completion = (100.0 * state.completed / state.planned) if state.planned else 0.0
    eta = None
    if state.elapsed_seconds and state.completed and state.completed < state.planned:
        eta = state.elapsed_seconds * (state.planned - state.completed) / state.completed
    header = Panel(
        f"{datetime.now().isoformat(timespec='seconds')}  "
        f"rows done={state.completed} running={len(state.running)} errors={state.errors} "
        f"total={state.planned} completion={completion:.1f}% "
        f"elapsed={_format_duration(state.elapsed_seconds)} ETA={_format_duration(eta)}",
        title="Axon benchmark",
        box=box.ROUNDED,
    )

    running = Table(title="Running jobs", box=box.ROUNDED)
    running.add_column("Lane")
    running.add_column("Mem")
    running.add_column("Util", justify="right")
    running.add_column("Task")
    running.add_column("Progress")
    running.add_column("ETA")
    gpu_stats = {lane: (mem, util) for lane, mem, util in _gpu_rows()}
    shown_lanes: set[str] = set()
    for job in state.running:
        mem, util = gpu_stats.get(job.lane, ("-", "-"))
        shown_lanes.add(job.lane)
        running.add_row(job.lane, mem, util, job.task, job.progress, job.eta)
    for lane, (mem, util) in gpu_stats.items():
        if lane not in shown_lanes:
            running.add_row(lane, mem, util, "idle", "", "")
    if not state.running and not gpu_stats:
        running.add_row("none", "", "", "no active jobs detected", "", "")

    timings = Table(title="Paired median fastest timings", box=box.ROUNDED)
    timings.add_column("Framework")
    timings.add_column("Rows", justify="right")
    timings.add_column("Median min warmup", justify="right")
    timings.add_column("Median min repeat", justify="right")
    for framework in ("HF", "Axon/torch", "Axon/JAX"):
        repeat_item = state.median_min_repeat_times.get(framework)
        if repeat_item is None:
            continue
        repeat_count, repeat_value = repeat_item
        warmup_item = state.median_min_warmup_times.get(framework)
        warmup_text = ""
        count = repeat_count
        if warmup_item is not None:
            warmup_count, warmup_value = warmup_item
            warmup_text = f"{warmup_value:.4f}s"
            count = min(count, warmup_count)
        timings.add_row(framework, str(count), warmup_text, f"{repeat_value:.4f}s")
    if not timings.rows:
        timings.add_row("none", "0", "", "")

    fastest = Table(title="Paired fastest counts", box=box.ROUNDED)
    fastest.add_column("Phase")
    fastest.add_column("Rows", justify="right")
    fastest.add_column("HF", justify="right")
    fastest.add_column("Axon/torch", justify="right")
    fastest.add_column("Axon/JAX", justify="right")
    fastest.add_column("Ties", justify="right")
    for phase in ("Warmup", "Repeat"):
        counts = state.fastest_counts.get(phase, {})
        if not counts.get("complete"):
            continue
        fastest.add_row(
            phase,
            str(counts.get("complete", 0)),
            str(counts.get("HF", 0)),
            str(counts.get("Axon/torch", 0)),
            str(counts.get("Axon/JAX", 0)),
            str(counts.get("ties", 0)),
        )
    if not fastest.rows:
        fastest.add_row("none", "0", "0", "0", "0", "0")

    failures = Table(title="Recent failures", box=box.ROUNDED)
    failures.add_column("Backend")
    failures.add_column("Axon")
    failures.add_column("Checkpoint")
    failures.add_column("Status")
    if state.failed_rows:
        for row in state.failed_rows:
            failures.add_row(
                row.get("backend", ""),
                Path(row.get("axon", "")).name,
                row.get("checkpoint", ""),
                row.get("masked_top1_eq") or row.get("returncode") or "ERROR",
            )
    else:
        failures.add_row("none", "", "", "")
    return Group(header, running, timings, fastest, failures)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rich monitor for axon-benchmark log directories.")
    parser.add_argument("run_dir", type=Path, help="Benchmark run directory under log/.")
    parser.add_argument("--refresh", type=float, default=2.0, help="Refresh interval in seconds.")
    parser.add_argument("--watch", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    run_dir = args.run_dir
    if not run_dir.exists():
        raise SystemExit(f"run directory does not exist: {run_dir}")
    if not args.watch:
        from rich.console import Console

        Console().print(_render(run_dir))
        return
    with Live(_render(run_dir), refresh_per_second=max(0.2, 1.0 / max(args.refresh, 0.2))) as live:
        while True:
            live.update(_render(run_dir))
            time.sleep(max(args.refresh, 0.2))


if __name__ == "__main__":
    main()
