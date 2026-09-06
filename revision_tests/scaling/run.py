#!/usr/bin/env python3
"""Run the frozen EACL 2027 checkpoint-scaling experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil
import torch
import yaml
from safetensors.torch import save_file

try:
    from .oracle import (
        PROTOCOL_ID,
        checkpoint_manifest,
        compare_output,
        discover_checkpoint,
        files_unchanged,
        sha256_file,
        validate_input_operation,
        verify_huggingface_revision,
    )
    from .validate_protocol import EXPECTED_IDS, EXPECTED_METHODS, load_cases
except ImportError:
    from oracle import (
        PROTOCOL_ID,
        checkpoint_manifest,
        compare_output,
        discover_checkpoint,
        files_unchanged,
        sha256_file,
        validate_input_operation,
        verify_huggingface_revision,
    )
    from validate_protocol import EXPECTED_IDS, EXPECTED_METHODS, load_cases

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--log-root", type=Path, default=REPO / "log" / "revision_tests")
    parser.add_argument("--brainsurgery-cli", type=Path, default=REPO / ".venv" / "bin" / "brainsurgery")
    parser.add_argument("--python", type=Path, default=REPO / ".venv" / "bin" / "python")
    parser.add_argument("--model", action="append", choices=EXPECTED_IDS, dest="models")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--sample-interval-ms", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--workload-note")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--keep-outputs", action="store_true")
    parser.add_argument("--publish-dir", type=Path)
    return parser.parse_args()


def require_args(args: argparse.Namespace) -> None:
    if not args.run_id or not all(char.isalnum() or char == "_" for char in args.run_id):
        raise SystemExit("--run-id must contain only letters, digits, and underscores")
    if args.repetitions < 1 or args.num_workers < 1:
        raise SystemExit("repetitions and num-workers must be positive")
    if args.sample_interval_ms < 1 or args.timeout < 1:
        raise SystemExit("sample interval and timeout must be positive")
    if not args.smoke and args.repetitions < 5:
        raise SystemExit("reported-size runs require at least five measured repetitions")
    if not args.smoke and args.num_workers != 1:
        raise SystemExit("the frozen reported protocol requires --num-workers 1")
    if not args.smoke and not args.workload_note:
        raise SystemExit("reported-size runs require --workload-note")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_value(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO, check=True, text=True, capture_output=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def command_output(command: list[str]) -> str:
    try:
        return subprocess.run(command, check=False, text=True, capture_output=True).stdout.strip()
    except OSError:
        return "unavailable"


def collect_environment(args: argparse.Namespace) -> dict[str, Any]:
    try:
        affinity = psutil.Process().cpu_affinity()
    except (AttributeError, psutil.AccessDenied, PermissionError):
        affinity = None
    packages = {}
    for package in ("brainsurgery", "torch", "safetensors", "psutil", "pyyaml"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "unavailable"
    cli_path = args.brainsurgery_cli.absolute()
    python_path = args.python.absolute()
    return {
        "protocol_id": PROTOCOL_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "python_executable": str(python_path),
        "brainsurgery_cli": str(cli_path),
        "brainsurgery_cli_sha256": sha256_file(cli_path) if cli_path.is_file() else None,
        "packages": packages,
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_affinity": affinity,
        "memory_bytes": psutil.virtual_memory().total,
        "disk": dict(psutil.disk_usage(str(args.log_root.resolve().parent))._asdict()),
        "filesystem": command_output(["df", "-P", str(args.log_root.resolve().parent)]),
        "gpu_inventory_only_not_used": command_output(
            ["nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version", "--format=csv,noheader"]
        ),
        "execution_device": "cpu",
        "cuda_visible_devices": "",
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_status_porcelain": git_value("status", "--short"),
        "cache_policy": "one_unmeasured_warmup_then_warm_cache_no_os_cache_drop",
        "num_workers": args.num_workers,
        "sample_interval_ms": args.sample_interval_ms,
        "operator_workload_note": args.workload_note,
    }


def initial_eligibility_reasons(args: argparse.Namespace, environment: dict[str, Any], model_ids: list[str]) -> list[str]:
    reasons = []
    if args.smoke:
        reasons.append("run was explicitly requested as a synthetic smoke preflight")
    if environment["system"] != "Linux":
        reasons.append("reported systems run requires Linux")
    if environment["git_commit"] == "unavailable":
        reasons.append("Git commit is unavailable")
    if environment["git_status_porcelain"]:
        reasons.append("Git worktree is dirty")
    if model_ids != EXPECTED_IDS:
        reasons.append("reported evaluation requires the complete ordered ten-checkpoint matrix")
    return reasons


def make_smoke_model(root: Path) -> dict[str, Any]:
    root.mkdir(parents=True)
    state = {
        "block.0.weight": torch.arange(24, dtype=torch.float32).reshape(6, 4),
        "block.0.bias": torch.tensor([1.0, -2.0, 3.0, -4.0], dtype=torch.float32),
        "block.1.weight": torch.arange(16, dtype=torch.float16).reshape(4, 4),
        "counter": torch.tensor([7], dtype=torch.int64),
    }
    save_file(state, str(root / "model.safetensors"))
    return {
        "id": "SMOKE",
        "display": "deterministic tiny fixture",
        "model_id": "local/synthetic",
        "revision": hashlib.sha256((HERE / "run.py").read_bytes()).hexdigest(),
        "input": str(root),
        "expected_layout": "single",
    }


def source_file_hashes(path: Path) -> dict[str, str]:
    layout = discover_checkpoint(path)
    files = list(layout["data_files"])
    if layout["index_file"] is not None:
        files.append(layout["index_file"])
    return {item.name: sha256_file(item) for item in files}


def directory_usage(path: Path) -> tuple[int, int]:
    logical = 0
    allocated = 0
    if not path.exists():
        return 0, 0
    for item in path.rglob("*"):
        try:
            is_file = item.is_file()
            stat = item.stat() if is_file else None
        except OSError:
            continue
        if stat is not None:
            logical += stat.st_size
            allocated += getattr(stat, "st_blocks", 0) * 512
    return logical, allocated


def run_monitored(
    command: list[str],
    *,
    stdout_path: Path,
    stderr_path: Path,
    temp_path: Path,
    timeout: int,
    interval_seconds: float,
    environment: dict[str, str],
) -> dict[str, Any]:
    started = time.perf_counter()
    peak_rss = 0
    peak_temp_logical = 0
    peak_temp_allocated = 0
    io_by_pid: dict[int, tuple[int, int]] = {}
    rss_sampled_pids: set[int] = set()
    sampling_failure_pids: set[int] = set()
    timed_out = False
    degraded = False
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(command, cwd=REPO, env=environment, stdout=stdout, stderr=stderr)
        try:
            root: psutil.Process | None = psutil.Process(process.pid)
        except psutil.NoSuchProcess:
            root = None
            degraded = True
        while process.poll() is None:
            if time.perf_counter() - started > timeout:
                timed_out = True
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                break
            processes = [root] if root is not None else []
            if root is not None:
                try:
                    processes.extend(root.children(recursive=True))
                except psutil.NoSuchProcess:
                    processes = []
                except (psutil.AccessDenied, PermissionError):
                    degraded = True
            rss = 0
            for child in processes:
                try:
                    rss += child.memory_info().rss
                    rss_sampled_pids.add(child.pid)
                    counters = child.io_counters()
                    previous = io_by_pid.get(child.pid, (0, 0))
                    io_by_pid[child.pid] = (
                        max(previous[0], counters.read_bytes),
                        max(previous[1], counters.write_bytes),
                    )
                except psutil.NoSuchProcess:
                    continue
                except (psutil.AccessDenied, AttributeError, PermissionError):
                    sampling_failure_pids.add(child.pid)
            peak_rss = max(peak_rss, rss)
            logical, allocated = directory_usage(temp_path)
            peak_temp_logical = max(peak_temp_logical, logical)
            peak_temp_allocated = max(peak_temp_allocated, allocated)
            time.sleep(interval_seconds)
        returncode = process.wait()
    logical, allocated = directory_usage(temp_path)
    unresolved_sampling_pids = {
        pid
        for pid in sampling_failure_pids
        if pid not in rss_sampled_pids or pid not in io_by_pid
    }
    degraded |= bool(unresolved_sampling_pids)
    return {
        "command": command,
        "command_shell_display": shlex.join(command),
        "returncode": 124 if timed_out else returncode,
        "timed_out": timed_out,
        "wall_seconds": time.perf_counter() - started,
        "peak_process_tree_rss_bytes": peak_rss,
        "process_tree_read_bytes": sum(value[0] for value in io_by_pid.values()),
        "process_tree_write_bytes": sum(value[1] for value in io_by_pid.values()),
        "peak_temp_logical_bytes": max(peak_temp_logical, logical),
        "peak_temp_allocated_bytes": max(peak_temp_allocated, allocated),
        "sampling_interval_seconds": interval_seconds,
        "resource_sampling_degraded": degraded,
        "gpu_peak_memory_bytes": None,
        "gpu_used": False,
    }


def write_plan(path: Path, input_path: Path, output_path: Path, operation: dict[str, Any]) -> None:
    plan = {
        "inputs": [f"model::{input_path}"],
        "transforms": [{"scale_": {"target": operation["target_regex"], "by": operation["factor"]}}],
        "output": {
            "path": str(output_path),
            "format": "safetensors",
            "shard": operation["output_shard_size"],
        },
    }
    path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")


def command_for(
    args: argparse.Namespace,
    method: str,
    input_path: Path,
    output_path: Path,
    plan_path: Path,
    arena_path: Path,
    operation: dict[str, Any],
) -> list[str]:
    if method == "python_pytorch":
        return [
            str(args.python.absolute()), str(HERE / "baseline.py"),
            "--input", str(input_path), "--output", str(output_path),
            "--target-regex", operation["target_regex"], "--factor", str(operation["factor"]),
            "--shard-size-bytes", str(operation["output_shard_size_bytes"]),
        ]
    provider = "inmemory" if method == "brainsurgery_inmemory" else "arena"
    command = [
        str(args.brainsurgery_cli.resolve()), str(plan_path), "--provider", provider,
        "--num-workers", str(args.num_workers), "--no-summarize", "--log-level", "warning",
    ]
    if provider == "arena":
        segment_size = "1MB" if args.smoke else operation["arena_segment_size"]
        command.extend(["--arena-root", str(arena_path), "--arena-segment-size", segment_size])
    return command


def execute_attempt(
    args: argparse.Namespace,
    *,
    model: dict[str, Any],
    input_path: Path,
    input_manifest: dict[str, Any],
    operation: dict[str, Any],
    attempts_root: Path,
    method: str,
    phase: str,
    repetition: int,
    schedule_index: int,
) -> dict[str, Any]:
    attempt_id = f"{schedule_index:03d}_{model['id']}_{phase}_{repetition:02d}_{method}"
    attempt_dir = attempts_root / attempt_id
    attempt_dir.mkdir(parents=True)
    output_path = attempt_dir / "output"
    arena_path = attempt_dir / "arena"
    plan_path = attempt_dir / "plan.yaml"
    write_plan(plan_path, input_path, output_path, operation)
    command = command_for(args, method, input_path, output_path, plan_path, arena_path, operation)
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    process = run_monitored(
        command,
        stdout_path=attempt_dir / "stdout.txt",
        stderr_path=attempt_dir / "stderr.txt",
        temp_path=arena_path,
        timeout=args.timeout,
        interval_seconds=args.sample_interval_ms / 1000,
        environment=environment,
    )
    validation = None
    validation_error = None
    if process["returncode"] == 0:
        try:
            validation = compare_output(
                input_path, output_path,
                target_regex=operation["target_regex"],
                factor=operation["factor"],
                shard_size_bytes=operation["output_shard_size_bytes"],
            )
        except Exception as exc:
            validation_error = f"{type(exc).__name__}: {exc}"
    correct = bool(validation and validation["passed"])
    output_bytes = (
        validation["output_manifest"]["checkpoint_file_bytes"] if validation else 0
    )
    wall = process["wall_seconds"]
    logical_processed = input_manifest["checkpoint_file_bytes"] + output_bytes
    result = {
        "protocol_id": PROTOCOL_ID,
        "attempt_id": attempt_id,
        "model_id": model["id"],
        "method": method,
        "phase": phase,
        "measured": phase == "measured",
        "repetition": repetition,
        "schedule_index": schedule_index,
        "process": process,
        "input_checkpoint_bytes": input_manifest["checkpoint_file_bytes"],
        "output_checkpoint_bytes": output_bytes,
        "effective_logical_bytes_per_second": logical_processed / wall if wall > 0 else None,
        "validation": validation,
        "validation_error": validation_error,
        "correct": correct,
        "timing_eligible": phase == "measured" and correct and not process["resource_sampling_degraded"],
        "output_removed_after_validation": False,
        "arena_leftover_after_exit": directory_usage(arena_path),
    }
    if correct and not args.keep_outputs:
        shutil.rmtree(output_path)
        result["output_removed_after_validation"] = True
    write_json(attempt_dir / "result.json", result)
    return result


def metric(values: list[float | int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "values": [], "median": None, "minimum": None, "maximum": None, "mean": None, "sample_standard_deviation": None}
    return {
        "count": len(values), "values": values, "median": statistics.median(values),
        "minimum": min(values), "maximum": max(values), "mean": statistics.mean(values),
        "sample_standard_deviation": statistics.stdev(values) if len(values) > 1 else None,
    }


def build_summary(
    args: argparse.Namespace,
    environment: dict[str, Any],
    models: list[dict[str, Any]],
    manifests: dict[str, dict[str, Any]],
    results: list[dict[str, Any]],
    eligibility_reasons: list[str],
    integrity: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    pairs: dict[str, Any] = {}
    for model in models:
        for method in EXPECTED_METHODS:
            attempts = [item for item in results if item["model_id"] == model["id"] and item["method"] == method and item["measured"]]
            correct = [item for item in attempts if item["correct"]]
            valid = [item for item in attempts if item["timing_eligible"]]
            key = f"{model['id']}:{method}"
            pairs[key] = {
                "model_id": model["id"], "display": model["display"],
                "family": model.get("family", "synthetic"),
                "analysis_role": model.get("analysis_role", "smoke"),
                "method": method,
                "measured_attempts": len(attempts), "correct_attempts": sum(item["correct"] for item in attempts),
                "timing_eligible_attempts": len(valid),
                "wall_seconds": metric([item["process"]["wall_seconds"] for item in valid]),
                "peak_rss_bytes": metric([item["process"]["peak_process_tree_rss_bytes"] for item in valid]),
                "process_read_bytes": metric([item["process"]["process_tree_read_bytes"] for item in valid]),
                "process_write_bytes": metric([item["process"]["process_tree_write_bytes"] for item in valid]),
                "peak_temp_allocated_bytes": metric([item["process"]["peak_temp_allocated_bytes"] for item in valid]),
                "effective_logical_bytes_per_second": metric([item["effective_logical_bytes_per_second"] for item in valid]),
                "output_checkpoint_bytes": metric([item["output_checkpoint_bytes"] for item in valid]),
                "output_shard_counts": sorted({item["validation"]["output_manifest"]["data_file_count"] for item in correct}),
            }
    final_reasons = list(eligibility_reasons)
    if any(not value["passed"] for value in integrity.values()):
        final_reasons.append("one or more source checkpoints changed during execution")
    expected_pairs = len(models) * len(EXPECTED_METHODS)
    correctness_complete = len(pairs) == expected_pairs and all(
        pair["correct_attempts"] == args.repetitions for pair in pairs.values()
    )
    measurement_complete = len(pairs) == expected_pairs and all(
        pair["timing_eligible_attempts"] == args.repetitions for pair in pairs.values()
    )
    if not correctness_complete:
        final_reasons.append("correct measured repetitions are incomplete")
    if not measurement_complete:
        final_reasons.append("correct, fully sampled measured repetitions are incomplete")
    return {
        "protocol_id": PROTOCOL_ID,
        "run_id": args.run_id,
        "run_directory": f"log/revision_tests/{args.run_id}/scaling",
        "git_commit": environment["git_commit"],
        "reported_eligible": not final_reasons,
        "eligibility_reasons": final_reasons,
        "correctness_complete": correctness_complete,
        "measurement_complete": measurement_complete,
        "execution_device": "cpu",
        "gpu_used": False,
        "warmup_attempts": sum(not item["measured"] for item in results),
        "measured_attempts": sum(item["measured"] for item in results),
        "correct_measured_attempts": sum(item["measured"] and item["correct"] for item in results),
        "models": {
            model["id"]: {
                "display": model["display"],
                "family": model.get("family", "synthetic"),
                "analysis_role": model.get("analysis_role", "smoke"),
                "model_id": model["model_id"],
                "nominal_parameter_count": model.get("nominal_parameter_count"),
                "revision": model["revision"],
                "input_manifest": manifests[model["id"]],
                "source_integrity": integrity[model["id"]],
            }
            for model in models
        },
        "pairs": pairs,
        "claim_boundary": "Frozen CPU checkpoint rewrite on the enumerated revisions and one recorded Linux system; not GPU, training, inference, usability, or general tool superiority evidence.",
    }


def render_markdown(summary: dict[str, Any]) -> str:
    reportable = summary["reported_eligible"]
    lines = [
        "# Checkpoint-scaling result", "", f"Protocol: `{summary['protocol_id']}`  ",
        f"Run: `{summary['run_id']}`  ", f"Commit: `{summary['git_commit']}`  ",
        f"Status: **{'REPORTABLE CANDIDATE' if reportable else 'NON-REPORTABLE PREFLIGHT'}**", "",
        "| Model | Method | Correct | Wall median (s) | Peak RSS (GiB) | Effective GiB/s | Output shards |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for pair in summary["pairs"].values():
        wall = pair["wall_seconds"]["median"]
        rss = pair["peak_rss_bytes"]["median"]
        throughput = pair["effective_logical_bytes_per_second"]["median"]
        visible = reportable and wall is not None and rss is not None and throughput is not None
        lines.append(
            f"| {pair['display']} | {pair['method']} | {pair['correct_attempts']}/{pair['measured_attempts']} | "
            f"{wall:.3f} | {rss / 1024**3:.3f} | {throughput / 1024**3:.3f} | {','.join(map(str, pair['output_shard_counts']))} |"
            if visible else
            f"| {pair['display']} | {pair['method']} | {pair['correct_attempts']}/{pair['measured_attempts']} | — | — | — | {','.join(map(str, pair['output_shard_counts'])) or '—'} |"
        )
    if not reportable:
        lines.extend(["", "Performance values are suppressed: " + "; ".join(summary["eligibility_reasons"]) + "."])
    lines.extend(["", summary["claim_boundary"], ""])
    return "\n".join(lines)


def render_latex(summary: dict[str, Any]) -> str:
    reportable = summary["reported_eligible"]
    rows = []
    names = {"python_pytorch": "Python/PyTorch", "brainsurgery_inmemory": "BrainSurgery (memory)", "brainsurgery_arena": "BrainSurgery (arena)"}
    for pair in summary["pairs"].values():
        wall = pair["wall_seconds"]["median"]
        rss = pair["peak_rss_bytes"]["median"]
        throughput = pair["effective_logical_bytes_per_second"]["median"]
        if reportable and wall is not None and rss is not None and throughput is not None:
            values = f"{wall:.2f} & {rss / 1024**3:.2f} & {throughput / 1024**3:.2f}"
        else:
            values = "-- & -- & --"
        rows.append(f"{pair['model_id']} & {names[pair['method']]} & {pair['correct_attempts']}/{pair['measured_attempts']} & {values} \\\\")
    warning = "% Human-audit reportable candidate." if reportable else "% NON-REPORTABLE PREFLIGHT: performance suppressed."
    caption = "Controlled CPU checkpoint rewrite; medians over correctness-validated measured repetitions." if reportable else "Non-reportable scaling-harness preflight; performance fields are suppressed."
    return (
        f"{warning}\n\\begin{{table*}}[t]\n\\centering\n\\small\n"
        "\\begin{tabular}{lllrrr}\n\\toprule\nModel & Method & Correct & Wall (s) & Peak RSS (GiB) & Effective GiB/s \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
        + f"\\caption{{{caption}}}\n\\label{{tab:checkpoint-scaling}}\n\\end{{table*}}\n"
    )


def render_text(summary: dict[str, Any]) -> str:
    if not summary["reported_eligible"]:
        return (
            "# Scaling preflight interpretation\n\n"
            f"The synthetic harness produced correct outputs for {summary['correct_measured_attempts']}/{summary['measured_attempts']} measured attempts. "
            "This validates orchestration, monitoring, sharded output, and the independent oracle only. Performance, scaling, efficiency, GPU, usability, and downstream claims are prohibited for this preflight.\n"
        )
    return (
        "# Scaling result interpretation\n\n"
        f"All {summary['correct_measured_attempts']}/{summary['measured_attempts']} measured model/method attempts passed the independent exact oracle. "
        "Use the four Pythia points for the primary within-family curve and show the GPT-2, OLMo, and Qwen2.5 pairs separately. "
        "Report the table as a controlled CPU checkpoint-rewrite comparison on the recorded Linux host, together with full repetition values and the stated claim boundary.\n"
    )


def render_csv(summary: dict[str, Any]) -> str:
    columns = [
        "model_key",
        "family",
        "analysis_role",
        "display",
        "upstream_model_id",
        "revision",
        "nominal_parameter_count",
        "stored_tensor_element_count",
        "logical_tensor_bytes",
        "checkpoint_file_bytes",
        "dtype_logical_bytes",
        "input_shards",
        "method",
        "correct_attempts",
        "measured_attempts",
        "median_wall_seconds",
        "median_peak_rss_bytes",
        "median_peak_temp_allocated_bytes",
        "median_effective_logical_bytes_per_second",
        "median_output_checkpoint_bytes",
        "output_shard_counts",
        "reported_eligible",
    ]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for pair in summary["pairs"].values():
        model = summary["models"][pair["model_id"]]
        manifest = model["input_manifest"]
        visible = summary["reported_eligible"]
        writer.writerow(
            {
                "model_key": pair["model_id"],
                "family": pair["family"],
                "analysis_role": pair["analysis_role"],
                "display": pair["display"],
                "upstream_model_id": model["model_id"],
                "revision": model["revision"],
                "nominal_parameter_count": model["nominal_parameter_count"],
                "stored_tensor_element_count": manifest["stored_tensor_element_count"],
                "logical_tensor_bytes": manifest["logical_tensor_bytes"],
                "checkpoint_file_bytes": manifest["checkpoint_file_bytes"],
                "dtype_logical_bytes": json.dumps(manifest["dtype_logical_bytes"], sort_keys=True),
                "input_shards": manifest["data_file_count"],
                "method": pair["method"],
                "correct_attempts": pair["correct_attempts"],
                "measured_attempts": pair["measured_attempts"],
                "median_wall_seconds": pair["wall_seconds"]["median"] if visible else "",
                "median_peak_rss_bytes": pair["peak_rss_bytes"]["median"] if visible else "",
                "median_peak_temp_allocated_bytes": pair["peak_temp_allocated_bytes"]["median"] if visible else "",
                "median_effective_logical_bytes_per_second": pair["effective_logical_bytes_per_second"]["median"] if visible else "",
                "median_output_checkpoint_bytes": pair["output_checkpoint_bytes"]["median"] if visible else "",
                "output_shard_counts": ";".join(map(str, pair["output_shard_counts"])),
                "reported_eligible": str(summary["reported_eligible"]).lower(),
            }
        )
    return output.getvalue()


def publish(run_dir: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite publish directory: {destination}")
    destination.mkdir(parents=True)
    for name in (
        "summary.json",
        "paper_data.csv",
        "paper_table.md",
        "paper_table.tex",
        "paper_text.md",
    ):
        shutil.copy2(run_dir / name, destination / name)


def main() -> int:
    args = parse_args()
    require_args(args)
    cases = load_cases()
    run_dir = args.log_root.resolve() / args.run_id / "scaling"
    if run_dir.exists():
        raise SystemExit(f"refusing to overwrite existing run directory: {run_dir}")
    run_dir.mkdir(parents=True)
    (run_dir / "command.txt").write_text(
        shlex.join([sys.executable, *sys.argv]) + "\n", encoding="utf-8"
    )
    write_json(run_dir / "arguments.json", {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})
    environment = collect_environment(args)
    write_json(run_dir / "environment.json", environment)

    if not args.brainsurgery_cli.resolve().is_file():
        raise SystemExit(f"BrainSurgery CLI does not exist: {args.brainsurgery_cli}")
    if not args.python.absolute().is_file():
        raise SystemExit(f"Python executable does not exist: {args.python}")
    fixture_root = run_dir / "fixture"
    if args.smoke:
        models = [make_smoke_model(fixture_root)]
    else:
        selected = args.models or EXPECTED_IDS
        models = [model for model in cases["models"] if model["id"] in selected]
    model_ids = [model["id"] for model in models]
    eligibility_reasons = initial_eligibility_reasons(args, environment, model_ids)
    if not args.smoke and eligibility_reasons:
        write_json(run_dir / "preflight_failure.json", {"reasons": eligibility_reasons})
        raise SystemExit("reported-run preflight failed: " + "; ".join(eligibility_reasons))

    source_manifest = {
        path.name: sha256_file(path)
        for path in (
            HERE / "README.md",
            HERE / "protocol.md",
            HERE / "cases.yaml",
            HERE / "baseline.py",
            HERE / "oracle.py",
            HERE / "validate_protocol.py",
            HERE / "download_models.py",
            HERE / "run.py",
        )
    }
    write_json(run_dir / "source_manifest.json", source_manifest)
    manifests: dict[str, dict[str, Any]] = {}
    hashes_before: dict[str, dict[str, str]] = {}
    manifest_hashes: dict[str, str] = {}
    input_paths: dict[str, Path] = {}
    for model in models:
        input_path = Path(model["input"]).resolve() if args.smoke else (REPO / model["input"]).resolve()
        input_paths[model["id"]] = input_path
        manifest = checkpoint_manifest(input_path, tensor_hashes=False)
        operation_contract = validate_input_operation(
            input_path, cases["operation"]["target_regex"]
        )
        revision = (
            {"passed": True, "scope": "synthetic_fixture"}
            if args.smoke
            else verify_huggingface_revision(input_path, model["revision"])
        )
        manifest.update(
            {
                "model_id": model["model_id"],
                "revision": model["revision"],
                "revision_evidence": revision,
                "operation_contract": operation_contract,
            }
        )
        if manifest["layout"] != model["expected_layout"]:
            raise SystemExit(f"{model['id']} input layout {manifest['layout']} != {model['expected_layout']}")
        if not revision["passed"]:
            raise SystemExit(f"{model['id']} lacks matching Hugging Face revision metadata")
        if not operation_contract["passed"]:
            raise SystemExit(
                f"{model['id']} does not satisfy the frozen floating-weight operation contract"
            )
        if not args.smoke and set(operation_contract["matched_dtype_counts"]) != {
            model["expected_weight_dtype"]
        }:
            raise SystemExit(
                f"{model['id']} matched weight dtype differs from the frozen matrix"
            )
        manifests[model["id"]] = manifest
        hashes_before[model["id"]] = source_file_hashes(input_path)
        manifest_path = run_dir / f"input_manifest_{model['id'].lower()}.json"
        write_json(manifest_path, manifest)
        manifest_hashes[manifest_path.name] = sha256_file(manifest_path)
    write_json(run_dir / "input_manifest_sha256.json", manifest_hashes)

    results: list[dict[str, Any]] = []
    attempts_root = run_dir / "attempts"
    attempts_root.mkdir()
    schedule_index = 0
    for model in models:
        input_path = input_paths[model["id"]]
        for method in EXPECTED_METHODS:
            schedule_index += 1
            print(f"[warmup] {model['id']} {method}", flush=True)
            result = execute_attempt(args, model=model, input_path=input_path, input_manifest=manifests[model["id"]], operation=cases["operation"], attempts_root=attempts_root, method=method, phase="warmup", repetition=0, schedule_index=schedule_index)
            results.append(result)
            if not result["correct"]:
                write_json(run_dir / "preflight_failure.json", {"failed_warmup": result["attempt_id"]})
                print(f"FAIL: warm-up {result['attempt_id']}; preserved under {run_dir.relative_to(REPO)}")
                return 1
        for repetition in range(1, args.repetitions + 1):
            shift = (repetition - 1) % len(EXPECTED_METHODS)
            order = EXPECTED_METHODS[shift:] + EXPECTED_METHODS[:shift]
            for method in order:
                schedule_index += 1
                print(f"[measured {repetition}/{args.repetitions}] {model['id']} {method}", flush=True)
                results.append(execute_attempt(args, model=model, input_path=input_path, input_manifest=manifests[model["id"]], operation=cases["operation"], attempts_root=attempts_root, method=method, phase="measured", repetition=repetition, schedule_index=schedule_index))

    integrity = {model["id"]: files_unchanged(input_paths[model["id"]], hashes_before[model["id"]]) for model in models}
    summary = build_summary(args, environment, models, manifests, results, eligibility_reasons, integrity)
    write_json(run_dir / "summary.json", summary)
    (run_dir / "paper_data.csv").write_text(render_csv(summary), encoding="utf-8")
    (run_dir / "paper_table.md").write_text(render_markdown(summary), encoding="utf-8")
    (run_dir / "paper_table.tex").write_text(render_latex(summary), encoding="utf-8")
    (run_dir / "paper_text.md").write_text(render_text(summary), encoding="utf-8")
    if args.publish_dir:
        publish(run_dir, args.publish_dir.resolve())
    print(f"Run: {run_dir.relative_to(REPO)}")
    print(f"Correct measured attempts: {summary['correct_measured_attempts']}/{summary['measured_attempts']}")
    print(f"Status: {'REPORTABLE CANDIDATE' if summary['reported_eligible'] else 'NON-REPORTABLE PREFLIGHT'}")
    return 0 if summary["correctness_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
