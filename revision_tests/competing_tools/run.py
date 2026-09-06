#!/usr/bin/env python3
"""Run the frozen competing-tool equivalence study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import shutil
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

try:
    from .oracle import PROTOCOL_ID, compare_output, validate_comparison_record
    from .prepare import brain_plan, mergekit_config, prepare, write_yaml
except ImportError:
    from oracle import PROTOCOL_ID, compare_output, validate_comparison_record
    from prepare import brain_plan, mergekit_config, prepare, write_yaml


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EXPECTED_PAIRS = (
    ("R01", "brainsurgery"),
    ("R01", "torch_state_bridge"),
    ("M01", "brainsurgery"),
    ("M01", "mergekit"),
    ("M02", "brainsurgery"),
    ("M02", "mergekit"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="Unique underscore-separated run ID")
    parser.add_argument("--log-root", type=Path, default=REPO / "log" / "revision_tests")
    parser.add_argument(
        "--brainsurgery-cli", type=Path, default=REPO / ".venv" / "bin" / "brainsurgery"
    )
    parser.add_argument("--mergekit-cli", type=Path, required=True)
    parser.add_argument("--torch-state-bridge-python", type=Path, required=True)
    parser.add_argument("--source-model", type=Path)
    parser.add_argument("--source-id")
    parser.add_argument("--source-revision")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--sample-interval-ms", type=int, default=10)
    parser.add_argument(
        "--workload-note",
        help="Operator observation of concurrent workload; required for non-smoke runs",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--keep-outputs", action="store_true")
    return parser.parse_args()


def require_args(args: argparse.Namespace) -> None:
    if not args.run_id or not all(char.isalnum() or char == "_" for char in args.run_id):
        raise SystemExit("--run-id must contain only letters, digits, and underscores")
    if args.repetitions < 1:
        raise SystemExit("--repetitions must be positive")
    if args.num_threads < 1:
        raise SystemExit("--num-threads must be positive")
    if args.timeout < 1 or args.sample_interval_ms < 1:
        raise SystemExit("timeout and sampling interval must be positive")
    if (args.source_model is None) != (args.source_revision is None):
        raise SystemExit("--source-model and --source-revision must be supplied together")
    if not args.smoke and args.source_model is None:
        raise SystemExit("reported-size runs require --source-model and --source-revision")
    if not args.smoke and not args.source_id:
        raise SystemExit("reported-size runs require --source-id")
    if not args.smoke and args.repetitions < 5:
        raise SystemExit("non-smoke runs require at least five measured repetitions")
    if not args.smoke and not args.workload_note:
        raise SystemExit("non-smoke runs require an explicit --workload-note")


def git_value(*arguments: str) -> str:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=REPO,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def python_snapshot(executable: Path, modules: tuple[str, ...]) -> dict[str, Any]:
    code = """
import hashlib, importlib.metadata, importlib.util, json, pathlib, platform, site, sys
module_files = {}
for name in sys.argv[1:]:
  spec = importlib.util.find_spec(name)
  origin = pathlib.Path(spec.origin) if spec is not None and spec.origin else None
  module_files[name] = {
    'path': str(origin) if origin else None,
    'sha256': hashlib.sha256(origin.read_bytes()).hexdigest() if origin and origin.is_file() else None,
  }
print(json.dumps({
  'executable': sys.executable,
  'python': platform.python_version(),
  'site_packages': site.getsitepackages(),
  'packages': sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions()),
  'module_files': module_files,
}, sort_keys=True))
"""
    completed = subprocess.run(
        [str(executable), "-c", code, *modules], check=True, text=True, capture_output=True
    )
    return json.loads(completed.stdout)


def collect_environment(args: argparse.Namespace) -> dict[str, Any]:
    mergekit_python = args.mergekit_cli.absolute().parent / "python"
    brainsurgery_python = args.brainsurgery_cli.absolute().parent / "python"
    snapshots = {
        "brainsurgery": python_snapshot(brainsurgery_python, ("brainsurgery",)),
        "mergekit": python_snapshot(mergekit_python, ("mergekit",)),
        "torch_state_bridge": python_snapshot(
            args.torch_state_bridge_python.absolute(), ("torch_state_bridge",)
        ),
    }
    site_sets = {name: set(snapshot["site_packages"]) for name, snapshot in snapshots.items()}
    shared_sites = {
        f"{left}:{right}": sorted(site_sets[left] & site_sets[right])
        for index, left in enumerate(site_sets)
        for right in list(site_sets)[index + 1 :]
        if site_sets[left] & site_sets[right]
    }
    try:
        cpu_affinity = psutil.Process().cpu_affinity()
    except (AttributeError, psutil.AccessDenied, PermissionError):
        cpu_affinity = None
    try:
        process_count = len(psutil.pids())
    except (psutil.AccessDenied, PermissionError):
        process_count = None
    filesystem = {}
    for command_name, command in (
        ("df", ["df", "-P", str(args.log_root.resolve().parent)]),
        (
            "findmnt",
            [
                "findmnt",
                "-no",
                "SOURCE,FSTYPE,OPTIONS",
                "--target",
                str(args.log_root.resolve().parent),
            ],
        ),
    ):
        try:
            filesystem[command_name] = subprocess.run(
                command, check=False, text=True, capture_output=True
            ).stdout.strip()
        except OSError:
            filesystem[command_name] = "unavailable"
    return {
        "protocol_id": PROTOCOL_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hostname": platform.node(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_affinity": cpu_affinity,
        "load_average": list(os.getloadavg()),
        "visible_process_count": process_count,
        "memory_bytes": psutil.virtual_memory().total,
        "disk": psutil.disk_usage(str(args.log_root.resolve().parent)),
        "filesystem": filesystem,
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_status_porcelain": git_value("status", "--short"),
        "num_threads": args.num_threads,
        "cache_policy": "warm_after_one_unmeasured_warmup_no_os_cache_drop",
        "sample_interval_ms": args.sample_interval_ms,
        "operator_workload_note": args.workload_note,
        "python_environments": snapshots,
        "shared_site_packages": shared_sites,
    }


def package_version(snapshot: dict[str, Any], package: str) -> str | None:
    normalized = package.lower().replace("_", "-")
    for name, version in snapshot["packages"]:
        if name.lower().replace("_", "-") == normalized:
            return version
    return None


def validate_environment(environment: dict[str, Any], smoke: bool) -> list[str]:
    reasons = []
    snapshots = environment["python_environments"]
    if package_version(snapshots["mergekit"], "mergekit") != "0.1.4":
        reasons.append("mergekit is not exactly 0.1.4")
    if package_version(snapshots["torch_state_bridge"], "torch-state-bridge") != "0.1.0":
        reasons.append("torch-state-bridge is not exactly 0.1.0")
    expected_tsb_hash = "48a04fbfa14ae1d56f7b70b657d45aedc3773930d6cfa99455c7d51f72389ff8"
    actual_tsb_hash = snapshots["torch_state_bridge"]["module_files"][
        "torch_state_bridge"
    ]["sha256"]
    if actual_tsb_hash != expected_tsb_hash:
        reasons.append("torch-state-bridge installed module hash differs from tools.yaml")
    frozen_shared = {"torch": "2.14.0", "safetensors": "0.5.3", "numpy": "2.4.6"}
    for package, expected in frozen_shared.items():
        versions = {package_version(snapshot, package) for snapshot in snapshots.values()}
        if versions != {expected}:
            reasons.append(
                f"tool environments do not all use {package} {expected}: {sorted(str(v) for v in versions)}"
            )
    if package_version(snapshots["mergekit"], "transformers") != "4.57.1":
        reasons.append("MergeKit environment does not use Transformers 4.57.1")
    if environment["shared_site_packages"]:
        reasons.append("tool environments share a site-packages directory")
    if environment["git_commit"] == "unavailable":
        reasons.append("Git commit is unavailable")
    if environment["git_status_porcelain"]:
        reasons.append("Git worktree is dirty")
    if environment["system"] != "Linux":
        reasons.append("reported systems run requires Linux")
    if smoke:
        reasons.append("run was explicitly requested as smoke")
    return reasons


def render_specification(
    case_id: str,
    tool: str,
    fixture_root: Path,
    output_dir: Path,
    attempt_dir: Path,
) -> tuple[list[str], Path]:
    if tool == "brainsurgery":
        path = attempt_dir / "specification.yaml"
        write_yaml(path, brain_plan(case_id, fixture_root, output_dir))
        return [], path
    if tool == "mergekit":
        path = attempt_dir / "specification.yaml"
        write_yaml(path, mergekit_config(case_id, fixture_root))
        return [], path
    if tool == "torch_state_bridge":
        path = attempt_dir / "rules.txt"
        path.write_text(
            "layer.{n}.weight, block.{n}.weight\nlayer.{n}.bias, block.{n}.bias\n",
            encoding="utf-8",
        )
        return [], path
    raise ValueError(f"unknown tool: {tool}")


def command_for(
    args: argparse.Namespace,
    case_id: str,
    tool: str,
    fixture_root: Path,
    output_dir: Path,
    specification: Path,
) -> list[str]:
    if tool == "brainsurgery":
        return [
            str(args.brainsurgery_cli.resolve()),
            str(specification),
            "--provider",
            "inmemory",
            "--num-workers",
            str(args.num_threads),
            "--no-summarize",
            "--log-level",
            "warning",
        ]
    if tool == "mergekit":
        return [
            str(args.mergekit_cli.resolve()),
            str(specification),
            str(output_dir),
            "--device",
            "cpu",
            "--no-copy-tokenizer",
            "--no-write-model-card",
            "--out-shard-size",
            "5B",
            "--random-seed",
            "0",
            "--num-threads",
            str(args.num_threads),
            "--quiet",
        ]
    if tool == "torch_state_bridge":
        return [
            str(args.torch_state_bridge_python.absolute()),
            str(HERE / "adapter_torch_state_bridge.py"),
            "--input",
            str(fixture_root / "rename" / "model.safetensors"),
            "--rules",
            str(specification),
            "--output",
            str(output_dir / "model.safetensors"),
        ]
    raise ValueError(f"unknown tool: {tool}")


def run_monitored(
    command: list[str],
    *,
    stdout_path: Path,
    stderr_path: Path,
    timeout: int,
    interval_seconds: float,
    environment: dict[str, str],
) -> dict[str, Any]:
    started = time.perf_counter()
    peak_rss = 0
    io_by_pid: dict[int, tuple[int, int]] = {}
    timed_out = False
    process_tree_sampling_degraded = False
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            command,
            cwd=REPO,
            env=environment,
            stdout=stdout,
            stderr=stderr,
        )
        root = psutil.Process(process.pid)
        while process.poll() is None:
            if time.perf_counter() - started > timeout:
                timed_out = True
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                break
            processes = [root]
            try:
                processes.extend(root.children(recursive=True))
            except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError):
                process_tree_sampling_degraded = True
            rss = 0
            for child in processes:
                try:
                    rss += child.memory_info().rss
                    io = child.io_counters()
                    old = io_by_pid.get(child.pid, (0, 0))
                    io_by_pid[child.pid] = (
                        max(old[0], io.read_bytes),
                        max(old[1], io.write_bytes),
                    )
                except (
                    psutil.NoSuchProcess,
                    psutil.AccessDenied,
                    AttributeError,
                    PermissionError,
                ):
                    process_tree_sampling_degraded = True
                    continue
            peak_rss = max(peak_rss, rss)
            time.sleep(interval_seconds)
        returncode = process.wait()
    duration = time.perf_counter() - started
    return {
        "command": command,
        "command_shell_display": shlex.join(command),
        "returncode": 124 if timed_out else returncode,
        "timed_out": timed_out,
        "wall_seconds": duration,
        "peak_process_tree_rss_bytes": peak_rss,
        "process_tree_read_bytes": sum(value[0] for value in io_by_pid.values()),
        "process_tree_write_bytes": sum(value[1] for value in io_by_pid.values()),
        "rss_sampling_interval_seconds": interval_seconds,
        "process_tree_sampling_degraded": process_tree_sampling_degraded,
    }


def directory_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def specification_metrics(path: Path) -> dict[str, int]:
    text = path.read_text(encoding="utf-8")
    return {
        "bytes": len(text.encode("utf-8")),
        "nonblank_noncomment_lines": sum(
            bool(line.strip()) and not line.lstrip().startswith("#") for line in text.splitlines()
        ),
    }


def execute_attempt(
    args: argparse.Namespace,
    *,
    fixture_root: Path,
    attempts_root: Path,
    case_id: str,
    tool: str,
    phase: str,
    repetition: int,
    schedule_index: int,
) -> dict[str, Any]:
    attempt_id = f"{schedule_index:03d}_{phase}_{repetition:02d}_{case_id}_{tool}"
    attempt_dir = attempts_root / attempt_id
    attempt_dir.mkdir(parents=True)
    output_dir = attempt_dir / "output"
    _unused, specification = render_specification(
        case_id, tool, fixture_root, output_dir, attempt_dir
    )
    command = command_for(args, case_id, tool, fixture_root, output_dir, specification)
    process_env = os.environ.copy()
    process_env.update(
        {
            "OMP_NUM_THREADS": str(args.num_threads),
            "MKL_NUM_THREADS": str(args.num_threads),
            "OPENBLAS_NUM_THREADS": str(args.num_threads),
            "PYTHONHASHSEED": "0",
            "TOKENIZERS_PARALLELISM": "false",
            "CUDA_VISIBLE_DEVICES": "",
        }
    )
    process = run_monitored(
        command,
        stdout_path=attempt_dir / "stdout.txt",
        stderr_path=attempt_dir / "stderr.txt",
        timeout=args.timeout,
        interval_seconds=args.sample_interval_ms / 1000,
        environment=process_env,
    )
    validation = None
    validation_error = None
    if process["returncode"] == 0:
        try:
            validation = compare_output(case_id, output_dir, fixture_root)
            validate_comparison_record(validation, case_id)
        except Exception as exc:  # Preserve diagnostics for arbitrary invalid output.
            validation_error = f"{type(exc).__name__}: {exc}"
    output_bytes = directory_bytes(output_dir) if output_dir.exists() else 0
    correct = bool(validation and validation["passed"])
    result = {
        "protocol_id": PROTOCOL_ID,
        "attempt_id": attempt_id,
        "phase": phase,
        "measured": phase == "measured",
        "repetition": repetition,
        "schedule_index": schedule_index,
        "case_id": case_id,
        "tool": tool,
        "specification": specification.name,
        "specification_metrics": specification_metrics(specification),
        "process": process,
        "output_bytes": output_bytes,
        "validation": validation,
        "validation_error": validation_error,
        "correct": correct,
        "timing_eligible": phase == "measured" and correct,
        "output_removed_after_validation": False,
    }
    if correct and not args.keep_outputs:
        shutil.rmtree(output_dir)
        result["output_removed_after_validation"] = True
    write_json(attempt_dir / "result.json", result)
    return result


def timing_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "values": [],
            "median": None,
            "minimum": None,
            "maximum": None,
            "mean": None,
            "sample_standard_deviation": None,
        }
    return {
        "count": len(values),
        "values": values,
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "mean": statistics.mean(values),
        "sample_standard_deviation": statistics.stdev(values) if len(values) > 1 else None,
    }


def current_source_manifest() -> dict[str, str]:
    return {
        path.name: sha256_file(path)
        for path in (
            HERE / "README.md",
            HERE / "protocol.md",
            HERE / "cases.yaml",
            HERE / "tools.yaml",
            HERE / "oracle.py",
            HERE / "prepare.py",
            HERE / "adapter_torch_state_bridge.py",
            HERE / "run.py",
        )
    }


def input_integrity_errors(
    source_manifest: dict[str, str], fixture_root: Path
) -> list[str]:
    errors = []
    if current_source_manifest() != source_manifest:
        errors.append("protocol source checksum changed during the run")
    fixture_manifest = json.loads(
        (fixture_root / "fixture_manifest.json").read_text(encoding="utf-8")
    )
    for relative, expected_hash in fixture_manifest["files"].items():
        path = fixture_root / relative
        if not path.is_file() or sha256_file(path) != expected_hash:
            errors.append(f"fixture checksum changed during the run: {relative}")
    return errors


def build_summary(
    args: argparse.Namespace,
    environment: dict[str, Any],
    eligibility_reasons: list[str],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    pairs = {}
    for case_id, tool in EXPECTED_PAIRS:
        pair_results = [
            result
            for result in results
            if result["case_id"] == case_id and result["tool"] == tool and result["measured"]
        ]
        correct = [result for result in pair_results if result["correct"]]
        pairs[f"{case_id}:{tool}"] = {
            "case_id": case_id,
            "tool": tool,
            "measured_attempts": len(pair_results),
            "correct_attempts": len(correct),
            "incorrect_or_failed_attempts": len(pair_results) - len(correct),
            "wall_seconds": timing_summary(
                [result["process"]["wall_seconds"] for result in correct]
            ),
            "peak_process_tree_rss_bytes": timing_summary(
                [result["process"]["peak_process_tree_rss_bytes"] for result in correct]
            ),
            "read_bytes": timing_summary(
                [result["process"]["process_tree_read_bytes"] for result in correct]
            ),
            "write_bytes": timing_summary(
                [result["process"]["process_tree_write_bytes"] for result in correct]
            ),
            "output_bytes": timing_summary([result["output_bytes"] for result in correct]),
            "specification_metrics": (
                pair_results[0]["specification_metrics"] if pair_results else None
            ),
            "tensors_checked": sorted(
                {
                    result["validation"]["tensors_checked"]
                    for result in correct
                    if result["validation"] is not None
                }
            ),
            "maximum_absolute_difference": max(
                (
                    result["validation"]["maximum_absolute_difference"]
                    for result in correct
                    if result["validation"] is not None
                    and result["validation"]["maximum_absolute_difference"] is not None
                ),
                default=None,
            ),
        }
    complete = all(pair["correct_attempts"] == args.repetitions for pair in pairs.values())
    final_reasons = list(eligibility_reasons)
    if any(
        result["measured"] and result["process"]["process_tree_sampling_degraded"]
        for result in results
    ):
        final_reasons.append("process-tree memory/I/O sampling was degraded")
    reported_eligible = not final_reasons and complete
    comparisons = {}
    for case_id in {case for case, _tool in EXPECTED_PAIRS}:
        case_pairs = [pair for pair in pairs.values() if pair["case_id"] == case_id]
        if len(case_pairs) != 2:
            continue
        brain = next(pair for pair in case_pairs if pair["tool"] == "brainsurgery")
        competitor = next(pair for pair in case_pairs if pair["tool"] != "brainsurgery")
        brain_wall = brain["wall_seconds"]["median"]
        competitor_wall = competitor["wall_seconds"]["median"]
        comparisons[case_id] = {
            "competitor": competitor["tool"],
            "brain_to_competitor_median_wall_ratio": (
                brain_wall / competitor_wall
                if brain_wall is not None and competitor_wall not in (None, 0)
                else None
            ),
        }
    return {
        "protocol_id": PROTOCOL_ID,
        "run_id": args.run_id,
        "fixture_scope": "real_shape_derived" if args.source_model else "tiny_smoke",
        "source_model": str(args.source_model.resolve()) if args.source_model else None,
        "source_id": args.source_id,
        "source_revision": args.source_revision,
        "reported_eligible": reported_eligible,
        "eligibility_reasons": final_reasons,
        "correctness_complete": complete,
        "warmup_attempts": sum(not result["measured"] for result in results),
        "measured_attempts": sum(result["measured"] for result in results),
        "correct_measured_attempts": sum(
            result["measured"] and result["correct"] for result in results
        ),
        "pairs": pairs,
        "comparisons": comparisons,
        "git_commit": environment["git_commit"],
    }


def render_table(summary: dict[str, Any]) -> str:
    table_lines = [
        "| Case | Tool | Correct runs | Median wall (s) | Median peak RSS (MiB) | Output (MiB) | Spec lines |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for pair in summary["pairs"].values():
        wall = pair["wall_seconds"]["median"]
        rss = pair["peak_process_tree_rss_bytes"]["median"]
        output_bytes = pair["output_bytes"]["median"]
        spec_lines = pair["specification_metrics"]["nonblank_noncomment_lines"]
        timing_usable = summary["reported_eligible"] and wall is not None and rss is not None
        wall_text = f"{wall:.6f}" if timing_usable else "—"
        rss_text = f"{rss / (1024 * 1024):.2f}" if timing_usable else "—"
        output_text = f"{output_bytes / (1024 * 1024):.2f}" if output_bytes else "—"
        table_lines.append(
            f"| {pair['case_id']} | {pair['tool']} | "
            f"{pair['correct_attempts']}/{pair['measured_attempts']} | "
            f"{wall_text} | {rss_text} | {output_text} | {spec_lines} |"
        )
    ratio_lines = [
        "| Case | Competitor | BrainSurgery / competitor median wall ratio |",
        "|---|---|---:|",
    ]
    for case_id, comparison in sorted(summary["comparisons"].items()):
        ratio = comparison["brain_to_competitor_median_wall_ratio"]
        ratio_lines.append(
            f"| {case_id} | {comparison['competitor']} | "
            + (
                f"{ratio:.4f} |"
                if summary["reported_eligible"] and ratio is not None
                else "— |"
            )
        )
    if summary["reported_eligible"]:
        status = "REPORTABLE CANDIDATE: all automated eligibility gates passed."
        scope = "reported-size candidate"
    else:
        reasons = "; ".join(summary["eligibility_reasons"]) or "incomplete correctness"
        status = f"NOT REPORTABLE: {reasons}."
        scope = "non-reportable"
    return (
        f"# Competing-tool run ({scope})\n\n**{status}**\n\n"
        + "\n".join(table_lines)
        + "\n\n"
        + "\n".join(ratio_lines)
        + "\n\nTimings include process startup, input loading, transformation, and output save. "
        "Specification lines are descriptive only and are not a usability measure.\n"
    )


def latex_tool_name(tool: str) -> str:
    names = {
        "brainsurgery": r"\textsc{BrainSurgery}",
        "mergekit": r"\textsc{MergeKit}",
        "torch_state_bridge": r"\texttt{torch-state-bridge}",
    }
    return names[tool]


def render_latex(summary: dict[str, Any]) -> str:
    reportable = summary["reported_eligible"]
    rows = []
    for pair in summary["pairs"].values():
        wall = pair["wall_seconds"]["median"]
        rss = pair["peak_process_tree_rss_bytes"]["median"]
        output_bytes = pair["output_bytes"]["median"]
        wall_text = f"{wall:.3f}" if reportable and wall is not None else "--"
        rss_text = f"{rss / (1024 * 1024):.1f}" if reportable and rss is not None else "--"
        output_text = f"{output_bytes / (1024 * 1024):.1f}" if output_bytes else "--"
        rows.append(
            f"{pair['case_id']} & {latex_tool_name(pair['tool'])} & "
            f"{pair['correct_attempts']}/{pair['measured_attempts']} & "
            f"{wall_text} & {rss_text} & {output_text} \\\\"
        )
    if reportable:
        caption = (
            "Controlled checkpoint-operation comparison. Runtime and peak RSS are "
            "medians over correctness-validated measured runs."
        )
        warning = "% All automated reporting gates passed; complete human audit before use."
    else:
        caption = (
            "Non-reportable integration preflight. Timing and memory are suppressed; "
            "correctness counts are shown only to validate the harness."
        )
        warning = "% NON-REPORTABLE PREFLIGHT: do not use this fragment as paper evidence."
    return (
        f"{warning}\n"
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\small\n"
        "\\begin{tabular}{llrrrr}\n"
        "\\toprule\n"
        "Case & Tool & Correct & Wall (s) & Peak RSS (MiB) & Output (MiB) \\\\\n"
        "\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n"
        "\\end{tabular}\n"
        f"\\caption{{{caption}}}\n"
        "\\label{tab:competing-tools}\n"
        "\\end{table*}\n"
    )


def render_narrative(summary: dict[str, Any]) -> str:
    correct = summary["correct_measured_attempts"]
    measured = summary["measured_attempts"]
    if not summary["reported_eligible"]:
        return (
            "# Competing-tool preflight interpretation\n\n"
            f"This non-reportable integration preflight produced correct outputs in "
            f"{correct}/{measured} measured tool/case attempts. It establishes that the "
            "frozen adapters, common-operation contracts, and independent oracle execute "
            "together. Because the reporting gates did not pass, no runtime, memory, "
            "efficiency, scaling, usability, downstream-quality, or general tool-ranking "
            "claim may be drawn from this run.\n"
        )
    case_text = []
    for case_id in sorted(summary["comparisons"]):
        case_pairs = [
            pair for pair in summary["pairs"].values() if pair["case_id"] == case_id
        ]
        maximum = max(
            (
                pair["maximum_absolute_difference"]
                for pair in case_pairs
                if pair["maximum_absolute_difference"] is not None
            ),
            default=None,
        )
        ratio = summary["comparisons"][case_id][
            "brain_to_competitor_median_wall_ratio"
        ]
        case_text.append(
            f"{case_id}: all outputs passed; maximum absolute difference "
            f"{maximum:.3g}; BrainSurgery/competitor median wall-time ratio {ratio:.3f}."
        )
    return (
        "# Competing-tool result interpretation\n\n"
        f"All {correct}/{measured} measured outputs passed the independent oracle. "
        + " ".join(case_text)
        + " These narrow, operation-matched results must not be interpreted as a "
        "general ranking of the tools or as usability evidence.\n"
    )


def write_reports(run_dir: Path, summary: dict[str, Any]) -> None:
    (run_dir / "paper_table.md").write_text(render_table(summary), encoding="utf-8")
    (run_dir / "paper_table.tex").write_text(render_latex(summary), encoding="utf-8")
    (run_dir / "paper_text.md").write_text(render_narrative(summary), encoding="utf-8")


def main() -> int:
    args = parse_args()
    require_args(args)
    run_dir = args.log_root.resolve() / args.run_id / "competing_tools"
    if run_dir.exists():
        raise SystemExit(f"refusing to overwrite existing run directory: {run_dir}")
    run_dir.mkdir(parents=True)
    write_json(
        run_dir / "arguments.json",
        vars(args)
        | {key: str(value) for key, value in vars(args).items() if isinstance(value, Path)},
    )

    environment = collect_environment(args)
    write_json(run_dir / "environment.json", environment)
    eligibility_reasons = validate_environment(environment, args.smoke)
    if not args.smoke and eligibility_reasons:
        write_json(run_dir / "preflight_failure.json", {"reasons": eligibility_reasons})
        raise SystemExit("reported-run preflight failed: " + "; ".join(eligibility_reasons))

    source_manifest = current_source_manifest()
    write_json(run_dir / "source_manifest.json", source_manifest)
    fixture_root = run_dir / "fixtures"
    prepare(
        fixture_root,
        source_model=args.source_model.resolve() if args.source_model else None,
        source_revision=args.source_revision,
        source_id=args.source_id,
    )

    results = []
    schedule_index = 0
    attempts_root = run_dir / "attempts"
    attempts_root.mkdir()
    for case_id, tool in EXPECTED_PAIRS:
        schedule_index += 1
        print(f"[warmup] {case_id} {tool}", flush=True)
        results.append(
            execute_attempt(
                args,
                fixture_root=fixture_root,
                attempts_root=attempts_root,
                case_id=case_id,
                tool=tool,
                phase="warmup",
                repetition=0,
                schedule_index=schedule_index,
            )
        )
    failed_warmups = [result["attempt_id"] for result in results if not result["correct"]]
    if failed_warmups:
        summary = build_summary(args, environment, eligibility_reasons, results)
        write_json(run_dir / "summary.json", summary)
        write_json(run_dir / "preflight_failure.json", {"failed_warmups": failed_warmups})
        write_reports(run_dir, summary)
        print("warm-up correctness failed: " + ", ".join(failed_warmups))
        print(f"raw results: {run_dir.relative_to(REPO)}")
        return 1
    integrity_errors = input_integrity_errors(source_manifest, fixture_root)
    if integrity_errors:
        write_json(run_dir / "preflight_failure.json", {"reasons": integrity_errors})
        print("input-integrity preflight failed: " + "; ".join(integrity_errors))
        return 1
    pairs = list(EXPECTED_PAIRS)
    for repetition in range(1, args.repetitions + 1):
        offset = (repetition - 1) % len(pairs)
        for case_id, tool in pairs[offset:] + pairs[:offset]:
            schedule_index += 1
            print(f"[measured {repetition}] {case_id} {tool}", flush=True)
            results.append(
                execute_attempt(
                    args,
                    fixture_root=fixture_root,
                    attempts_root=attempts_root,
                    case_id=case_id,
                    tool=tool,
                    phase="measured",
                    repetition=repetition,
                    schedule_index=schedule_index,
                )
            )

    integrity_errors = input_integrity_errors(source_manifest, fixture_root)
    summary = build_summary(
        args, environment, eligibility_reasons + integrity_errors, results
    )
    write_json(run_dir / "summary.json", summary)
    write_reports(run_dir, summary)
    print(f"raw results: {run_dir.relative_to(REPO)}")
    print(
        f"correct measured attempts: {summary['correct_measured_attempts']}/"
        f"{summary['measured_attempts']}"
    )
    print(f"reported_eligible={str(summary['reported_eligible']).lower()}")
    return 0 if summary["correctness_complete"] and not integrity_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
