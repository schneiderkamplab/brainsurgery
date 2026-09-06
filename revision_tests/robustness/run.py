#!/usr/bin/env python3
"""Run the EACL 2027 BrainSurgery robustness evaluation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CASES_PATH = HERE / "cases.yaml"
FAULT_INJECTOR = HERE / "fault_injector.py"
PROTOCOL_ID = "eacl2027_robustness_v1"
SHARD_INDEX = "model.safetensors.index.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        help="Unique underscore-separated run id (default: timestamp, test, commit)",
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=REPO / "log" / "revision_tests",
        help="Raw artifact root (default: log/revision_tests)",
    )
    parser.add_argument(
        "--brainsurgery-cli",
        type=Path,
        default=REPO / ".venv" / "bin" / "brainsurgery",
        help="BrainSurgery CLI executable",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=REPO / ".venv" / "bin" / "python",
        help="Python used for the isolated fault-injection process",
    )
    parser.add_argument(
        "--publish-dir",
        type=Path,
        help="Optional new directory for compact, committable results",
    )
    parser.add_argument("--timeout", type=int, default=120, help="Seconds per process")
    parser.add_argument(
        "--interrupt-timeout",
        type=int,
        default=30,
        help="Seconds to wait for the first-shard marker",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases_doc = load_cases()
    repo_info = collect_repo_info()
    commit_short = repo_info["commit"][:8] if repo_info["commit"] else "unknown"
    timestamp = datetime.now(UTC).strftime("%Y_%m_%dT%H%M%SZ")
    run_id = args.run_id or f"{timestamp}_robustness_macos_{commit_short}"
    require_safe_run_id(run_id)

    run_dir = args.log_root.resolve() / run_id / "robustness"
    if run_dir.exists():
        raise SystemExit(f"refusing to overwrite existing run directory: {run_dir}")
    run_dir.mkdir(parents=True)

    cli = args.brainsurgery_cli.absolute()
    # Do not resolve the virtual-environment interpreter symlink: executing its
    # base interpreter path directly would bypass the virtual environment.
    python = args.python.absolute()
    if not cli.is_file():
        raise SystemExit(f"BrainSurgery CLI does not exist: {cli}")
    if not python.is_file():
        raise SystemExit(f"Python executable does not exist: {python}")

    command_display = f".venv/bin/python revision_tests/robustness/run.py --run-id {run_id}"
    if args.publish_dir:
        command_display += f" --publish-dir {args.publish_dir.as_posix()}"
    (run_dir / "command.txt").write_text(command_display + "\n", encoding="utf-8")
    write_json(run_dir / "environment.json", collect_environment(repo_info, run_id))

    fixtures, fixture_manifest = create_fixtures(run_dir / "fixtures")
    write_json(run_dir / "fixture_manifest.json", fixture_manifest)
    source_manifest = {
        path.name: sha256_file(path)
        for path in (CASES_PATH, HERE / "protocol.md", FAULT_INJECTOR, HERE / "run.py")
    }
    write_json(run_dir / "source_manifest.json", source_manifest)

    controls = run_auditor_controls(run_dir / "auditor_controls")
    write_json(run_dir / "auditor_controls.json", controls)

    case_results: list[dict[str, Any]] = []
    plan_hashes: dict[str, str] = {}
    for case in cases_doc["cases"]:
        print(f"[{case['id']}] {case['name']}", flush=True)
        result, plan_sha = run_case(
            case=case,
            fixtures=fixtures,
            run_dir=run_dir,
            cli=cli,
            python=python,
            timeout=args.timeout,
            interrupt_timeout=args.interrupt_timeout,
        )
        case_results.append(result)
        plan_hashes[case["id"]] = plan_sha
        print(f"[{case['id']}] {'PASS' if result['evaluation_passed'] else 'FAIL'}", flush=True)

    write_json(
        run_dir / "plan_manifest.json",
        {
            "protocol_id": PROTOCOL_ID,
            "case_definition_sha256": sha256_file(CASES_PATH),
            "generated_plan_sha256": plan_hashes,
        },
    )
    summary = build_summary(run_id, repo_info, case_results, controls, source_manifest)
    write_json(run_dir / "summary.json", summary)
    (run_dir / "paper_table.md").write_text(render_paper_table(summary), encoding="utf-8")

    if args.publish_dir:
        publish_compact_results(run_dir, args.publish_dir.resolve())

    print(f"Run: {repo_relative(run_dir)}")
    print(
        f"Evaluation: {summary['cases_passed']}/{summary['cases_total']} cases; "
        f"sources unchanged: {summary['sources_unchanged']}/{summary['cases_total']}"
    )
    print(
        f"Safety finding: {summary['observed_safe_cases']}/{summary['cases_total']} safe; "
        f"{summary['partial_or_mixed_output_findings']} partial/mixed-output finding(s)"
    )
    print(f"Overall evaluation: {'PASS' if summary['evaluation_passed'] else 'FAIL'}")
    return 0 if summary["evaluation_passed"] else 1


def load_cases() -> dict[str, Any]:
    raw = yaml.safe_load(CASES_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("protocol_id") != PROTOCOL_ID:
        raise SystemExit(f"{CASES_PATH} does not declare protocol {PROTOCOL_ID}")
    if raw.get("provider") != "inmemory":
        raise SystemExit("robustness protocol v1 requires the inmemory provider")
    cases = raw.get("cases")
    if not isinstance(cases, list):
        raise SystemExit("cases.yaml must contain a cases list")
    expected_ids = [f"R{index:02d}" for index in range(1, 20)]
    actual_ids = [case.get("id") for case in cases if isinstance(case, dict)]
    if actual_ids != expected_ids:
        raise SystemExit(f"expected frozen case ids {expected_ids}, got {actual_ids}")
    return raw


def fixture_state() -> dict[str, torch.Tensor]:
    return {
        "layer.0.weight": torch.arange(16, dtype=torch.float32).reshape(4, 4),
        "layer.1.weight": torch.arange(16, 32, dtype=torch.float32).reshape(4, 4),
        "math.a": torch.tensor([[1.0, -2.0], [3.0, 4.0]], dtype=torch.float32),
        "unchanged.sentinel": torch.tensor([11, 13, 17, 19], dtype=torch.int64),
        "large.a": torch.arange(1024, dtype=torch.float32),
        "large.b": torch.arange(1024, 2048, dtype=torch.float32),
    }


def create_fixtures(root: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    root.mkdir()
    valid_dir = root / "valid"
    valid_dir.mkdir()
    valid = valid_dir / "model.safetensors"
    state = fixture_state()
    save_file(state, str(valid), metadata={"protocol": PROTOCOL_ID})

    corrupt_dir = root / "corrupt"
    corrupt_dir.mkdir()
    corrupt = corrupt_dir / "model.safetensors"
    corrupt.write_bytes(b"deterministic-corrupt-safetensors-fixture\n")

    truncated_dir = root / "truncated"
    truncated_dir.mkdir()
    truncated = truncated_dir / "model.safetensors"
    valid_bytes = valid.read_bytes()
    truncated.write_bytes(valid_bytes[: len(valid_bytes) // 2])

    missing_shard = root / "missing_shard"
    missing_shard.mkdir()
    present_name = "model-00001-of-00002.safetensors"
    missing_name = "model-00002-of-00002.safetensors"
    save_file({"math.a": state["math.a"]}, str(missing_shard / present_name))
    index = {
        "metadata": {"total_size": state["math.a"].numel() * state["math.a"].element_size()},
        "weight_map": {"math.a": present_name, "large.a": missing_name},
    }
    (missing_shard / SHARD_INDEX).write_text(json.dumps(index, indent=2), encoding="utf-8")

    paths = {
        "valid": valid,
        "corrupt": corrupt,
        "truncated": truncated,
        "missing_shard": missing_shard,
        "missing": root / "missing" / "model.safetensors",
    }
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "logical_tensor_count": len(state),
        "logical_payload_bytes": sum(t.numel() * t.element_size() for t in state.values()),
        "logical_tensors": {
            name: {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "sha256": tensor_sha256(tensor),
            }
            for name, tensor in sorted(state.items())
        },
        "fixtures": {name: tree_manifest(path) for name, path in paths.items()},
    }
    return paths, manifest


def run_case(
    *,
    case: dict[str, Any],
    fixtures: dict[str, Path],
    run_dir: Path,
    cli: Path,
    python: Path,
    timeout: int,
    interrupt_timeout: int,
) -> tuple[dict[str, Any], str]:
    case_dir = run_dir / "cases" / f"{case['id']}_{case['name']}"
    case_dir.mkdir(parents=True)
    source = fixtures[case["input"]]
    source_before = tree_manifest(source)

    output_mode = case.get("output_mode", "single")
    if case.get("output_setup") == "blocked_parent":
        blocker = case_dir / "blocked_parent"
        blocker.write_text("regular file blocks directory creation\n", encoding="utf-8")
        output = blocker / "output.safetensors"
    elif output_mode == "sharded":
        output = case_dir / "output_shards"
    else:
        output = case_dir / "output.safetensors"

    output_setup = case.get("output_setup")
    preexisting = output_setup in {"preexisting", "preexisting_sharded"}
    if output_setup == "preexisting":
        save_file(
            {"preexisting.sentinel": torch.tensor([23, 29, 31], dtype=torch.int64)},
            str(output),
            metadata={"purpose": "preexisting_destination"},
        )
    elif output_setup == "preexisting_sharded":
        output.mkdir()
        old_shard = "preexisting.safetensors"
        old_state = {"preexisting.sentinel": torch.tensor([23, 29, 31], dtype=torch.int64)}
        save_file(old_state, str(output / old_shard))
        (output / SHARD_INDEX).write_text(
            json.dumps(
                {
                    "metadata": {"total_size": 3 * 8},
                    "weight_map": {"preexisting.sentinel": old_shard},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    output_before = tree_manifest(output)

    plan_path = case_dir / "plan.yaml"
    write_plan(case, source, output, plan_path)
    marker = case_dir / "first_shard_complete.marker"
    invocation = invoke(
        case=case,
        plan_path=plan_path,
        marker=marker,
        cli=cli,
        python=python,
        stdout_path=case_dir / "stdout.txt",
        stderr_path=case_dir / "stderr.txt",
        timeout=timeout,
        interrupt_timeout=interrupt_timeout,
    )

    source_after = tree_manifest(source)
    artifact = classify_artifact(output, output_before, preexisting=preexisting)
    combined_diagnostic = (
        (case_dir / "stdout.txt").read_text(encoding="utf-8")
        + "\n"
        + (case_dir / "stderr.txt").read_text(encoding="utf-8")
    )
    diagnostic_pattern = case.get("diagnostic_regex")
    diagnostic_applicable = diagnostic_pattern is not None
    diagnostic_matched = (
        re.search(diagnostic_pattern, combined_diagnostic, flags=re.DOTALL) is not None
        if diagnostic_applicable
        else None
    )
    process_matched = process_matches(case["expected_process"], invocation)
    source_unchanged = source_before == source_after
    artifact_matched = artifact["state"] == case["expected_artifact"]
    safe_destination = artifact["state"] in {"absent", "preexisting_unchanged"} or (
        case["expected_process"] == "success" and artifact["state"] == "valid_complete"
    )
    observed_safe = source_unchanged and safe_destination
    safety_matched = observed_safe == (case["expected_safety"] == "safe")
    evaluation_passed = bool(
        process_matched
        and (diagnostic_matched is not False)
        and source_unchanged
        and artifact_matched
        and safety_matched
        and (case["driver"] != "interrupt" or invocation["marker_observed"])
    )

    result = {
        "id": case["id"],
        "name": case["name"],
        "category": case["category"],
        "driver": case["driver"],
        "expected_process": case["expected_process"],
        "observed_process": invocation["process_outcome"],
        "returncode": invocation["returncode"],
        "signal": invocation["signal"],
        "timed_out": invocation["timed_out"],
        "duration_seconds": invocation["duration_seconds"],
        "process_matched": process_matched,
        "diagnostic_applicable": diagnostic_applicable,
        "diagnostic_matched": diagnostic_matched,
        "exception_type": extract_exception_type(combined_diagnostic),
        "source_before": source_before,
        "source_after": source_after,
        "source_unchanged": source_unchanged,
        "expected_artifact": case["expected_artifact"],
        "artifact": artifact,
        "artifact_matched": artifact_matched,
        "expected_safety": case["expected_safety"],
        "observed_safe": observed_safe,
        "safety_matched": safety_matched,
        "interruption_marker_observed": invocation["marker_observed"],
        "evaluation_passed": evaluation_passed,
    }
    write_json(case_dir / "result.json", result)
    return result, sha256_file(plan_path)


def write_plan(case: dict[str, Any], source: Path, output: Path, plan_path: Path) -> None:
    if case.get("plan_kind") == "malformed_yaml":
        plan_path.write_text("inputs: [unterminated\n", encoding="utf-8")
        return
    if case.get("plan_kind") == "non_mapping":
        plan_path.write_text("- not\n- a_mapping\n", encoding="utf-8")
        return
    output_spec: str | dict[str, str]
    if case.get("output_mode") == "sharded":
        output_spec = {"path": str(output), "format": "safetensors", "shard": "1KB"}
    else:
        output_spec = str(output)
    plan = {
        "inputs": [f"model::{source}"],
        "transforms": case["transforms"],
        "output": output_spec,
    }
    plan_path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")


def invoke(
    *,
    case: dict[str, Any],
    plan_path: Path,
    marker: Path,
    cli: Path,
    python: Path,
    stdout_path: Path,
    stderr_path: Path,
    timeout: int,
    interrupt_timeout: int,
) -> dict[str, Any]:
    driver = case["driver"]
    if driver == "cli":
        command = [
            str(cli),
            str(plan_path),
            "--provider",
            "inmemory",
            "--num-workers",
            "1",
            "--no-summarize",
        ]
    else:
        mode = "exception_after_first" if driver == "inject_exception" else "pause_after_first"
        command = [str(python), str(FAULT_INJECTOR), "--mode", mode]
        if driver == "interrupt":
            command.extend(["--marker", str(marker)])
        command.append(str(plan_path))

    env = os.environ.copy()
    env.update({"PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1"})
    start = time.perf_counter()
    if driver != "interrupt":
        try:
            completed = subprocess.run(
                command,
                cwd=REPO,
                env=env,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
            stdout_path.write_text(completed.stdout, encoding="utf-8")
            stderr_path.write_text(completed.stderr, encoding="utf-8")
            return invocation_result(
                command, completed.returncode, time.perf_counter() - start, False, False
            )
        except subprocess.TimeoutExpired as exc:
            stdout_path.write_text(decode_timeout_stream(exc.stdout), encoding="utf-8")
            stderr_path.write_text(decode_timeout_stream(exc.stderr), encoding="utf-8")
            return invocation_result(command, 124, time.perf_counter() - start, True, False)

    process = subprocess.Popen(
        command,
        cwd=REPO,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    deadline = time.monotonic() + interrupt_timeout
    marker_observed = False
    while time.monotonic() < deadline:
        if marker.is_file():
            marker_observed = True
            break
        if process.poll() is not None:
            break
        time.sleep(0.05)
    timed_out = False
    if process.poll() is None:
        if marker_observed:
            process.terminate()
        else:
            timed_out = True
            process.kill()
    try:
        stdout, stderr = process.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.kill()
        stdout, stderr = process.communicate()
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    return invocation_result(
        command, process.returncode, time.perf_counter() - start, timed_out, marker_observed
    )


def invocation_result(
    command: list[str], returncode: int, duration: float, timed_out: bool, marker: bool
) -> dict[str, Any]:
    signal_number = -returncode if returncode < 0 else None
    if returncode == 0:
        outcome = "success"
    elif signal_number == signal.SIGTERM:
        outcome = "interrupted"
    else:
        outcome = "failure"
    return {
        "command": display_command(command),
        "returncode": returncode,
        "signal": signal_number,
        "process_outcome": outcome,
        "timed_out": timed_out,
        "marker_observed": marker,
        "duration_seconds": duration,
    }


def process_matches(expected: str, invocation: dict[str, Any]) -> bool:
    return invocation["process_outcome"] == expected and not invocation["timed_out"]


def classify_artifact(path: Path, before: dict[str, Any], *, preexisting: bool) -> dict[str, Any]:
    after = tree_manifest(path)
    if not after["exists"]:
        return {"state": "absent", "manifest": after, "load_error": None}
    if preexisting and before == after:
        return {"state": "preexisting_unchanged", "manifest": after, "load_error": None}
    if path.is_file():
        try:
            tensors = load_file(str(path), device="cpu")
            return {
                "state": "valid_complete",
                "manifest": after,
                "tensor_count": len(tensors),
                "load_error": None,
            }
        except Exception as exc:
            return {
                "state": "invalid_file",
                "manifest": after,
                "load_error": f"{type(exc).__name__}: {exc}",
            }

    index_path = path / SHARD_INDEX
    if not index_path.is_file():
        return {
            "state": "partial_without_index",
            "manifest": after,
            "load_error": "shard index absent",
        }
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index["weight_map"]
        state: dict[str, torch.Tensor] = {}
        for shard_name in sorted(set(weight_map.values())):
            shard = load_file(str(path / shard_name), device="cpu")
            overlap = set(state) & set(shard)
            if overlap:
                raise ValueError(f"duplicate tensors: {sorted(overlap)}")
            state.update(shard)
        if set(state) != set(weight_map):
            raise ValueError("weight map does not exactly cover loaded tensors")
        referenced_shards = set(weight_map.values())
        visible_shards = {item.name for item in path.glob("*.safetensors")}
        if visible_shards != referenced_shards:
            state_name = (
                "preexisting_changed_with_unindexed_shard" if preexisting else "partial_or_invalid"
            )
            return {
                "state": state_name,
                "manifest": after,
                "tensor_count": len(state),
                "unindexed_shards": sorted(visible_shards - referenced_shards),
                "load_error": "visible safetensors do not exactly match the shard index",
            }
        return {
            "state": "valid_complete",
            "manifest": after,
            "tensor_count": len(state),
            "load_error": None,
        }
    except Exception as exc:
        return {
            "state": "partial_or_invalid",
            "manifest": after,
            "load_error": f"{type(exc).__name__}: {exc}",
        }


def run_auditor_controls(root: Path) -> dict[str, Any]:
    root.mkdir()
    absent = root / "absent.safetensors"
    valid = root / "valid.safetensors"
    save_file({"x": torch.tensor([1.0])}, str(valid))
    valid_before = tree_manifest(valid)
    partial = root / "partial"
    partial.mkdir()
    save_file({"x": torch.tensor([1.0])}, str(partial / "model-00001-of-00002.safetensors"))
    mixed = root / "mixed"
    mixed.mkdir()
    save_file({"old": torch.tensor([1.0])}, str(mixed / "old.safetensors"))
    (mixed / SHARD_INDEX).write_text(
        json.dumps({"metadata": {"total_size": 4}, "weight_map": {"old": "old.safetensors"}}),
        encoding="utf-8",
    )
    mixed_before = tree_manifest(mixed)
    save_file({"new": torch.tensor([2.0])}, str(mixed / "unindexed.safetensors"))
    observed = {
        "absent": classify_artifact(absent, tree_manifest(absent), preexisting=False)["state"],
        "valid_complete": classify_artifact(valid, {"exists": False}, preexisting=False)["state"],
        "preexisting_unchanged": classify_artifact(valid, valid_before, preexisting=True)["state"],
        "partial_without_index": classify_artifact(partial, {"exists": False}, preexisting=False)[
            "state"
        ],
        "preexisting_changed_with_unindexed_shard": classify_artifact(
            mixed, mixed_before, preexisting=True
        )["state"],
    }
    expected = {name: name for name in observed}
    return {"expected": expected, "observed": observed, "passed": observed == expected}


def build_summary(
    run_id: str,
    repo_info: dict[str, Any],
    cases: list[dict[str, Any]],
    controls: dict[str, Any],
    source_manifest: dict[str, str],
) -> dict[str, Any]:
    applicable_diagnostics = [case for case in cases if case["diagnostic_applicable"]]
    prepublication = [case for case in cases if case["id"] <= "R15"]
    return {
        "protocol_id": PROTOCOL_ID,
        "run_id": run_id,
        "repository_commit": repo_info["commit"],
        "relevant_paths_clean": repo_info["relevant_paths_clean"],
        "source_manifest": source_manifest,
        "cases_total": len(cases),
        "cases_passed": sum(case["evaluation_passed"] for case in cases),
        "expected_non_success_detected": sum(
            case["process_matched"] for case in cases if case["expected_process"] != "success"
        ),
        "expected_non_success_total": sum(case["expected_process"] != "success" for case in cases),
        "diagnostics_matched": sum(case["diagnostic_matched"] is True for case in cases),
        "diagnostics_applicable": len(applicable_diagnostics),
        "sources_unchanged": sum(case["source_unchanged"] for case in cases),
        "prepublication_outputs_withheld_or_preserved": sum(
            case["artifact"]["state"] in {"absent", "preexisting_unchanged"}
            for case in prepublication
        ),
        "prepublication_failure_cases": len(prepublication),
        "preexisting_destinations_preserved": sum(
            case["artifact"]["state"] == "preexisting_unchanged" for case in cases
        ),
        "preexisting_destination_cases": sum(
            case["expected_artifact"].startswith("preexisting_") for case in cases
        ),
        "observed_safe_cases": sum(case["observed_safe"] for case in cases),
        "partial_or_mixed_output_findings": sum(
            case["artifact"]["state"]
            in {"partial_without_index", "preexisting_changed_with_unindexed_shard"}
            for case in cases
        ),
        "unsafe_case_ids": [case["id"] for case in cases if not case["observed_safe"]],
        "auditor_controls": controls,
        "evaluation_passed": all(case["evaluation_passed"] for case in cases)
        and controls["passed"],
        "claim_boundary": (
            "The result characterizes 19 deterministic cases using the in-memory provider, "
            "safetensors, and the recorded local filesystem. It supports rejection, source-"
            "preservation, and diagnostic claims only for these cases. Because injected save "
            "failure and interruption expose partial or mixed shard directories, it does not "
            "support transactional or crash-safe publication claims."
        ),
        "cases": cases,
    }


def render_paper_table(summary: dict[str, Any]) -> str:
    lines = [
        "# Robustness and failure-semantics result",
        "",
        f"Protocol: `{summary['protocol_id']}`",
        f"Run: `{summary['run_id']}`",
        f"Commit: `{summary['repository_commit']}`",
        "",
        "| Case | Failure class | Process | Diagnostic | Input unchanged | Artifact | "
        "Safe | Evaluation |",
        "|---|---|---|---:|---:|---|---:|---|",
    ]
    for case in summary["cases"]:
        diagnostic = (
            "n/a"
            if not case["diagnostic_applicable"]
            else ("yes" if case["diagnostic_matched"] else "no")
        )
        lines.append(
            "| {id} | {category} | {process} | {diagnostic} | {source} | {artifact} | "
            "{safe} | {evaluation} |".format(
                id=case["id"],
                category=case["category"].replace("_", " "),
                process=case["observed_process"],
                diagnostic=diagnostic,
                source="yes" if case["source_unchanged"] else "no",
                artifact=case["artifact"]["state"].replace("_", " "),
                safe="yes" if case["observed_safe"] else "no",
                evaluation="PASS" if case["evaluation_passed"] else "FAIL",
            )
        )
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- Evaluation cases: {summary['cases_passed']}/{summary['cases_total']} passed.",
            "- Expected non-success outcomes: "
            f"{summary['expected_non_success_detected']}/"
            f"{summary['expected_non_success_total']} detected.",
            f"- Applicable diagnostics: {summary['diagnostics_matched']}/"
            f"{summary['diagnostics_applicable']} matched.",
            f"- Source inputs: {summary['sources_unchanged']}/{summary['cases_total']} unchanged.",
            "- Failures before publication: "
            f"{summary['prepublication_outputs_withheld_or_preserved']}/"
            f"{summary['prepublication_failure_cases']} withheld output or preserved "
            "the destination.",
            "- Pre-existing destinations: "
            f"{summary['preexisting_destinations_preserved']}/"
            f"{summary['preexisting_destination_cases']} preserved.",
            f"- Observed-safe cases: {summary['observed_safe_cases']}/{summary['cases_total']}.",
            "- Partial or mixed-output findings: "
            f"{summary['partial_or_mixed_output_findings']} "
            f"({', '.join(summary['unsafe_case_ids'])}).",
            "",
            "## Claim boundary",
            "",
            summary["claim_boundary"],
            "",
        ]
    )
    return "\n".join(lines)


def publish_compact_results(run_dir: Path, publish_dir: Path) -> None:
    if publish_dir.exists():
        raise SystemExit(f"refusing to overwrite publish directory: {publish_dir}")
    publish_dir.mkdir(parents=True)
    for name in (
        "command.txt",
        "environment.json",
        "fixture_manifest.json",
        "source_manifest.json",
        "auditor_controls.json",
        "plan_manifest.json",
        "summary.json",
        "paper_table.md",
    ):
        shutil.copy2(run_dir / name, publish_dir / name)


def collect_repo_info() -> dict[str, Any]:
    commit = run_capture(["git", "rev-parse", "HEAD"])
    status = run_capture(["git", "status", "--porcelain"])
    relevant_status = run_capture(
        ["git", "status", "--porcelain", "--", "brainsurgery", "revision_tests/robustness"]
    )
    return {
        "commit": commit,
        "worktree_clean": not bool(status),
        "relevant_paths_clean": not bool(relevant_status),
        "status_porcelain": status.splitlines(),
        "relevant_status_porcelain": relevant_status.splitlines(),
    }


def collect_environment(repo_info: dict[str, Any], run_id: str) -> dict[str, Any]:
    cpu = run_capture(["sysctl", "-n", "machdep.cpu.brand_string"], required=False)
    memory = run_capture(["sysctl", "-n", "hw.memsize"], required=False)
    return {
        "run_id": run_id,
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "macos_version": platform.mac_ver()[0],
        "cpu": cpu or platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "memory_bytes": int(memory) if memory.isdigit() else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "safetensors": package_version("safetensors"),
        "pyyaml": package_version("PyYAML"),
        "brainsurgery": package_version("brainsurgery"),
        "protocol_id": PROTOCOL_ID,
        "subprocess_environment": {"PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1"},
        "repository": repo_info,
    }


def tree_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "kind": None, "files": []}
    if path.is_file():
        return {
            "exists": True,
            "kind": "file",
            "files": [
                {
                    "path": path.name,
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            ],
        }
    files = []
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        files.append(
            {
                "path": item.relative_to(path).as_posix(),
                "size": item.stat().st_size,
                "sha256": sha256_file(item),
            }
        )
    return {"exists": True, "kind": "directory", "files": files}


def extract_exception_type(text: str) -> str | None:
    matches = re.findall(r"(?:^|\n)([A-Za-z_][\w.]*(?:Error|Exception)):\s", text)
    return matches[-1] if matches else None


def tensor_sha256(tensor: torch.Tensor) -> str:
    data = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def run_capture(command: list[str], *, required: bool = True) -> str:
    completed = subprocess.run(command, cwd=REPO, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        if required:
            raise SystemExit(
                f"command failed ({display_command(command)}): {completed.stderr.strip()}"
            )
        return ""
    return completed.stdout.strip()


def display_command(command: list[str]) -> str:
    rendered = shlex.join(command)
    return rendered.replace(str(REPO), "<repo>")


def decode_timeout_stream(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value or ""


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return str(path)


def require_safe_run_id(run_id: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_]+", run_id):
        raise SystemExit("run id must contain only letters, digits, and underscores")


if __name__ == "__main__":
    sys.exit(main())
