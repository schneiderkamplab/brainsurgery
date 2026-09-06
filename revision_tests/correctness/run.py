#!/usr/bin/env python3
"""Run the EACL 2027 BrainSurgery correctness/preservation evaluation."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from oracle import clone_state, expected_state, fixture_state


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CASES_PATH = HERE / "cases.yaml"
PROTOCOL_ID = "eacl2027_correctness_v1"
FIXTURE_METADATA = {
    "format": "pt",
    "revision_test_protocol": PROTOCOL_ID,
    "purpose": "independent_correctness_fixture",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        help="Unique underscore-separated run id (default: timestamp, backend, commit)",
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
        "--publish-dir",
        type=Path,
        help="Optional new directory for compact, committable results",
    )
    parser.add_argument("--timeout", type=int, default=120, help="Seconds per CLI process")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases_doc = load_cases()
    repo_info = collect_repo_info()
    commit_short = repo_info["commit"][:8] if repo_info["commit"] else "unknown"
    timestamp = datetime.now(UTC).strftime("%Y_%m_%dT%H%M%SZ")
    run_id = args.run_id or f"{timestamp}_correctness_macos_{commit_short}"
    require_safe_run_id(run_id)

    run_dir = args.log_root.resolve() / run_id / "correctness"
    if run_dir.exists():
        raise SystemExit(f"refusing to overwrite existing run directory: {run_dir}")
    run_dir.mkdir(parents=True)

    cli = args.brainsurgery_cli.resolve()
    if not cli.is_file():
        raise SystemExit(f"BrainSurgery CLI does not exist: {cli}")

    command_display = f".venv/bin/python revision_tests/correctness/run.py --run-id {run_id}"
    if args.publish_dir:
        command_display += f" --publish-dir {args.publish_dir.as_posix()}"
    (run_dir / "command.txt").write_text(command_display + "\n", encoding="utf-8")

    environment = collect_environment(repo_info=repo_info, run_id=run_id)
    write_json(run_dir / "environment.json", environment)

    fixture_dir = run_dir / "fixture"
    fixture_dir.mkdir()
    fixture_path = fixture_dir / "model.safetensors"
    fixture = fixture_state()
    save_file(fixture, str(fixture_path), metadata=FIXTURE_METADATA)
    fixture_file_sha = sha256_file(fixture_path)
    fixture_manifest = {
        "protocol_id": PROTOCOL_ID,
        "file": "fixture/model.safetensors",
        "file_sha256": fixture_file_sha,
        "metadata": read_metadata(fixture_path),
        "tensor_payload_bytes": state_nbytes(fixture),
        "tensors": state_manifest(fixture),
    }
    write_json(run_dir / "fixture_manifest.json", fixture_manifest)

    source_manifest = {
        path.name: sha256_file(path)
        for path in (CASES_PATH, HERE / "protocol.md", HERE / "oracle.py", HERE / "run.py")
    }
    write_json(run_dir / "source_manifest.json", source_manifest)

    validator_controls = run_validator_controls(fixture)
    write_json(run_dir / "validator_controls.json", validator_controls)

    case_results: list[dict[str, Any]] = []
    generated_plans: dict[str, dict[str, str]] = {}
    for case in cases_doc["cases"]:
        print(f"[{case['id']}] {case['name']}", flush=True)
        result, plan_hashes = run_case(
            case=case,
            fixture_path=fixture_path,
            fixture_file_sha=fixture_file_sha,
            fixture=fixture,
            run_dir=run_dir,
            cli=cli,
            provider=cases_doc["provider"],
            timeout=args.timeout,
        )
        case_results.append(result)
        generated_plans[case["id"]] = plan_hashes
        outcome = "PASS" if result["passed"] else "FAIL"
        print(f"[{case['id']}] {outcome}", flush=True)

    plan_manifest = {
        "protocol_id": PROTOCOL_ID,
        "case_definition_sha256": sha256_file(CASES_PATH),
        "generated_plan_sha256": generated_plans,
    }
    write_json(run_dir / "plan_manifest.json", plan_manifest)

    summary = build_summary(
        run_id=run_id,
        repo_info=repo_info,
        case_results=case_results,
        validator_controls=validator_controls,
        fixture_manifest=fixture_manifest,
        source_manifest=source_manifest,
    )
    write_json(run_dir / "summary.json", summary)
    (run_dir / "paper_table.md").write_text(render_paper_table(summary), encoding="utf-8")

    if args.publish_dir:
        publish_compact_results(run_dir=run_dir, publish_dir=args.publish_dir.resolve())

    print(f"Run: {repo_relative(run_dir)}")
    print(
        f"Result: {summary['cases_passed']}/{summary['cases_total']} cases; "
        f"{summary['untouched_tensors_exact']}/{summary['untouched_tensors_checked']} "
        "untouched tensor checks exact"
    )
    print(f"Overall: {'PASS' if summary['passed'] else 'FAIL'}")
    return 0 if summary["passed"] else 1


def load_cases() -> dict[str, Any]:
    raw = yaml.safe_load(CASES_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("protocol_id") != PROTOCOL_ID:
        raise SystemExit(f"{CASES_PATH} does not declare protocol {PROTOCOL_ID}")
    if raw.get("provider") != "inmemory":
        raise SystemExit("correctness protocol v1 requires the inmemory provider")
    cases = raw.get("cases")
    if not isinstance(cases, list) or not cases:
        raise SystemExit("cases.yaml must contain a non-empty cases list")
    expected_ids = [f"C{index:02d}" for index in range(1, 11)]
    actual_ids = [case.get("id") for case in cases if isinstance(case, dict)]
    if actual_ids != expected_ids:
        raise SystemExit(f"expected frozen case ids {expected_ids}, got {actual_ids}")
    return raw


def run_case(
    *,
    case: dict[str, Any],
    fixture_path: Path,
    fixture_file_sha: str,
    fixture: dict[str, torch.Tensor],
    run_dir: Path,
    cli: Path,
    provider: str,
    timeout: int,
) -> tuple[dict[str, Any], dict[str, str]]:
    case_dir = run_dir / "cases" / f"{case['id']}_{case['name']}"
    case_dir.mkdir(parents=True)
    output_dir = case_dir / "output"
    output_dir.mkdir()
    expected = expected_state(case["id"])
    output_mode = case["output_mode"]

    if output_mode == "single":
        output_path = output_dir / "model.safetensors"
        output_spec: str | dict[str, str] = str(output_path)
    elif output_mode == "sharded_reload":
        output_path = output_dir / "sharded"
        output_spec = {
            "path": str(output_path),
            "format": "safetensors",
            "shard": case["shard_size"],
        }
    else:
        raise ValueError(f"unsupported output mode: {output_mode}")

    plan = {
        "inputs": [f"model::{fixture_path}"],
        "transforms": case["transforms"],
        "output": output_spec,
    }
    plan_path = case_dir / "plan.yaml"
    plan_path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")
    invocation = invoke_cli(
        cli=cli,
        plan_path=plan_path,
        provider=provider,
        summarize_path=case_dir / "executed_plan.yaml",
        stdout_path=case_dir / "stdout.txt",
        stderr_path=case_dir / "stderr.txt",
        timeout=timeout,
    )

    comparison: dict[str, Any]
    sharding: dict[str, Any] | None = None
    reload_comparison: dict[str, Any] | None = None
    reload_invocation: dict[str, Any] | None = None
    plan_hashes = {"primary": sha256_file(plan_path)}
    output_metadata: list[dict[str, str] | None] = []

    if invocation["returncode"] == 0:
        try:
            if output_mode == "single":
                actual = load_file(str(output_path), device="cpu")
                output_metadata = [read_metadata(output_path)]
            else:
                actual, sharding, output_metadata = load_sharded_independently(output_path)
            comparison = compare_states(
                actual=actual,
                expected=expected,
                source=fixture,
                write_set=set(case["write_set"]),
            )

            if output_mode == "sharded_reload":
                reload_path = output_dir / "reloaded.safetensors"
                reload_plan = {
                    "inputs": [f"model::{output_path}"],
                    "transforms": [{"scale_": {"target": r"math\.a", "by": 1.0}}],
                    "output": str(reload_path),
                }
                reload_plan_path = case_dir / "reload_plan.yaml"
                reload_plan_path.write_text(
                    yaml.safe_dump(reload_plan, sort_keys=False), encoding="utf-8"
                )
                plan_hashes["reload"] = sha256_file(reload_plan_path)
                reload_invocation = invoke_cli(
                    cli=cli,
                    plan_path=reload_plan_path,
                    provider=provider,
                    summarize_path=case_dir / "reload_executed_plan.yaml",
                    stdout_path=case_dir / "reload_stdout.txt",
                    stderr_path=case_dir / "reload_stderr.txt",
                    timeout=timeout,
                )
                if reload_invocation["returncode"] == 0:
                    reloaded = load_file(str(reload_path), device="cpu")
                    reload_comparison = compare_states(
                        actual=reloaded,
                        expected=expected,
                        source=fixture,
                        write_set={"math.a"},
                    )
                else:
                    reload_comparison = failed_comparison("reload CLI failed")
        except Exception as exc:  # preserve evaluation failure as data
            comparison = failed_comparison(f"output validation failed: {type(exc).__name__}: {exc}")
    else:
        comparison = failed_comparison("primary CLI failed")

    source_sha_after = sha256_file(fixture_path)
    input_unchanged = source_sha_after == fixture_file_sha
    sharding_passed = sharding is None or all(
        bool(sharding[key])
        for key in (
            "multiple_shards",
            "weight_map_exact",
            "index_total_size_exact",
            "assigned_tensors_present",
            "no_duplicate_tensors",
        )
    )
    reload_passed = reload_comparison is None or (
        reload_invocation is not None
        and reload_invocation["returncode"] == 0
        and reload_comparison["passed"]
    )
    passed = bool(
        invocation["returncode"] == 0
        and comparison["passed"]
        and input_unchanged
        and sharding_passed
        and reload_passed
    )

    result = {
        "id": case["id"],
        "name": case["name"],
        "classification": case["classification"],
        "write_set": case["write_set"],
        "output_mode": output_mode,
        "invocation": invocation,
        "comparison": comparison,
        "input_file_sha256_before": fixture_file_sha,
        "input_file_sha256_after": source_sha_after,
        "input_file_unchanged": input_unchanged,
        "sharding": sharding,
        "reload_invocation": reload_invocation,
        "reload_comparison": reload_comparison,
        "secondary_container_metadata": {
            "input": FIXTURE_METADATA,
            "outputs": output_metadata,
            "all_outputs_equal_input": bool(output_metadata)
            and all(metadata == FIXTURE_METADATA for metadata in output_metadata),
            "primary_endpoint": False,
        },
        "passed": passed,
    }
    write_json(case_dir / "result.json", result)
    return result, plan_hashes


def invoke_cli(
    *,
    cli: Path,
    plan_path: Path,
    provider: str,
    summarize_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    timeout: int,
) -> dict[str, Any]:
    command = [
        str(cli),
        str(plan_path),
        "--provider",
        provider,
        "--num-workers",
        "1",
        "--summary-mode",
        "resolve",
        "--summarize-path",
        str(summarize_path),
    ]
    env = os.environ.copy()
    env.update({"PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1"})
    start = time.perf_counter()
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
        duration = time.perf_counter() - start
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        return {
            "command": display_command(command),
            "returncode": completed.returncode,
            "timed_out": False,
            "duration_seconds": duration,
        }
    except subprocess.TimeoutExpired as exc:
        duration = time.perf_counter() - start
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        return {
            "command": display_command(command),
            "returncode": 124,
            "timed_out": True,
            "duration_seconds": duration,
        }


def compare_states(
    *,
    actual: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
    source: dict[str, torch.Tensor],
    write_set: set[str],
) -> dict[str, Any]:
    actual_names = set(actual)
    expected_names = set(expected)
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    tensor_checks: list[dict[str, Any]] = []
    for name in sorted(expected_names & actual_names):
        expected_tensor = expected[name]
        actual_tensor = actual[name]
        shape_equal = tuple(actual_tensor.shape) == tuple(expected_tensor.shape)
        dtype_equal = actual_tensor.dtype == expected_tensor.dtype
        bytes_equal = dtype_equal and shape_equal and tensor_sha256(actual_tensor) == tensor_sha256(
            expected_tensor
        )
        tensor_checks.append(
            {
                "name": name,
                "expected_shape": list(expected_tensor.shape),
                "actual_shape": list(actual_tensor.shape),
                "expected_dtype": dtype_name(expected_tensor),
                "actual_dtype": dtype_name(actual_tensor),
                "expected_sha256": tensor_sha256(expected_tensor),
                "actual_sha256": tensor_sha256(actual_tensor),
                "shape_equal": shape_equal,
                "dtype_equal": dtype_equal,
                "bytes_equal": bytes_equal,
            }
        )

    untouched_names = sorted((set(source) & expected_names) - write_set)
    untouched_checks = []
    for name in untouched_names:
        exact = name in actual and tensor_identity(actual[name], source[name])
        untouched_checks.append({"name": name, "exact": exact})

    names_equal = not missing and not unexpected
    all_tensors_exact = names_equal and all(check["bytes_equal"] for check in tensor_checks)
    all_untouched_exact = all(check["exact"] for check in untouched_checks)
    return {
        "passed": names_equal and all_tensors_exact and all_untouched_exact,
        "names_equal": names_equal,
        "missing_names": missing,
        "unexpected_names": unexpected,
        "expected_tensor_count": len(expected),
        "actual_tensor_count": len(actual),
        "tensors_exact": sum(check["bytes_equal"] for check in tensor_checks),
        "tensor_checks": tensor_checks,
        "untouched_checked": len(untouched_checks),
        "untouched_exact": sum(check["exact"] for check in untouched_checks),
        "untouched_checks": untouched_checks,
    }


def failed_comparison(reason: str) -> dict[str, Any]:
    return {
        "passed": False,
        "reason": reason,
        "names_equal": False,
        "missing_names": [],
        "unexpected_names": [],
        "expected_tensor_count": 0,
        "actual_tensor_count": 0,
        "tensors_exact": 0,
        "tensor_checks": [],
        "untouched_checked": 0,
        "untouched_exact": 0,
        "untouched_checks": [],
    }


def load_sharded_independently(
    output_dir: Path,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], list[dict[str, str] | None]]:
    index_path = output_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    metadata = index.get("metadata")
    if not isinstance(weight_map, dict):
        raise ValueError("shard index has no weight_map mapping")

    state: dict[str, torch.Tensor] = {}
    shard_metadata: list[dict[str, str] | None] = []
    assigned_tensors_present = True
    duplicate_names: list[str] = []
    shard_names = sorted(set(weight_map.values()))
    for shard_name in shard_names:
        shard_path = output_dir / shard_name
        shard = load_file(str(shard_path), device="cpu")
        shard_metadata.append(read_metadata(shard_path))
        for name, tensor in shard.items():
            if name in state:
                duplicate_names.append(name)
            state[name] = tensor
        for tensor_name, assigned_shard in weight_map.items():
            if assigned_shard == shard_name and tensor_name not in shard:
                assigned_tensors_present = False

    total_size = state_nbytes(state)
    index_total_size = metadata.get("total_size") if isinstance(metadata, dict) else None
    details = {
        "shard_count": len(shard_names),
        "shard_names": shard_names,
        "multiple_shards": len(shard_names) > 1,
        "weight_map_exact": set(weight_map) == set(state),
        "index_total_size": index_total_size,
        "observed_total_size": total_size,
        "index_total_size_exact": index_total_size == total_size,
        "assigned_tensors_present": assigned_tensors_present,
        "duplicate_tensor_names": sorted(duplicate_names),
        "no_duplicate_tensors": not duplicate_names,
    }
    return state, details, shard_metadata


def run_validator_controls(source: dict[str, torch.Tensor]) -> dict[str, Any]:
    expected = clone_state(source)
    baseline = compare_states(
        actual=clone_state(expected), expected=expected, source=source, write_set=set()
    )

    value_corrupt = clone_state(expected)
    value_corrupt["unchanged.sentinel"][0, 0] += 1.0
    value_result = compare_states(
        actual=value_corrupt, expected=expected, source=source, write_set=set()
    )

    dtype_corrupt = clone_state(expected)
    dtype_corrupt["embedding.weight"] = dtype_corrupt["embedding.weight"].to(torch.float64)
    dtype_result = compare_states(
        actual=dtype_corrupt, expected=expected, source=source, write_set=set()
    )

    key_corrupt = clone_state(expected)
    del key_corrupt["layer.0.bias"]
    key_result = compare_states(
        actual=key_corrupt, expected=expected, source=source, write_set=set()
    )

    controls = {
        "oracle_import_guard_passed": oracle_import_guard(),
        "baseline_accepted": baseline["passed"],
        "value_corruption_detected": not value_result["passed"],
        "dtype_corruption_detected": not dtype_result["passed"],
        "missing_key_detected": not key_result["passed"],
    }
    controls["passed"] = all(controls.values())
    return controls


def build_summary(
    *,
    run_id: str,
    repo_info: dict[str, Any],
    case_results: list[dict[str, Any]],
    validator_controls: dict[str, Any],
    fixture_manifest: dict[str, Any],
    source_manifest: dict[str, str],
) -> dict[str, Any]:
    tensors_checked = sum(result["comparison"]["expected_tensor_count"] for result in case_results)
    tensors_exact = sum(result["comparison"]["tensors_exact"] for result in case_results)
    untouched_checked = sum(result["comparison"]["untouched_checked"] for result in case_results)
    untouched_exact = sum(result["comparison"]["untouched_exact"] for result in case_results)
    metadata_equal = sum(
        result["secondary_container_metadata"]["all_outputs_equal_input"]
        for result in case_results
    )
    cases_passed = sum(result["passed"] for result in case_results)
    return {
        "protocol_id": PROTOCOL_ID,
        "run_id": run_id,
        "run_directory": f"log/revision_tests/{run_id}/correctness",
        "repository_commit": repo_info["commit"],
        "relevant_paths_clean": repo_info["relevant_paths_clean"],
        "cases_total": len(case_results),
        "cases_passed": cases_passed,
        "oracle_tensors_checked": tensors_checked,
        "oracle_tensors_exact": tensors_exact,
        "untouched_tensors_checked": untouched_checked,
        "untouched_tensors_exact": untouched_exact,
        "input_files_unchanged": sum(result["input_file_unchanged"] for result in case_results),
        "metadata_cases_equal": metadata_equal,
        "metadata_cases_checked": len(case_results),
        "metadata_is_primary_endpoint": False,
        "validator_controls": validator_controls,
        "fixture_file_sha256": fixture_manifest["file_sha256"],
        "source_manifest": source_manifest,
        "cases": case_results,
        "passed": cases_passed == len(case_results) and validator_controls["passed"],
        "claim_boundary": (
            "Exact tensor-state correctness for the enumerated cases; custom safetensors "
            "metadata and arbitrary sidecar files are not covered by the primary claim."
        ),
    }


def render_paper_table(summary: dict[str, Any]) -> str:
    lines = [
        "# Correctness and preservation result",
        "",
        f"Protocol: `{summary['protocol_id']}`",
        f"Run: `{summary['run_id']}`",
        f"Commit: `{summary['repository_commit']}`",
        "",
        "| Case | Operation | Class | Oracle exact | Untouched exact | Input unchanged | Result |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for result in summary["cases"]:
        comparison = result["comparison"]
        lines.append(
            "| {id} | {name} | {classification} | {exact}/{checked} | "
            "{untouched_exact}/{untouched_checked} | {input_unchanged} | {outcome} |".format(
                id=result["id"],
                name=result["name"].replace("_", " "),
                classification=result["classification"].replace("_", " "),
                exact=comparison["tensors_exact"],
                checked=comparison["expected_tensor_count"],
                untouched_exact=comparison["untouched_exact"],
                untouched_checked=comparison["untouched_checked"],
                input_unchanged="yes" if result["input_file_unchanged"] else "no",
                outcome="PASS" if result["passed"] else "FAIL",
            )
        )
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- Cases: {summary['cases_passed']}/{summary['cases_total']} passed.",
            "- Oracle tensors: "
            f"{summary['oracle_tensors_exact']}/{summary['oracle_tensors_checked']} exact.",
            "- Untouched tensor checks: "
            f"{summary['untouched_tensors_exact']}/"
            f"{summary['untouched_tensors_checked']} exact.",
            "- Source checkpoint checks: "
            f"{summary['input_files_unchanged']}/{summary['cases_total']} unchanged.",
            "- Verifier controls: " + render_control_count(summary["validator_controls"]),
            "- Safetensors custom metadata preserved: "
            f"{summary['metadata_cases_equal']}/{summary['metadata_cases_checked']} cases "
            "(secondary observation, not a primary endpoint).",
            "",
            "## Claim boundary",
            "",
            summary["claim_boundary"],
            "",
        ]
    )
    return "\n".join(lines)


def publish_compact_results(*, run_dir: Path, publish_dir: Path) -> None:
    if publish_dir.exists():
        raise SystemExit(f"refusing to overwrite publish directory: {publish_dir}")
    publish_dir.mkdir(parents=True)
    for name in (
        "command.txt",
        "environment.json",
        "fixture_manifest.json",
        "source_manifest.json",
        "validator_controls.json",
        "plan_manifest.json",
        "summary.json",
        "paper_table.md",
    ):
        shutil.copy2(run_dir / name, publish_dir / name)


def collect_repo_info() -> dict[str, Any]:
    commit = run_capture(["git", "rev-parse", "HEAD"])
    status = run_capture(["git", "status", "--porcelain"])
    relevant_status = run_capture(
        ["git", "status", "--porcelain", "--", "brainsurgery", "revision_tests/correctness"]
    )
    return {
        "commit": commit,
        "worktree_clean": not bool(status),
        "relevant_paths_clean": not bool(relevant_status),
        "status_porcelain": status.splitlines(),
        "relevant_status_porcelain": relevant_status.splitlines(),
    }


def collect_environment(*, repo_info: dict[str, Any], run_id: str) -> dict[str, Any]:
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
        "torch_num_threads": torch.get_num_threads(),
        "subprocess_environment": {"PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1"},
        "protocol_id": PROTOCOL_ID,
        "repository": repo_info,
    }


def state_manifest(state: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {
        name: {
            "shape": list(tensor.shape),
            "dtype": dtype_name(tensor),
            "payload_bytes": tensor.numel() * tensor.element_size(),
            "sha256": tensor_sha256(tensor),
        }
        for name, tensor in sorted(state.items())
    }


def state_nbytes(state: dict[str, torch.Tensor]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in state.values())


def tensor_identity(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        tuple(left.shape) == tuple(right.shape)
        and left.dtype == right.dtype
        and tensor_sha256(left) == tensor_sha256(right)
    )


def tensor_sha256(tensor: torch.Tensor) -> str:
    # Reshape first so scalar tensors can be reinterpreted as bytes. PyTorch
    # rejects a direct dtype-changing view on a zero-dimensional tensor.
    data = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
    return hashlib.sha256(data).hexdigest()


def dtype_name(tensor: torch.Tensor) -> str:
    return str(tensor.dtype).removeprefix("torch.")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_metadata(path: Path) -> dict[str, str] | None:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return handle.metadata()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def run_capture(command: list[str], *, required: bool = True) -> str:
    completed = subprocess.run(command, cwd=REPO, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        if required:
            raise RuntimeError(f"command failed ({completed.returncode}): {shlex.join(command)}")
        return ""
    return completed.stdout.strip()


def oracle_import_guard() -> bool:
    tree = ast.parse((HERE / "oracle.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.startswith("brainsurgery") for alias in node.names):
                return False
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").startswith("brainsurgery"):
                return False
    return True


def display_command(command: list[str]) -> str:
    display_args = []
    for item in command:
        path = Path(item)
        if path.is_absolute():
            try:
                item = path.relative_to(REPO).as_posix()
            except ValueError:
                pass
        display_args.append(item)
    return shlex.join(display_args)


def render_control_count(controls: dict[str, Any]) -> str:
    values = [bool(value) for key, value in controls.items() if key != "passed"]
    return f"{sum(values)}/{len(values)} passed."


def require_safe_run_id(run_id: str) -> None:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
    if not run_id or any(character not in allowed for character in run_id):
        raise SystemExit("run id must contain only letters, digits, and underscores")


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
