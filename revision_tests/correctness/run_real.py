#!/usr/bin/env python3
"""Run exact tensor preservation checks on the three real base checkpoints."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors import safe_open

from run import (
    HERE,
    REPO,
    collect_environment,
    collect_repo_info,
    dtype_name,
    invoke_cli,
    read_metadata,
    repo_relative,
    require_safe_run_id,
    sha256_file,
    tensor_sha256,
    write_json,
)


REAL_CASES_PATH = HERE / "real_cases.yaml"
REAL_PROTOCOL_ID = "eacl2027_real_preservation_v1"


@dataclass(frozen=True)
class CheckpointLayout:
    root: Path
    tensor_files: dict[str, Path]
    data_files: tuple[Path, ...]
    index_file: Path | None
    kind: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="Unique underscore-separated run id")
    parser.add_argument(
        "--log-root", type=Path, default=REPO / "log" / "revision_tests"
    )
    parser.add_argument(
        "--brainsurgery-cli",
        type=Path,
        default=REPO / ".venv" / "bin" / "brainsurgery",
    )
    parser.add_argument("--publish-dir", type=Path)
    parser.add_argument("--timeout", type=int, default=900)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases_doc = load_real_cases()
    repo_info = collect_repo_info()
    commit_short = repo_info["commit"][:8] if repo_info["commit"] else "unknown"
    timestamp = datetime.now(UTC).strftime("%Y_%m_%dT%H%M%SZ")
    run_id = args.run_id or f"{timestamp}_real_preservation_macos_{commit_short}"
    require_safe_run_id(run_id)

    run_dir = args.log_root.resolve() / run_id / "real_preservation"
    if run_dir.exists():
        raise SystemExit(f"refusing to overwrite existing run directory: {run_dir}")
    run_dir.mkdir(parents=True)
    cli = args.brainsurgery_cli.resolve()
    if not cli.is_file():
        raise SystemExit(f"BrainSurgery CLI does not exist: {cli}")

    command = f".venv/bin/python revision_tests/correctness/run_real.py --run-id {run_id}"
    if args.publish_dir:
        command += f" --publish-dir {args.publish_dir.as_posix()}"
    (run_dir / "command.txt").write_text(command + "\n", encoding="utf-8")

    environment = collect_environment(repo_info=repo_info, run_id=run_id)
    environment["protocol_id"] = REAL_PROTOCOL_ID
    environment["measurement_use"] = "correctness_only_not_performance"
    write_json(run_dir / "environment.json", environment)

    source_manifest = {
        path.name: sha256_file(path)
        for path in (REAL_CASES_PATH, HERE / "real_protocol.md", HERE / "run_real.py")
    }
    write_json(run_dir / "source_manifest.json", source_manifest)

    case_results = []
    for case in cases_doc["cases"]:
        print(f"[{case['id']}] {case['target']}", flush=True)
        result = run_real_case(
            case=case,
            run_dir=run_dir,
            cli=cli,
            provider=cases_doc["provider"],
            timeout=args.timeout,
        )
        case_results.append(result)
        print(f"[{case['id']}] {'PASS' if result['passed'] else 'FAIL'}", flush=True)

    summary = build_real_summary(
        run_id=run_id,
        repo_info=repo_info,
        source_manifest=source_manifest,
        cases=case_results,
    )
    write_json(run_dir / "summary.json", summary)
    (run_dir / "paper_table.md").write_text(render_real_table(summary), encoding="utf-8")
    if args.publish_dir:
        publish(run_dir=run_dir, publish_dir=args.publish_dir.resolve())

    print(f"Run: {repo_relative(run_dir)}")
    print(
        f"Result: {summary['cases_passed']}/{summary['cases_total']} checkpoints; "
        f"{summary['tensors_exact']}/{summary['tensors_checked']} tensors exact"
    )
    print(f"Overall: {'PASS' if summary['passed'] else 'FAIL'}")
    return 0 if summary["passed"] else 1


def load_real_cases() -> dict[str, Any]:
    raw = yaml.safe_load(REAL_CASES_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("protocol_id") != REAL_PROTOCOL_ID:
        raise SystemExit(f"real_cases.yaml must declare {REAL_PROTOCOL_ID}")
    cases = raw.get("cases")
    expected_ids = ["R01", "R02", "R03"]
    if not isinstance(cases, list) or [case.get("id") for case in cases] != expected_ids:
        raise SystemExit("real_cases.yaml must contain the frozen R01..R03 matrix")
    return raw


def run_real_case(
    *,
    case: dict[str, Any],
    run_dir: Path,
    cli: Path,
    provider: str,
    timeout: int,
) -> dict[str, Any]:
    input_path = (REPO / case["input"]).resolve()
    if not input_path.exists():
        raise SystemExit(f"missing frozen input for {case['id']}: {case['input']}")

    case_dir = run_dir / "cases" / f"{case['id']}_{case['target']}"
    case_dir.mkdir(parents=True)
    output_path = case_dir / "output" / "model.safetensors"
    output_path.parent.mkdir()

    input_layout = discover_checkpoint(input_path)
    selected = case["identity_tensor"]
    if selected not in input_layout.tensor_files:
        raise SystemExit(f"identity tensor missing for {case['id']}: {selected}")
    if not read_tensor(input_layout, selected).is_floating_point():
        raise SystemExit(f"identity tensor is not floating point for {case['id']}: {selected}")

    source_files = list(input_layout.data_files)
    if input_layout.index_file:
        source_files.append(input_layout.index_file)
    revision_check = verify_huggingface_revision(
        input_path=input_path,
        source_files=source_files,
        expected_revision=case["revision"],
    )
    source_hashes_before = {repo_relative(path): sha256_file(path) for path in source_files}
    input_manifest = checkpoint_manifest(input_layout)
    input_manifest.update(
        {
            "case_id": case["id"],
            "target": case["target"],
            "model_id": case["model_id"],
            "revision": case["revision"],
            "input": case["input"],
            "identity_tensor": selected,
            "revision_check": revision_check,
            "files_sha256": source_hashes_before,
        }
    )
    write_json(case_dir / "input_manifest.json", input_manifest)

    plan = {
        "inputs": [f"model::{input_path}"],
        "transforms": [{"scale_": {"target": re.escape(selected), "by": 1.0}}],
        "output": str(output_path),
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

    if invocation["returncode"] == 0:
        try:
            output_layout = discover_checkpoint(output_path)
            comparison = compare_checkpoints(
                actual=output_layout,
                expected=input_layout,
                write_set={selected},
            )
            output_metadata = metadata_list(output_layout)
        except Exception as exc:
            comparison = comparison_failure(
                f"output validation failed: {type(exc).__name__}: {exc}"
            )
            output_metadata = []
    else:
        comparison = comparison_failure("BrainSurgery CLI failed")
        output_metadata = []

    source_hashes_after = {repo_relative(path): sha256_file(path) for path in source_files}
    source_files_unchanged = source_hashes_after == source_hashes_before
    input_metadata = metadata_list(input_layout)
    passed = bool(
        revision_check["passed"]
        and invocation["returncode"] == 0
        and comparison["passed"]
        and source_files_unchanged
    )
    result = {
        "id": case["id"],
        "target": case["target"],
        "model_id": case["model_id"],
        "revision": case["revision"],
        "input": case["input"],
        "identity_tensor": selected,
        "revision_check": revision_check,
        "plan_sha256": sha256_file(plan_path),
        "invocation": invocation,
        "comparison": comparison,
        "source_files_sha256_before": source_hashes_before,
        "source_files_sha256_after": source_hashes_after,
        "source_files_unchanged": source_files_unchanged,
        "secondary_container_metadata": {
            "input": input_metadata,
            "output": output_metadata,
            "equal": input_metadata == output_metadata,
            "primary_endpoint": False,
        },
        "passed": passed,
    }
    write_json(case_dir / "result.json", result)
    return result


def discover_checkpoint(path: Path) -> CheckpointLayout:
    if path.is_file():
        names = safetensors_keys(path)
        return CheckpointLayout(
            root=path.parent,
            tensor_files={name: path for name in names},
            data_files=(path,),
            index_file=None,
            kind="single_file",
        )

    index_path = path / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not all(
            isinstance(name, str) and isinstance(file_name, str)
            for name, file_name in weight_map.items()
        ):
            raise ValueError(f"invalid safetensors weight map: {index_path}")
        tensor_files = {name: path / file_name for name, file_name in weight_map.items()}
        data_files = tuple(sorted(set(tensor_files.values())))
        verify_index_assignments(tensor_files)
        return CheckpointLayout(
            root=path,
            tensor_files=tensor_files,
            data_files=data_files,
            index_file=index_path,
            kind="indexed_sharded",
        )

    files = tuple(sorted(path.glob("*.safetensors")))
    if len(files) != 1:
        raise ValueError(f"unsupported safetensors checkpoint layout: {path}")
    names = safetensors_keys(files[0])
    return CheckpointLayout(
        root=path,
        tensor_files={name: files[0] for name in names},
        data_files=files,
        index_file=None,
        kind="directory_single_file",
    )


def verify_index_assignments(tensor_files: dict[str, Path]) -> None:
    actual_by_file: dict[Path, set[str]] = {}
    for file_path in set(tensor_files.values()):
        actual_by_file[file_path] = set(safetensors_keys(file_path))
    for name, file_path in tensor_files.items():
        if name not in actual_by_file[file_path]:
            raise ValueError(f"index assigns missing tensor {name!r} to {file_path.name}")
    assigned = set(tensor_files)
    actual = set().union(*actual_by_file.values())
    if assigned != actual:
        raise ValueError("index tensor-name set does not match shard tensor-name set")


def checkpoint_manifest(layout: CheckpointLayout) -> dict[str, Any]:
    tensors = {}
    total_bytes = 0
    for name in sorted(layout.tensor_files):
        tensor = read_tensor(layout, name)
        payload_bytes = tensor.numel() * tensor.element_size()
        total_bytes += payload_bytes
        tensors[name] = {
            "shape": list(tensor.shape),
            "dtype": dtype_name(tensor),
            "payload_bytes": payload_bytes,
            "sha256": tensor_sha256(tensor),
        }
    return {
        "layout": layout.kind,
        "tensor_count": len(tensors),
        "tensor_payload_bytes": total_bytes,
        "tensors": tensors,
    }


def compare_checkpoints(
    *,
    actual: CheckpointLayout,
    expected: CheckpointLayout,
    write_set: set[str],
) -> dict[str, Any]:
    actual_names = set(actual.tensor_files)
    expected_names = set(expected.tensor_files)
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    checks = []
    for name in sorted(actual_names & expected_names):
        actual_tensor = read_tensor(actual, name)
        expected_tensor = read_tensor(expected, name)
        shape_equal = tuple(actual_tensor.shape) == tuple(expected_tensor.shape)
        dtype_equal = actual_tensor.dtype == expected_tensor.dtype
        expected_sha256 = tensor_sha256(expected_tensor)
        actual_sha256 = tensor_sha256(actual_tensor)
        bytes_equal = bool(
            shape_equal and dtype_equal and actual_sha256 == expected_sha256
        )
        checks.append(
            {
                "name": name,
                "expected_shape": list(expected_tensor.shape),
                "actual_shape": list(actual_tensor.shape),
                "expected_dtype": dtype_name(expected_tensor),
                "actual_dtype": dtype_name(actual_tensor),
                "expected_sha256": expected_sha256,
                "actual_sha256": actual_sha256,
                "shape_equal": shape_equal,
                "dtype_equal": dtype_equal,
                "bytes_equal": bytes_equal,
                "untouched": name not in write_set,
            }
        )
    untouched = [check for check in checks if check["untouched"]]
    names_equal = not missing and not unexpected
    return {
        "passed": names_equal and all(check["bytes_equal"] for check in checks),
        "names_equal": names_equal,
        "missing_names": missing,
        "unexpected_names": unexpected,
        "expected_tensor_count": len(expected_names),
        "actual_tensor_count": len(actual_names),
        "tensors_exact": sum(check["bytes_equal"] for check in checks),
        "tensor_checks": checks,
        "untouched_checked": len(untouched),
        "untouched_exact": sum(check["bytes_equal"] for check in untouched),
    }


def safetensors_keys(path: Path) -> list[str]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return list(handle.keys())


def read_tensor(layout: CheckpointLayout, name: str) -> torch.Tensor:
    path = layout.tensor_files[name]
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def metadata_list(layout: CheckpointLayout) -> list[dict[str, str] | None]:
    return [read_metadata(path) for path in layout.data_files]


def verify_huggingface_revision(
    *,
    input_path: Path,
    source_files: list[Path],
    expected_revision: str,
) -> dict[str, Any]:
    model_root = input_path if input_path.is_dir() else input_path.parent
    records = []
    for source_file in source_files:
        relative = source_file.relative_to(model_root)
        metadata_file = model_root / ".cache" / "huggingface" / "download" / Path(
            relative.as_posix() + ".metadata"
        )
        observed_revision = None
        if metadata_file.is_file():
            lines = metadata_file.read_text(encoding="utf-8").splitlines()
            observed_revision = lines[0] if lines else None
        records.append(
            {
                "file": repo_relative(source_file),
                "metadata_found": metadata_file.is_file(),
                "observed_revision": observed_revision,
                "matches_expected": observed_revision == expected_revision,
            }
        )
    return {
        "expected_revision": expected_revision,
        "files": records,
        "passed": bool(records) and all(record["matches_expected"] for record in records),
    }


def build_real_summary(
    *,
    run_id: str,
    repo_info: dict[str, Any],
    source_manifest: dict[str, str],
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "protocol_id": REAL_PROTOCOL_ID,
        "run_id": run_id,
        "run_directory": f"log/revision_tests/{run_id}/real_preservation",
        "repository_commit": repo_info["commit"],
        "relevant_paths_clean": repo_info["relevant_paths_clean"],
        "cases_total": len(cases),
        "cases_passed": sum(case["passed"] for case in cases),
        "tensors_checked": sum(
            case["comparison"]["expected_tensor_count"] for case in cases
        ),
        "tensors_exact": sum(case["comparison"]["tensors_exact"] for case in cases),
        "untouched_tensors_checked": sum(
            case["comparison"]["untouched_checked"] for case in cases
        ),
        "untouched_tensors_exact": sum(
            case["comparison"]["untouched_exact"] for case in cases
        ),
        "source_checkpoint_sets_unchanged": sum(
            case["source_files_unchanged"] for case in cases
        ),
        "metadata_cases_equal": sum(
            case["secondary_container_metadata"]["equal"] for case in cases
        ),
        "source_manifest": source_manifest,
        "cases": cases,
        "passed": all(case["passed"] for case in cases),
        "claim_boundary": (
            "Exact tensor-state preservation for an explicit identity operation on the "
            "three enumerated checkpoint revisions; not a performance measurement."
        ),
    }


def render_real_table(summary: dict[str, Any]) -> str:
    lines = [
        "# Real-checkpoint preservation result",
        "",
        f"Protocol: `{summary['protocol_id']}`",
        f"Run: `{summary['run_id']}`",
        f"Commit: `{summary['repository_commit']}`",
        "",
        "| Target | Revision | Tensors exact | Untouched exact | Sources unchanged | Result |",
        "|---|---|---:|---:|---:|---|",
    ]
    for case in summary["cases"]:
        comparison = case["comparison"]
        lines.append(
            "| {target} | `{revision}` | {exact}/{checked} | {untouched_exact}/"
            "{untouched_checked} | {sources} | {result} |".format(
                target=case["target"],
                revision=case["revision"][:12],
                exact=comparison["tensors_exact"],
                checked=comparison["expected_tensor_count"],
                untouched_exact=comparison["untouched_exact"],
                untouched_checked=comparison["untouched_checked"],
                sources="yes" if case["source_files_unchanged"] else "no",
                result="PASS" if case["passed"] else "FAIL",
            )
        )
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- Checkpoints: {summary['cases_passed']}/{summary['cases_total']} passed.",
            f"- Tensors: {summary['tensors_exact']}/{summary['tensors_checked']} exact.",
            "- Untouched tensors: "
            f"{summary['untouched_tensors_exact']}/"
            f"{summary['untouched_tensors_checked']} exact.",
            "- Source checkpoint sets: "
            f"{summary['source_checkpoint_sets_unchanged']}/"
            f"{summary['cases_total']} unchanged.",
            "- Custom safetensors metadata preserved: "
            f"{summary['metadata_cases_equal']}/{summary['cases_total']} cases "
            "(secondary observation).",
            "- Runtime is intentionally not reported as a performance result.",
            "",
            "## Claim boundary",
            "",
            summary["claim_boundary"],
            "",
        ]
    )
    return "\n".join(lines)


def comparison_failure(reason: str) -> dict[str, Any]:
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
    }


def publish(*, run_dir: Path, publish_dir: Path) -> None:
    if publish_dir.exists():
        raise SystemExit(f"refusing to overwrite publish directory: {publish_dir}")
    publish_dir.mkdir(parents=True)
    for name in (
        "command.txt",
        "environment.json",
        "source_manifest.json",
        "summary.json",
        "paper_table.md",
    ):
        shutil.copy2(run_dir / name, publish_dir / name)


if __name__ == "__main__":
    raise SystemExit(main())
