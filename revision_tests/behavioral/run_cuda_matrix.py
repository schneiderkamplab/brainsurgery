#!/usr/bin/env python3
"""Run the frozen ten-checkpoint CUDA behavioral extension."""

from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch
import yaml

from revision_tests.scaling.oracle import (
    compare_output,
    discover_checkpoint,
    sha256_file,
    validate_input_operation,
    verify_huggingface_revision,
)
from revision_tests.scaling.validate_protocol import EXPECTED_IDS, load_cases

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PROTOCOL_PATH = HERE / "matrix_protocol.yaml"
PROTOCOL_ID = "eacl2027_behavioral_matrix_v2"


def git_value(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        default=f"eacl2027_behavioral_matrix_cuda_{git_value('rev-parse', '--short', 'HEAD')}",
    )
    parser.add_argument("--model", action="append", choices=EXPECTED_IDS, dest="models")
    parser.add_argument("--smoke-limit", type=int)
    parser.add_argument("--keep-transformed", action="store_true")
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_protocol() -> dict[str, Any]:
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict) or protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"matrix_protocol.yaml must declare {PROTOCOL_ID}")
    if protocol.get("expected_model_ids") != EXPECTED_IDS:
        raise ValueError(f"expected_model_ids must equal {EXPECTED_IDS}")
    operation = protocol.get("operation")
    if not isinstance(operation, dict):
        raise ValueError("operation is missing")
    if operation.get("target_regex") != r".*\.weight" or operation.get("factor") != 1.0:
        raise ValueError("the matrix operation must remain a .weight multiply-by-one")
    if operation.get("output_shard_size_bytes") != 256 * 1024 * 1024:
        raise ValueError("the output shard budget must remain 256 MiB")
    re.compile(operation["target_regex"])
    return protocol


def checkpoint_hashes(path: Path) -> dict[str, str]:
    layout = discover_checkpoint(path)
    files = list(layout["data_files"])
    if layout["index_file"] is not None:
        files.append(layout["index_file"])
    return {item.relative_to(layout["root"]).as_posix(): sha256_file(item) for item in files}


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO, check=True)


def write_plan(
    path: Path,
    source: Path,
    transformed: Path,
    operation: dict[str, Any],
) -> None:
    plan = {
        "inputs": [f"model::{source}"],
        "transforms": [
            {"scale_": {"target": operation["target_regex"], "by": operation["factor"]}}
        ],
        "output": {
            "path": str(transformed),
            "format": "safetensors",
            "shard": operation["output_shard_size"],
        },
    }
    path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")


def copy_model_sidecars(source: Path, transformed: Path) -> list[str]:
    copied = []
    for name, required in (("config.json", True), ("generation_config.json", False)):
        source_file = source / name
        if required and not source_file.is_file():
            raise RuntimeError(f"missing required model sidecar: {source_file}")
        if source_file.is_file():
            shutil.copy2(source_file, transformed / name)
            copied.append(name)
    return copied


def run_case(
    *,
    args: argparse.Namespace,
    case: dict[str, Any],
    operation: dict[str, Any],
    run_root: Path,
    git_commit: str,
) -> dict[str, Any]:
    case_id = case["id"]
    source = Path(case["input"])
    transformed = Path("models") / "behavioral_matrix_v2" / case_id.lower()
    case_root = run_root / case_id.lower()
    if transformed.exists():
        raise RuntimeError(f"refusing to overwrite {transformed}")
    case_root.mkdir(parents=True)

    revision = verify_huggingface_revision(source, case["revision"])
    contract = validate_input_operation(source, operation["target_regex"])
    if not revision["passed"] or not contract["passed"]:
        raise RuntimeError(f"{case_id}: source revision or operation contract failed")
    source_before = checkpoint_hashes(source)

    plan_path = case_root / "executed_plan.yaml"
    write_plan(plan_path, source, transformed, operation)
    run(
        [
            str(REPO / ".venv/bin/brainsurgery"),
            str(plan_path),
            "--provider",
            "inmemory",
            "--num-workers",
            "1",
            "--no-summarize",
            "--log-level",
            "warning",
        ]
    )

    copied_sidecars = copy_model_sidecars(source, transformed)

    comparison = compare_output(
        source,
        transformed,
        target_regex=operation["target_regex"],
        factor=operation["factor"],
        shard_size_bytes=operation["output_shard_size_bytes"],
    )
    source_after = checkpoint_hashes(source)
    tensor_validation = {
        "protocol_id": PROTOCOL_ID,
        "model_id": case_id,
        "source_revision": revision,
        "source_unchanged": source_before == source_after,
        "comparison": comparison,
        "passed": revision["passed"] and source_before == source_after and comparison["passed"],
    }
    write_json(case_root / "tensor_validation.json", tensor_validation)
    if not tensor_validation["passed"]:
        raise RuntimeError(f"{case_id}: independent tensor gate failed")
    print(
        f"{case_id}: tensor gate {comparison['tensors_passed']}/"
        f"{comparison['tensors_checked']} exact",
        flush=True,
    )

    common = [
        "--tokenizer",
        str(source),
        "--tokenizer-revision",
        case["revision"],
        "--config-revision",
        case["revision"],
        "--device",
        "cuda:0",
        "--dtype",
        case["expected_weight_dtype"],
        "--local-files-only",
    ]
    smoke = ["--smoke-limit", str(args.smoke_limit)] if args.smoke_limit is not None else []
    runner = str(HERE / "run_model.py")
    python = str(REPO / ".venv/bin/python")
    run(
        [
            python,
            runner,
            "--role",
            "reference",
            "--model",
            str(source),
            "--revision",
            case["revision"],
            *common,
            *smoke,
            "--output",
            str(case_root / "reference"),
        ]
    )
    run(
        [
            python,
            runner,
            "--role",
            "transformed",
            "--model",
            str(transformed),
            "--config",
            str(source),
            "--revision",
            f"{git_commit}_lossless_{case_id.lower()}_v2",
            *common,
            *smoke,
            "--output",
            str(case_root / "transformed"),
        ]
    )
    run(
        [
            python,
            str(HERE / "analyze.py"),
            "--reference",
            str(case_root / "reference"),
            "--transformed",
            str(case_root / "transformed"),
            "--output",
            str(case_root / "comparison.json"),
        ]
    )
    behavioral = json.loads((case_root / "comparison.json").read_text(encoding="utf-8"))
    passed = behavioral["decision"] == "PASS"
    if passed and not args.keep_transformed:
        shutil.rmtree(transformed)
    return {
        "id": case_id,
        "display": case["display"],
        "family": case["family"],
        "nominal_parameter_count": case["nominal_parameter_count"],
        "model_id": case["model_id"],
        "revision": case["revision"],
        "dtype": case["expected_weight_dtype"],
        "tensor_count": comparison["tensors_checked"],
        "tensor_exact": comparison["tensors_passed"],
        "behavioral": behavioral["aggregate"],
        "reported_eligible": behavioral["reported_eligible"],
        "passed": passed,
        "transformed_removed_after_pass": passed and not args.keep_transformed,
    }


def main() -> int:
    args = parse_args()
    if not args.run_id or not all(character.isalnum() or character == "_" for character in args.run_id):
        raise SystemExit("--run-id must contain only letters, digits, and underscores")
    if args.smoke_limit is not None and not 1 <= args.smoke_limit <= 70:
        raise SystemExit("--smoke-limit must be between 1 and 70")
    if platform.system() != "Linux":
        raise SystemExit("the reported matrix requires Linux")
    if git_value("status", "--porcelain"):
        raise SystemExit("the Git checkout must be clean")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    protocol = load_protocol()
    cases_doc = load_cases()
    selected_ids = args.models or EXPECTED_IDS
    cases = [case for case in cases_doc["models"] if case["id"] in selected_ids]
    run_root = REPO / "log" / "revision_tests" / args.run_id / "behavioral_matrix"
    if run_root.exists():
        raise SystemExit(f"refusing to overwrite {run_root}")
    for case in cases:
        if not (REPO / case["input"]).exists():
            raise SystemExit(f"missing checkpoint: {case['input']}")

    git_commit = git_value("rev-parse", "HEAD")
    print(torch.cuda.get_device_name(0), flush=True)
    run([str(REPO / ".venv/bin/python"), str(HERE / "validate_manifest.py")])
    run([str(REPO / ".venv/bin/python"), "-m", "pytest", "-q", str(HERE / "test_analysis.py")])

    results = []
    for index, case in enumerate(cases, 1):
        print(f"[{index}/{len(cases)}] {case['id']} {case['display']}", flush=True)
        results.append(
            run_case(
                args=args,
                case=case,
                operation=protocol["operation"],
                run_root=run_root,
                git_commit=git_commit,
            )
        )

    full_matrix = selected_ids == EXPECTED_IDS
    reportable = (
        args.smoke_limit is None
        and full_matrix
        and all(result["passed"] and result["reported_eligible"] for result in results)
    )
    summary = {
        "protocol_id": PROTOCOL_ID,
        "run_id": args.run_id,
        "git_commit": git_commit,
        "device": "cuda:0",
        "gpu": torch.cuda.get_device_name(0),
        "smoke_limit": args.smoke_limit,
        "complete_ordered_matrix": full_matrix,
        "reported_eligible": reportable,
        "models_passed": sum(result["passed"] for result in results),
        "models_total": len(results),
        "results": results,
        "claim_boundary": (
            "Lossless regression evidence for the enumerated checkpoints, prompt manifest, "
            "software environment, and GPU; not evidence of general model quality."
        ),
    }
    write_json(run_root / "summary.json", summary)
    print(
        f"matrix {'PASS' if reportable else 'NON-REPORTABLE'}: "
        f"{summary['models_passed']}/{summary['models_total']} models; "
        f"reported_eligible={str(reportable).lower()}",
        flush=True,
    )
    return 0 if all(result["passed"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
