#!/usr/bin/env python3
"""Validate the frozen scaling matrix and, optionally, local checkpoints."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

try:
    from .oracle import (
        PROTOCOL_ID,
        discover_checkpoint,
        validate_input_operation,
        verify_huggingface_revision,
    )
except ImportError:
    from oracle import (
        PROTOCOL_ID,
        discover_checkpoint,
        validate_input_operation,
        verify_huggingface_revision,
    )

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CASES = HERE / "cases.yaml"
EXPECTED_IDS = [
    "P01",
    "P02",
    "P03",
    "P04",
    "G01",
    "G02",
    "O01",
    "O02",
    "Q01",
    "Q02",
]
EXPECTED_METHODS = ["python_pytorch", "brainsurgery_inmemory", "brainsurgery_arena"]
EXPECTED_FAMILIES = {
    "pythia": 4,
    "gpt2": 2,
    "olmo": 2,
    "qwen2_5": 2,
}


def load_cases() -> dict:
    doc = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or doc.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"cases.yaml must declare {PROTOCOL_ID}")
    if doc.get("methods") != EXPECTED_METHODS:
        raise ValueError(f"methods must equal {EXPECTED_METHODS}")
    models = doc.get("models")
    if not isinstance(models, list) or [item.get("id") for item in models] != EXPECTED_IDS:
        raise ValueError(f"models must be the ordered {EXPECTED_IDS} matrix")
    operation = doc.get("operation")
    if not isinstance(operation, dict):
        raise ValueError("operation is missing")
    if operation.get("factor") != 0.5:
        raise ValueError("factor must remain 0.5")
    if operation.get("output_shard_size_bytes") != 512 * 1024 * 1024:
        raise ValueError("output shard budget must remain 512 MiB")
    pattern = re.compile(operation.get("target_regex", ""))
    if pattern.pattern != r".*\.weight":
        raise ValueError("target regex has changed")
    for model in models:
        if not re.fullmatch(r"[0-9a-f]{40}", model.get("revision", "")):
            raise ValueError(f"{model.get('id')} revision is not a 40-character commit")
        if model.get("expected_layout") not in {"single", "sharded"}:
            raise ValueError(f"invalid expected layout for {model.get('id')}")
        if model.get("expected_weight_dtype") not in {"float16", "float32", "bfloat16"}:
            raise ValueError(f"invalid expected weight dtype for {model.get('id')}")
        if not isinstance(model.get("nominal_parameter_count"), int) or model[
            "nominal_parameter_count"
        ] <= 0:
            raise ValueError(f"invalid nominal parameter count for {model.get('id')}")
        expected_role = "primary_scaling" if model.get("family") == "pythia" else "architecture_pair"
        if model.get("analysis_role") != expected_role:
            raise ValueError(f"invalid analysis role for {model.get('id')}")
        path = Path(model.get("input", ""))
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe model path for {model.get('id')}")
    family_counts = {
        family: sum(model.get("family") == family for model in models)
        for family in EXPECTED_FAMILIES
    }
    if family_counts != EXPECTED_FAMILIES:
        raise ValueError(f"family matrix changed: {family_counts}")
    return doc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-models", action="store_true")
    args = parser.parse_args()
    doc = load_cases()
    checked = 0
    if args.check_models:
        for model in doc["models"]:
            path = REPO / model["input"]
            layout = discover_checkpoint(path)
            if layout["layout"] != model["expected_layout"]:
                raise SystemExit(
                    f"{model['id']} layout is {layout['layout']}, expected {model['expected_layout']}"
                )
            revision = verify_huggingface_revision(path, model["revision"])
            if not revision["passed"]:
                raise SystemExit(
                    f"{model['id']} checkpoint files lack metadata for revision {model['revision']}"
                )
            contract = validate_input_operation(path, doc["operation"]["target_regex"])
            if not contract["passed"]:
                raise SystemExit(f"{model['id']} violates the frozen operation contract")
            if set(contract["matched_dtype_counts"]) != {model["expected_weight_dtype"]}:
                raise SystemExit(
                    f"{model['id']} weight dtypes {sorted(contract['matched_dtype_counts'])} "
                    f"do not equal {model['expected_weight_dtype']}"
                )
            checked += 1
    suffix = f" and {checked} local checkpoints" if args.check_models else ""
    print(f"PASS: {PROTOCOL_ID} matrix{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
