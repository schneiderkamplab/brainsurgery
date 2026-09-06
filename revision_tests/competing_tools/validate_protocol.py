#!/usr/bin/env python3
"""Validate the frozen competing-tool definitions and local source record."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
PROTOCOL_ID = "eacl2027_competing_tools_v1"
EXPECTED_CASE_TOOLS = {
    "R01": ["brainsurgery", "torch_state_bridge"],
    "M01": ["brainsurgery", "mergekit"],
    "M02": ["brainsurgery", "mergekit"],
}


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a mapping")
    return value


def validate() -> dict[str, Any]:
    cases_doc = load_yaml(HERE / "cases.yaml")
    tools_doc = load_yaml(HERE / "tools.yaml")
    errors = []
    for name, document in (("cases.yaml", cases_doc), ("tools.yaml", tools_doc)):
        if document.get("protocol_id") != PROTOCOL_ID:
            errors.append(f"{name}: incorrect protocol_id")
    cases = cases_doc.get("cases")
    if not isinstance(cases, list):
        errors.append("cases.yaml: cases must be a list")
        cases = []
    ids = [case.get("id") for case in cases if isinstance(case, dict)]
    if ids != list(EXPECTED_CASE_TOOLS):
        errors.append(f"case order/IDs differ: {ids}")
    for case in cases:
        if not isinstance(case, dict):
            errors.append("case is not a mapping")
            continue
        case_id = case.get("id")
        required = {
            "id",
            "name",
            "classification",
            "tools",
            "input_set",
            "output_contract",
            "formula" if case_id != "R01" else "selection",
            "comparison",
        }
        missing = required - case.keys()
        if missing:
            errors.append(f"{case_id}: missing fields {sorted(missing)}")
        if case.get("tools") != EXPECTED_CASE_TOOLS.get(case_id):
            errors.append(f"{case_id}: tool set/order differs")
        if case.get("comparison") not in {"exact", "tolerance"}:
            errors.append(f"{case_id}: invalid comparison")
    tolerance = cases_doc.get("arithmetic_tolerance")
    if tolerance != {"absolute": 1e-6, "relative": 1e-6}:
        errors.append("arithmetic tolerance differs from the frozen 1e-6 rule")

    tools = tools_doc.get("tools", {})
    if set(tools) != {
        "brainsurgery",
        "mergekit",
        "torch_state_bridge",
        "orbax_checkpoint",
    }:
        errors.append("tools.yaml: unexpected tool set")
    if tools.get("mergekit", {}).get("version") != "0.1.4":
        errors.append("MergeKit version is not 0.1.4")
    if tools.get("torch_state_bridge", {}).get("version") != "0.1.0":
        errors.append("torch-state-bridge version is not 0.1.0")
    if tools.get("orbax_checkpoint", {}).get("executable_baseline") is not False:
        errors.append("Orbax must remain a documented non-executable comparison")
    hashes = (
        tools.get("mergekit", {}).get("wheel_sha256"),
        tools.get("torch_state_bridge", {}).get("wheel_sha256"),
    )
    if any(
        not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes
    ):
        errors.append("competitor wheel hashes must be lowercase SHA-256 values")
    if errors:
        raise ValueError("protocol validation failed:\n- " + "\n- ".join(errors))
    return {"protocol_id": PROTOCOL_ID, "case_count": len(cases), "pair_count": 6}


def main() -> int:
    try:
        result = validate()
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(
        f"PASS: {result['case_count']} neutral cases and "
        f"{result['pair_count']} declared tool/case memberships"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
