#!/usr/bin/env python3
"""Validate the frozen behavioral prompt manifest and its compact summary."""

from __future__ import annotations

import hashlib
import json
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
PROTOCOL_ID = "eacl2027_behavioral_v1"
LABELS = ["A", "B", "C", "D"]
REQUIRED_FIELDS = {
    "protocol_id",
    "ordinal",
    "prompt_id",
    "source",
    "source_name",
    "source_item_id",
    "source_row_index_zero_based",
    "source_revision",
    "split",
    "license",
    "citation_key",
    "selection_stratum",
    "selection_rank_one_based",
    "selection_rank_sha256",
    "language_code",
    "language",
    "script",
    "instruction_language",
    "task_category",
    "prompt_template_id",
    "normalization",
    "prompt",
    "prompt_sha256",
    "expected",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"line {line_number} is not a JSON object")
        rows.append(value)
    return rows


def expected_summary(
    rows: list[dict[str, Any]], config: dict[str, Any], path: Path
) -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "manifest": path.name,
        "manifest_sha256": sha256_file(path),
        "prompt_count": len(rows),
        "multiple_choice_count": sum(
            row.get("expected", {}).get("kind") == "multiple_choice" for row in rows
        ),
        "source_counts": dict(sorted(Counter(row.get("source") for row in rows).items())),
        "language_counts": dict(sorted(Counter(row.get("language_code") for row in rows).items())),
        "task_category_counts": dict(
            sorted(Counter(row.get("task_category") for row in rows).items())
        ),
        "license_counts": dict(sorted(Counter(row.get("license") for row in rows).items())),
        "selection_seed": config["selection_seed"],
        "source_fingerprints": config["sources"],
    }


def validate(
    manifest_path: Path = HERE / "prompt_manifest.jsonl",
    summary_path: Path = HERE / "manifest_summary.json",
    sources_path: Path = HERE / "sources.yaml",
) -> dict[str, Any]:
    errors: list[str] = []
    config = yaml.safe_load(sources_path.read_text(encoding="utf-8"))
    rows = read_jsonl(manifest_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    if config.get("protocol_id") != PROTOCOL_ID:
        errors.append(f"sources protocol_id is not {PROTOCOL_ID}")
    if len(rows) != 70:
        errors.append(f"expected 70 prompts, found {len(rows)}")
    if [row.get("ordinal") for row in rows] != list(range(1, len(rows) + 1)):
        errors.append("ordinals are not the contiguous sequence 1..N")

    prompt_ids: set[str] = set()
    record_keys: set[tuple[Any, Any, Any]] = set()
    belebele_ids: dict[str, set[str]] = defaultdict(set)
    mmlu_counts: Counter[str] = Counter()
    human_eval_count = 0

    for position, row in enumerate(rows, 1):
        label = f"row {position} ({row.get('prompt_id', '<missing>')})"
        missing = REQUIRED_FIELDS - row.keys()
        if missing:
            errors.append(f"{label}: missing fields {sorted(missing)}")
            continue
        if row["protocol_id"] != PROTOCOL_ID:
            errors.append(f"{label}: wrong protocol_id")
        if row["prompt_id"] in prompt_ids:
            errors.append(f"{label}: duplicate prompt_id")
        prompt_ids.add(row["prompt_id"])
        record_key = (row["source"], row["source_item_id"], row["language_code"])
        if record_key in record_keys:
            errors.append(f"{label}: duplicate source/item/language tuple")
        record_keys.add(record_key)

        source = config.get("sources", {}).get(row["source"])
        if source is None:
            errors.append(f"{label}: unknown source {row['source']!r}")
        else:
            expected_revision = source.get("data_revision", source["repository_revision"])
            for field, expected in (
                ("source_name", source["name"]),
                ("source_revision", expected_revision),
                ("split", source["split"]),
                ("license", source["license"]),
                ("citation_key", source["citation_key"]),
            ):
                if row[field] != expected:
                    errors.append(f"{label}: {field} does not match sources.yaml")

        prompt = row["prompt"]
        if not isinstance(prompt, str) or not prompt:
            errors.append(f"{label}: prompt must be a non-empty string")
        else:
            if "\r" in prompt or unicodedata.normalize("NFC", prompt) != prompt:
                errors.append(f"{label}: prompt is not normalized to NFC/LF")
            if sha256_bytes(prompt.encode("utf-8")) != row["prompt_sha256"]:
                errors.append(f"{label}: prompt_sha256 mismatch")
        if row["normalization"] != "Unicode_NFC_and_LF":
            errors.append(f"{label}: unexpected normalization declaration")
        if (
            not isinstance(row["source_row_index_zero_based"], int)
            or row["source_row_index_zero_based"] < 0
        ):
            errors.append(f"{label}: invalid source row index")
        if (
            not isinstance(row["selection_rank_one_based"], int)
            or row["selection_rank_one_based"] < 1
        ):
            errors.append(f"{label}: invalid selection rank")
        rank_hash = row["selection_rank_sha256"]
        if not isinstance(rank_hash, str) or len(rank_hash) != 64:
            errors.append(f"{label}: invalid selection SHA-256")

        expected = row["expected"]
        if not isinstance(expected, dict) or expected.get("kind") not in {
            "multiple_choice",
            "regression_only",
        }:
            errors.append(f"{label}: invalid expected block")
        elif expected["kind"] == "multiple_choice":
            if expected.get("labels") != LABELS or expected.get("correct_label") not in LABELS:
                errors.append(f"{label}: invalid multiple-choice labels or answer")
        elif set(expected) != {"kind"}:
            errors.append(f"{label}: regression-only row contains answer data")

        if row["source"] == "belebele":
            belebele_ids[row["language_code"]].add(row["source_item_id"])
        elif row["source"] == "mmlu":
            mmlu_counts[row["selection_stratum"]] += 1
        elif row["source"] == "human_eval":
            human_eval_count += 1

    expected_languages = set(config["strata"]["belebele_languages"])
    if set(belebele_ids) != expected_languages:
        errors.append("Belebele language strata do not match sources.yaml")
    if belebele_ids:
        parallel_sets = list(belebele_ids.values())
        if any(len(ids) != 5 for ids in parallel_sets):
            errors.append("each Belebele language must contain five source identifiers")
        if any(ids != parallel_sets[0] for ids in parallel_sets[1:]):
            errors.append("Belebele language strata do not share the same parallel identifiers")
    expected_subjects = set(config["strata"]["mmlu_subjects"])
    if set(mmlu_counts) != expected_subjects or any(
        mmlu_counts[subject] != 5 for subject in expected_subjects
    ):
        errors.append("MMLU must contain five prompts in every declared subject")
    if human_eval_count != config["sample_sizes"]["human_eval"]:
        errors.append(f"HumanEval count is {human_eval_count}, expected 10")

    computed_summary = expected_summary(rows, config, manifest_path)
    if summary != computed_summary:
        differing = sorted(
            key
            for key in set(summary) | set(computed_summary)
            if summary.get(key) != computed_summary.get(key)
        )
        errors.append(f"manifest_summary.json differs in fields: {differing}")
    if errors:
        raise ValueError("behavioral manifest validation failed:\n- " + "\n- ".join(errors))
    return computed_summary


def main() -> int:
    try:
        summary = validate()
    except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(
        "PASS: "
        f"{summary['prompt_count']}/70 prompts, "
        f"{summary['multiple_choice_count']}/60 multiple-choice, "
        f"manifest {summary['manifest_sha256']}"
    )
    print(f"sources: {json.dumps(summary['source_counts'], sort_keys=True)}")
    print(f"languages: {json.dumps(summary['language_counts'], sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
