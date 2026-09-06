"""Negative controls for the independent behavioral bundle analyzer."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

from revision_tests.behavioral.analyze import (
    BundleError,
    compare,
    prompt_ids_sha256,
    sha256_file,
)

MANIFEST = Path(__file__).with_name("prompt_manifest.jsonl")


def _manifest_rows() -> list[dict[str, Any]]:
    return [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line]


def _metadata(rows: list[dict[str, Any]]) -> dict[str, Any]:
    prompt_ids = [row["prompt_id"] for row in rows]
    return {
        "protocol_id": "eacl2027_behavioral_v1",
        "manifest_sha256": sha256_file(MANIFEST),
        "selected_prompt_ids_sha256": prompt_ids_sha256(prompt_ids),
        "prompt_count": len(rows),
        "tokenizer_fingerprint": "synthetic-tokenizer",
        "model_architecture_fingerprint": "synthetic-config",
        "device_fingerprint": {"type": "synthetic"},
        "platform": "synthetic",
        "python": "synthetic",
        "torch": "synthetic",
        "transformers": "synthetic",
        "dtype": "float32",
        "max_new_tokens": 32,
        "do_sample": False,
        "deterministic_algorithms": True,
        "seed": 0,
        "tokenizer_call": {"add_special_tokens": "tokenizer_default", "truncation": False},
        "choice_candidates": [" A", " B", " C", " D"],
        "reported_eligible": False,
    }


def _predictions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    predictions = []
    for row in rows:
        correct = row["expected"].get("correct_label")
        predictions.append(
            {
                "prompt_id": row["prompt_id"],
                "ordinal": row["ordinal"],
                "input_token_count": 12,
                "generated_token_ids": [11, 12, 13],
                "generated_text": "synthetic",
                "next_token_top1_id": 2,
                "mcq_predicted_label": correct,
                "mcq_choice_mean_logprobs": (
                    {"A": -4.0, "B": -3.0, "C": -2.0, "D": -1.0} if correct else None
                ),
                "correct_label": correct,
                "logits_key": f"p{row['ordinal']:04d}",
            }
        )
    return predictions


def _write_bundle(
    root: Path,
    rows: list[dict[str, Any]],
    *,
    metadata_change: tuple[str, Any] | None = None,
    prediction_change: tuple[int, str, Any] | None = None,
    logit_change: tuple[int, int, float] | None = None,
) -> None:
    root.mkdir()
    metadata = _metadata(rows)
    if metadata_change:
        metadata[metadata_change[0]] = metadata_change[1]
    predictions = _predictions(rows)
    if prediction_change:
        index, field, value = prediction_change
        predictions[index][field] = value
    logits = {
        f"p{row['ordinal']:04d}": torch.tensor([0.0, 1.0, 2.0, 1.5], dtype=torch.float32)
        for row in rows
    }
    if logit_change:
        row_index, column, value = logit_change
        logits[predictions[row_index]["logits_key"]][column] = value
    (root / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "predictions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions),
        encoding="utf-8",
    )
    save_file(logits, root / "last_token_logits.safetensors")


@pytest.fixture
def exact_bundles(tmp_path: Path) -> tuple[Path, Path]:
    rows = _manifest_rows()[:2]
    reference = tmp_path / "reference"
    transformed = tmp_path / "transformed"
    _write_bundle(reference, rows)
    _write_bundle(transformed, rows)
    return reference, transformed


def test_exact_synthetic_bundles_pass(exact_bundles: tuple[Path, Path]) -> None:
    result = compare(*exact_bundles, MANIFEST)
    assert result["decision"] == "PASS"
    assert result["reported_eligible"] is False
    assert result["aggregate"]["exact_logits"] == 2


def test_changed_logit_is_detected(tmp_path: Path) -> None:
    rows = _manifest_rows()[:2]
    reference, transformed = tmp_path / "reference", tmp_path / "transformed"
    _write_bundle(reference, rows)
    _write_bundle(transformed, rows, logit_change=(1, 0, 0.25))
    result = compare(reference, transformed, MANIFEST)
    assert result["decision"] == "FAIL"
    assert result["aggregate"]["exact_logits"] == 1
    assert result["mismatches"][0]["failed_endpoints"] == ["logits"]


def test_changed_generated_token_is_detected(tmp_path: Path) -> None:
    rows = _manifest_rows()[:2]
    reference, transformed = tmp_path / "reference", tmp_path / "transformed"
    _write_bundle(reference, rows)
    _write_bundle(transformed, rows, prediction_change=(0, "generated_token_ids", [99]))
    result = compare(reference, transformed, MANIFEST)
    assert result["decision"] == "FAIL"
    assert result["aggregate"]["greedy_id_matches"] == 1
    assert "greedy_ids" in result["mismatches"][0]["failed_endpoints"]


def test_substituted_prompt_id_is_rejected(tmp_path: Path) -> None:
    rows = _manifest_rows()[:2]
    reference, transformed = tmp_path / "reference", tmp_path / "transformed"
    _write_bundle(reference, rows)
    changed_rows = deepcopy(rows)
    changed_rows[1]["prompt_id"] = "substituted_prompt"
    _write_bundle(transformed, changed_rows)
    with pytest.raises(BundleError, match="metadata|prompt IDs"):
        compare(reference, transformed, MANIFEST)


def test_incompatible_metadata_is_rejected(tmp_path: Path) -> None:
    rows = _manifest_rows()[:2]
    reference, transformed = tmp_path / "reference", tmp_path / "transformed"
    _write_bundle(reference, rows)
    _write_bundle(transformed, rows, metadata_change=("dtype", "float16"))
    with pytest.raises(BundleError, match="incompatible run metadata"):
        compare(reference, transformed, MANIFEST)
