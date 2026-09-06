#!/usr/bin/env python3
"""Compare reference and transformed behavioral result bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "prompt_manifest.jsonl"
PROTOCOL_ID = "eacl2027_behavioral_v1"
COMPATIBILITY_FIELDS = (
    "protocol_id",
    "manifest_sha256",
    "selected_prompt_ids_sha256",
    "prompt_count",
    "tokenizer_fingerprint",
    "model_architecture_fingerprint",
    "device_fingerprint",
    "platform",
    "python",
    "torch",
    "transformers",
    "dtype",
    "max_new_tokens",
    "do_sample",
    "deterministic_algorithms",
    "seed",
    "tokenizer_call",
    "choice_candidates",
)
PREDICTION_FIELDS = {
    "prompt_id",
    "ordinal",
    "input_token_count",
    "generated_token_ids",
    "generated_text",
    "next_token_top1_id",
    "mcq_predicted_label",
    "mcq_choice_mean_logprobs",
    "correct_label",
    "logits_key",
}


class BundleError(ValueError):
    """A result bundle is incomplete, incompatible, or internally inconsistent."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--transformed", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise BundleError(f"{path}: line {number} is not an object")
        rows.append(row)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_ids_sha256(prompt_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(prompt_ids).encode("utf-8")).hexdigest()


def load_bundle(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, torch.Tensor]]:
    required = ("metadata.json", "predictions.jsonl", "last_token_logits.safetensors")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise BundleError(f"{root}: missing {missing}")
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    predictions = read_jsonl(root / "predictions.jsonl")
    logits = load_file(root / "last_token_logits.safetensors", device="cpu")
    if not isinstance(metadata, dict):
        raise BundleError(f"{root}: metadata is not an object")
    if metadata.get("protocol_id") != PROTOCOL_ID:
        raise BundleError(f"{root}: wrong or missing protocol_id")
    if metadata.get("prompt_count") != len(predictions):
        raise BundleError(f"{root}: prompt_count does not match predictions")
    seen_ids: set[str] = set()
    seen_keys: set[str] = set()
    for index, row in enumerate(predictions, 1):
        missing_fields = PREDICTION_FIELDS - row.keys()
        if missing_fields:
            raise BundleError(f"{root}: prediction {index} missing {sorted(missing_fields)}")
        if row["prompt_id"] in seen_ids:
            raise BundleError(f"{root}: duplicate prompt_id {row['prompt_id']}")
        seen_ids.add(row["prompt_id"])
        key = row["logits_key"]
        if key in seen_keys or key not in logits:
            raise BundleError(f"{root}: invalid or duplicate logits key {key!r}")
        seen_keys.add(key)
        tensor = logits[key]
        if tensor.dtype != torch.float32 or tensor.ndim != 1:
            raise BundleError(f"{root}: {key} must be a one-dimensional float32 tensor")
    if set(logits) != seen_keys:
        raise BundleError(f"{root}: unreferenced logits tensors are present")
    return metadata, predictions, logits


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().contiguous().numpy().tobytes()


def _diagnostics(reference: torch.Tensor, transformed: torch.Tensor) -> dict[str, Any]:
    if reference.shape != transformed.shape:
        return {
            "shape_equal": False,
            "byte_exact": False,
            "max_absolute_difference": None,
            "mean_absolute_difference": None,
            "cosine_similarity": None,
        }
    byte_exact = _tensor_bytes(reference) == _tensor_bytes(transformed)
    difference = (reference.to(torch.float64) - transformed.to(torch.float64)).abs()
    cosine = torch.nn.functional.cosine_similarity(
        reference.to(torch.float64).unsqueeze(0),
        transformed.to(torch.float64).unsqueeze(0),
    ).item()
    return {
        "shape_equal": True,
        "byte_exact": byte_exact,
        "max_absolute_difference": float(difference.max().item()),
        "mean_absolute_difference": float(difference.mean().item()),
        "cosine_similarity": float(cosine) if math.isfinite(cosine) else None,
    }


def _rates(counts: Counter[str]) -> dict[str, Any]:
    total = counts["prompts"]
    mcq = counts["mcq"]
    result: dict[str, Any] = {
        "prompts": total,
        "exact_logits": counts["exact_logits"],
        "top1_matches": counts["top1_matches"],
        "greedy_id_matches": counts["greedy_id_matches"],
        "generated_text_matches": counts["generated_text_matches"],
        "mcq": mcq,
        "mcq_prediction_matches": counts["mcq_prediction_matches"],
        "reference_mcq_correct": counts["reference_mcq_correct"],
        "transformed_mcq_correct": counts["transformed_mcq_correct"],
    }
    result["rates"] = {
        "exact_logits": counts["exact_logits"] / total if total else None,
        "top1_matches": counts["top1_matches"] / total if total else None,
        "greedy_id_matches": counts["greedy_id_matches"] / total if total else None,
        "mcq_prediction_matches": counts["mcq_prediction_matches"] / mcq if mcq else None,
        "reference_mcq_accuracy": counts["reference_mcq_correct"] / mcq if mcq else None,
        "transformed_mcq_accuracy": counts["transformed_mcq_correct"] / mcq if mcq else None,
    }
    return result


def compare(reference_dir: Path, transformed_dir: Path, manifest_path: Path) -> dict[str, Any]:
    manifest_rows = read_jsonl(manifest_path)
    manifest_by_id = {row["prompt_id"]: row for row in manifest_rows}
    if len(manifest_by_id) != len(manifest_rows):
        raise BundleError("manifest contains duplicate prompt IDs")
    reference_meta, reference_rows, reference_logits = load_bundle(reference_dir)
    transformed_meta, transformed_rows, transformed_logits = load_bundle(transformed_dir)

    incompatible = {
        field: {"reference": reference_meta.get(field), "transformed": transformed_meta.get(field)}
        for field in COMPATIBILITY_FIELDS
        if reference_meta.get(field) != transformed_meta.get(field)
    }
    if incompatible:
        raise BundleError(f"incompatible run metadata: {json.dumps(incompatible, sort_keys=True)}")
    expected_manifest_sha = sha256_file(manifest_path)
    if reference_meta.get("manifest_sha256") != expected_manifest_sha:
        raise BundleError("bundle manifest checksum does not match --manifest")

    reference_ids = [row["prompt_id"] for row in reference_rows]
    transformed_ids = [row["prompt_id"] for row in transformed_rows]
    if reference_ids != transformed_ids:
        raise BundleError("reference and transformed prompt IDs/order differ")
    unknown = [prompt_id for prompt_id in reference_ids if prompt_id not in manifest_by_id]
    if unknown:
        raise BundleError(f"bundle contains prompt IDs absent from manifest: {unknown}")
    if prompt_ids_sha256(reference_ids) != reference_meta.get("selected_prompt_ids_sha256"):
        raise BundleError("selected prompt ID checksum is inconsistent")
    expected_prefix = [row["prompt_id"] for row in manifest_rows[: len(reference_rows)]]
    if reference_ids != expected_prefix:
        raise BundleError("bundle prompts are not the declared leading manifest subset")

    aggregate: Counter[str] = Counter()
    grouped: dict[str, dict[str, Counter[str]]] = {
        "source": defaultdict(Counter),
        "language": defaultdict(Counter),
        "stratum": defaultdict(Counter),
    }
    mismatches = []
    max_absolute_difference = 0.0
    absolute_difference_sum = 0.0
    cosine_values: list[float] = []

    for reference_row, transformed_row in zip(reference_rows, transformed_rows, strict=True):
        prompt_id = reference_row["prompt_id"]
        manifest = manifest_by_id[prompt_id]
        if (
            reference_row["ordinal"] != manifest["ordinal"]
            or transformed_row["ordinal"] != manifest["ordinal"]
        ):
            raise BundleError(f"{prompt_id}: ordinal does not match manifest")
        if reference_row["input_token_count"] != transformed_row["input_token_count"]:
            raise BundleError(f"{prompt_id}: tokenized input lengths differ")
        for row, role in ((reference_row, "reference"), (transformed_row, "transformed")):
            expected_label = manifest["expected"].get("correct_label")
            if row["correct_label"] != expected_label:
                raise BundleError(f"{prompt_id}: {role} correct_label does not match manifest")

        diagnostics = _diagnostics(
            reference_logits[reference_row["logits_key"]],
            transformed_logits[transformed_row["logits_key"]],
        )
        exact_logits = diagnostics["byte_exact"]
        top1_match = reference_row["next_token_top1_id"] == transformed_row["next_token_top1_id"]
        greedy_match = (
            reference_row["generated_token_ids"] == transformed_row["generated_token_ids"]
        )
        text_match = reference_row["generated_text"] == transformed_row["generated_text"]
        mcq = manifest["expected"]["kind"] == "multiple_choice"
        mcq_match = reference_row["mcq_predicted_label"] == transformed_row["mcq_predicted_label"]

        flags = {
            "prompts": True,
            "exact_logits": exact_logits,
            "top1_matches": top1_match,
            "greedy_id_matches": greedy_match,
            "generated_text_matches": text_match,
            "mcq": mcq,
            "mcq_prediction_matches": mcq and mcq_match,
            "reference_mcq_correct": mcq
            and reference_row["mcq_predicted_label"] == reference_row["correct_label"],
            "transformed_mcq_correct": mcq
            and transformed_row["mcq_predicted_label"] == transformed_row["correct_label"],
        }
        for name, passed in flags.items():
            if passed:
                aggregate[name] += 1
        for dimension, value in (
            ("source", manifest["source"]),
            ("language", manifest["language_code"]),
            ("stratum", manifest["selection_stratum"]),
        ):
            for name, passed in flags.items():
                if passed:
                    grouped[dimension][value][name] += 1

        if diagnostics["max_absolute_difference"] is not None:
            max_absolute_difference = max(
                max_absolute_difference, diagnostics["max_absolute_difference"]
            )
            absolute_difference_sum += diagnostics["mean_absolute_difference"]
        if diagnostics["cosine_similarity"] is not None:
            cosine_values.append(diagnostics["cosine_similarity"])
        failures = []
        if not exact_logits:
            failures.append("logits")
        if not top1_match:
            failures.append("top1")
        if not greedy_match:
            failures.append("greedy_ids")
        if mcq and not mcq_match:
            failures.append("mcq_prediction")
        if failures:
            mismatches.append(
                {"prompt_id": prompt_id, "failed_endpoints": failures, "logits": diagnostics}
            )

    prompt_count = len(reference_rows)
    mcq_count = aggregate["mcq"]
    exact_pass = (
        aggregate["exact_logits"] == prompt_count
        and aggregate["top1_matches"] == prompt_count
        and aggregate["greedy_id_matches"] == prompt_count
        and aggregate["mcq_prediction_matches"] == mcq_count
    )
    reported_eligible = bool(
        reference_meta.get("reported_eligible")
        and transformed_meta.get("reported_eligible")
        and prompt_count == len(manifest_rows) == 70
    )
    return {
        "protocol_id": PROTOCOL_ID,
        "decision": "PASS" if exact_pass else "FAIL",
        "reported_eligible": reported_eligible,
        "claim": "lossless behavioral regression" if reported_eligible else "non-reportable smoke",
        "reference": str(reference_dir),
        "transformed": str(transformed_dir),
        "manifest": str(manifest_path),
        "manifest_sha256": expected_manifest_sha,
        "aggregate": _rates(aggregate),
        "by_source": {key: _rates(value) for key, value in sorted(grouped["source"].items())},
        "by_language": {key: _rates(value) for key, value in sorted(grouped["language"].items())},
        "by_stratum": {key: _rates(value) for key, value in sorted(grouped["stratum"].items())},
        "logit_diagnostics": {
            "maximum_absolute_difference": max_absolute_difference,
            "mean_of_per_prompt_mean_absolute_differences": (
                absolute_difference_sum / prompt_count if prompt_count else None
            ),
            "mean_cosine_similarity": (
                sum(cosine_values) / len(cosine_values) if cosine_values else None
            ),
        },
        "mismatches": mismatches,
    }


def main() -> int:
    args = parse_args()
    try:
        result = compare(args.reference, args.transformed, args.manifest)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (BundleError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        f"{result['decision']}: {result['aggregate']['exact_logits']}/"
        f"{result['aggregate']['prompts']} exact logit vectors; "
        f"reported_eligible={str(result['reported_eligible']).lower()}"
    )
    return 0 if result["decision"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
