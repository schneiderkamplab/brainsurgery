"""Negative controls and invariants for the scaling harness."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from revision_tests.scaling import baseline, oracle
from revision_tests.scaling.run import render_csv, render_latex, render_markdown
from revision_tests.scaling.validate_protocol import load_cases


def fixture(path: Path) -> Path:
    path.mkdir()
    save_file(
        {
            "block.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
            "block.bias": torch.tensor([1.0, -2.0, 3.0]),
            "counter": torch.tensor([7], dtype=torch.int64),
        },
        str(path / "model.safetensors"),
    )
    return path


def transform_with_baseline(source: Path, output: Path, shard_size: int = 64) -> None:
    state = baseline.load_state(source)
    state["block.weight"] = state["block.weight"] * 0.5
    baseline.save_sharded(state, output, shard_size)


def test_protocol_matrix_is_frozen() -> None:
    doc = load_cases()
    assert [model["id"] for model in doc["models"]] == [
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
    assert [model["family"] for model in doc["models"]].count("pythia") == 4
    assert doc["operation"]["output_shard_size_bytes"] == 512 * 1024 * 1024
    assert doc["models"][3]["revision"] == "bb1e3e710cdf6b524461d543cfb5ba773f0a81b6"
    assert doc["models"][-1]["revision"] == "d149729398750b98c0af14eb82c78cfe92750796"


def test_independent_oracle_accepts_exact_output(tmp_path: Path) -> None:
    source = fixture(tmp_path / "source")
    assert oracle.validate_input_operation(source, r".*\.weight")["passed"]
    output = tmp_path / "output"
    transform_with_baseline(source, output)
    result = oracle.compare_output(
        source, output, target_regex=r".*\.weight", factor=0.5, shard_size_bytes=64
    )
    assert result["passed"]
    assert result["matched_tensor_count"] == 1
    assert result["tensors_passed"] == 3


def test_oracle_rejects_unscaled_selected_tensor(tmp_path: Path) -> None:
    source = fixture(tmp_path / "source")
    output = tmp_path / "output"
    baseline.save_sharded(baseline.load_state(source), output, 64)
    result = oracle.compare_output(
        source, output, target_regex=r".*\.weight", factor=0.5, shard_size_bytes=64
    )
    assert not result["passed"]
    check = next(item for item in result["tensor_checks"] if item["name"] == "block.weight")
    assert not check["passed"]


def test_oracle_rejects_changed_non_target(tmp_path: Path) -> None:
    source = fixture(tmp_path / "source")
    output = tmp_path / "output"
    transform_with_baseline(source, output)
    state = baseline.load_state(output)
    state["block.bias"] = state["block.bias"] + 1
    replacement = tmp_path / "replacement"
    baseline.save_sharded(state, replacement, 64)
    result = oracle.compare_output(
        source, replacement, target_regex=r".*\.weight", factor=0.5, shard_size_bytes=64
    )
    assert not result["passed"]
    check = next(item for item in result["tensor_checks"] if item["name"] == "block.bias")
    assert not check["selected"]
    assert not check["passed"]


def test_oracle_rejects_bad_index_coverage(tmp_path: Path) -> None:
    source = fixture(tmp_path / "source")
    output = tmp_path / "output"
    transform_with_baseline(source, output)
    index_path = output / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    del index["weight_map"]["counter"]
    index_path.write_text(json.dumps(index), encoding="utf-8")
    try:
        oracle.discover_checkpoint(output)
    except ValueError as exc:
        assert "index" in str(exc)
    else:
        raise AssertionError("invalid index was accepted")


def test_paper_reports_suppress_nonreportable_performance() -> None:
    summary = {
        "protocol_id": oracle.PROTOCOL_ID,
        "run_id": "smoke",
        "git_commit": "abc",
        "reported_eligible": False,
        "eligibility_reasons": ["smoke"],
        "pairs": {
            "SMOKE:python_pytorch": {
                "model_id": "SMOKE",
                "display": "tiny",
                "family": "synthetic",
                "analysis_role": "smoke",
                "method": "python_pytorch",
                "correct_attempts": 1,
                "measured_attempts": 1,
                "wall_seconds": {"median": 123.456},
                "peak_rss_bytes": {"median": 1024**3},
                "effective_logical_bytes_per_second": {"median": 2 * 1024**3},
                "output_shard_counts": [1],
            }
        },
        "models": {
            "SMOKE": {
                "model_id": "local/synthetic",
                "revision": "abc",
                "input_manifest": {
                    "stored_tensor_element_count": 45,
                    "logical_tensor_bytes": 152,
                    "checkpoint_file_bytes": 432,
                    "dtype_logical_bytes": {"float32": 144, "int64": 8},
                    "data_file_count": 1,
                },
                "nominal_parameter_count": None,
            }
        },
        "claim_boundary": "narrow",
    }
    assert "123.456" not in render_markdown(deepcopy(summary))
    assert "123.46" not in render_latex(deepcopy(summary))
    assert "123.456" not in render_csv(deepcopy(summary))
    assert "NON-REPORTABLE" in render_markdown(summary)


def test_source_integrity_detects_mutation(tmp_path: Path) -> None:
    source = fixture(tmp_path / "source")
    before = oracle.checkpoint_manifest(source, tensor_hashes=False)["files"]
    expected = {name: item["sha256"] for name, item in before.items()}
    state = load_file(str(source / "model.safetensors"))
    state["block.bias"] = state["block.bias"] + 2
    save_file(state, str(source / "replacement.safetensors"))
    (source / "model.safetensors").unlink()
    (source / "replacement.safetensors").rename(source / "model.safetensors")
    assert not oracle.files_unchanged(source, expected)["passed"]


def test_revision_metadata_is_checked_for_every_checkpoint_file(tmp_path: Path) -> None:
    source = fixture(tmp_path / "source")
    metadata = source / ".cache" / "huggingface" / "download" / "model.safetensors.metadata"
    metadata.parent.mkdir(parents=True)
    revision = "a" * 40
    metadata.write_text(f"{revision}\nunused-etag\n", encoding="utf-8")
    assert oracle.verify_huggingface_revision(source, revision)["passed"]
    assert not oracle.verify_huggingface_revision(source, "b" * 40)["passed"]
