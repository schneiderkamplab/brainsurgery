from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import tests.conftest as test_fixtures
from brainsurgery.synapse import lower_axon_program_to_synapse_spec, parse_axon_program_from_path
from brainsurgery.synapse.axon_test_matrix import run_axon_test_matrix
from tests.model_downloads import MATRIX_AXON_MODEL_DIR_PAIRS, MODEL_SPECS


def _load_axon_spec(path: Path) -> dict[str, Any]:
    modules = parse_axon_program_from_path(path)
    return lower_axon_program_to_synapse_spec(modules)


def test_bert_fixture_is_declared() -> None:
    assert hasattr(test_fixtures, "bert_local_path")


def test_bert_is_registered_in_download_specs_and_matrix() -> None:
    assert MODEL_SPECS["bert"].repo_id == "google-bert/bert-base-uncased"
    assert MODEL_SPECS["bert"].local_dir == "bert"
    assert ("bert", "bert") in MATRIX_AXON_MODEL_DIR_PAIRS


def test_matrix_resolves_bert_to_bert_axon(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    examples_dir = tmp_path / "examples"
    models_dir = tmp_path / "models"
    examples_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    (examples_dir / "bert.axon").write_text(
        "bert :: Tensor -> Tensor\nbert x = x\n",
        encoding="utf-8",
    )
    (models_dir / "bert").mkdir(parents=True, exist_ok=True)

    exit_code = run_axon_test_matrix(
        examples_dir=examples_dir,
        models_dir=models_dir,
        dry_run=True,
        include=["bert"],
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "bert.axon" in out
    assert "/models/bert" in out


def test_bert_axon_lowers_with_expected_symbols(repo_root: Path) -> None:
    spec = _load_axon_spec(repo_root / "examples" / "bert.axon")

    assert spec.get("synapse") == 1
    model = spec.get("model", {})
    assert model.get("outputs") == {"logits": "logits"}

    symbols = model.get("symbols", {})
    assert symbols.get("D") == 768
    assert symbols.get("V") in (30522, None)
    assert symbols.get("L") == 12
    assert symbols.get("H") == 12
    assert symbols.get("FFN") == 3072
    assert symbols.get("EPS") == 1.0e-12

    blocks = model.get("blocks", {})
    assert "bert_block" in blocks

    graph = model.get("graph", [])
    assert isinstance(graph, list)
    position_nodes = [
        node_spec
        for item in graph
        if isinstance(item, dict)
        for node_spec in item.values()
        if isinstance(node_spec, dict) and node_spec.get("_op") == "position_ids"
    ]
    assert len(position_nodes) == 1
    assert position_nodes[0].get("_args") == ["input_ids", "null"]

    decoder_nodes = [
        node_spec
        for item in graph
        if isinstance(item, dict)
        for node_spec in item.values()
        if isinstance(node_spec, dict)
        and node_spec.get("_op") == "linear"
        and (
            node_spec.get("weight") == "@@bert.embeddings.word_embeddings.weight"
            or node_spec.get("_params", {}).get("weight")
            == "@@bert.embeddings.word_embeddings.weight"
        )
    ]
    assert len(decoder_nodes) == 1
    assert decoder_nodes[0].get("bias_path") == "@@cls.predictions.bias"

    block_graph = blocks["bert_block"].get("graph", [])
    assert isinstance(block_graph, list)
    attention_nodes = [
        node_spec
        for item in block_graph
        if isinstance(item, dict)
        for node_spec in item.values()
        if isinstance(node_spec, dict) and node_spec.get("_op") == "attention"
    ]
    assert len(attention_nodes) == 1
    assert attention_nodes[0].get("mask") == "attn_mask"
    assert attention_nodes[0].get("padding_mask") is True
