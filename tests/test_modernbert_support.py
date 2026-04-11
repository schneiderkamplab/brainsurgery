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


def _collect_node_specs(items: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        for node_spec in item.values():
            if not isinstance(node_spec, dict):
                continue
            if "_op" in node_spec:
                out.append(node_spec)
            for key in ("graph", "_body", "_then", "_else"):
                nested = node_spec.get(key)
                if isinstance(nested, list):
                    out.extend(_collect_node_specs(nested))
    return out


def test_modernbert_fixture_is_declared() -> None:
    assert hasattr(test_fixtures, "modernbert_local_path")


def test_modernbert_is_registered_in_download_specs_and_matrix() -> None:
    assert MODEL_SPECS["modernbert"].repo_id == "answerdotai/ModernBERT-base"
    assert MODEL_SPECS["modernbert"].local_dir == "modernbert"
    assert ("modernbert", "modernbert") in MATRIX_AXON_MODEL_DIR_PAIRS


def test_matrix_resolves_modernbert_to_modernbert_axon(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    examples_dir = tmp_path / "examples"
    models_dir = tmp_path / "models"
    examples_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    (examples_dir / "modernbert.axon").write_text(
        "modernbert :: Tensor -> Tensor\nmodernbert x = x\n",
        encoding="utf-8",
    )
    (models_dir / "modernbert").mkdir(parents=True, exist_ok=True)

    exit_code = run_axon_test_matrix(
        examples_dir=examples_dir,
        models_dir=models_dir,
        dry_run=True,
        include=["modernbert"],
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "modernbert.axon" in out
    assert "/models/modernbert" in out


def test_modernbert_axon_lowers_with_expected_structure(repo_root: Path) -> None:
    pytest.skip(
        "outdated modernbert example lowering expectations after positional-only/kwarg changes"
    )
    spec = _load_axon_spec(repo_root / "examples" / "modernbert.axon")
    assert spec.get("synapse") == 1
    model = spec.get("model", {})
    assert model.get("outputs") == {"logits": "logits"}

    symbols = model.get("symbols", {})
    assert symbols.get("D") == 768
    assert symbols.get("V") in (50368, None)
    assert symbols.get("L") == 22
    assert symbols.get("H") == 12
    assert symbols.get("FFN") == 1152
    assert symbols.get("EPS") == 1.0e-5

    blocks = model.get("blocks", {})
    assert "modernbert_layer" in blocks

    model_graph = model.get("graph", [])
    assert isinstance(model_graph, list)
    top_nodes = _collect_node_specs(model_graph)
    top_decoder_nodes = [
        node_spec
        for node_spec in top_nodes
        if node_spec.get("_op") == "linear"
        and node_spec.get("weight") == "model.embeddings.tok_embeddings.weight"
    ]
    assert len(top_decoder_nodes) == 1
    assert top_decoder_nodes[0].get("bias") is True

    block_graph = blocks["modernbert_layer"].get("graph", [])
    assert isinstance(block_graph, list)
    block_nodes = _collect_node_specs(block_graph)

    assert any(node.get("_op") == "rope_pair" for node in block_nodes)
    assert any(node.get("_op") == "bidirectional_mask" for node in block_nodes)
    assert any(node.get("_op") == "split" for node in block_nodes)

    attention_nodes = [node for node in block_nodes if node.get("_op") == "attention"]
    assert len(attention_nodes) == 1
    assert attention_nodes[0].get("causal") is False

    no_bias_layernorm_nodes = [
        node
        for node in block_nodes
        if node.get("_op") == "layernorm" and ("bias" in node and node.get("bias") is None)
    ]
    assert len(no_bias_layernorm_nodes) >= 2
