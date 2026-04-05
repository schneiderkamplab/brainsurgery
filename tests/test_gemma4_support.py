from __future__ import annotations

from pathlib import Path
from typing import Any

from brainsurgery.synapse import lower_axon_program_to_synapse_spec, parse_axon_program_from_path


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


def test_gemma4_e_axon_lowers_with_expected_structure(repo_root: Path) -> None:
    spec = _load_axon_spec(repo_root / "examples" / "gemma4_e.axon")
    assert spec.get("synapse") == 1
    model = spec.get("model", {})
    assert model.get("outputs") == {"logits": "logits", "new_kv": "new_kv"}

    model_graph = model.get("graph", [])
    assert isinstance(model_graph, list)
    top_nodes = _collect_node_specs(model_graph)
    assert not any("_param_root" in node and isinstance(node["_param_root"], dict) for node in top_nodes)
    assert any(
        node.get("_op") == "config_float"
        and node.get("_args") == "rope_parameters.full_attention.partial_rotary_factor"
        for node in top_nodes
    )

    blocks = model.get("blocks", {})
    assert "gemma4_e_block" in blocks
    block_graph = blocks["gemma4_e_block"].get("graph", [])
    assert isinstance(block_graph, list)
    block_nodes = _collect_node_specs(block_graph)
    all_block_nodes = _collect_node_specs(
        [
            {"block": block_spec}
            for block_spec in blocks.values()
            if isinstance(block_spec, dict) and isinstance(block_spec.get("graph"), list)
        ]
    )

    rope_nodes = [node for node in block_nodes if node.get("_op") == "rope_pair"]
    assert len(rope_nodes) == 1
    assert rope_nodes[0].get("rope_mode") is not None
    assert rope_nodes[0].get("partial_rotary_factor") is not None

    rms_nodes = [node for node in all_block_nodes if node.get("_op") == "rmsnorm"]
    assert any(node.get("with_scale") is False for node in rms_nodes)

    assert any(node.get("_op") == "activations_tanh" for node in top_nodes)


def test_gemma4_dense_axon_lowers_with_expected_structure(repo_root: Path) -> None:
    spec = _load_axon_spec(repo_root / "examples" / "gemma4_dense.axon")
    assert spec.get("synapse") == 1
    model = spec.get("model", {})
    assert model.get("outputs") == {"logits": "logits", "new_kv": "new_kv"}

    blocks = model.get("blocks", {})
    assert "gemma4_dense_block" in blocks
    block_graph = blocks["gemma4_dense_block"].get("graph", [])
    assert isinstance(block_graph, list)
    block_nodes = _collect_node_specs(block_graph)

    rope_nodes = [node for node in block_nodes if node.get("_op") == "rope_pair"]
    assert len(rope_nodes) == 1
    assert rope_nodes[0].get("rope_mode") is not None
    assert rope_nodes[0].get("partial_rotary_factor") is not None


def test_gemma4_moe_axon_lowers_with_expected_structure(repo_root: Path) -> None:
    spec = _load_axon_spec(repo_root / "examples" / "gemma4_moe.axon")
    assert spec.get("synapse") == 1
    model = spec.get("model", {})
    assert model.get("outputs") == {"logits": "logits", "new_kv": "new_kv"}

    blocks = model.get("blocks", {})
    assert "gemma4_moe_block" in blocks
    block_graph = blocks["gemma4_moe_block"].get("graph", [])
    assert isinstance(block_graph, list)
    block_nodes = _collect_node_specs(block_graph)
    all_block_nodes = _collect_node_specs(
        [
            {"block": block_spec}
            for block_spec in blocks.values()
            if isinstance(block_spec, dict) and isinstance(block_spec.get("graph"), list)
        ]
    )

    assert any(node.get("_op") == "gemma4_router" for node in block_nodes)
    assert any(node.get("_op") == "gemma4_moe_experts" for node in block_nodes)
    assert any(
        node.get("_op") == "rmsnorm" and node.get("with_scale") is False for node in all_block_nodes
    )
