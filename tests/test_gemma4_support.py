from __future__ import annotations

import json
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
    assert not any(
        "_param_root" in node and isinstance(node["_param_root"], dict) for node in top_nodes
    )
    all_nodes = _collect_node_specs(
        [{"model": model} for model in [model] if isinstance(model, dict)]
        + [
            {"block": block_spec}
            for block_spec in model.get("blocks", {}).values()
            if isinstance(block_spec, dict)
        ]
    )
    assert any(
        node.get("_op") == "config_float"
        and node.get("_args") == "rope_parameters.full_attention.partial_rotary_factor"
        for node in all_nodes
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


def test_gemma4_e_inline_config_lowers_equivalently_to_imported_form(
    repo_root: Path, tmp_path: Path
) -> None:
    inline_path = (
        repo_root / "brainsurgery" / "synapse" / "models" / "gemma" / "generic-gemma-4-e.axon"
    )
    inline_text = inline_path.read_text(encoding="utf-8")
    config_start = inline_text.index("CFG = ")
    model_start = inline_text.index("\nrms :: @Path")
    config_text = inline_text[config_start:model_start].strip()
    imported_cfg_path = tmp_path / "gemma4_config.axon"
    imported_cfg_path.write_text(config_text + "\n", encoding="utf-8")

    imported_text = inline_text[:config_start]
    imported_text += (
        "import gemma4_config (D, V, L, H, KVH, HD, GHD, FFN, PLI, EPS, WIN_LOCAL, WIN_FULL, "
        "THETA_LOCAL, THETA_FULL, ROPE_SCALE_FULL, ROTARY_PARTIAL_FULL, ROPE_PERIOD, "
        "LOGIT_SOFTCAP, PER_LAYER_INPUT_SCALE, PER_LAYER_PROJ_SCALE, NUM_KV_SHARED, "
        "USE_DOUBLE_WIDE_MLP)\n\n"
    )
    imported_text += inline_text[model_start + 1 :]
    imported_path = tmp_path / "gemma4_e_imported.axon"
    imported_path.write_text(imported_text, encoding="utf-8")

    inline_spec = _load_axon_spec(inline_path)
    imported_spec = _load_axon_spec(imported_path)

    inline_blocks = inline_spec["model"]["blocks"]
    imported_blocks = imported_spec["model"]["blocks"]
    assert json.dumps(
        inline_blocks["gemma4_apply_per_layer_input"]["graph"],
        sort_keys=True,
    ) == json.dumps(
        imported_blocks["gemma4_apply_per_layer_input"]["graph"],
        sort_keys=True,
    )

    imported_main = imported_spec["model"]["graph"]
    inline_main = inline_spec["model"]["graph"]
    imported_norm = next(
        node_spec
        for item in imported_main
        for node_spec in item.values()
        if isinstance(node_spec, dict) and node_spec.get("_op") == "rmsnorm"
    )
    inline_norm = next(
        node_spec
        for item in inline_main
        for node_spec in item.values()
        if isinstance(node_spec, dict) and node_spec.get("_op") == "rmsnorm"
    )
    assert imported_norm.get("_param_root") == "model"
    assert inline_norm.get("_param_root") == "model"
