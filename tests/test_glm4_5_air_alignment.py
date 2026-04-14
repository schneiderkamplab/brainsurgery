from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from brainsurgery.synapse import lower_axon_program_to_synapse_spec, parse_axon_program_from_path


def _load_axon_spec(path: Path) -> dict[str, Any]:
    modules = parse_axon_program_from_path(path)
    return lower_axon_program_to_synapse_spec(modules)


def test_glm4_5_air_axon_key_alignment(repo_root: Path, glm4_5_air_local_path: Path) -> None:
    index_path = glm4_5_air_local_path / "model.safetensors.index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = payload.get("weight_map")
    assert isinstance(weight_map, dict)
    keys = set(weight_map.keys())

    # Core top-level paths.
    assert "model.embed_tokens.weight" in keys
    assert "model.norm.weight" in keys
    assert "lm_head.weight" in keys

    # Layer 0 is dense-only per first_k_dense_replace=1.
    assert "model.layers.0.mlp.gate_proj.weight" in keys
    assert "model.layers.0.mlp.up_proj.weight" in keys
    assert "model.layers.0.mlp.down_proj.weight" in keys
    assert "model.layers.0.mlp.gate.weight" not in keys
    assert "model.layers.0.mlp.shared_experts.gate_proj.weight" not in keys

    # Later layers are MoE with routed + shared experts.
    assert "model.layers.1.mlp.gate.weight" in keys
    assert "model.layers.1.mlp.gate.e_score_correction_bias" in keys
    assert "model.layers.1.mlp.shared_experts.gate_proj.weight" in keys
    assert "model.layers.1.mlp.experts.0.gate_proj.weight" in keys

    spec = _load_axon_spec(repo_root / "brainsurgery" / "synapse" / "models" / "glm" / "glm_4_5_air.axon")
    model = spec.get("model", {})
    blocks = model.get("blocks", {})
    assert "glm4_moe_dense_block" in blocks
    assert "glm4_moe_sparse_block" in blocks
