from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from brainsurgery.synapse import lower_axon_program_to_synapse_spec, parse_axon_program_from_path
from tests.test_flags import LONG_TEST_ENV, run_long_tests_enabled

_RUN_LONG = run_long_tests_enabled()


def _load_axon_spec(path: Path) -> dict[str, Any]:
    modules = parse_axon_program_from_path(path)
    return lower_axon_program_to_synapse_spec(modules)


@pytest.mark.skipif(not _RUN_LONG, reason=f"set {LONG_TEST_ENV}=1 to enable long tests")
def test_phi3_mini_4k_instruct_axon_key_alignment(
    repo_root: Path, phi3_mini_4k_instruct_local_path: Path
) -> None:
    index_path = phi3_mini_4k_instruct_local_path / "model.safetensors.index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = payload.get("weight_map")
    assert isinstance(weight_map, dict)
    keys = set(weight_map.keys())

    assert "model.embed_tokens.weight" in keys
    assert "model.norm.weight" in keys
    assert "lm_head.weight" in keys

    assert "model.layers.0.self_attn.qkv_proj.weight" in keys
    assert "model.layers.0.self_attn.o_proj.weight" in keys
    assert "model.layers.0.self_attn.q_proj.weight" not in keys
    assert "model.layers.0.self_attn.k_proj.weight" not in keys
    assert "model.layers.0.self_attn.v_proj.weight" not in keys

    assert "model.layers.0.mlp.gate_up_proj.weight" in keys
    assert "model.layers.0.mlp.down_proj.weight" in keys
    assert "model.layers.0.mlp.gate_proj.weight" not in keys
    assert "model.layers.0.mlp.up_proj.weight" not in keys

    spec = _load_axon_spec(repo_root / "examples" / "phi3_mini_4k_instruct.axon")
    model = spec.get("model", {})
    blocks = model.get("blocks", {})
    assert "phi3_block" in blocks
