from __future__ import annotations

from pathlib import Path
from typing import Any

from brainsurgery.synapse import lower_axon_program_to_synapse_spec, parse_axon_program_from_path


def _load_axon_spec(path: Path) -> dict[str, Any]:
    modules = parse_axon_program_from_path(path)
    return lower_axon_program_to_synapse_spec(modules)


def test_glm4_5_air_axon_lowers_with_expected_symbols(repo_root: Path) -> None:
    spec = _load_axon_spec(repo_root / "brainsurgery" / "synapse" / "models" / "glm" / "glm_4_5_air.axon")

    assert spec.get("synapse") == 1
    model = spec.get("model", {})
    assert model.get("outputs") == {"logits": "logits", "new_kv": "new_kv"}

    symbols = model.get("symbols", {})
    assert symbols.get("D") == 4096
    assert symbols.get("V") == 151552
    assert symbols.get("L") == 46
    assert symbols.get("H") == 96
    assert symbols.get("KVH") == 8
    assert symbols.get("QD") == 12288
    assert symbols.get("KVD") == 1024
    assert symbols.get("FFN") == 10944
    assert symbols.get("E") == 128
    assert symbols.get("EPT") == 8
    assert symbols.get("EM") == 1408
    assert symbols.get("EPS") == 1.0e-05
    assert symbols.get("THETA") == 1000000.0
    assert symbols.get("C") == 131072

    blocks = model.get("blocks", {})
    assert "glm4_moe_attn" in blocks
    assert "glm4_moe_dense_block" in blocks
    assert "glm4_moe_sparse_block" in blocks
    assert "routed_expert_ffn" in blocks
    assert "dense_ffn" in blocks
