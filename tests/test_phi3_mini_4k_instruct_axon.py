from __future__ import annotations

from pathlib import Path
from typing import Any

from brainsurgery.synapse import lower_axon_program_to_synapse_spec, parse_axon_program_from_path


def _load_axon_spec(path: Path) -> dict[str, Any]:
    modules = parse_axon_program_from_path(path)
    return lower_axon_program_to_synapse_spec(modules)


def test_phi3_mini_4k_instruct_axon_lowers_with_expected_symbols(repo_root: Path) -> None:
    spec = _load_axon_spec(repo_root / "examples" / "phi3_mini_4k_instruct.axon")

    assert spec.get("synapse") == 1
    model = spec.get("model", {})
    assert model.get("outputs") == {"logits": "logits", "new_kv": "new_kv"}

    symbols = model.get("symbols", {})
    assert symbols.get("D") == 3072
    assert symbols.get("V") == 32064
    assert symbols.get("L") == 32
    assert symbols.get("H") == 32
    assert symbols.get("KVH") == 32
    assert symbols.get("QD") == 3072
    assert symbols.get("KVD") == 3072
    assert symbols.get("FFN") == 8192
    assert symbols.get("EPS") == 1.0e-05
    assert symbols.get("THETA") == 10000.0
    assert symbols.get("C") == 2047

    blocks = model.get("blocks", {})
    assert "phi3_block" in blocks
