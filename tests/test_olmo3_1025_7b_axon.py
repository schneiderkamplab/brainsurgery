from __future__ import annotations

from pathlib import Path
from typing import Any

from brainsurgery.synapse import lower_axon_program_to_synapse_spec, parse_axon_program_from_path


def _load_axon_spec(path: Path) -> dict[str, Any]:
    modules = parse_axon_program_from_path(path)
    return lower_axon_program_to_synapse_spec(modules)


def test_olmo3_1025_7b_axon_lowers_with_expected_symbols(repo_root: Path) -> None:
    spec = _load_axon_spec(repo_root / "examples" / "olmo3_1025_7b.axon")

    assert spec.get("synapse") == 1
    model = spec.get("model", {})
    assert model.get("outputs") == {"logits": "logits", "new_kv": "new_kv"}

    symbols = model.get("symbols", {})
    assert symbols.get("D") == 4096
    assert symbols.get("V") == 100278
    assert symbols.get("L") == 32
    assert symbols.get("H") == 32
    assert symbols.get("KVH") == 32
    assert symbols.get("FFN") == 11008
    assert symbols.get("EPS") == 1.0e-06
    assert symbols.get("THETA") == 500000.0
    assert symbols.get("C") == 65536
    assert symbols.get("SW") == 4096
    assert symbols.get("FULL_ATTN_PERIOD") == 4
    assert symbols.get("ROPE_SCALE") == 8.0
    assert symbols.get("ROPE_BETA_FAST") == 32.0
    assert symbols.get("ROPE_BETA_SLOW") == 1.0
    assert symbols.get("ROPE_CONTEXT") == 8192

    blocks = model.get("blocks", {})
    assert "olmo3_block" in blocks
