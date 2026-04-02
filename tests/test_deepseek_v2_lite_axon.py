from __future__ import annotations

from pathlib import Path
from typing import Any

from brainsurgery.synapse import lower_axon_program_to_synapse_spec, parse_axon_program_from_path


def _load_axon_spec(path: Path) -> dict[str, Any]:
    modules = parse_axon_program_from_path(path)
    return lower_axon_program_to_synapse_spec(modules)


def test_deepseek_v2_lite_axon_lowers_with_expected_symbols(repo_root: Path) -> None:
    spec = _load_axon_spec(repo_root / "examples" / "deepseek_v2_lite.axon")

    assert spec.get("synapse") == 1
    model = spec.get("model", {})
    assert model.get("outputs") == {"logits": "logits", "new_kv": "new_kv"}

    symbols = model.get("symbols", {})
    assert symbols.get("D") == 2048
    assert symbols.get("V") == 102400
    assert symbols.get("L") == 27
    assert symbols.get("H") == 16
    assert symbols.get("QKNOPE") == 128
    assert symbols.get("QKROPE") == 64
    assert symbols.get("VHD") == 128
    assert symbols.get("QHD") in (192, "QKNOPE + QKROPE", "QKNOPE+QKROPE")
    assert symbols.get("KVR") == 512
    assert symbols.get("KVPROJ") in (4096, "H * (QKNOPE + VHD)", "H*(QKNOPE+VHD)")
    assert symbols.get("FFN") == 10944
    assert symbols.get("E") == 64
    assert symbols.get("EPT") == 6
    assert symbols.get("EM") == 1408
    assert symbols.get("SE") == 2
    assert symbols.get("EPS") == 1.0e-06
    assert symbols.get("THETA") == 10000.0
    assert symbols.get("C") == 163840
    assert symbols.get("FIRST_DENSE") == 1

    blocks = model.get("blocks", {})
    assert "deepseek_v2_lite_attn" in blocks
    assert "deepseek_v2_lite_dense_block" in blocks
    assert "deepseek_v2_lite_moe_block" in blocks
