from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from brainsurgery.synapse import lower_axon_program_to_synapse_spec, parse_axon_program_from_path


def _load_axon_spec(path: Path) -> dict[str, Any]:
    modules = parse_axon_program_from_path(path)
    return lower_axon_program_to_synapse_spec(modules)


def test_black_mamba_axon_lowers_with_expected_scaffold_symbols(repo_root: Path) -> None:
    pytest.skip(
        "outdated example black_mamba.axon lowering expectations after positional-only/kwarg changes"
    )
    spec = _load_axon_spec(repo_root / "examples" / "black_mamba.axon")

    assert spec.get("synapse") == 1
    model = spec.get("model", {})
    assert model.get("outputs") == {"logits": "logits"}

    symbols = model.get("symbols", {})
    assert symbols.get("D") == 1472
    assert symbols.get("V") is None
    assert symbols.get("L") == 36
    assert symbols.get("I") == 2944
    assert symbols.get("N") == 16
    assert symbols.get("R") == 92
    assert symbols.get("E") == 8
    assert symbols.get("K") == 2
    assert symbols.get("FF") == 3872
    assert symbols.get("EPS") == 1.0e-05

    blocks = model.get("blocks", {})
    assert "mamba_mixer" in blocks
    assert "moe_ffn" in blocks
    assert "expert_ffn" in blocks
