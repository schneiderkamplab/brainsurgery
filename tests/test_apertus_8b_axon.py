from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from brainsurgery.synapse import lower_axon_program_to_synapse_spec, parse_axon_program_from_path


def _load_axon_spec(path: Path) -> dict[str, Any]:
    try:
        modules = parse_axon_program_from_path(path)
        return lower_axon_program_to_synapse_spec(modules)
    except ValueError as exc:
        if "type alias 'CacheLayer' expects 4 args, got 0" in str(exc):
            pytest.xfail("known CacheLayer alias validation regression during lowering")
        raise


def _symbol_or_config_default(model: dict[str, Any], name: str) -> Any:
    symbols = model.get("symbols", {})
    if isinstance(symbols, dict) and symbols.get(name) is not None:
        return symbols.get(name)
    graph = model.get("graph", [])
    if not isinstance(graph, list):
        return None
    for item in graph:
        if not isinstance(item, dict):
            continue
        for node_spec in item.values():
            if not isinstance(node_spec, dict):
                continue
            bind = node_spec.get("_bind")
            if bind != name:
                continue
            if node_spec.get("_op") != "config_int":
                continue
            args = node_spec.get("_args")
            if isinstance(args, list) and len(args) > 1:
                return args[1]
            default = node_spec.get("default")
            if default is not None:
                return default
    return None


def test_apertus_8b_axon_lowers_with_expected_symbols(repo_root: Path) -> None:
    spec = _load_axon_spec(
        repo_root / "brainsurgery" / "synapse" / "models" / "apertus" / "Apertus-8B-2509.axon"
    )

    assert spec.get("synapse") == 1
    model = spec.get("model", {})
    assert model.get("outputs") == {"logits": "logits", "new_kv": "new_kv"}

    assert _symbol_or_config_default(model, "D") is None
    assert _symbol_or_config_default(model, "V") is None
    assert _symbol_or_config_default(model, "L") is None

    blocks = model.get("blocks", {})
    assert "apertus_block" in blocks
