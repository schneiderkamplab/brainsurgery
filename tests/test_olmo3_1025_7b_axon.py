from __future__ import annotations

from pathlib import Path
from typing import Any

from brainsurgery.synapse import lower_axon_program_to_synapse_spec, parse_axon_program_from_path


def _load_axon_spec(path: Path) -> dict[str, Any]:
    modules = parse_axon_program_from_path(path)
    return lower_axon_program_to_synapse_spec(modules)


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


def test_olmo3_1025_7b_axon_lowers_with_expected_symbols(repo_root: Path) -> None:
    spec = _load_axon_spec(
        repo_root / "brainsurgery" / "synapse" / "models" / "olmo3" / "Olmo-3-1025-7B.axon"
    )

    assert spec.get("synapse") == 1
    model = spec.get("model", {})
    assert model.get("outputs") == {"logits": "logits", "new_kv": "new_kv"}

    assert _symbol_or_config_default(model, "D") is None
    assert _symbol_or_config_default(model, "V") is None
    assert _symbol_or_config_default(model, "L") is None

    blocks = model.get("blocks", {})
    assert "olmo3_block" in blocks
