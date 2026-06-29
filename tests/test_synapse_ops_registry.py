from __future__ import annotations

from brainsurgery.synapse.ops import OP_MODULES, get_op_module


def test_synapse_ops_registry_discovery_has_expected_core_ops() -> None:
    # Smoke check for discovery + registration.
    assert "embedding" in OP_MODULES
    assert "linear" in OP_MODULES


def test_synapse_ops_registry_exports_required_interface() -> None:
    required = ("OP_NAME", "LOWERING_TYPE_SIGNATURE")
    for op_name, module in OP_MODULES.items():
        assert op_name == module.OP_NAME
        for attr in required:
            assert hasattr(module, attr)
        signature = module.LOWERING_TYPE_SIGNATURE
        assert isinstance(signature, dict)
        assert {"args", "kwargs", "returns"} <= set(signature)
        type_rule = getattr(module, "type_rule", None)
        assert type_rule is None or callable(type_rule)


def test_get_op_module_returns_none_for_unknown_op() -> None:
    assert get_op_module("__does_not_exist__") is None
