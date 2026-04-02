from __future__ import annotations

from collections import Counter
from importlib import import_module
from pkgutil import iter_modules
from types import ModuleType
from typing import Any

_REQUIRED_EXPORTS: tuple[str, ...] = (
    "OP_NAME",
    "interpret",
    "compile",
    "uses_node_path",
    "LOWERING_TYPE_SIGNATURE",
)


def _discovered_module_names() -> list[str]:
    package_path = globals().get("__path__", [])
    module_names = sorted(
        module_info.name
        for module_info in iter_modules(package_path)
        if not module_info.name.startswith("_")
    )
    duplicates = sorted(name for name, count in Counter(module_names).items() if count > 1)
    if duplicates:
        names = ", ".join(duplicates)
        raise RuntimeError(f"Duplicate synapse op module names discovered in {__name__}: {names}")
    return module_names


def _require_module_export(module: ModuleType, name: str) -> Any:
    if not hasattr(module, name):
        raise RuntimeError(
            f"Synapse op module {module.__name__!r} is missing required export {name!r}"
        )
    return getattr(module, name)


def _load_discovered_op_modules() -> dict[str, Any]:
    loaded_modules: dict[str, Any] = {}
    for module_name in _discovered_module_names():
        qualified_name = f"{__name__}.{module_name}"
        try:
            module = import_module(qualified_name)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to import discovered synapse op module: {qualified_name}"
            ) from exc

        for export_name in _REQUIRED_EXPORTS:
            _require_module_export(module, export_name)

        type_signature = _require_module_export(module, "LOWERING_TYPE_SIGNATURE")
        if not isinstance(type_signature, dict):
            raise RuntimeError(
                f"Synapse op module {qualified_name!r} export 'LOWERING_TYPE_SIGNATURE' must be a dict"
            )
        for key in ("args", "kwargs", "returns"):
            if key not in type_signature:
                raise RuntimeError(
                    f"Synapse op module {qualified_name!r} LOWERING_TYPE_SIGNATURE must contain key {key!r}"
                )

        op_name = _require_module_export(module, "OP_NAME")
        if not isinstance(op_name, str) or not op_name:
            raise RuntimeError(
                f"Synapse op module {qualified_name!r} has invalid OP_NAME: {op_name!r}"
            )

        for callable_name in ("interpret", "compile", "uses_node_path"):
            exported = _require_module_export(module, callable_name)
            if not callable(exported):
                raise RuntimeError(
                    f"Synapse op module {qualified_name!r} export {callable_name!r} must be callable"
                )

        existing = loaded_modules.get(op_name)
        if existing is not None:
            raise RuntimeError(
                f"Duplicate synapse OP_NAME registered: {op_name!r} in "
                f"{existing.__name__!r} and {qualified_name!r}"
            )
        loaded_modules[op_name] = module

    return loaded_modules


OP_MODULES: dict[str, Any] = _load_discovered_op_modules()


def get_op_module(op_name: str) -> Any | None:
    if op_name == "_ir_const":
        return OP_MODULES.get("_ir_expr")
    module = OP_MODULES.get(op_name)
    if module is not None:
        return module
    if op_name.startswith("activations_"):
        return OP_MODULES.get("activation")
    return None


def get_op_lowering_signature(op_name: str) -> dict[str, Any] | None:
    module = get_op_module(op_name)
    if module is None:
        return None
    signature: dict[str, Any] = {}
    if hasattr(module, "LOWERING_ARITY"):
        signature["arity"] = getattr(module, "LOWERING_ARITY")
    if hasattr(module, "LOWERING_ALLOWED_KWARGS"):
        signature["allowed_kwargs"] = getattr(module, "LOWERING_ALLOWED_KWARGS")
    if hasattr(module, "LOWERING_REQUIRED_KWARGS"):
        signature["required_kwargs"] = getattr(module, "LOWERING_REQUIRED_KWARGS")
    if hasattr(module, "LOWERING_KWARG_KINDS"):
        signature["kwarg_kinds"] = getattr(module, "LOWERING_KWARG_KINDS")
    return signature or None


def get_op_lowering_type_signature(op_name: str) -> dict[str, Any] | None:
    module = get_op_module(op_name)
    if module is None:
        return None
    signature = getattr(module, "LOWERING_TYPE_SIGNATURE", None)
    if isinstance(signature, dict):
        return signature
    return None


def get_op_lowering_normalizer(op_name: str) -> Any | None:
    module = get_op_module(op_name)
    if module is None:
        return None
    normalize = getattr(module, "lowering_normalize_kwargs", None)
    if callable(normalize):
        return normalize
    return None


def get_op_lowering_infer_metadata(op_name: str) -> Any | None:
    module = get_op_module(op_name)
    if module is None:
        return None
    infer = getattr(module, "lowering_infer_metadata", None)
    if callable(infer):
        return infer
    return None


def get_op_lowering_known_output_arity(op_name: str) -> Any | None:
    module = get_op_module(op_name)
    if module is None:
        return None
    arity = getattr(module, "lowering_known_output_arity", None)
    if callable(arity):
        return arity
    return None


def get_op_lowering_validator(op_name: str) -> Any | None:
    module = get_op_module(op_name)
    if module is None:
        return None
    validate = getattr(module, "lowering_validate_signature", None)
    if callable(validate):
        return validate
    return None


__all__ = [
    "OP_MODULES",
    "get_op_module",
    "get_op_lowering_signature",
    "get_op_lowering_type_signature",
    "get_op_lowering_normalizer",
    "get_op_lowering_infer_metadata",
    "get_op_lowering_known_output_arity",
    "get_op_lowering_validator",
]
