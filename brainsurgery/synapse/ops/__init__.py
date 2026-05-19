from __future__ import annotations

from collections import Counter
from importlib import import_module
from pkgutil import iter_modules
from types import ModuleType
from typing import Any

_REQUIRED_EXPORTS: tuple[str, ...] = (
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
        raise RuntimeError(f"Duplicate Axon primitive op module names discovered in {__name__}: {names}")
    return module_names


def _require_module_export(module: ModuleType, name: str) -> Any:
    if not hasattr(module, name):
        raise RuntimeError(
            f"Axon primitive op module {module.__name__!r} is missing required export {name!r}"
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
                f"Failed to import discovered Axon primitive op module: {qualified_name}"
            ) from exc

        for export_name in _REQUIRED_EXPORTS:
            _require_module_export(module, export_name)

        type_signature = _require_module_export(module, "LOWERING_TYPE_SIGNATURE")
        if not isinstance(type_signature, dict):
            raise RuntimeError(
                f"Axon primitive op module {qualified_name!r} export 'LOWERING_TYPE_SIGNATURE' must be a dict"
            )
        for key in ("args", "kwargs", "returns"):
            if key not in type_signature:
                raise RuntimeError(
                    f"Axon primitive op module {qualified_name!r} LOWERING_TYPE_SIGNATURE must contain key {key!r}"
                )

        op_name = _require_module_export(module, "OP_NAME")
        if not isinstance(op_name, str) or not op_name:
            raise RuntimeError(
                f"Axon primitive op module {qualified_name!r} has invalid OP_NAME: {op_name!r}"
            )

        existing = loaded_modules.get(op_name)
        if existing is not None:
            raise RuntimeError(
                f"Duplicate Axon primitive OP_NAME registered: {op_name!r} in "
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


def get_op_lowering_type_signature(op_name: str) -> dict[str, Any] | None:
    module = get_op_module(op_name)
    if module is None:
        return None
    signature = getattr(module, "LOWERING_TYPE_SIGNATURE", None)
    if isinstance(signature, dict):
        return signature
    return None


def get_op_parameter_names(op_name: str) -> tuple[str, ...] | None:
    module = get_op_module(op_name)
    if module is None:
        return None
    names_by_op = getattr(module, "LOWERING_PARAM_NAMES_BY_OP", None)
    if isinstance(names_by_op, dict):
        names = names_by_op.get(op_name)
        if names is not None:
            return tuple(names)
    names = getattr(module, "LOWERING_PARAM_NAMES", None)
    if names is not None:
        return tuple(names)
    signature = getattr(module, "LOWERING_TYPE_SIGNATURE", None)
    if not isinstance(signature, dict):
        return None
    positional = tuple(f"arg{idx}" for idx, _ in enumerate(signature.get("args", ())))
    named = tuple(signature.get("kwargs", {}).keys())
    return positional + named


def get_op_parameter_defaults(op_name: str) -> dict[str, Any]:
    module = get_op_module(op_name)
    if module is None:
        return {}
    defaults_by_op = getattr(module, "LOWERING_PARAM_DEFAULTS_BY_OP", None)
    if isinstance(defaults_by_op, dict) and op_name in defaults_by_op:
        defaults = defaults_by_op[op_name]
        if not isinstance(defaults, dict):
            raise RuntimeError(
                f"Axon primitive op module {module.__name__!r} export "
                "'LOWERING_PARAM_DEFAULTS_BY_OP' entries must be dicts"
            )
        return dict(defaults)
    defaults = getattr(module, "LOWERING_PARAM_DEFAULTS", {})
    if not isinstance(defaults, dict):
        raise RuntimeError(
            f"Axon primitive op module {module.__name__!r} export "
            "'LOWERING_PARAM_DEFAULTS' must be a dict"
        )
    return dict(defaults)


def get_op_type_rule(op_name: str) -> Any | None:
    module = get_op_module(op_name)
    if module is None:
        return None
    rule = getattr(module, "type_rule", None)
    if callable(rule):
        return rule
    return None


def get_op_semantics(op_name: str) -> dict[str, Any]:
    module = get_op_module(op_name)
    if module is None:
        return {}
    semantics = getattr(module, "PRIMITIVE_SEMANTICS", {})
    if not isinstance(semantics, dict):
        raise RuntimeError(
            f"Axon primitive op module {module.__name__!r} export "
            "'PRIMITIVE_SEMANTICS' must be a dict"
        )
    return dict(semantics)


__all__ = [
    "OP_MODULES",
    "get_op_module",
    "get_op_lowering_type_signature",
    "get_op_parameter_defaults",
    "get_op_parameter_names",
    "get_op_semantics",
    "get_op_type_rule",
]
