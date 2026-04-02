from __future__ import annotations

from typing import Any, Mapping

from ._config_common import _resolve_scalar_ref


def _uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return False


def _resolve_root_arg(
    *,
    op_name: str,
    node_spec: dict[str, Any],
    env: dict[str, Any],
    symbols: Mapping[str, int | float | bool],
) -> str:
    raw = node_spec.get("_args")
    value = _resolve_scalar_ref(raw, env, symbols)
    if not isinstance(value, str):
        raise ValueError(f"{op_name} root argument must resolve to string")
    return value


def _resolve_default(
    *,
    op_name: str,
    node_spec: dict[str, Any],
    env: dict[str, Any],
    symbols: Mapping[str, int | float | bool],
) -> tuple[bool, str]:
    if "default" not in node_spec:
        return False, ""
    raw_default = node_spec.get("default")
    value = _resolve_scalar_ref(raw_default, env, symbols)
    if not isinstance(value, str):
        raise ValueError(f"{op_name} default must resolve to string")
    return True, value


def _param_root_exists(state: Mapping[str, Any], root: str) -> bool:
    if root == "":
        return True
    prefix = f"{root}."
    for key in state.keys():
        if not isinstance(key, str):
            continue
        if key == root or key.startswith(prefix):
            return True
    return False


def _compile_value_expr(*, emitter: Any, value: Any, env: dict[str, str]) -> str:
    if isinstance(value, str):
        if value in env:
            return env[value]
        symbols = getattr(emitter, "symbols", {})
        if isinstance(symbols, dict) and value in symbols:
            return repr(symbols[value])
    return repr(value)


__all__ = [
    "_compile_value_expr",
    "_param_root_exists",
    "_resolve_default",
    "_resolve_root_arg",
    "_uses_node_path",
]
