from __future__ import annotations

from typing import Any

OP_NAME = "_ir_const"


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return False


def interpret(
    model: Any,
    node_spec: dict[str, Any],
    env: dict[str, Any],
    *,
    node_path: str,
    scope: str,
    symbols: dict[str, int],
) -> None:
    raise NotImplementedError(f"TinyGrad interpret for '{OP_NAME}' not yet implemented")


def compile(
    emitter: Any,
    node_spec: dict[str, Any],
    env: dict[str, str],
    *,
    node_path_var: str,
    scope_var: str,
    indent: str,
) -> list[str]:
    del node_path_var, scope_var
    out_name = str(node_spec.get("_bind"))
    out_var = emitter._assign_out_var(env, out_name)
    value = node_spec.get("value")
    if isinstance(value, str):
        value_code = emitter._expr_code(value, env)
    else:
        value_code = repr(value)
    return [f"{indent}{out_var} = {value_code}"]


__all__ = [
    "OP_NAME",
    "interpret",
    "compile",
    "uses_node_path",
]
