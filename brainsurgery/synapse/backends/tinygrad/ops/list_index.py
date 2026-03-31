from __future__ import annotations

from typing import Any

OP_NAME = "list_index"


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
    lines: list[str] = []

    def assign_out_var(out_name: str) -> str:
        return emitter._assign_out_var(env, out_name)

    def read(name: str) -> str:
        return emitter._read_env_var(env, name)

    ins = node_spec.get("_args")
    if not isinstance(ins, list) or len(ins) != 2:
        raise ValueError("list_index expects [collection,index]")
    coll = read(str(ins[0]))
    idx_expr = emitter._expr_code(ins[1], env)
    out_name = str(node_spec.get("_bind"))
    out_var = assign_out_var(out_name)
    lines.append(f"{indent}if {coll} is None:")
    lines.append(f"{indent}    {out_var} = None")
    lines.append(f"{indent}else:")
    lines.append(f"{indent}    try:")
    lines.append(f"{indent}        {out_var} = {coll}[int({idx_expr})]")
    lines.append(f"{indent}    except (IndexError, KeyError, TypeError):")
    lines.append(f"{indent}        {out_var} = None")
    return lines


__all__ = [
    "OP_NAME",
    "interpret",
    "compile",
    "uses_node_path",
]
