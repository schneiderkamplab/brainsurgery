from __future__ import annotations

from typing import Any

OP_NAME = "list_index"
LOWERING_ARITY = (2, 2)
LOWERING_ALLOWED_KWARGS: set[str] = set()
LOWERING_REQUIRED_KWARGS: set[str] = set()
LOWERING_KWARG_KINDS: dict[str, Any] = {}


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return False


def lowering_validate_signature(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> None:
    del args, kwargs, ctx
    if not isinstance(out, str):
        raise ValueError("list_index requires a single scalar output binding")


def interpret(
    model: Any,
    node_spec: dict[str, Any],
    env: dict[str, Any],
    *,
    node_path: str,
    scope: str,
    symbols: dict[str, int],
) -> None:
    ins = node_spec.get("_args")
    if not isinstance(ins, list) or len(ins) != 2:
        raise ValueError("list_index expects [collection, index]")
    collection = env.get(ins[0])
    out_name = model._require_name(node_spec.get("_bind"), field="list_index._bind")
    if collection is None:
        env[out_name] = None
        return
    idx = int(model._eval_expr(ins[1], env, symbols))
    try:
        env[out_name] = collection[idx]
    except (IndexError, KeyError, TypeError):
        env[out_name] = None
    return


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


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any", "Any"),
    "kwargs": dict(LOWERING_KWARG_KINDS),
    "returns": ("Any",),
}

__all__ = [
    "LOWERING_ARITY",
    "LOWERING_ALLOWED_KWARGS",
    "LOWERING_REQUIRED_KWARGS",
    "LOWERING_KWARG_KINDS",
    "OP_NAME",
    "lowering_validate_signature",
    "interpret",
    "compile",
    "uses_node_path",
    "LOWERING_TYPE_SIGNATURE",
]
