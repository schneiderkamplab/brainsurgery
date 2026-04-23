from __future__ import annotations

from typing import Any

OP_NAME = "list_length"
LOWERING_ARITY = (1, 1)
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
        raise ValueError("list_length requires a single scalar output binding")


def interpret(
    model: Any,
    node_spec: dict[str, Any],
    env: dict[str, Any],
    *,
    node_path: str,
    scope: str,
    symbols: dict[str, int],
) -> None:
    del node_path, scope, symbols
    ins = node_spec.get("_args")
    if not isinstance(ins, list) or len(ins) != 1:
        raise ValueError("list_length expects [collection]")
    collection_name = str(ins[0])
    if collection_name not in env:
        raise ValueError(f"list_length missing input {collection_name!r}")
    collection = env[collection_name]
    if collection is None:
        raise ValueError("list_length expects non-null list input")
    out_name = model._require_name(node_spec.get("_bind"), field="list_length._bind")
    env[out_name] = len(collection)


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
    lines: list[str] = []

    def assign_out_var(out_name: str) -> str:
        return emitter._assign_out_var(env, out_name)

    def read(name: str) -> str:
        return emitter._read_env_var(env, name)

    ins = node_spec.get("_args")
    if not isinstance(ins, list) or len(ins) != 1:
        raise ValueError("list_length expects [collection]")
    coll = read(str(ins[0]))
    out_name = str(node_spec.get("_bind"))
    out_var = assign_out_var(out_name)
    lines.append(f"{indent}if {coll} is None:")
    lines.append(f"{indent}    raise ValueError('list_length expects non-null list input')")
    lines.append(f"{indent}{out_var} = len({coll})")
    return lines


LOWERING_TYPE_SIGNATURE = {
    "args": ("List[_T]",),
    "kwargs": dict(LOWERING_KWARG_KINDS),
    "returns": ("Dim",),
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
