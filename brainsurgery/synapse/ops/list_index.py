from __future__ import annotations

from typing import Any

from ..axon.ast import TypeList, TypeOptional

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
    collection_name = str(ins[0])
    if collection_name not in env:
        raise ValueError(f"list_index missing input {collection_name!r}")
    collection = env[collection_name]
    out_name = model._require_name(node_spec.get("_bind"), field="list_index._bind")
    if collection is None:
        raise ValueError("list_index expects non-null list input")
    idx = int(model._eval_expr(ins[1], env, symbols))
    try:
        env[out_name] = collection[idx]
    except (IndexError, KeyError, TypeError) as exc:
        raise ValueError(f"list_index invalid access at index {idx}") from exc
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
    lines.append(f"{indent}    raise ValueError('list_index expects non-null list input')")
    lines.append(f"{indent}{out_var} = {coll}[int({idx_expr})]")
    return lines


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any", "Int"),
    "kwargs": dict(LOWERING_KWARG_KINDS),
    "returns": ("Any",),
}


def type_rule(
    *,
    arg_types: tuple[Any, ...],
    kwarg_types: dict[str, Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    helpers: Any,
) -> Any | None:
    del kwarg_types, args, kwargs, helpers
    if len(arg_types) < 1:
        return None
    collection_tp = arg_types[0]
    if isinstance(collection_tp, TypeOptional):
        collection_tp = collection_tp.inner
    if isinstance(collection_tp, TypeList):
        return collection_tp.item
    return None


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
    "type_rule",
]
