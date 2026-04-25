from __future__ import annotations

from typing import Any

from ..axon.ast import TypeOptional

OP_NAME = "require"
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
        raise ValueError("require requires a single scalar output binding")


def interpret(
    model: Any,
    node_spec: dict[str, Any],
    env: dict[str, Any],
    *,
    node_path: str,
    scope: str,
    symbols: dict[str, int | float | bool],
) -> None:
    del node_path, scope
    ins = node_spec.get("_args")
    if isinstance(ins, list):
        if len(ins) != 1:
            raise ValueError("require expects one input")
        value = model._eval_expr(ins[0], env, symbols)
    else:
        value = model._eval_expr(ins, env, symbols)
    if value is None:
        raise ValueError("require expected non-null value")
    out_name = model._require_name(node_spec.get("_bind"), field="require._bind")
    env[out_name] = value


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
    ins = node_spec.get("_args")
    if isinstance(ins, list):
        if len(ins) != 1:
            raise ValueError("require expects one input")
        raw_arg = ins[0]
    else:
        raw_arg = ins
    value_expr = emitter._expr_code(raw_arg, env)
    out_name = str(node_spec.get("_bind"))
    out_var = emitter._assign_out_var(env, out_name)
    return [
        f"{indent}if {value_expr} is None:",
        f"{indent}    raise ValueError('require expected non-null value')",
        f"{indent}{out_var} = {value_expr}",
    ]


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any",),
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
    if not arg_types:
        return None
    arg_tp = arg_types[0]
    if isinstance(arg_tp, TypeOptional):
        return arg_tp.inner
    return arg_tp


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
