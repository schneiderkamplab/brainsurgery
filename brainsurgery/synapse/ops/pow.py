from __future__ import annotations

from typing import Any

import torch

from ._broadcast import broadcast_last_dim, broadcast_shape

OP_NAME = "pow"
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
        raise ValueError("pow requires a single scalar output binding")


def lowering_infer_metadata(
    *,
    args: list[str],
    out: str | list[str],
    kwargs: dict[str, Any],
    ctx: Any,
) -> bool:
    del kwargs
    if not isinstance(out, str):
        return False
    first_in = args[0].strip() if args else None
    second_in = args[1].strip() if len(args) > 1 else None
    first_dim = ctx.tensor_last_dim.get(first_in) if isinstance(first_in, str) else None
    second_dim = ctx.tensor_last_dim.get(second_in) if isinstance(second_in, str) else None
    first_shape = ctx.tensor_shape.get(first_in) if isinstance(first_in, str) else None
    second_shape = ctx.tensor_shape.get(second_in) if isinstance(second_in, str) else None
    broadcasted_shape = broadcast_shape(first_shape, second_shape)
    if first_shape is not None and second_shape is not None and broadcasted_shape is None:
        raise ValueError(
            f"pow requires broadcastable shapes; got {first_shape!r} and {second_shape!r}"
        )
    unified = broadcast_last_dim(first_dim, second_dim)
    if first_dim is not None and second_dim is not None and unified is None:
        raise ValueError(
            f"pow requires broadcastable last-dim; got {first_dim!r} and {second_dim!r}"
        )
    if unified is not None:
        ctx.tensor_last_dim[out] = unified
    if broadcasted_shape is not None:
        ctx.tensor_shape[out] = broadcasted_shape
    return True


def interpret(
    model: Any,
    node_spec: dict[str, Any],
    env: dict[str, Any],
    *,
    node_path: str,
    scope: str,
    symbols: dict[str, int],
) -> None:
    del node_path, scope
    inputs = node_spec.get("_args")
    if not isinstance(inputs, list) or len(inputs) != 2:
        raise ValueError("pow expects two inputs")
    out = model._require_name(node_spec.get("_bind"), field="pow._bind")
    base = model._eval_expr(inputs[0], env, symbols)
    exp = model._eval_expr(inputs[1], env, symbols)
    env[out] = torch.pow(base, exp)


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
    inputs = node_spec.get("_args")
    if not isinstance(inputs, list) or len(inputs) != 2:
        raise ValueError("pow expects two inputs")
    base = emitter._expr_code(inputs[0], env)
    exp = emitter._expr_code(inputs[1], env)
    out_var = emitter._assign_out_var(env, str(node_spec.get("_bind")))
    return [f"{indent}{out_var} = torch.pow({base}, {exp})"]


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any", "Any"),
    "kwargs": dict(LOWERING_KWARG_KINDS),
    "returns": "dynamic",
}

__all__ = [
    "LOWERING_ARITY",
    "LOWERING_ALLOWED_KWARGS",
    "LOWERING_REQUIRED_KWARGS",
    "LOWERING_KWARG_KINDS",
    "OP_NAME",
    "lowering_validate_signature",
    "lowering_infer_metadata",
    "interpret",
    "compile",
    "uses_node_path",
    "LOWERING_TYPE_SIGNATURE",
]
