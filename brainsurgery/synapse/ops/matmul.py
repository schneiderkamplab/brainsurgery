from __future__ import annotations

from typing import Any

import torch

from ._broadcast import broadcast_shape

OP_NAME = "matmul"
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
        raise ValueError("matmul requires a single scalar output binding")


def lowering_infer_metadata(
    *,
    args: list[str],
    out: str | list[str],
    kwargs: dict[str, Any],
    ctx: Any,
) -> bool:
    del kwargs
    if not isinstance(out, str) or len(args) != 2:
        return False
    left_name = args[0].strip() if isinstance(args[0], str) else None
    right_name = args[1].strip() if isinstance(args[1], str) else None
    left_shape = ctx.tensor_shape.get(left_name) if isinstance(left_name, str) else None
    right_shape = ctx.tensor_shape.get(right_name) if isinstance(right_name, str) else None
    if not (isinstance(left_shape, tuple) and isinstance(right_shape, tuple)):
        return False
    if len(left_shape) < 2 or len(right_shape) < 2:
        return False
    batch = broadcast_shape(left_shape[:-2], right_shape[:-2])
    if batch is None:
        raise ValueError(
            f"matmul requires broadcastable batch dims; got {left_shape!r} and {right_shape!r}"
        )
    if left_shape[-1] != right_shape[-2]:
        return False
    out_shape = batch + (left_shape[-2], right_shape[-1])
    ctx.tensor_shape[out] = out_shape
    ctx.tensor_last_dim[out] = right_shape[-1]
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
    del node_path, scope, symbols
    inputs = node_spec.get("_args")
    if not isinstance(inputs, list) or len(inputs) != 2:
        raise ValueError("matmul expects two inputs")
    out = model._require_name(node_spec.get("_bind"), field="matmul._bind")
    left = model._read_tensor_input(inputs[0], env)
    right = model._read_tensor_input(inputs[1], env)
    env[out] = torch.matmul(left, right)


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
        raise ValueError("matmul expects two inputs")
    left = emitter._read_env_var(env, str(inputs[0]))
    right = emitter._read_env_var(env, str(inputs[1]))
    out_var = emitter._assign_out_var(env, str(node_spec.get("_bind")))
    return [f"{indent}{out_var} = torch.matmul({left}, {right})"]


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any", "Any"),
    "kwargs": dict(LOWERING_KWARG_KINDS),
    "returns": "dynamic",
}


def type_rule(
    *,
    arg_types: tuple[Any, ...],
    kwarg_types: dict[str, Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    helpers: Any,
) -> Any | None:
    del kwarg_types, args, kwargs
    if len(arg_types) != 2:
        return None
    left_dims = helpers.type_dims(arg_types[0])
    right_dims = helpers.type_dims(arg_types[1])
    if left_dims is None or right_dims is None:
        return None
    if len(left_dims) < 2 or len(right_dims) < 2:
        return None
    batch = broadcast_shape(left_dims[:-2], right_dims[:-2])
    if batch is None:
        return None
    if left_dims[-1] != right_dims[-2]:
        return None
    return helpers.type_tensor(dims=(*batch, left_dims[-2], right_dims[-1]))


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
    "type_rule",
]
