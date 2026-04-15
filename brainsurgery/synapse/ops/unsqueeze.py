from __future__ import annotations

from typing import Any

import torch

OP_NAME = "unsqueeze"
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
    del ctx
    if not isinstance(out, str):
        raise ValueError("unsqueeze requires a single scalar output binding")
    if kwargs:
        raise ValueError("unsqueeze does not accept kwargs")
    if len(args) != 2:
        raise ValueError("unsqueeze requires exactly two positional args: tensor, dim")


def lowering_infer_metadata(
    *,
    args: list[str],
    out: str | list[str],
    kwargs: dict[str, Any],
    ctx: Any,
) -> bool:
    if not isinstance(out, str) or not args:
        return False
    if len(args) < 2:
        return False
    dim = _coerce_int_literal(args[1])
    if dim is None:
        return False
    source_name = str(args[0]).strip()
    source_shape = ctx.tensor_shape.get(source_name)
    if isinstance(source_shape, tuple):
        rank = len(source_shape)
        target_dim = dim if dim >= 0 else rank + 1 + dim
        if target_dim < 0 or target_dim > rank:
            return False
        new_shape = source_shape[:target_dim] + (1,) + source_shape[target_dim:]
        ctx.tensor_shape[out] = new_shape
        ctx.tensor_last_dim[out] = new_shape[-1]
        return True
    first_dim = ctx.tensor_last_dim.get(source_name)
    if first_dim is not None:
        if dim in {-1, 1}:
            ctx.tensor_last_dim[out] = 1 if dim == -1 else first_dim
            return True
        ctx.tensor_last_dim[out] = first_dim
        return True
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
    del node_path, scope
    raw_args = node_spec.get("_args")
    if not isinstance(raw_args, list) or len(raw_args) != 2:
        raise ValueError("unsqueeze expects two positional args: tensor, dim")
    src = model._read_tensor_input(raw_args[0], env)
    raw_dim = model._eval_expr(raw_args[1], env, symbols)
    dim = _coerce_int_literal(raw_dim)
    if dim is None:
        raise ValueError("unsqueeze.dim must be int")
    out = model._require_name(node_spec.get("_bind"), field="unsqueeze._bind")
    env[out] = torch.unsqueeze(src, dim)


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
    raw_args = node_spec.get("_args")
    if not isinstance(raw_args, list) or len(raw_args) != 2:
        raise ValueError("unsqueeze expects two positional args: tensor, dim")
    src = emitter._read_env_var(env, str(raw_args[0]))
    out_var = emitter._assign_out_var(env, str(node_spec.get("_bind")))
    dim = _coerce_int_literal(raw_args[1])
    if dim is not None:
        return [f"{indent}{out_var} = torch.unsqueeze({src}, {dim})"]
    dim_name = str(raw_args[1])
    if dim_name not in env:
        raise ValueError("unsqueeze.dim must be int")
    dim_expr = emitter._read_env_var(env, dim_name)
    dim_var = emitter._fresh("dim")
    return [
        f"{indent}{dim_var} = {dim_expr}",
        f"{indent}if isinstance({dim_var}, bool) or not isinstance({dim_var}, int):",
        f"{indent}    raise ValueError('unsqueeze.dim must be int')",
        f"{indent}{out_var} = torch.unsqueeze({src}, int({dim_var}))",
    ]


def _coerce_int_literal(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str):
        token = value.strip()
        if not token:
            return None
        if token.startswith("-"):
            digits = token[1:]
            if digits.isdigit():
                return -int(digits)
            return None
        if token.isdigit():
            return int(token)
    return None


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any", "int"),
    "kwargs": {},
    "returns": ("Tensor",),
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
