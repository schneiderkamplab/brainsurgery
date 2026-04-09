from __future__ import annotations

from typing import Any

import torch

from ._broadcast import _normalize_dim_token

OP_NAME = "transpose"
LOWERING_ARITY = (1, 1)
LOWERING_ALLOWED_KWARGS: set[str] = {"dim1", "dim2"}
LOWERING_REQUIRED_KWARGS: set[str] = {"dim1", "dim2"}
LOWERING_KWARG_KINDS: dict[str, Any] = {"dim1": "int", "dim2": "int"}


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return False


def lowering_validate_signature(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> None:
    del args, ctx
    if not isinstance(out, str):
        raise ValueError("transpose requires a single scalar output binding")
    if "dim1" not in kwargs or "dim2" not in kwargs:
        raise ValueError("transpose requires dim1 and dim2")


def lowering_infer_metadata(
    *,
    args: list[str],
    out: str | list[str],
    kwargs: dict[str, Any],
    ctx: Any,
) -> bool:
    if not isinstance(out, str) or not args:
        return False
    source_name = str(args[0]).strip()
    source_shape = ctx.tensor_shape.get(source_name)
    raw_dim1 = kwargs.get("dim1")
    raw_dim2 = kwargs.get("dim2")
    if (
        not isinstance(source_shape, tuple)
        or not isinstance(raw_dim1, int)
        or not isinstance(raw_dim2, int)
    ):
        return False
    rank = len(source_shape)
    dim1 = raw_dim1 if raw_dim1 >= 0 else rank + raw_dim1
    dim2 = raw_dim2 if raw_dim2 >= 0 else rank + raw_dim2
    if not (0 <= dim1 < rank and 0 <= dim2 < rank):
        return False
    new_shape = list(source_shape)
    new_shape[dim1], new_shape[dim2] = new_shape[dim2], new_shape[dim1]
    out_shape = tuple(_normalize_dim_token(v) for v in new_shape)
    ctx.tensor_shape[out] = out_shape
    ctx.tensor_last_dim[out] = out_shape[-1]
    return True


def _resolve_int(
    model: Any, raw: Any, env: dict[str, Any], symbols: dict[str, int], name: str
) -> int:
    value = model._eval_expr(raw, env, symbols)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"transpose.{name} must resolve to int")
    return int(value)


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
    src = model._read_tensor_input(node_spec.get("_args"), env)
    dim1 = _resolve_int(model, node_spec.get("dim1"), env, symbols, "dim1")
    dim2 = _resolve_int(model, node_spec.get("dim2"), env, symbols, "dim2")
    out = model._require_name(node_spec.get("_bind"), field="transpose._bind")
    env[out] = torch.transpose(src, dim1, dim2)


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
    src = emitter._read_env_var(env, str(node_spec.get("_args")))
    out_var = emitter._assign_out_var(env, str(node_spec.get("_bind")))
    dim1_expr = emitter._expr_code(node_spec.get("dim1"), env)
    dim2_expr = emitter._expr_code(node_spec.get("dim2"), env)
    return [f"{indent}{out_var} = torch.transpose({src}, int({dim1_expr}), int({dim2_expr}))"]


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any",),
    "kwargs": dict(LOWERING_KWARG_KINDS),
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
