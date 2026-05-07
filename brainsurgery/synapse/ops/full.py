from __future__ import annotations

from typing import Any

import torch

from ._tensor_create import (
    arg_or_default,
    raw_args,
    resolve_dtype,
    resolve_shape,
    shape_expr_code,
    type_from_shape_args,
    uses_node_path,
)

OP_NAME = "full"
LOWERING_ARITY = (3, 4)
LOWERING_ALLOWED_KWARGS: set[str] = set()
LOWERING_REQUIRED_KWARGS: set[str] = set()
LOWERING_KWARG_KINDS: dict[str, Any] = {}


def lowering_validate_signature(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> None:
    del ctx
    if not isinstance(out, str):
        raise ValueError("full requires a single scalar output binding")
    if kwargs:
        unknown = ", ".join(sorted(str(key) for key in kwargs))
        raise ValueError(f"full unsupported kwargs: {unknown}")
    if len(args) < 3 or len(args) > 4:
        raise ValueError(f"full expects 3..4 positional args, got {len(args)}")


def lowering_infer_metadata(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> bool:
    if not isinstance(out, str) or len(args) < 2 or kwargs:
        return False
    shape = args[1]
    if not isinstance(shape, list) or not shape:
        return False
    ctx.tensor_shape[out] = tuple(shape)
    ctx.tensor_last_dim[out] = shape[-1]
    return True


def interpret(
    model: Any,
    node_spec: dict[str, Any],
    env: dict[str, Any],
    *,
    node_path: str,
    scope: str,
    symbols: dict[str, int | float],
) -> None:
    del node_path, scope
    args = raw_args(node_spec)
    if len(args) < 3:
        raise ValueError("full requires positional args: ref shape value [dtype]")
    ref = model._read_tensor_input(args[0], env)
    shape = resolve_shape(model, args[1], env, symbols)
    value_raw = args[2]
    value = (
        env[value_raw]
        if isinstance(value_raw, str) and value_raw in env
        else model._eval_expr(value_raw, env, symbols)
    )
    dtype = resolve_dtype(model._eval_expr(arg_or_default(args, 3, None), env, symbols))
    out = model._require_name(node_spec.get("_bind"), field="full._bind")
    env[out] = torch.full(shape, value, device=ref.device, dtype=dtype or ref.dtype)


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
    args = raw_args(node_spec)
    if len(args) < 3:
        raise ValueError("full requires positional args: ref shape value [dtype]")
    ref = emitter._read_env_var(env, str(args[0]))
    value_ref = args[2]
    value_code = (
        emitter._read_env_var(env, str(value_ref))
        if isinstance(value_ref, str) and str(value_ref) in env
        else emitter._expr_code(value_ref, env)
    )
    out_var = emitter._assign_out_var(env, str(node_spec.get("_bind")))
    shape_var = emitter._fresh("shape")
    dtype_var = emitter._fresh("dtype")
    dtype_expr = emitter._expr_code(arg_or_default(args, 3, None), env)
    return [
        f"{indent}{shape_var} = tuple(int(v) for v in {shape_expr_code(emitter, args[1], env)})",
        f"{indent}if len({shape_var}) == 0:",
        f"{indent}    raise ValueError('full shape must be a non-empty list')",
        f"{indent}{dtype_var} = self._dtype_from_name({dtype_expr}) if {dtype_expr} is not None else None",
        f"{indent}{out_var} = torch.full({shape_var}, {value_code}, device={ref}.device, dtype=({dtype_var} or {ref}.dtype))",
    ]


LOWERING_TYPE_SIGNATURE = {
    "args": ("Tensor[..REF]", "List[Dim]", "Any", "Any"),
    "kwargs": {},
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
    del arg_types, kwarg_types, kwargs
    return type_from_shape_args(args, helpers)


__all__ = [
    "OP_NAME",
    "LOWERING_ARITY",
    "LOWERING_ALLOWED_KWARGS",
    "LOWERING_REQUIRED_KWARGS",
    "LOWERING_KWARG_KINDS",
    "lowering_validate_signature",
    "lowering_infer_metadata",
    "interpret",
    "compile",
    "uses_node_path",
    "LOWERING_TYPE_SIGNATURE",
    "type_rule",
]
