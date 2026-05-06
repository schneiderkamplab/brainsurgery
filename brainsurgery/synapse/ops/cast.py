from __future__ import annotations

from typing import Any

import torch

OP_NAME = "cast"
LOWERING_ARITY = (1, 2)
LOWERING_ALLOWED_KWARGS: set[str] = set()
LOWERING_REQUIRED_KWARGS: set[str] = set()
LOWERING_KWARG_KINDS: dict[str, Any] = {}


def _raw_args(node_spec: dict[str, Any]) -> list[Any]:
    raw = node_spec.get("_args")
    if isinstance(raw, list):
        return list(raw)
    if raw is None:
        return []
    return [raw]


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return False


def lowering_validate_signature(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> None:
    del ctx
    if not isinstance(out, str):
        raise ValueError("cast requires a single scalar output binding")
    if kwargs:
        unknown = ", ".join(sorted(str(key) for key in kwargs))
        raise ValueError(f"cast unsupported kwargs: {unknown}")
    if len(args) != 2:
        raise ValueError("cast requires positional args: x dtype")


def lowering_infer_metadata(
    *,
    args: list[str],
    out: str | list[str],
    kwargs: dict[str, Any],
    ctx: Any,
) -> bool:
    del kwargs
    if not isinstance(out, str) or not args:
        return False
    source_name = str(args[0]).strip()
    source_shape = ctx.tensor_shape.get(source_name)
    if isinstance(source_shape, tuple):
        ctx.tensor_shape[out] = source_shape
        if source_shape:
            ctx.tensor_last_dim[out] = source_shape[-1]
        return True
    if source_name in ctx.tensor_last_dim:
        ctx.tensor_last_dim[out] = ctx.tensor_last_dim[source_name]
        return True
    return False


def _resolve_dtype(raw: Any) -> torch.dtype:
    if not isinstance(raw, str):
        raise ValueError("cast.dtype must be a string")
    value = raw.strip().lower()
    if value in {"long", "int64"}:
        return torch.long
    if value in {"bool"}:
        return torch.bool
    if value in {"float", "float32", "fp32"}:
        return torch.float32
    raise ValueError("cast.dtype must be one of: long, int64, bool, float32")


def _resolve_dtype_code(raw_expr: str) -> str:
    return (
        "(\n"
        f"            torch.long if str({raw_expr}).strip().lower() in ('long', 'int64') else\n"
        f"            torch.bool if str({raw_expr}).strip().lower() in ('bool',) else\n"
        f"            torch.float32 if str({raw_expr}).strip().lower() in ('float', 'float32', 'fp32') else\n"
        "            None\n"
        "        )"
    )


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
    args = _raw_args(node_spec)
    if len(args) != 2:
        raise ValueError("cast requires positional args: x dtype")
    src = model._read_tensor_input(args[0], env)
    dtype_raw = model._eval_expr(args[1], env, symbols)
    dtype = _resolve_dtype(dtype_raw)
    out = model._require_name(node_spec.get("_bind"), field="cast._bind")
    env[out] = src.to(dtype=dtype)


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
    args = _raw_args(node_spec)
    if len(args) != 2:
        raise ValueError("cast requires positional args: x dtype")
    src = emitter._read_env_var(env, str(args[0]))
    out_var = emitter._assign_out_var(env, str(node_spec.get("_bind")))
    dtype_expr = emitter._expr_code(args[1], env)
    dtype_var = emitter._fresh("dtype")
    return [
        f"{indent}{dtype_var}_raw = {dtype_expr}",
        f"{indent}{dtype_var} = {_resolve_dtype_code(f'{dtype_var}_raw')}",
        f"{indent}if {dtype_var} is None:",
        f"{indent}    raise ValueError('cast.dtype must be one of: long, int64, bool, float32')",
        f"{indent}{out_var} = {src}.to(dtype={dtype_var})",
    ]


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
    if not arg_types:
        return None
    input_dims = helpers.type_dims(arg_types[0])
    if input_dims is None:
        return None
    return helpers.type_tensor(dims=input_dims)

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
