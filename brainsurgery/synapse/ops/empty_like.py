from __future__ import annotations

from typing import Any

import torch

OP_NAME = "empty_like"
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


def _arg_or_default(args: list[Any], index: int, default: Any) -> Any:
    if index >= len(args):
        return default
    value = args[index]
    if isinstance(value, str) and value.strip().lower() in {"null", "none"}:
        return default
    return value


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return False


def lowering_validate_signature(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> None:
    del ctx
    if not isinstance(out, str):
        raise ValueError("empty_like requires a single scalar output binding")
    if kwargs:
        unknown = ", ".join(sorted(str(key) for key in kwargs))
        raise ValueError(f"empty_like unsupported kwargs: {unknown}")
    if len(args) < 1 or len(args) > 2:
        raise ValueError(f"empty_like expects 1..2 positional args, got {len(args)}")


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    token = dtype_name.strip().lower()
    if token in {"float32", "fp32"}:
        return torch.float32
    if token in {"float16", "fp16", "half"}:
        return torch.float16
    if token in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if token in {"int64", "long"}:
        return torch.int64
    if token in {"int32", "int"}:
        return torch.int32
    if token in {"bool", "boolean"}:
        return torch.bool
    raise ValueError(f"Unsupported empty_like dtype: {dtype_name}")


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
    args = _raw_args(node_spec)
    if len(args) < 1:
        raise ValueError("empty_like requires positional args: x [dtype]")
    x = model._read_tensor_input(args[0], env)
    dtype_raw = model._eval_expr(_arg_or_default(args, 1, None), env, symbols)
    out = model._require_name(node_spec.get("_bind"), field="empty_like._bind")
    if dtype_raw is None:
        env[out] = torch.empty_like(x)
        return
    if not isinstance(dtype_raw, str):
        raise ValueError("empty_like dtype must resolve to string when provided")
    env[out] = torch.empty_like(x, dtype=_resolve_dtype(dtype_raw))


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
    if len(args) < 1:
        raise ValueError("empty_like requires positional args: x [dtype]")
    src = emitter._read_env_var(env, str(args[0]))
    out_var = emitter._assign_out_var(env, str(node_spec.get("_bind")))
    dtype_expr = emitter._expr_code(_arg_or_default(args, 1, None), env)
    dtype_var = emitter._fresh("dtype_name")
    resolved_var = emitter._fresh("dtype_obj")
    lines = [
        f"{indent}{dtype_var} = {dtype_expr}",
        f"{indent}if {dtype_var} is None:",
        f"{indent}    {out_var} = torch.empty_like({src})",
        f"{indent}else:",
        f"{indent}    if not isinstance({dtype_var}, str):",
        f"{indent}        raise ValueError('empty_like dtype must resolve to string when provided')",
        f"{indent}    {dtype_var} = {dtype_var}.strip().lower()",
        f"{indent}    if {dtype_var} in ('float32', 'fp32'):",
        f"{indent}        {resolved_var} = torch.float32",
        f"{indent}    elif {dtype_var} in ('float16', 'fp16', 'half'):",
        f"{indent}        {resolved_var} = torch.float16",
        f"{indent}    elif {dtype_var} in ('bfloat16', 'bf16'):",
        f"{indent}        {resolved_var} = torch.bfloat16",
        f"{indent}    elif {dtype_var} in ('int64', 'long'):",
        f"{indent}        {resolved_var} = torch.int64",
        f"{indent}    elif {dtype_var} in ('int32', 'int'):",
        f"{indent}        {resolved_var} = torch.int32",
        f"{indent}    elif {dtype_var} in ('bool', 'boolean'):",
        f"{indent}        {resolved_var} = torch.bool",
        f"{indent}    else:",
        f"{indent}        raise ValueError(f'Unsupported empty_like dtype: {{{dtype_var}}}')",
        f"{indent}    {out_var} = torch.empty_like({src}, dtype={resolved_var})",
    ]
    return lines


LOWERING_TYPE_SIGNATURE = {
    "args": ("Tensor[..S]", "Any"),
    "kwargs": {},
    "returns": ("Tensor[..S]",),
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
    if len(arg_types) < 1:
        return None
    input_dims = helpers.type_dims(arg_types[0])
    if input_dims is None:
        return None
    return helpers.type_tensor(dims=input_dims)


__all__ = [
    "OP_NAME",
    "LOWERING_ARITY",
    "LOWERING_ALLOWED_KWARGS",
    "LOWERING_REQUIRED_KWARGS",
    "LOWERING_KWARG_KINDS",
    "lowering_validate_signature",
    "interpret",
    "compile",
    "uses_node_path",
    "LOWERING_TYPE_SIGNATURE",
    "type_rule",
]
