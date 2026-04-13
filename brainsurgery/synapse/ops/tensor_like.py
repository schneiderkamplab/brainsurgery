from __future__ import annotations

from typing import Any

import torch

OP_NAME = "tensor_like"
LOWERING_ARITY = (2, 3)
LOWERING_ALLOWED_KWARGS: set[str] = set()
LOWERING_REQUIRED_KWARGS: set[str] = set()
LOWERING_KWARG_KINDS: dict[str, Any] = {}


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return False


def _raw_args(node_spec: dict[str, Any]) -> list[Any]:
    raw = node_spec.get("_args")
    if isinstance(raw, list):
        return list(raw)
    if raw is None:
        return []
    return [raw]


def _resolve_dtype(dtype_raw: Any) -> torch.dtype | None:
    if dtype_raw is None:
        return None
    text = str(dtype_raw).strip().lower()
    if text in {"", "null", "none", "default"}:
        return None
    if text in {"float32", "fp32", "single"}:
        return torch.float32
    if text in {"float16", "fp16", "half"}:
        return torch.float16
    if text in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if text in {"int64", "long"}:
        return torch.int64
    if text in {"int32", "int"}:
        return torch.int32
    if text in {"bool"}:
        return torch.bool
    raise ValueError(
        "_tensor_like dtype must be one of: float32, float16, bfloat16, int64, int32, bool, or null"
    )


def lowering_validate_signature(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> None:
    del kwargs, ctx
    if not isinstance(out, str):
        raise ValueError("_tensor_like requires a single output binding")
    if len(args) < 2 or len(args) > 3:
        raise ValueError("_tensor_like requires positional args: value ref [dtype]")


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
    if len(args) < 2 or len(args) > 3:
        raise ValueError("_tensor_like requires positional args: value ref [dtype]")
    out = model._require_name(node_spec.get("_bind"), field="_tensor_like._bind")
    value = model._eval_expr(args[0], env, symbols)
    ref = model._read_tensor_input(args[1], env)
    dtype = _resolve_dtype(model._eval_expr(args[2], env, symbols) if len(args) >= 3 else None)
    target_dtype = ref.dtype if dtype is None else dtype
    if torch.is_tensor(value):
        env[out] = value.to(device=ref.device, dtype=target_dtype)
        return
    env[out] = torch.tensor(value, device=ref.device, dtype=target_dtype)


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
    if len(args) < 2 or len(args) > 3:
        raise ValueError("_tensor_like requires positional args: value ref [dtype]")
    out_var = emitter._assign_out_var(env, str(node_spec.get("_bind")))
    value_expr = emitter._expr_code(args[0], env)
    ref = emitter._read_env_var(env, str(args[1]))
    dtype_expr = emitter._expr_code(args[2], env) if len(args) >= 3 else "None"
    dtype_var = emitter._fresh("tensor_like_dtype")
    dtype_torch = emitter._fresh("tensor_like_dtype_torch")
    value_var = emitter._fresh("tensor_like_value")
    return [
        f"{indent}{dtype_var} = {dtype_expr}",
        f"{indent}{dtype_torch} = None",
        f"{indent}if {dtype_var} is not None:",
        f"{indent}    _dt = str({dtype_var}).strip().lower()",
        f"{indent}    if _dt in ('', 'none', 'null', 'default'):",
        f"{indent}        {dtype_torch} = None",
        f"{indent}    elif _dt in ('float32', 'fp32', 'single'):",
        f"{indent}        {dtype_torch} = torch.float32",
        f"{indent}    elif _dt in ('float16', 'fp16', 'half'):",
        f"{indent}        {dtype_torch} = torch.float16",
        f"{indent}    elif _dt in ('bfloat16', 'bf16'):",
        f"{indent}        {dtype_torch} = torch.bfloat16",
        f"{indent}    elif _dt in ('int64', 'long'):",
        f"{indent}        {dtype_torch} = torch.int64",
        f"{indent}    elif _dt in ('int32', 'int'):",
        f"{indent}        {dtype_torch} = torch.int32",
        f"{indent}    elif _dt in ('bool',):",
        f"{indent}        {dtype_torch} = torch.bool",
        f"{indent}    else:",
        f'{indent}        raise ValueError("_tensor_like dtype must be one of: float32, float16, bfloat16, int64, int32, bool, or null")',
        f"{indent}{value_var} = {value_expr}",
        f"{indent}if torch.is_tensor({value_var}):",
        f"{indent}    {out_var} = {value_var}.to(device={ref}.device, dtype=({ref}.dtype if {dtype_torch} is None else {dtype_torch}))",
        f"{indent}else:",
        f"{indent}    {out_var} = torch.tensor({value_var}, device={ref}.device, dtype=({ref}.dtype if {dtype_torch} is None else {dtype_torch}))",
    ]


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any", "Tensor", "Any"),
    "kwargs": {},
    "returns": ("Tensor",),
}


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
]
