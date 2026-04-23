from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

OP_NAME = "softmax"
LOWERING_ARITY = (3, 3)
LOWERING_ALLOWED_KWARGS: set[str] = set()
LOWERING_REQUIRED_KWARGS: set[str] = set()
LOWERING_KWARG_KINDS: dict[str, Any] = {}
_SUPPORTED_DTYPES: set[str] = {"float32", "float16", "bfloat16"}


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
    if not isinstance(out, str):
        raise ValueError("softmax requires a single scalar output binding")
    if kwargs:
        unknown = ", ".join(sorted(str(key) for key in kwargs))
        raise ValueError(f"softmax unsupported kwargs: {unknown}")
    if len(args) != 3:
        raise ValueError(f"softmax expects exactly 3 positional args, got {len(args)}")


def interpret(
    model: Any,
    node_spec: dict[str, Any],
    env: dict[str, Any],
    *,
    node_path: str,
    scope: str,
    symbols: dict[str, int],
) -> None:
    args = _raw_args(node_spec)
    if len(args) != 3:
        raise ValueError("softmax requires positional args: x dim dtype")
    x = model._read_tensor_input(args[0], env)
    out = model._require_name(node_spec.get("_bind"), field="softmax._bind")
    dim = int(model._eval_expr(args[1], env, symbols))
    dtype_expr = args[2]
    dtype_name = None if dtype_expr is None else model._eval_expr(dtype_expr, env, symbols)
    if dtype_name is None:
        env[out] = F.softmax(x, dim=dim)
    else:
        if not isinstance(dtype_name, str):
            raise ValueError("softmax dtype must be a string when provided")
        dtype_map: dict[str, torch.dtype] = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if dtype_name not in _SUPPORTED_DTYPES:
            raise ValueError(f"Unsupported softmax dtype: {dtype_name}")
        env[out] = F.softmax(x, dim=dim, dtype=dtype_map[dtype_name])
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

    def infer_param(param_name: str) -> str:
        return emitter._infer_param_expr(node_spec, node_path_var, param_name)

    def read(name: str) -> str:
        return emitter._read_env_var(env, name)

    args = _raw_args(node_spec)
    if len(args) != 3:
        raise ValueError("softmax requires positional args: x dim dtype")
    src = read(str(args[0]))
    out_name = str(node_spec.get("_bind"))
    out_var = assign_out_var(out_name)
    dim = emitter._expr_code(args[1], env)
    dtype_expr_raw = args[2]
    if dtype_expr_raw is None:
        lines.append(f"{indent}{out_var} = F.softmax({src}, dim=int({dim}))")
    else:
        dtype_expr = emitter._expr_code(dtype_expr_raw, env)
        dtype_raw = emitter._fresh("dtype_raw")
        dtype_var = emitter._fresh("dtype")
        lines.append(f"{indent}{dtype_raw} = {dtype_expr}")
        lines.append(f"{indent}if {dtype_raw} is None:")
        lines.append(f"{indent}    {out_var} = F.softmax({src}, dim=int({dim}))")
        lines.append(f"{indent}else:")
        lines.append(f"{indent}    if not isinstance({dtype_raw}, str):")
        lines.append(
            f"{indent}        raise ValueError('softmax dtype must be a string when provided')"
        )
        lines.append(f"{indent}    {dtype_raw} = {dtype_raw}.strip().lower()")
        lines.append(f"{indent}    if {dtype_raw} == 'float32':")
        lines.append(f"{indent}        {dtype_var} = torch.float32")
        lines.append(f"{indent}    elif {dtype_raw} == 'float16':")
        lines.append(f"{indent}        {dtype_var} = torch.float16")
        lines.append(f"{indent}    elif {dtype_raw} == 'bfloat16':")
        lines.append(f"{indent}        {dtype_var} = torch.bfloat16")
        lines.append(f"{indent}    else:")
        lines.append(
            f"{indent}        raise ValueError(f'Unsupported softmax dtype: {{{dtype_raw}}}')"
        )
        lines.append(f"{indent}    {out_var} = F.softmax({src}, dim=int({dim}), dtype={dtype_var})")
    return lines


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any", "Any", "Any"),
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
    if len(arg_types) != 3:
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
    "interpret",
    "compile",
    "uses_node_path",
    "LOWERING_TYPE_SIGNATURE",
    "type_rule",
]
