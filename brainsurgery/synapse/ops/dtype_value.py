from __future__ import annotations

from typing import Any

import torch

OP_NAME = "dtype_value"
LOWERING_ARITY = (2, 2)
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
        raise ValueError("dtype_value requires a single scalar output binding")
    if kwargs:
        unknown = ", ".join(sorted(str(key) for key in kwargs))
        raise ValueError(f"dtype_value unsupported kwargs: {unknown}")
    if len(args) != 2:
        raise ValueError(f"dtype_value expects 2 positional args, got {len(args)}")


def _resolve_kind(
    model: Any, raw: Any, env: dict[str, Any], symbols: dict[str, int | float]
) -> str:
    value = model._eval_expr(raw, env, symbols)
    if not isinstance(value, str):
        raise ValueError("dtype_value kind must resolve to string")
    kind = value.strip().lower()
    if kind not in {"min", "max", "eps", "tiny", "inf", "-inf"}:
        raise ValueError("dtype_value kind must be one of: min,max,eps,tiny,inf,-inf")
    return kind


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
    if len(args) != 2:
        raise ValueError("dtype_value requires positional args: x kind")
    x = model._read_tensor_input(args[0], env)
    if not torch.is_tensor(x):
        raise ValueError("dtype_value first argument must resolve to tensor")
    if not torch.is_floating_point(x):
        raise ValueError("dtype_value expects floating-point tensor input")
    kind = _resolve_kind(model, args[1], env, symbols)
    info = torch.finfo(x.dtype)
    if kind == "min":
        out_value = float(info.min)
    elif kind == "max":
        out_value = float(info.max)
    elif kind == "eps":
        out_value = float(info.eps)
    elif kind == "tiny":
        out_value = float(info.tiny)
    elif kind == "inf":
        out_value = float("inf")
    else:
        out_value = float("-inf")
    out = model._require_name(node_spec.get("_bind"), field="dtype_value._bind")
    env[out] = out_value


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
        raise ValueError("dtype_value requires positional args: x kind")
    src = emitter._read_env_var(env, str(args[0]))
    kind_expr = emitter._expr_code(args[1], env)
    kind_var = emitter._fresh("dtype_value_kind")
    out_var = emitter._assign_out_var(env, str(node_spec.get("_bind")))
    lines: list[str] = [
        f"{indent}if not torch.is_tensor({src}):",
        f"{indent}    raise ValueError('dtype_value first argument must resolve to tensor')",
        f"{indent}if not torch.is_floating_point({src}):",
        f"{indent}    raise ValueError('dtype_value expects floating-point tensor input')",
        f"{indent}{kind_var} = str({kind_expr}).strip().lower()",
        f"{indent}if {kind_var} == 'min':",
        f"{indent}    {out_var} = float(torch.finfo({src}.dtype).min)",
        f"{indent}elif {kind_var} == 'max':",
        f"{indent}    {out_var} = float(torch.finfo({src}.dtype).max)",
        f"{indent}elif {kind_var} == 'eps':",
        f"{indent}    {out_var} = float(torch.finfo({src}.dtype).eps)",
        f"{indent}elif {kind_var} == 'tiny':",
        f"{indent}    {out_var} = float(torch.finfo({src}.dtype).tiny)",
        f"{indent}elif {kind_var} == 'inf':",
        f"{indent}    {out_var} = float('inf')",
        f"{indent}elif {kind_var} == '-inf':",
        f"{indent}    {out_var} = float('-inf')",
        f"{indent}else:",
        f"{indent}    raise ValueError('dtype_value kind must be one of: min,max,eps,tiny,inf,-inf')",
    ]
    return lines


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any", "String"),
    "kwargs": {},
    "returns": ("Float",),
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
