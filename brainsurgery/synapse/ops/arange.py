from __future__ import annotations

from typing import Any

import torch

from ..axon.ast import AxonExprAscribe, AxonExprNull, AxonExprParen, DimExprBinary

OP_NAME = "arange"
LOWERING_ARITY = (1, 3)
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
    if isinstance(value, str) and value.strip().lower() == "null":
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
        raise ValueError("arange requires a single scalar output binding")
    if kwargs:
        unknown = ", ".join(sorted(str(key) for key in kwargs))
        raise ValueError(f"arange unsupported kwargs: {unknown}")
    if len(args) < 1 or len(args) > 3:
        raise ValueError("arange requires positional args: x [start end]")


def lowering_infer_metadata(
    *,
    args: list[str],
    out: str | list[str],
    kwargs: dict[str, Any],
    ctx: Any,
) -> bool:
    del args, kwargs, ctx
    return isinstance(out, str)


def _resolve_bound(
    model: Any,
    raw: Any,
    env: dict[str, Any],
    symbols: dict[str, int],
    *,
    field: str,
) -> int:
    value = model._eval_expr(raw, env, symbols)
    if isinstance(value, bool):
        raise ValueError(f"arange.{field} must resolve to int")
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and float(value).is_integer():
        return int(value)
    raise ValueError(f"arange.{field} must resolve to int")


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
    if len(args) < 1 or len(args) > 3:
        raise ValueError("arange requires positional args: x [start end]")
    src = model._read_tensor_input(args[0], env)
    start = _resolve_bound(model, _arg_or_default(args, 1, 0), env, symbols, field="start")
    end_raw = _arg_or_default(args, 2, None)
    if end_raw is None:
        end = int(src.shape[-2] if src.ndim >= 2 else src.shape[-1])
    else:
        end_value = model._eval_expr(end_raw, env, symbols)
        if end_value is None:
            end = int(src.shape[-2] if src.ndim >= 2 else src.shape[-1])
        elif isinstance(end_value, bool):
            raise ValueError("arange.end must resolve to int or null")
        elif isinstance(end_value, int):
            end = int(end_value)
        elif isinstance(end_value, float) and float(end_value).is_integer():
            end = int(end_value)
        else:
            raise ValueError("arange.end must resolve to int or null")
    out = model._require_name(node_spec.get("_bind"), field="arange._bind")
    env[out] = torch.arange(start, end, device=src.device, dtype=torch.long)


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
    if len(args) < 1 or len(args) > 3:
        raise ValueError("arange requires positional args: x [start end]")
    src = emitter._read_env_var(env, str(args[0]))
    out_var = emitter._assign_out_var(env, str(node_spec.get("_bind")))
    start_expr = emitter._expr_code(_arg_or_default(args, 1, 0), env)
    end_raw = _arg_or_default(args, 2, None)
    start_var = emitter._fresh("arange_start")
    end_var = emitter._fresh("arange_end")
    end_lines: list[str]
    if end_raw is None:
        end_lines = [
            f"{indent}{end_var} = int({src}.shape[-2] if {src}.ndim >= 2 else {src}.shape[-1])"
        ]
    else:
        end_expr = emitter._expr_code(end_raw, env)
        end_raw_var = emitter._fresh("arange_end_raw")
        end_lines = [
            f"{indent}{end_raw_var} = {end_expr}",
            f"{indent}{end_var} = int({src}.shape[-2] if {end_raw_var} is None and {src}.ndim >= 2 else ({src}.shape[-1] if {end_raw_var} is None else {end_raw_var}))",
        ]
    return [
        f"{indent}{start_var} = int({start_expr})",
        *end_lines,
        f"{indent}{out_var} = torch.arange({start_var}, {end_var}, device={src}.device, dtype=torch.long)",
    ]


LOWERING_TYPE_SIGNATURE = {
    "args": ("Tensor[..S]", "?Dim", "?Dim"),
    "kwargs": dict(LOWERING_KWARG_KINDS),
    "returns": ("IdxTensor[..R]",),
}


def _dim_sub(left: Any, right: Any) -> Any:
    if right == 0:
        return left
    if left == right:
        return 0
    if isinstance(left, int) and isinstance(right, int):
        return left - right
    return DimExprBinary(op="-", left=left, right=right)


def _is_null_expr(expr: Any) -> bool:
    current = expr
    while isinstance(current, AxonExprAscribe | AxonExprParen):
        current = current.expr if isinstance(current, AxonExprAscribe) else current.inner
    return isinstance(current, AxonExprNull)


def type_rule(
    *,
    arg_types: tuple[Any, ...],
    kwarg_types: dict[str, Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    helpers: Any,
) -> Any | None:
    del kwarg_types, kwargs
    if len(args) < 3:
        return None
    start = helpers.expr_to_dim_token(args[1])
    if start is None:
        return None
    end = helpers.expr_to_dim_token(args[2])
    if end is None and _is_null_expr(args[2]) and arg_types:
        ref_dims = helpers.type_dims(arg_types[0])
        if ref_dims:
            end = ref_dims[-2] if len(ref_dims) >= 2 else ref_dims[-1]
    if end is None:
        return None
    return helpers.type_tensor(dims=(_dim_sub(end, start),))

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
