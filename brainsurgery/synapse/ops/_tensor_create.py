from __future__ import annotations

from typing import Any

import torch

from ..axon.ast import AxonExprAscribe, AxonExprName, AxonExprParen


def raw_args(node_spec: dict[str, Any]) -> list[Any]:
    raw = node_spec.get("_args")
    if isinstance(raw, list):
        return list(raw)
    if raw is None:
        return []
    return [raw]


def arg_or_default(args: list[Any], index: int, default: Any) -> Any:
    if index >= len(args):
        return default
    value = args[index]
    if isinstance(value, str) and value.strip().lower() in {"null", "none"}:
        return default
    return value


def resolve_dtype(dtype_raw: Any) -> torch.dtype | None:
    if dtype_raw is None:
        return None
    token = str(dtype_raw).strip().lower()
    if token in {"", "null", "none", "default"}:
        return None
    if token in {"float32", "fp32", "single"}:
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
    raise ValueError(
        "tensor creation dtype must be one of: float32, float16, bfloat16, "
        "int64, int32, bool, or null"
    )


def resolve_shape(model: Any, raw_shape: Any, env: dict[str, Any], symbols: dict[str, Any]) -> tuple[int, ...]:
    if isinstance(raw_shape, list):
        raw_items = raw_shape
    else:
        evaluated = model._eval_expr(raw_shape, env, symbols)
        if not isinstance(evaluated, list | tuple) or not evaluated:
            raise ValueError("tensor creation shape must be a non-empty list")
        raw_items = list(evaluated)
    resolved: list[int] = []
    for item in raw_items:
        value = model._eval_expr(item, env, symbols)
        if isinstance(value, bool):
            raise ValueError("tensor creation shape entries must resolve to dims")
        if isinstance(value, int):
            resolved.append(int(value))
            continue
        if isinstance(value, float) and float(value).is_integer():
            resolved.append(int(value))
            continue
        raise ValueError("tensor creation shape entries must resolve to dims")
    return tuple(resolved)


def shape_expr_code(emitter: Any, raw_shape: Any, env: dict[str, str]) -> str:
    def expr_payload(value: Any) -> Any:
        if isinstance(value, str):
            token = value.strip()
            if token.isidentifier():
                return {"_expr": "name", "id": token}
        return value

    if isinstance(raw_shape, list) and raw_shape:
        return f"({', '.join(f'int({emitter._expr_code(expr_payload(item), env)})' for item in raw_shape)},)"
    return emitter._expr_code(expr_payload(raw_shape), env)


def shape_dim_tokens(shape_expr: Any, helpers: Any) -> tuple[Any, ...] | None:
    current = shape_expr
    while True:
        while isinstance(current, AxonExprAscribe | AxonExprParen):
            current = current.expr if isinstance(current, AxonExprAscribe) else current.inner
        if isinstance(current, AxonExprName):
            resolved = helpers.resolve_name_expr(current.name)
            if resolved is None:
                return None
            current = resolved
            continue
        break
    items = getattr(current, "items", None)
    if not isinstance(items, tuple):
        return None
    dims: list[Any] = []
    for item in items:
        token = helpers.expr_to_dim_token(item)
        if token is None:
            return None
        dims.append(token)
    return tuple(dims)


def type_from_shape_args(args: tuple[Any, ...], helpers: Any) -> Any | None:
    if len(args) < 2:
        return None
    dims = shape_dim_tokens(args[1], helpers)
    if dims is None:
        return None
    return helpers.type_tensor(dims=dims)


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return False


__all__ = [
    "arg_or_default",
    "raw_args",
    "resolve_dtype",
    "resolve_shape",
    "shape_expr_code",
    "type_from_shape_args",
    "uses_node_path",
]
