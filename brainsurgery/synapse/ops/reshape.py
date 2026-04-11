from __future__ import annotations

from typing import Any

import torch

OP_NAME = "reshape"
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


def _name_expr(value: str) -> dict[str, Any]:
    return {"_expr": "name", "id": value}


def _expr_payload(value: Any) -> Any:
    if isinstance(value, str):
        token = value.strip()
        if token.isidentifier():
            return _name_expr(token)
    return value


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return False


def lowering_validate_signature(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> None:
    del ctx
    if not isinstance(out, str):
        raise ValueError("reshape requires a single scalar output binding")
    if kwargs:
        unknown = ", ".join(sorted(str(key) for key in kwargs))
        raise ValueError(f"reshape unsupported kwargs: {unknown}")
    if len(args) != 2:
        raise ValueError(f"reshape expects 2 positional args, got {len(args)}")


def lowering_infer_metadata(
    *,
    args: list[str],
    out: str | list[str],
    kwargs: dict[str, Any],
    ctx: Any,
) -> bool:
    if not isinstance(out, str):
        return False
    if len(args) != 2:
        return False
    shape = args[1]
    if not isinstance(shape, list) or not shape:
        return False
    ctx.tensor_shape[out] = tuple(shape)
    ctx.tensor_last_dim[out] = shape[-1]
    return True


def _resolve_shape(
    model: Any,
    raw_shape: Any,
    env: dict[str, Any],
    symbols: dict[str, int],
) -> tuple[int, ...]:
    if isinstance(raw_shape, list):
        raw_items = raw_shape
    else:
        evaluated = model._eval_expr(raw_shape, env, symbols)
        if not isinstance(evaluated, list | tuple) or not evaluated:
            raise ValueError("reshape.shape must be a non-empty list")
        raw_items = list(evaluated)
    resolved: list[int] = []
    for item in raw_items:
        value = model._eval_expr(item, env, symbols)
        if isinstance(value, bool):
            raise ValueError("reshape.shape entries must resolve to ints")
        if isinstance(value, int):
            resolved.append(int(value))
            continue
        if isinstance(value, float) and float(value).is_integer():
            resolved.append(int(value))
            continue
        raise ValueError("reshape.shape entries must resolve to ints")
    return tuple(resolved)


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
        raise ValueError("reshape requires positional args: x shape")
    src = model._read_tensor_input(args[0], env)
    shape = _resolve_shape(model, args[1], env, symbols)
    out = model._require_name(node_spec.get("_bind"), field="reshape._bind")
    env[out] = torch.reshape(src, shape)


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
    lines: list[str] = []
    args = _raw_args(node_spec)
    if len(args) != 2:
        raise ValueError("reshape requires positional args: x shape")
    src = emitter._read_env_var(env, str(args[0]))
    out_var = emitter._assign_out_var(env, str(node_spec.get("_bind")))
    raw_shape = args[1]
    if isinstance(raw_shape, list) and raw_shape:
        shape_expr = f"({', '.join(f'int({emitter._expr_code(_expr_payload(item), env)})' for item in raw_shape)},)"
    else:
        shape_expr = emitter._expr_code(_expr_payload(raw_shape), env)
    shape_var = emitter._fresh("shape")
    lines.append(f"{indent}{shape_var} = tuple(int(v) for v in {shape_expr})")
    lines.append(f"{indent}if len({shape_var}) == 0:")
    lines.append(f"{indent}    raise ValueError('reshape.shape must be a non-empty list')")
    lines.append(f"{indent}{out_var} = torch.reshape({src}, {shape_var})")
    return lines


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any", "Any"),
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
