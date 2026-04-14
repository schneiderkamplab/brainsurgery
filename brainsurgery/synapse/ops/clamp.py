from __future__ import annotations

from typing import Any

import torch

OP_NAME = "clamp"
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


def _is_null_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() == "null":
        return True
    return False


def _arg_or_none(args: list[Any], index: int) -> Any:
    if index >= len(args):
        return None
    value = args[index]
    if _is_null_like(value):
        return None
    return value


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
        raise ValueError("clamp requires a single output binding")
    if kwargs:
        unknown = ", ".join(sorted(str(key) for key in kwargs))
        raise ValueError(f"clamp unsupported kwargs: {unknown}")
    if len(args) < 1 or len(args) > 3:
        raise ValueError(f"clamp expects 1..3 positional args, got {len(args)}")
    min_arg = _arg_or_none(args, 1)
    max_arg = _arg_or_none(args, 2)
    if min_arg is None and max_arg is None:
        raise ValueError("clamp requires at least one of min/max")


def lowering_infer_metadata(
    *,
    args: list[str],
    out: str | list[str],
    kwargs: dict[str, Any],
    ctx: Any,
) -> bool:
    del kwargs
    if not isinstance(out, str):
        return False
    first_in = args[0].strip() if args else None
    if isinstance(first_in, str) and first_in.isidentifier():
        first_dim = ctx.tensor_last_dim.get(first_in)
        if first_dim is not None:
            ctx.tensor_last_dim[out] = first_dim
    return True


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
    if not args:
        raise ValueError("clamp requires positional args: x [min] [max]")
    x = model._read_tensor_input(args[0], env)
    out = model._require_name(node_spec.get("_bind"), field="clamp._bind")
    min_arg = _arg_or_none(args, 1)
    max_arg = _arg_or_none(args, 2)
    if min_arg is None and max_arg is None:
        raise ValueError("clamp requires at least one of min/max")
    min_value = float(model._eval_expr(min_arg, env, symbols)) if min_arg is not None else None
    max_value = float(model._eval_expr(max_arg, env, symbols)) if max_arg is not None else None
    env[out] = torch.clamp(x, min=min_value, max=max_value)


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
    if not args:
        raise ValueError("clamp requires positional args: x [min] [max]")
    lines: list[str] = []
    src = emitter._read_env_var(env, str(args[0]))
    out_var = emitter._assign_out_var(env, str(node_spec.get("_bind")))
    min_arg = _arg_or_none(args, 1)
    max_arg = _arg_or_none(args, 2)
    if min_arg is None and max_arg is None:
        raise ValueError("clamp requires at least one of min/max")
    min_code = emitter._expr_code(_expr_payload(min_arg), env) if min_arg is not None else "None"
    max_code = emitter._expr_code(_expr_payload(max_arg), env) if max_arg is not None else "None"
    lines.append(f"{indent}{out_var} = torch.clamp({src}, min={min_code}, max={max_code})")
    return lines


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any", "Any", "Any"),
    "kwargs": {},
    "returns": "dynamic",
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
