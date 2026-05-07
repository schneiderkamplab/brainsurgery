from __future__ import annotations

from typing import Any

import torch

OP_NAME = "sum"
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
        raise ValueError("sum requires a single scalar output binding")
    if kwargs:
        unknown = ", ".join(sorted(str(key) for key in kwargs))
        raise ValueError(f"sum unsupported kwargs: {unknown}")
    if len(args) < 1 or len(args) > 3:
        raise ValueError(f"sum expects 1..3 positional args, got {len(args)}")


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
    if not args:
        raise ValueError("sum requires positional args: x [dim] [keepdim]")
    x = model._read_tensor_input(args[0], env)
    dim = int(model._eval_expr(_arg_or_default(args, 1, -1), env, symbols))
    keepdim = bool(model._eval_expr(_arg_or_default(args, 2, False), env, symbols))
    out = model._require_name(node_spec.get("_bind"), field="sum._bind")
    env[out] = torch.sum(x, dim=dim, keepdim=keepdim)


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
        raise ValueError("sum requires positional args: x [dim] [keepdim]")
    x = emitter._read_env_var(env, str(args[0]))
    dim = emitter._expr_code(_arg_or_default(args, 1, -1), env)
    keepdim = emitter._expr_code(_arg_or_default(args, 2, False), env)
    out_var = emitter._assign_out_var(env, str(node_spec.get("_bind")))
    return [f"{indent}{out_var} = torch.sum({x}, dim=int({dim}), keepdim=bool({keepdim}))"]


LOWERING_TYPE_SIGNATURE = {
    "args": ("Tensor[..S]", "Int", "Bool"),
    "kwargs": {},
    "returns": "dynamic",
}


def _dim_value(expr: Any, helpers: Any) -> int | None:
    token = helpers.expr_to_dim_token(expr)
    if isinstance(token, int):
        return int(token)
    if isinstance(token, str):
        try:
            return int(token)
        except ValueError:
            return None
    return None


def _bool_value(expr: Any) -> bool | None:
    value = getattr(expr, "value", None)
    if isinstance(value, bool):
        return value
    return None


def type_rule(
    *,
    arg_types: tuple[Any, ...],
    kwarg_types: dict[str, Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    helpers: Any,
) -> Any | None:
    del kwarg_types, kwargs
    if not arg_types:
        return None
    dims = helpers.type_dims(arg_types[0])
    if dims is None:
        return None
    dim = _dim_value(args[1], helpers) if len(args) > 1 else -1
    if dim is None:
        return None
    rank = len(dims)
    if rank == 0:
        return helpers.type_tensor(dims=())
    if dim < 0:
        dim += rank
    if dim < 0 or dim >= rank:
        return None
    keepdim = _bool_value(args[2]) if len(args) > 2 else False
    if keepdim:
        out_dims = list(dims)
        out_dims[dim] = 1
    else:
        out_dims = [value for idx, value in enumerate(dims) if idx != dim]
    return helpers.type_tensor(dims=tuple(out_dims))


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
