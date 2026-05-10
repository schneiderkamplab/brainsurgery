from __future__ import annotations

from typing import Any


OP_NAME = "sum"


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
    "LOWERING_TYPE_SIGNATURE",
    "type_rule",
]
