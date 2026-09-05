from __future__ import annotations

from typing import Any

from ..axon.ast import AxonExprAscribe, AxonExprInt, AxonExprParen


OP_NAME = "unsqueeze"


LOWERING_TYPE_SIGNATURE = {
    "args": ("Tensor[..S]", "Int"),
    "kwargs": {},
    "returns": ("Tensor[..R]",),
}


def _unwrap_expr(expr: Any) -> Any:
    while isinstance(expr, AxonExprAscribe | AxonExprParen):
        expr = expr.expr if isinstance(expr, AxonExprAscribe) else expr.inner
    return expr


def type_rule(
    *,
    arg_types: tuple[Any, ...],
    kwarg_types: dict[str, Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    helpers: Any,
) -> Any | None:
    del kwarg_types, kwargs
    if len(arg_types) != 2 or len(args) != 2:
        return None
    dims = helpers.type_dims(arg_types[0])
    if dims is None:
        return None
    raw_dim = _unwrap_expr(args[1])
    if isinstance(raw_dim, AxonExprInt):
        dim_value = raw_dim.value
    elif isinstance(raw_dim, int):
        dim_value = raw_dim
    else:
        return None
    rank = len(dims) + 1
    dim = dim_value if dim_value >= 0 else rank + dim_value
    if dim < 0 or dim > len(dims):
        return None
    out_dims = list(dims)
    out_dims.insert(dim, 1)
    return helpers.type_tensor(dims=tuple(out_dims))


__all__ = [
    "OP_NAME",
    "LOWERING_TYPE_SIGNATURE",
    "type_rule",
]
