from __future__ import annotations

from typing import Any


from ..axon.ast import AxonExprAscribe, AxonExprInt, AxonExprParen, DimExprBinary

OP_NAME = "repeat"


LOWERING_TYPE_SIGNATURE = {
    "args": ("Tensor[..S]", "Any", "Any"),
    "kwargs": {},
    "returns": ("Tensor[..R]",),
}


def _unwrap_expr(expr: Any) -> Any:
    while isinstance(expr, AxonExprAscribe | AxonExprParen):
        expr = expr.expr if isinstance(expr, AxonExprAscribe) else expr.inner
    return expr


def _dim_mul(left: Any, right: Any) -> Any:
    if isinstance(left, int) and isinstance(right, int):
        return left * right
    if isinstance(right, DimExprBinary) and right.op == "/" and right.right == left:
        return right.left
    if isinstance(left, DimExprBinary) and left.op == "/" and left.right == right:
        return left.left
    if left == 1:
        return right
    if right == 1:
        return left
    return DimExprBinary(op="*", left=left, right=right)


def _expr_to_dim_token(expr: Any, helpers: Any, seen: frozenset[str] = frozenset()) -> Any | None:
    token = helpers.expr_to_dim_token(expr)
    if isinstance(token, str) and token not in seen:
        resolved = helpers.resolve_name_expr(token)
        if resolved is not None:
            return _expr_to_dim_token(resolved, helpers, seen | {token})
    return token


def type_rule(
    *,
    arg_types: tuple[Any, ...],
    kwarg_types: dict[str, Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    helpers: Any,
) -> Any | None:
    del kwarg_types, kwargs
    if len(arg_types) != 3 or len(args) != 3:
        return None
    dims = helpers.type_dims(arg_types[0])
    if dims is None:
        return None
    repeats = _expr_to_dim_token(args[1], helpers)
    if repeats is None:
        return None
    raw_dim = _unwrap_expr(args[2])
    if isinstance(raw_dim, AxonExprInt):
        dim_value = raw_dim.value
    elif isinstance(raw_dim, int):
        dim_value = raw_dim
    else:
        return None
    rank = len(dims)
    dim = dim_value if dim_value >= 0 else rank + dim_value
    if dim < 0 or dim >= rank:
        return None
    out_dims = list(dims)
    out_dims[dim] = _dim_mul(out_dims[dim], repeats)
    return helpers.type_tensor(dims=tuple(out_dims))

__all__ = [
    "OP_NAME",
    "LOWERING_TYPE_SIGNATURE",
    "type_rule",
]
