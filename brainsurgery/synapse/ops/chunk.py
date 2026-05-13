from __future__ import annotations

from typing import Any

from ..axon.ast import AxonExprAscribe, AxonExprInt, AxonExprParen, DimExprBinary

from ..axon.ast import TypeList

OP_NAME = "chunk"


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any",),
    "kwargs": {'dim': 'int', 'parts': 'int'},
    "returns": "dynamic",
}
LOWERING_PARAM_NAMES = ("x", "dim", "parts")
LOWERING_PARAM_DEFAULTS = {"dim": -1, "parts": None}


def type_rule(
    *,
    arg_types: tuple[Any, ...],
    kwarg_types: dict[str, Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    helpers: Any,
) -> Any | None:
    del kwarg_types
    if len(arg_types) < 1:
        return None
    input_dims = helpers.type_dims(arg_types[0])
    if input_dims is None:
        return None
    item_dims = tuple(input_dims)
    dim_expr = kwargs.get("dim") if "dim" in kwargs else args[1] if len(args) > 1 else None
    parts_expr = kwargs.get("parts") if "parts" in kwargs else args[2] if len(args) > 2 else None
    dim_token = _expr_dim_token(dim_expr, helpers) if dim_expr is not None else -1
    parts_token = _expr_dim_token(parts_expr, helpers) if parts_expr is not None else None
    if isinstance(dim_token, int) and parts_token is not None:
        rank = len(item_dims)
        dim_idx = dim_token if dim_token >= 0 else rank + dim_token
        if 0 <= dim_idx < rank:
            updated = list(item_dims)
            updated[dim_idx] = _dim_div(updated[dim_idx], parts_token)
            item_dims = tuple(updated)
    item_tp = helpers.type_tensor(dims=item_dims)
    return TypeList(item=item_tp)


def _expr_dim_token(expr: Any, helpers: Any) -> Any | None:
    while isinstance(expr, AxonExprAscribe | AxonExprParen):
        expr = expr.expr if isinstance(expr, AxonExprAscribe) else expr.inner
    if isinstance(expr, AxonExprInt):
        return expr.value
    return helpers.expr_to_dim_token(expr)


def _dim_div(left: Any, right: Any) -> Any:
    if right == 1:
        return left
    if isinstance(left, int) and isinstance(right, int) and right != 0 and left % right == 0:
        return left // right
    if isinstance(left, DimExprBinary) and left.op == "*" and isinstance(right, int):
        if left.left == right:
            return left.right
        if left.right == right:
            return left.left
        if isinstance(left.left, int) and left.left % right == 0:
            quotient = left.left // right
            return left.right if quotient == 1 else DimExprBinary(op="*", left=quotient, right=left.right)
        if isinstance(left.right, int) and left.right % right == 0:
            quotient = left.right // right
            return left.left if quotient == 1 else DimExprBinary(op="*", left=left.left, right=quotient)
    return DimExprBinary(op="/", left=left, right=right)


__all__ = [
    "OP_NAME",
    "LOWERING_TYPE_SIGNATURE",
    "LOWERING_PARAM_DEFAULTS",
    "LOWERING_PARAM_NAMES",
    "type_rule",
]
