from __future__ import annotations

from typing import Any


from ..axon.ast import AxonExprAscribe, AxonExprNull, AxonExprParen, DimExprBinary

OP_NAME = "arange"


LOWERING_TYPE_SIGNATURE = {
    "args": ("Tensor[..S]", "?Dim", "?Dim"),
    "kwargs": {},
    "returns": ("IdxTensor[..R]",),
}


def _dim_sub(left: Any, right: Any) -> Any:
    if right == 0:
        return left
    if left == right:
        return 0
    if isinstance(left, int) and isinstance(right, int):
        return left - right
    if isinstance(left, DimExprBinary) and left.op == "+":
        if left.left == right:
            return left.right
        if left.right == right:
            return left.left
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
    "OP_NAME",
    "LOWERING_TYPE_SIGNATURE",
    "type_rule",
]
