from __future__ import annotations

from typing import Any


from ..axon.ast import AxonExprAscribe, AxonExprInt, AxonExprParen, TypeTuple, TypeTensor

OP_NAME = "topk"


LOWERING_TYPE_SIGNATURE = {
    "args": ("Tensor[..S]", "Any", "Any", "Any", "Any"),
    "kwargs": {},
    "returns": ("Tensor[..S]", "Tensor[..S]"),
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
    if len(arg_types) != 5 or len(args) != 5:
        return None
    input_dims = helpers.type_dims(arg_types[0])
    if input_dims is None:
        return None
    if any(isinstance(dim, str) and dim.startswith("..") for dim in input_dims):
        return None
    k_dim = helpers.expr_to_dim_token(args[1])
    if k_dim is None:
        return None
    raw_axis = _unwrap_expr(args[2])
    if isinstance(raw_axis, AxonExprInt):
        axis_value = raw_axis.value
    elif isinstance(raw_axis, int):
        axis_value = raw_axis
    elif type(getattr(raw_axis, "value", None)) is int:
        axis_value = raw_axis.value
    else:
        return None
    rank = len(input_dims)
    axis = axis_value if axis_value >= 0 else rank + axis_value
    if axis < 0 or axis >= rank:
        return None
    out_dims = list(input_dims)
    out_dims[axis] = k_dim
    dims = tuple(out_dims)
    return TypeTuple(
        items=(
            TypeTensor(base="Tensor", dims=dims),
            TypeTensor(base="Tensor", dims=dims),
        )
    )

__all__ = [
    "OP_NAME",
    "LOWERING_TYPE_SIGNATURE",
    "type_rule",
]
