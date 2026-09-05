from __future__ import annotations

from typing import Any

from ..axon.ast import AxonExprAscribe, AxonExprInt, AxonExprParen, TypeDim

OP_NAME = "tensor_size"


LOWERING_TYPE_SIGNATURE = {
    "args": ("Tensor[..S]", "Int"),
    "kwargs": {},
    "returns": ("Dim",),
}


PRIMITIVE_SEMANTICS = {
    # The scalar Dim result equals the selected axis of the input tensor shape.
    "dim_output_from_tensor_axis": {
        "output": 0,
        "tensor_arg": 0,
        "axis_arg": 1,
    },
}


def _expr_int(expr: Any) -> int | None:
    while isinstance(expr, AxonExprAscribe | AxonExprParen):
        expr = expr.expr if isinstance(expr, AxonExprAscribe) else expr.inner
    if isinstance(expr, AxonExprInt):
        return expr.value
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
    if len(arg_types) != 2 or len(args) != 2:
        return TypeDim()
    dims = helpers.type_dims(arg_types[0])
    dim_index = _expr_int(args[1])
    if dims is None or dim_index is None:
        return TypeDim()
    axis = dim_index if dim_index >= 0 else len(dims) + dim_index
    if axis < 0 or axis >= len(dims):
        return TypeDim()
    return TypeDim()


__all__ = [
    "OP_NAME",
    "LOWERING_TYPE_SIGNATURE",
    "PRIMITIVE_SEMANTICS",
    "type_rule",
]
