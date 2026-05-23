from __future__ import annotations

from typing import Any

from ..axon.ast import AxonExprAscribe, AxonExprParen, DimExprBinary


OP_NAME = "activation"


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any",),
    "kwargs": {},
    "returns": "dynamic",
}
LOWERING_PARAM_NAMES_BY_OP = {
    "activations_gegelu": ("x", "limit"),
    "activations_gelu": ("x",),
    "activations_gelu_new": ("x",),
    "activations_gelu_pytorch_tanh": ("x",),
    "activations_relu": ("x",),
    "activations_relu2": ("x",),
    "activations_sigmoid": ("x",),
    "activations_silu": ("x",),
    "activations_swiglu": ("x",),
    "activations_tanh": ("x",),
    "activations_xielu": ("x", "alpha_p", "alpha_n", "beta", "eps"),
}


def type_rule(
    *,
    arg_types: tuple[Any, ...],
    kwarg_types: dict[str, Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    helpers: Any,
) -> Any | None:
    del kwarg_types, args, kwargs
    if len(arg_types) < 1:
        return None
    input_dims = helpers.type_dims(arg_types[0])
    if input_dims is None:
        return None
    return helpers.type_tensor(dims=input_dims)


def _gegelu_type_rule(
    *,
    arg_types: tuple[Any, ...],
    kwarg_types: dict[str, Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    helpers: Any,
) -> Any | None:
    del kwarg_types, args, kwargs
    if len(arg_types) < 1:
        return None
    input_dims = helpers.type_dims(arg_types[0])
    if input_dims is None or not input_dims:
        return None
    out_dims = (*input_dims[:-1], _dim_div(input_dims[-1], 2))
    return helpers.type_tensor(dims=out_dims)


def _dim_div(left: Any, right: Any) -> Any:
    while isinstance(left, AxonExprAscribe | AxonExprParen):
        left = left.expr if isinstance(left, AxonExprAscribe) else left.inner
    if right == 1:
        return left
    if isinstance(left, int) and left % right == 0:
        return left // right
    if isinstance(left, DimExprBinary) and left.op == "*":
        if left.left == right:
            return left.right
        if left.right == right:
            return left.left
        if isinstance(left.left, int) and left.left % right == 0:
            quotient = left.left // right
            if quotient == 1:
                return left.right
            return DimExprBinary(op="*", left=quotient, right=left.right)
        if isinstance(left.right, int) and left.right % right == 0:
            quotient = left.right // right
            if quotient == 1:
                return left.left
            return DimExprBinary(op="*", left=left.left, right=quotient)
    return DimExprBinary(op="/", left=left, right=right)


TYPE_RULES_BY_OP = {
    "activations_gegelu": _gegelu_type_rule,
}


__all__ = [
    "OP_NAME",
    "LOWERING_TYPE_SIGNATURE",
    "LOWERING_PARAM_NAMES_BY_OP",
    "TYPE_RULES_BY_OP",
    "type_rule",
]
