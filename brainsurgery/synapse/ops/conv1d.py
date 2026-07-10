from __future__ import annotations

from typing import Any


from ..axon.ast import AxonExprAscribe, AxonExprInt, AxonExprParen, DimExprBinary

OP_NAME = "conv1d"


LOWERING_TYPE_SIGNATURE = {
    "args": (
        "Tensor[B,C,S]",
        "Tensor[O,I,K]",
        "Tensor[O]",
        "Int",
        "Dim",
        "Dim",
        "Int",
        "Dim",
    ),
    "kwargs": {},
    "returns": "dynamic",
}
LOWERING_PARAM_NAMES = (
    "x",
    "weight",
    "bias",
    "stride",
    "padding_left",
    "padding_right",
    "dilation",
    "groups",
)
PRIMITIVE_SEMANTICS = {
    "effect": "total_pure",
    "usage": "unrestricted",
}


def _dim_add(left: Any, right: Any) -> Any:
    if left == 0:
        return right
    if right == 0:
        return left
    if isinstance(left, int) and isinstance(right, int):
        return left + right
    return DimExprBinary(op="+", left=left, right=right)


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


def _dim_mul(left: Any, right: Any) -> Any:
    if left == 0 or right == 0:
        return 0
    if left == 1:
        return right
    if right == 1:
        return left
    if isinstance(left, int) and isinstance(right, int):
        return left * right
    return DimExprBinary(op="*", left=left, right=right)


def _dim_token(expr: Any, helpers: Any) -> Any | None:
    current = expr
    while isinstance(current, AxonExprAscribe | AxonExprParen):
        current = current.expr if isinstance(current, AxonExprAscribe) else current.inner
    if isinstance(current, AxonExprInt):
        return current.value
    return helpers.expr_to_dim_token(current)


def type_rule(
    *,
    arg_types: tuple[Any, ...],
    kwarg_types: dict[str, Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    helpers: Any,
) -> Any | None:
    del kwarg_types, kwargs
    if len(arg_types) != 8 or len(args) != 8:
        return None
    x_dims = helpers.type_dims(arg_types[0])
    w_dims = helpers.type_dims(arg_types[1])
    b_dims = helpers.type_dims(arg_types[2])
    if x_dims is None or w_dims is None or b_dims is None:
        return None
    if len(x_dims) != 3 or len(w_dims) != 3 or len(b_dims) != 1:
        return None
    batch, in_channels, seq = x_dims
    out_channels, per_group_channels, kernel = w_dims
    if not helpers.dim_equivalent(out_channels, b_dims[0]):
        helpers.unify_dim(out_channels, b_dims[0])
    stride = _dim_token(args[3], helpers)
    pad_left = _dim_token(args[4], helpers)
    pad_right = _dim_token(args[5], helpers)
    dilation = _dim_token(args[6], helpers)
    groups = _dim_token(args[7], helpers)
    if stride != 1 or dilation != 1:
        return None
    if groups is not None:
        expected_in = _dim_mul(per_group_channels, groups)
        if not helpers.dim_equivalent(in_channels, expected_in):
            helpers.unify_dim(in_channels, expected_in)
    if pad_left is None or pad_right is None:
        return None
    padded = _dim_add(_dim_add(seq, pad_left), pad_right)
    out_seq = _dim_add(_dim_sub(padded, kernel), 1)
    return helpers.type_tensor(dims=(batch, out_channels, out_seq))


__all__ = [
    "LOWERING_PARAM_NAMES",
    "LOWERING_TYPE_SIGNATURE",
    "OP_NAME",
    "PRIMITIVE_SEMANTICS",
    "type_rule",
]
