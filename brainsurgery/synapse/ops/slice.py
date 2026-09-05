from __future__ import annotations

from typing import Any

from ..axon.ast import AxonExprAscribe, AxonExprName, AxonExprParen, DimExprBinary

OP_NAME = "slice"


def _dim_sub(left: Any, right: Any) -> Any:
    if right == 0:
        return left
    if isinstance(left, int) and isinstance(right, int):
        return left - right
    if isinstance(right, DimExprBinary) and right.op == "-" and right.left == left:
        return right.right
    return DimExprBinary(op="-", left=left, right=right)


def _arg_dim_token(value: Any, helpers: Any) -> Any | None:
    while isinstance(value, AxonExprAscribe | AxonExprParen):
        value = value.expr if isinstance(value, AxonExprAscribe) else value.inner
    token = helpers.expr_to_dim_token(value)
    if isinstance(value, AxonExprName):
        resolved = helpers.resolve_name_expr(value.name)
        if resolved is not None:
            resolved_token = helpers.expr_to_dim_token(resolved)
            if resolved_token is not None:
                return resolved_token
    return token


LOWERING_TYPE_SIGNATURE = {
    "args": ("Tensor[..S]", "Int", "Dim", "Dim"),
    "kwargs": {},
    "returns": ("Tensor[..R]",),
}


def type_rule(
    *,
    arg_types: tuple[Any, ...],
    kwarg_types: dict[str, Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    helpers: Any,
) -> Any | None:
    del kwarg_types, kwargs
    if len(arg_types) != 4 or len(args) != 4:
        return None
    input_dims = helpers.type_dims(arg_types[0])
    if input_dims is None:
        return None
    if any(isinstance(item, str) and item.startswith("..") for item in input_dims):
        # A variadic shape pack has unknown rank.  Inferring a concrete slice
        # axis from the tuple position would collapse the pack incorrectly
        # (for example Tensor[..S] sliced on -1 is not rank-1 Tensor[S]).
        return None
    dim_token = _arg_dim_token(args[1], helpers)
    if not isinstance(dim_token, int):
        return None
    rank = len(input_dims)
    dim = dim_token if dim_token >= 0 else rank + dim_token
    if dim < 0 or dim >= rank:
        return None
    start = _arg_dim_token(args[2], helpers)
    end = _arg_dim_token(args[3], helpers)
    if start is None or end is None:
        return None
    out_dims = list(input_dims)
    axis_dim = out_dims[dim]
    if (
        end == axis_dim
        and isinstance(start, DimExprBinary)
        and start.op == "-"
        and isinstance(start.left, str)
    ):
        out_dims[dim] = start.right
    else:
        out_dims[dim] = _dim_sub(end, start)
    return helpers.type_tensor(dims=tuple(out_dims))

__all__ = [
    "OP_NAME",
    "LOWERING_TYPE_SIGNATURE",
    "type_rule",
]
