from __future__ import annotations

from typing import Any

from ..axon.ast import AxonExpr, AxonExprList, TypeList, TypeTuple

OP_NAME = "split"


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any",),
    "kwargs": {'dim': 'int', 'sizes': 'list_dim'},
    "returns": "dynamic",
}


def type_rule(
    *,
    arg_types: tuple[Any, ...],
    kwarg_types: dict[str, Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    helpers: Any,
) -> Any | None:
    del kwarg_types
    if not arg_types:
        return None
    dims = helpers.type_dims(arg_types[0])
    if dims is None:
        return None
    dim_token = None
    if len(args) > 1 and isinstance(args[1], AxonExpr):
        dim_token = helpers.expr_to_dim_token(args[1])
    elif "dim" in kwargs and isinstance(kwargs["dim"], AxonExpr):
        dim_token = helpers.expr_to_dim_token(kwargs["dim"])
    axis = -1 if dim_token is None else dim_token
    if isinstance(axis, int):
        if axis < 0:
            axis += len(dims)
        if 0 <= axis < len(dims):
            sizes_expr = None
            if len(args) > 2:
                sizes_expr = args[2]
            elif "sizes" in kwargs:
                sizes_expr = kwargs["sizes"]
            if isinstance(sizes_expr, AxonExprList):
                part_dims = []
                for item in sizes_expr.items:
                    token = helpers.expr_to_dim_token(item)
                    if token is None:
                        break
                    out_dims = list(dims)
                    out_dims[axis] = token
                    part_dims.append(tuple(out_dims))
                else:
                    return TypeTuple(
                        items=tuple(helpers.type_tensor(dims=item) for item in part_dims)
                    )
    return TypeList(item=helpers.type_tensor(dims=dims))

__all__ = [
    "OP_NAME",
    "LOWERING_TYPE_SIGNATURE",
    "type_rule",
]
