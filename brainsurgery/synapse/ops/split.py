from __future__ import annotations

from typing import Any

from ..axon.ast import AxonExpr, AxonExprAscribe, AxonExprList, AxonExprParen, TypeList, TypeTuple

OP_NAME = "split"


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any",),
    "kwargs": {'dim': 'int', 'sizes': 'list_dim'},
    "returns": "dynamic",
}
LOWERING_PARAM_NAMES = ("x", "dim", "sizes")
LOWERING_PARAM_DEFAULTS = {"dim": -1, "sizes": None}


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
    dim_expr = args[1] if len(args) > 1 else kwargs.get("dim")
    dim_token = _expr_dim_token(dim_expr, helpers) if dim_expr is not None else None
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
            size_items = _list_items(sizes_expr)
            if size_items is not None:
                part_dims = []
                for item in size_items:
                    token = _expr_dim_token(item, helpers)
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


def _expr_dim_token(expr: Any, helpers: Any) -> Any | None:
    while isinstance(expr, AxonExprAscribe | AxonExprParen):
        expr = expr.expr if isinstance(expr, AxonExprAscribe) else expr.inner
    return helpers.expr_to_dim_token(expr)


def _list_items(expr: Any) -> tuple[Any, ...] | None:
    if isinstance(expr, AxonExprList):
        return tuple(expr.items)
    if getattr(getattr(expr, "op", None), "name", None) == "core.list":
        return tuple(getattr(expr, "inputs", ()))
    return None

__all__ = [
    "OP_NAME",
    "LOWERING_TYPE_SIGNATURE",
    "LOWERING_PARAM_DEFAULTS",
    "LOWERING_PARAM_NAMES",
    "type_rule",
]
