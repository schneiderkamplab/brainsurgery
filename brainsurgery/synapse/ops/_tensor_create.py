from __future__ import annotations

from typing import Any


from ..axon.ast import AxonExprAscribe, AxonExprName, AxonExprParen


def shape_dim_tokens(shape_expr: Any, helpers: Any) -> tuple[Any, ...] | None:
    current = shape_expr
    while True:
        while isinstance(current, AxonExprAscribe | AxonExprParen):
            current = current.expr if isinstance(current, AxonExprAscribe) else current.inner
        if isinstance(current, AxonExprName):
            resolved = helpers.resolve_name_expr(current.name)
            if resolved is None:
                return None
            current = resolved
            continue
        break
    items = getattr(current, "items", None)
    if items is None and getattr(getattr(current, "op", None), "name", None) == "core.list":
        items = getattr(current, "inputs", None)
    if not isinstance(items, tuple):
        return None
    dims: list[Any] = []
    for item in items:
        token = helpers.expr_to_dim_token(item)
        if token is None:
            return None
        dims.append(token)
    return tuple(dims)


def type_from_shape_args(args: tuple[Any, ...], helpers: Any) -> Any | None:
    if len(args) < 2:
        return None
    dims = shape_dim_tokens(args[1], helpers)
    if dims is None:
        return None
    return helpers.type_tensor(dims=dims)


__all__ = [
    "type_from_shape_args",
]
