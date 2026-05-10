from __future__ import annotations

from typing import Any


from ..axon.ast import AxonExprAscribe, AxonExprName, AxonExprParen

OP_NAME = "reshape"


LOWERING_TYPE_SIGNATURE = {
    "args": ("Tensor[..S]", "List[Dim]"),
    "kwargs": {},
    "returns": ("Tensor[..R]",),
}


def _shape_dim_tokens(shape_expr: Any, helpers: Any) -> tuple[Any, ...] | None:
    current = shape_expr
    while True:
        while isinstance(current, AxonExprAscribe | AxonExprParen):
            if isinstance(current, AxonExprAscribe):
                current = current.expr
            else:
                current = current.inner
        if isinstance(current, AxonExprName):
            resolved = helpers.resolve_name_expr(current.name)
            if resolved is None:
                return None
            current = resolved
            continue
        break
    items = getattr(current, "items", None)
    if not isinstance(items, tuple):
        return None
    dims: list[Any] = []
    for item in items:
        token = helpers.expr_to_dim_token(item)
        if token is None:
            return None
        dims.append(token)
    return tuple(dims)


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
        return None
    shape_dims = _shape_dim_tokens(args[1], helpers)
    if shape_dims is None:
        return None
    return helpers.type_tensor(dims=shape_dims)


__all__ = [
    "OP_NAME",
    "LOWERING_TYPE_SIGNATURE",
    "type_rule",
]
