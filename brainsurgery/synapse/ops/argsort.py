from __future__ import annotations

from typing import Any

from ..axon.ast import AxonExprAscribe, AxonExprInt, AxonExprParen

OP_NAME = "argsort"


LOWERING_TYPE_SIGNATURE = {
    "args": ("Tensor[..S]", "Int", "Bool", "Bool"),
    "kwargs": {},
    "returns": ("IdxTensor[..S]",),
}

LOWERING_PARAM_NAMES = ("x", "dim", "descending", "stable")


def _literal_int(expr: Any) -> int | None:
    while isinstance(expr, AxonExprAscribe | AxonExprParen):
        expr = expr.expr if isinstance(expr, AxonExprAscribe) else expr.inner
    if isinstance(expr, AxonExprInt):
        return expr.value
    value = getattr(expr, "value", None)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def type_rule(
    *,
    arg_types: tuple[Any, ...],
    kwarg_types: dict[str, Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    helpers: Any,
) -> Any | None:
    del kwarg_types, kwargs
    if not arg_types:
        return None
    dims = helpers.type_dims(arg_types[0])
    if dims is None:
        return None
    if len(args) >= 2 and (dim := _literal_int(args[1])) is not None:
        rank = len(dims)
        if dim < -rank or dim >= rank:
            raise ValueError(f"argsort dimension {dim} is out of range for rank-{rank} tensor")
    return helpers.type_tensor(dims=tuple(dims))


__all__ = [
    "OP_NAME",
    "LOWERING_TYPE_SIGNATURE",
    "LOWERING_PARAM_NAMES",
    "type_rule",
]
