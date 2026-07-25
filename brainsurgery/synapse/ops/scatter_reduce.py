from __future__ import annotations

from typing import Any

from ..axon.ast import AxonExprAscribe, AxonExprInt, AxonExprParen, AxonExprString

OP_NAME = "scatter_reduce"

SUPPORTED_REDUCTIONS = frozenset({"sum", "prod", "mean", "max", "min", "amax", "amin"})

LOWERING_TYPE_SIGNATURE = {
    "args": ("Tensor[..S]", "IdxTensor[..I]", "Tensor[..I]", "Int", "String", "Bool"),
    "kwargs": {},
    "returns": ("Tensor[..S]",),
}

LOWERING_PARAM_NAMES = ("x", "index", "src", "dim", "reduce", "include_self")


def _unwrap(expr: Any) -> Any:
    while isinstance(expr, AxonExprAscribe | AxonExprParen):
        expr = expr.expr if isinstance(expr, AxonExprAscribe) else expr.inner
    return expr


def _literal_int(expr: Any) -> int | None:
    expr = _unwrap(expr)
    if isinstance(expr, AxonExprInt):
        return expr.value
    value = getattr(expr, "value", None)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _literal_string(expr: Any) -> str | None:
    expr = _unwrap(expr)
    if isinstance(expr, AxonExprString):
        return expr.value
    value = getattr(expr, "value", None)
    return value if isinstance(value, str) else None


def type_rule(
    *,
    arg_types: tuple[Any, ...],
    kwarg_types: dict[str, Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    helpers: Any,
) -> Any | None:
    del kwarg_types, kwargs
    if len(arg_types) < 3:
        return None
    x_dims = helpers.type_dims(arg_types[0])
    index_dims = helpers.type_dims(arg_types[1])
    src_dims = helpers.type_dims(arg_types[2])
    if x_dims is None:
        return None
    rank = len(x_dims)
    for label, dims in (("index", index_dims), ("source", src_dims)):
        if dims is not None and len(dims) != rank:
            raise ValueError(
                f"scatter_reduce {label} rank {len(dims)} does not match input rank {rank}"
            )
    if len(args) >= 4 and (dim := _literal_int(args[3])) is not None:
        if dim < -rank or dim >= rank:
            raise ValueError(
                f"scatter_reduce dimension {dim} is out of range for rank-{rank} tensor"
            )
    if len(args) >= 5 and (reduction := _literal_string(args[4])) is not None:
        if reduction not in SUPPORTED_REDUCTIONS:
            supported = ", ".join(sorted(SUPPORTED_REDUCTIONS))
            raise ValueError(
                f"unsupported scatter_reduce reduction {reduction!r}; expected one of: {supported}"
            )
    return helpers.type_tensor(dims=tuple(x_dims))


__all__ = [
    "OP_NAME",
    "SUPPORTED_REDUCTIONS",
    "LOWERING_TYPE_SIGNATURE",
    "LOWERING_PARAM_NAMES",
    "type_rule",
]
