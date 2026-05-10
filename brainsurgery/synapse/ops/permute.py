from __future__ import annotations

from typing import Any


OP_NAME = "permute"


LOWERING_TYPE_SIGNATURE = {
    "args": ("Tensor[..S]", "Any"),
    "kwargs": {},
    "returns": ("Tensor[..R]",),
}


def _permute_indices(expr: Any, *, rank: int, helpers: Any) -> tuple[int, ...] | None:
    while True:
        resolved = None
        name = getattr(expr, "name", None)
        if isinstance(name, str):
            resolved = helpers.resolve_name_expr(name)
        if resolved is None:
            break
        expr = resolved
    items = getattr(expr, "items", None)
    if not isinstance(items, tuple):
        return None
    raw: list[int] = []
    for item in items:
        token = helpers.expr_to_dim_token(item)
        if not isinstance(token, int):
            return None
        raw.append(token)
    if len(raw) != rank:
        return None
    normalized = tuple(idx if idx >= 0 else rank + idx for idx in raw)
    if any(idx < 0 or idx >= rank for idx in normalized):
        return None
    if len(set(normalized)) != rank:
        return None
    return normalized


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
    input_dims = helpers.type_dims(arg_types[0])
    if input_dims is None:
        return None
    order = _permute_indices(args[1], rank=len(input_dims), helpers=helpers)
    if order is None:
        return None
    return helpers.type_tensor(dims=tuple(input_dims[idx] for idx in order))


__all__ = [
    "OP_NAME",
    "LOWERING_TYPE_SIGNATURE",
    "type_rule",
]
