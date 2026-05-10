from __future__ import annotations

from typing import Any


from ._broadcast import broadcast_shape

OP_NAME = "matmul"


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any", "Any"),
    "kwargs": {},
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
    del kwarg_types, args, kwargs
    if len(arg_types) != 2:
        return None
    left_dims = helpers.type_dims(arg_types[0])
    right_dims = helpers.type_dims(arg_types[1])
    if left_dims is None or right_dims is None:
        return None
    if len(left_dims) < 2 or len(right_dims) < 2:
        return None
    batch = broadcast_shape(left_dims[:-2], right_dims[:-2])
    if batch is None:
        return None
    if left_dims[-1] != right_dims[-2]:
        return None
    return helpers.type_tensor(dims=(*batch, left_dims[-2], right_dims[-1]))


__all__ = [
    "OP_NAME",
    "LOWERING_TYPE_SIGNATURE",
    "type_rule",
]
