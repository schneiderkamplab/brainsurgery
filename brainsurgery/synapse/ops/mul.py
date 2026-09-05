from __future__ import annotations

from typing import Any

from ._broadcast import broadcast_last_dim, broadcast_shape

OP_NAME = "mul"


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
    if left_dims is None and right_dims is None:
        return None
    if left_dims is None:
        return helpers.type_tensor(dims=right_dims)
    if right_dims is None:
        return helpers.type_tensor(dims=left_dims)
    out_dims = (
        helpers.broadcast_tensor_dims(left_dims, right_dims)
        if hasattr(helpers, "broadcast_tensor_dims")
        else broadcast_shape(left_dims, right_dims)
    )
    if out_dims is None:
        return None
    return helpers.type_tensor(dims=out_dims)


__all__ = [
    "OP_NAME",
    "LOWERING_TYPE_SIGNATURE",
    "type_rule",
]
