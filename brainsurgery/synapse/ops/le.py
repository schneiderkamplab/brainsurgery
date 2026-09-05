from __future__ import annotations

from typing import Any

from ._broadcast import broadcast_last_dim, broadcast_shape

OP_NAME = "le"


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any", "Any"),
    "kwargs": {},
    "returns": "Tensor[..R]",
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
    dims: tuple[Any, ...] | None = None
    for arg_type in arg_types:
        arg_dims = helpers.type_dims(arg_type)
        if arg_dims is None:
            continue
        if dims is None:
            dims = arg_dims
            continue
        dims = (
            helpers.broadcast_tensor_dims(dims, arg_dims)
            if hasattr(helpers, "broadcast_tensor_dims")
            else broadcast_shape(dims, arg_dims)
        )
        if dims is None:
            return None
    if dims is None:
        return None
    return helpers.type_tensor(dims=dims)


__all__ = [
    "OP_NAME",
    "LOWERING_TYPE_SIGNATURE",
    "type_rule",
]
