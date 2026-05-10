from __future__ import annotations

from typing import Any


OP_NAME = "tensor_like"


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any", "Tensor[..S]", "Any"),
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
    if not arg_types:
        return None
    value_dims = helpers.type_dims(arg_types[0])
    if value_dims is not None:
        return helpers.type_tensor(dims=value_dims)
    return helpers.type_tensor(dims=())


__all__ = [
    "OP_NAME",
    "LOWERING_TYPE_SIGNATURE",
    "type_rule",
]
