from __future__ import annotations

from typing import Any


OP_NAME = "l2norm"


LOWERING_TYPE_SIGNATURE = {
    "args": ("Tensor[..S]",),
    "kwargs": {'eps': 'number'},
    "returns": ("Tensor[..S]",),
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
    if len(arg_types) < 1:
        return None
    input_dims = helpers.type_dims(arg_types[0])
    if input_dims is None:
        return None
    return helpers.type_tensor(dims=input_dims)

__all__ = [
    "OP_NAME",
    "LOWERING_TYPE_SIGNATURE",
    "type_rule",
]
