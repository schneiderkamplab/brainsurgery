from __future__ import annotations

from typing import Any


OP_NAME = "fill"


LOWERING_TYPE_SIGNATURE = {
    "args": ("Tensor[..S]", "Any", "Any"),
    "kwargs": {},
    "returns": ("Tensor[..S]",),
}

PRIMITIVE_SEMANTICS = {
    "usage": "affine",
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
    "PRIMITIVE_SEMANTICS",
    "type_rule",
]
