from __future__ import annotations

from typing import Any


OP_NAME = "tensor_like"


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any", "Tensor[..S]", "Any"),
    "kwargs": {},
    "returns": "dynamic",
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
    if len(arg_types) < 2:
        return None
    ref_dims = helpers.type_dims(arg_types[1])
    if ref_dims is None:
        return None
    return helpers.type_tensor(dims=ref_dims)


__all__ = [
    "OP_NAME",
    "LOWERING_TYPE_SIGNATURE",
    "PRIMITIVE_SEMANTICS",
    "type_rule",
]
