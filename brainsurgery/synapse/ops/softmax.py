from __future__ import annotations

from typing import Any


OP_NAME = "softmax"
_SUPPORTED_DTYPES: set[str] = {"float32", "float16", "bfloat16"}


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any", "Any", "Any"),
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
    if len(arg_types) != 3:
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
