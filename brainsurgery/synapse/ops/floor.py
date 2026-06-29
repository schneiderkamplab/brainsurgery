from __future__ import annotations

import math
from typing import Any

from brainsurgery.synapse.axon.ast import TypeInt


OP_NAME = "floor"


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any",),
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
    input_dims = helpers.type_dims(arg_types[0])
    if input_dims is not None:
        return helpers.type_tensor(dims=input_dims)
    return TypeInt()


__all__ = [
    "OP_NAME",
    "LOWERING_TYPE_SIGNATURE",
    "type_rule",
]
