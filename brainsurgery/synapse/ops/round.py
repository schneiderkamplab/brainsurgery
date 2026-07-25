from __future__ import annotations

from typing import Any

from ..axon.ast import TypeAny, TypeDim, TypeFloat, TypeInt, TypeTensor

OP_NAME = "round"


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any",),
    "kwargs": {},
    "returns": "dynamic",
}

LOWERING_PARAM_NAMES = ("x",)


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
    input_type = arg_types[0]
    if isinstance(input_type, TypeTensor):
        return input_type
    if isinstance(input_type, TypeAny):
        return TypeAny()
    if isinstance(input_type, TypeFloat | TypeInt | TypeDim):
        return TypeInt()
    raise ValueError("round expects a numeric scalar or tensor")


__all__ = [
    "OP_NAME",
    "LOWERING_TYPE_SIGNATURE",
    "LOWERING_PARAM_NAMES",
    "type_rule",
]
