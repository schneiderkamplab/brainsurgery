from __future__ import annotations

from typing import Any

from ..axon.ast import TypeTensor

OP_NAME = "acos"


LOWERING_TYPE_SIGNATURE = {
    "args": ("Tensor[..S]",),
    "kwargs": {},
    "returns": ("Tensor[..S]",),
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
    del kwarg_types, args, kwargs, helpers
    if len(arg_types) != 1 or not isinstance(arg_types[0], TypeTensor):
        return None
    return arg_types[0]


__all__ = [
    "OP_NAME",
    "LOWERING_TYPE_SIGNATURE",
    "LOWERING_PARAM_NAMES",
    "type_rule",
]
