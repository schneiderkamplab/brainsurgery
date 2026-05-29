from __future__ import annotations

from typing import Any


OP_NAME = "assign_unit_slice"


LOWERING_TYPE_SIGNATURE = {
    "args": ("Tensor[..S]", "Int", "Int", "Tensor[..R]"),
    "kwargs": {},
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
    del kwarg_types, args, kwargs, helpers
    if not arg_types:
        return None
    return arg_types[0]


__all__ = [
    "OP_NAME",
    "LOWERING_TYPE_SIGNATURE",
    "type_rule",
]
