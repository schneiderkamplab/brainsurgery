from __future__ import annotations

from typing import Any

from ..axon.ast import TypeList, TypeOptional

OP_NAME = "list_append"


LOWERING_TYPE_SIGNATURE = {
    "args": ("List[_T]", "_T"),
    "kwargs": {},
    "returns": ("List[_T]",),
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
    values_type = arg_types[0]
    if isinstance(values_type, TypeOptional):
        values_type = values_type.inner
    if isinstance(values_type, TypeList):
        return values_type
    return None


__all__ = [
    "OP_NAME",
    "LOWERING_TYPE_SIGNATURE",
    "type_rule",
]
