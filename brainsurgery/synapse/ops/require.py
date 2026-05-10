from __future__ import annotations

from typing import Any

from ..axon.ast import TypeOptional

OP_NAME = "require"


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any",),
    "kwargs": {},
    "returns": ("Any",),
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
    arg_tp = arg_types[0]
    if isinstance(arg_tp, TypeOptional):
        return arg_tp.inner
    return arg_tp


__all__ = [
    "OP_NAME",
    "LOWERING_TYPE_SIGNATURE",
    "type_rule",
]
