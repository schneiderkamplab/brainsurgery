from __future__ import annotations

from typing import Any

from ..axon.ast import TypeList, TypeOptional

OP_NAME = "list_index"


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any", "Int"),
    "kwargs": {},
    "returns": ("Any",),
}

PRIMITIVE_SEMANTICS = {
    # Indexing can fail for null/non-list/out-of-range inputs. Wrappers such as
    # Cache.index guard those cases with lazy selects, so optimization must not
    # hoist the primitive out of the guarded branch.
    "effect": "partial_pure",
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
    if len(arg_types) < 1:
        return None
    collection_tp = arg_types[0]
    if isinstance(collection_tp, TypeOptional):
        collection_tp = collection_tp.inner
    if isinstance(collection_tp, TypeList):
        return collection_tp.item
    return None


__all__ = [
    "OP_NAME",
    "LOWERING_TYPE_SIGNATURE",
    "PRIMITIVE_SEMANTICS",
    "type_rule",
]
