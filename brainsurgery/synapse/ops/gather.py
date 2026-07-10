from __future__ import annotations

from typing import Any


from ..axon.ast import TypeTensor

OP_NAME = "gather"


LOWERING_TYPE_SIGNATURE = {
    "args": ("Tensor[..S]", "IdxTensor[..I]", "Any"),
    "kwargs": {},
    "returns": ("Tensor[..R]",),
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
    if len(arg_types) < 2:
        return None
    source = arg_types[0]
    index = arg_types[1]
    if not isinstance(source, TypeTensor) or not isinstance(index, TypeTensor):
        return None
    return TypeTensor(base=source.base, dims=index.dims)


__all__ = [
    "OP_NAME",
    "LOWERING_TYPE_SIGNATURE",
    "type_rule",
]
