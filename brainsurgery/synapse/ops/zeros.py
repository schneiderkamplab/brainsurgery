from __future__ import annotations

from typing import Any


from ._tensor_create import (
    type_from_shape_args,
)

OP_NAME = "zeros"


LOWERING_TYPE_SIGNATURE = {
    "args": ("Tensor[..REF]", "List[Dim]", "Any"),
    "kwargs": {},
    "returns": "dynamic",
}

PRIMITIVE_SEMANTICS = {
    # Fresh tensors are pure values, but their storage identity must not be
    # shared by CSE/hoisting. Backend ownership rewrites may update them
    # in-place after proving uniqueness.
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
    del arg_types, kwarg_types, kwargs
    return type_from_shape_args(args, helpers)


__all__ = [
    "OP_NAME",
    "LOWERING_TYPE_SIGNATURE",
    "PRIMITIVE_SEMANTICS",
    "type_rule",
]
