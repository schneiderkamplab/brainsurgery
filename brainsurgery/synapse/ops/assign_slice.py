from __future__ import annotations

from typing import Any


OP_NAME = "assign_slice"


LOWERING_TYPE_SIGNATURE = {
    "args": ("Tensor[..S]", "Tensor[..R]", "Int", "Dim", "Dim"),
    "kwargs": {},
    "returns": ("Tensor[..S]",),
}

PRIMITIVE_SEMANTICS = {
    # Functional slice update: return a tensor equal to x with the selected
    # slice replaced by src. Backend codegen may introduce affine in-place
    # fast paths only when ownership has been proven for the specific site.
    "effect": "total_pure",
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
    "PRIMITIVE_SEMANTICS",
    "type_rule",
]
