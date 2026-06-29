from __future__ import annotations

from typing import Any


OP_NAME = "expert_linear"


LOWERING_TYPE_SIGNATURE = {
    "args": ("Path", "Tensor[..S,Din]", "IdxTensor[..S]", "?Dim", "?Bool", "?Bool", "?Path", "?Path"),
    "kwargs": {},
    "returns": ("Tensor[..S,dim]",),
}


def type_rule(
    *,
    arg_types: tuple[Any, ...],
    kwarg_types: dict[str, Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    helpers: Any,
) -> Any | None:
    del kwarg_types, kwargs
    if len(arg_types) < 3 or len(args) < 3:
        return None
    input_dims = helpers.type_dims(arg_types[1])
    index_dims = helpers.type_dims(arg_types[2])
    if input_dims is None or index_dims is None or len(input_dims) < 1:
        return None
    if len(input_dims) - 1 != len(index_dims):
        return None
    for input_dim, index_dim in zip(input_dims[:-1], index_dims, strict=True):
        if not helpers.dim_equivalent(input_dim, index_dim):
            return None
    out_dim = helpers.expr_to_dim_token(args[3]) if len(args) >= 4 else None
    if out_dim is None:
        out_dim = input_dims[-1]
    return helpers.type_tensor(dims=(*input_dims[:-1], out_dim))


__all__ = [
    "OP_NAME",
    "LOWERING_TYPE_SIGNATURE",
    "type_rule",
]
