from __future__ import annotations

from typing import Any


from ._broadcast import _normalize_dim_token

OP_NAME = "transpose"


LOWERING_TYPE_SIGNATURE = {
    "args": ("Tensor[..S]", "Any", "Any"),
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
    del kwarg_types, kwargs
    if not arg_types or not args:
        return None
    input_dims = helpers.type_dims(arg_types[0])
    if input_dims is None:
        return None
    rank = len(input_dims)
    if rank == 0:
        return None
    dim1_token = helpers.expr_to_dim_token(args[1]) if len(args) >= 2 else 1
    dim2_token = helpers.expr_to_dim_token(args[2]) if len(args) >= 3 else 2
    if not isinstance(dim1_token, int) or not isinstance(dim2_token, int):
        return None
    dim1 = dim1_token if dim1_token >= 0 else rank + dim1_token
    dim2 = dim2_token if dim2_token >= 0 else rank + dim2_token
    if not (0 <= dim1 < rank and 0 <= dim2 < rank):
        return None
    out_dims = list(input_dims)
    out_dims[dim1], out_dims[dim2] = out_dims[dim2], out_dims[dim1]
    return helpers.type_tensor(dims=tuple(out_dims))


__all__ = [
    "OP_NAME",
    "LOWERING_TYPE_SIGNATURE",
    "type_rule",
]
