from __future__ import annotations

from typing import Any


OP_NAME = "linear"


LOWERING_TYPE_SIGNATURE = {
    "args": ("Path", "Tensor[..S,Din]", "?Dim", "?Bool", "?Bool", "?Int", "?Path", "?Path"),
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
    if len(arg_types) < 2 or len(args) < 2:
        return None
    input_dims = helpers.type_dims(arg_types[1])
    if input_dims is None or len(input_dims) < 1:
        return None
    out_dim = helpers.expr_to_dim_token(args[2]) if len(args) >= 3 else None
    if out_dim is None:
        out_dim = input_dims[-1]
    return helpers.type_tensor(dims=(*input_dims[:-1], out_dim))


__all__ = [
    "OP_NAME",
    "LOWERING_TYPE_SIGNATURE",
    "type_rule",
]
