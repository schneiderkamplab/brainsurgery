from __future__ import annotations

from typing import Any


OP_NAME = "embedding"


LOWERING_TYPE_SIGNATURE = {
    "args": ("Path", "Tensor[..S]", "?Dim"),
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
    del kwarg_types
    if len(arg_types) < 2:
        return None
    input_dims = helpers.type_dims(arg_types[1])
    if input_dims is None:
        return None
    dim_expr = args[2] if len(args) >= 3 else kwargs.get("dim")
    dim_token = helpers.expr_to_dim_token(dim_expr)
    if dim_token is None:
        return None
    return helpers.type_tensor(dims=(*input_dims, dim_token))


__all__ = [
    "OP_NAME",
    "LOWERING_TYPE_SIGNATURE",
    "type_rule",
]
