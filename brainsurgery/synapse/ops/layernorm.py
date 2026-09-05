from __future__ import annotations

from typing import Any


OP_NAME = "layernorm"


LOWERING_TYPE_SIGNATURE = {
    "args": ("Path", "Tensor[..S]", "?Float", "?Dim", "?Path", "?Bool", "?Path"),
    "kwargs": {},
    "returns": ("Tensor[..S]",),
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
    if len(arg_types) < 2:
        return None
    input_dims = helpers.type_dims(arg_types[1])
    if input_dims is None:
        return None
    dim_expr = args[3] if len(args) >= 4 else None
    dim_token = helpers.expr_to_dim_token(dim_expr)
    if dim_token is not None and input_dims:
        last = input_dims[-1]
        if last != dim_token:
            if isinstance(last, int) and isinstance(dim_token, int):
                raise ValueError(
                    f"Axon typecheck failed: _layernorm dim {dim_token!r} mismatches input last dim {last!r}"
                )
    return helpers.type_tensor(dims=input_dims)


__all__ = [
    "OP_NAME",
    "LOWERING_TYPE_SIGNATURE",
    "type_rule",
]
