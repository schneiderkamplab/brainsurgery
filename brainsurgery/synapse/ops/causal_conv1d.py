from __future__ import annotations

from typing import Any


OP_NAME = "causal_conv1d"


LOWERING_TYPE_SIGNATURE = {
    "args": ("Tensor[B,S,C]", "Tensor[C,1,K]", "Tensor[C]", "String"),
    "kwargs": {},
    "returns": "dynamic",
}
LOWERING_PARAM_NAMES = ("x", "weight", "bias", "activation")
PRIMITIVE_SEMANTICS = {
    "effect": "total_pure",
    "usage": "unrestricted",
}


def type_rule(
    *,
    arg_types: tuple[Any, ...],
    kwarg_types: dict[str, Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    helpers: Any,
) -> Any | None:
    del kwarg_types, args, kwargs
    if len(arg_types) != 4:
        return None
    x_dims = helpers.type_dims(arg_types[0])
    w_dims = helpers.type_dims(arg_types[1])
    b_dims = helpers.type_dims(arg_types[2])
    if x_dims is None or w_dims is None or b_dims is None:
        return None
    if len(x_dims) != 3 or len(w_dims) != 3 or len(b_dims) != 1:
        return None
    if not helpers.dim_equivalent(x_dims[2], w_dims[0]):
        helpers.unify_dim(x_dims[2], w_dims[0])
    if not helpers.dim_equivalent(x_dims[2], b_dims[0]):
        helpers.unify_dim(x_dims[2], b_dims[0])
    return helpers.type_tensor(dims=x_dims)


__all__ = [
    "LOWERING_PARAM_NAMES",
    "LOWERING_TYPE_SIGNATURE",
    "OP_NAME",
    "PRIMITIVE_SEMANTICS",
    "type_rule",
]
