from __future__ import annotations

from typing import Any


OP_NAME = "mamba_scan"


LOWERING_TYPE_SIGNATURE = {
    "args": (
        "Tensor[B,S,DM]",
        "Tensor[B,S,DM]",
        "Tensor[B,S,N]",
        "Tensor[B,S,N]",
        "Tensor[DM,N]",
        "Tensor[DM]",
    ),
    "kwargs": {},
    "returns": "dynamic",
}
LOWERING_PARAM_NAMES = ("u", "delta", "b", "c", "a", "d")
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
    if len(arg_types) != 6:
        return None
    u_dims = helpers.type_dims(arg_types[0])
    delta_dims = helpers.type_dims(arg_types[1])
    b_dims = helpers.type_dims(arg_types[2])
    c_dims = helpers.type_dims(arg_types[3])
    a_dims = helpers.type_dims(arg_types[4])
    d_dims = helpers.type_dims(arg_types[5])
    if None in {u_dims, delta_dims, b_dims, c_dims, a_dims, d_dims}:
        return None
    if (
        len(u_dims) != 3
        or len(delta_dims) != 3
        or len(b_dims) != 3
        or len(c_dims) != 3
        or len(a_dims) != 2
        or len(d_dims) != 1
    ):
        return None
    for left, right in (
        (u_dims[0], delta_dims[0]),
        (u_dims[1], delta_dims[1]),
        (u_dims[2], delta_dims[2]),
        (u_dims[0], b_dims[0]),
        (u_dims[1], b_dims[1]),
        (b_dims[0], c_dims[0]),
        (b_dims[1], c_dims[1]),
        (b_dims[2], c_dims[2]),
        (u_dims[2], a_dims[0]),
        (b_dims[2], a_dims[1]),
        (u_dims[2], d_dims[0]),
    ):
        if not helpers.dim_equivalent(left, right):
            helpers.unify_dim(left, right)
    return helpers.type_tensor(dims=u_dims)


__all__ = [
    "LOWERING_PARAM_NAMES",
    "LOWERING_TYPE_SIGNATURE",
    "OP_NAME",
    "PRIMITIVE_SEMANTICS",
    "type_rule",
]
