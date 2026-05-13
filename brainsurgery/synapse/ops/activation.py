from __future__ import annotations

from typing import Any


OP_NAME = "activation"


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any",),
    "kwargs": {},
    "returns": "dynamic",
}
LOWERING_PARAM_NAMES_BY_OP = {
    "activations_gegelu": ("x", "limit"),
    "activations_gelu": ("x",),
    "activations_gelu_new": ("x",),
    "activations_gelu_pytorch_tanh": ("x",),
    "activations_relu": ("x",),
    "activations_relu2": ("x",),
    "activations_sigmoid": ("x",),
    "activations_silu": ("x",),
    "activations_swiglu": ("x",),
    "activations_tanh": ("x",),
    "activations_xielu": ("x", "alpha_p", "alpha_n", "beta", "eps"),
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
    if len(arg_types) < 1:
        return None
    input_dims = helpers.type_dims(arg_types[0])
    if input_dims is None:
        return None
    return helpers.type_tensor(dims=input_dims)


__all__ = [
    "OP_NAME",
    "LOWERING_TYPE_SIGNATURE",
    "LOWERING_PARAM_NAMES_BY_OP",
    "type_rule",
]
