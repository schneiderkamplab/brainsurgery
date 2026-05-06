from __future__ import annotations

from typing import Any

import torch

OP_NAME = "l2norm"
LOWERING_ARITY = (1, 1)
LOWERING_ALLOWED_KWARGS: set[str] = {"eps"}
LOWERING_REQUIRED_KWARGS: set[str] = set()
LOWERING_KWARG_KINDS: dict[str, Any] = {
    "eps": "number",
}


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return False


def lowering_infer_metadata(
    *,
    args: list[str],
    out: str | list[str],
    kwargs: dict[str, Any],
    ctx: Any,
) -> bool:
    del kwargs
    if not isinstance(out, str):
        return False
    first_in = args[0].strip() if args else None
    if isinstance(first_in, str) and first_in.isidentifier():
        if first_in in ctx.tensor_last_dim:
            ctx.tensor_last_dim[out] = ctx.tensor_last_dim[first_in]
        if first_in in ctx.tensor_shape:
            ctx.tensor_shape[out] = ctx.tensor_shape[first_in]
    return True


def interpret(
    model: Any,
    node_spec: dict[str, Any],
    env: dict[str, Any],
    *,
    node_path: str,
    scope: str,
    symbols: dict[str, int | float | bool],
) -> None:
    del node_path, scope
    x = model._read_tensor_input(node_spec.get("_args"), env)
    eps = float(model._eval_expr(node_spec.get("eps", 1.0e-6), env, symbols))
    out = model._require_name(node_spec.get("_bind"), field="l2norm._bind")
    x_fp32 = x.float()
    env[out] = (x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + eps)).to(
        dtype=x.dtype
    )


def compile(
    emitter: Any,
    node_spec: dict[str, Any],
    env: dict[str, str],
    *,
    node_path_var: str,
    scope_var: str,
    indent: str,
) -> list[str]:
    del node_path_var, scope_var
    src = emitter._read_env_var(env, str(node_spec.get("_args")))
    out_name = str(node_spec.get("_bind"))
    out_var = emitter._assign_out_var(env, out_name)
    eps = emitter._expr_code(node_spec.get("eps", 1.0e-6), env)
    src_fp32 = emitter._fresh("src_fp32")
    lines = [
        f"{indent}{src_fp32} = {src}.float()",
        f"{indent}{out_var} = ({src_fp32} * torch.rsqrt({src_fp32}.pow(2).mean(dim=-1, keepdim=True) + float({eps}))).to(dtype={src}.dtype)",
    ]
    return lines


LOWERING_TYPE_SIGNATURE = {
    "args": ("Tensor[..S]",),
    "kwargs": dict(LOWERING_KWARG_KINDS),
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
    del kwarg_types, args, kwargs
    if len(arg_types) < 1:
        return None
    input_dims = helpers.type_dims(arg_types[0])
    if input_dims is None:
        return None
    return helpers.type_tensor(dims=input_dims)

__all__ = [
    "OP_NAME",
    "LOWERING_ARITY",
    "LOWERING_ALLOWED_KWARGS",
    "LOWERING_REQUIRED_KWARGS",
    "LOWERING_KWARG_KINDS",
    "lowering_infer_metadata",
    "interpret",
    "compile",
    "uses_node_path",
    "LOWERING_TYPE_SIGNATURE",
    "type_rule",
]
