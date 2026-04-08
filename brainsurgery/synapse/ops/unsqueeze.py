from __future__ import annotations

from typing import Any

import torch

OP_NAME = "unsqueeze"
LOWERING_ARITY = (1, 1)
LOWERING_ALLOWED_KWARGS: set[str] = {"dim"}
LOWERING_REQUIRED_KWARGS: set[str] = {"dim"}
LOWERING_KWARG_KINDS: dict[str, Any] = {"dim": "int"}


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return False


def lowering_validate_signature(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> None:
    del args, ctx
    if not isinstance(out, str):
        raise ValueError("unsqueeze requires a single scalar output binding")
    if "dim" not in kwargs:
        raise ValueError("unsqueeze requires dim")


def lowering_infer_metadata(
    *,
    args: list[str],
    out: str | list[str],
    kwargs: dict[str, Any],
    ctx: Any,
) -> bool:
    if not isinstance(out, str) or not args:
        return False
    dim = kwargs.get("dim")
    if not isinstance(dim, int):
        return False
    source_name = str(args[0]).strip()
    source_shape = ctx.tensor_shape.get(source_name)
    if isinstance(source_shape, tuple):
        rank = len(source_shape)
        target_dim = dim if dim >= 0 else rank + 1 + dim
        if target_dim < 0 or target_dim > rank:
            return False
        new_shape = source_shape[:target_dim] + (1,) + source_shape[target_dim:]
        ctx.tensor_shape[out] = new_shape
        ctx.tensor_last_dim[out] = new_shape[-1]
        return True
    first_dim = ctx.tensor_last_dim.get(source_name)
    if first_dim is not None:
        if dim in {-1, 1}:
            ctx.tensor_last_dim[out] = 1 if dim == -1 else first_dim
            return True
        ctx.tensor_last_dim[out] = first_dim
        return True
    return False


def interpret(
    model: Any,
    node_spec: dict[str, Any],
    env: dict[str, Any],
    *,
    node_path: str,
    scope: str,
    symbols: dict[str, int],
) -> None:
    del node_path, scope, symbols
    src = model._read_tensor_input(node_spec.get("_args"), env)
    raw_dim = node_spec.get("dim")
    if isinstance(raw_dim, bool) or not isinstance(raw_dim, int):
        raise ValueError("unsqueeze.dim must be int")
    out = model._require_name(node_spec.get("_bind"), field="unsqueeze._bind")
    env[out] = torch.unsqueeze(src, int(raw_dim))


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
    out_var = emitter._assign_out_var(env, str(node_spec.get("_bind")))
    raw_dim = node_spec.get("dim")
    if isinstance(raw_dim, bool) or not isinstance(raw_dim, int):
        raise ValueError("unsqueeze.dim must be int")
    return [f"{indent}{out_var} = torch.unsqueeze({src}, {int(raw_dim)})"]


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any",),
    "kwargs": dict(LOWERING_KWARG_KINDS),
    "returns": ("Tensor",),
}

__all__ = [
    "LOWERING_ARITY",
    "LOWERING_ALLOWED_KWARGS",
    "LOWERING_REQUIRED_KWARGS",
    "LOWERING_KWARG_KINDS",
    "OP_NAME",
    "lowering_validate_signature",
    "lowering_infer_metadata",
    "interpret",
    "compile",
    "uses_node_path",
    "LOWERING_TYPE_SIGNATURE",
]
