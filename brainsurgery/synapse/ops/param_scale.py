from __future__ import annotations

from typing import Any

import torch

OP_NAME = "param_scale"
LOWERING_ARITY = (1, 1)
LOWERING_ALLOWED_KWARGS: set[str] = {"scale"}
LOWERING_REQUIRED_KWARGS: set[str] = set()
LOWERING_KWARG_KINDS: dict[str, Any] = {"scale": "param"}


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return True


def lowering_validate_signature(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> None:
    del args, kwargs, ctx
    if not isinstance(out, str):
        raise ValueError("param_scale requires a single output binding")


def lowering_infer_metadata(
    *,
    args: list[str],
    out: str | list[str],
    kwargs: dict[str, Any],
    ctx: Any,
) -> bool:
    del kwargs
    if not isinstance(out, str) or not args:
        return False
    src = args[0].strip()
    if isinstance(src, str) and src.isidentifier() and src in ctx.tensor_last_dim:
        ctx.tensor_last_dim[out] = ctx.tensor_last_dim[src]
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
    del scope, symbols
    src_name = node_spec.get("_args")
    if not isinstance(src_name, str):
        raise ValueError("param_scale expects a single tensor input")
    out_name = model._require_name(node_spec.get("_bind"), field="param_scale._bind")
    x = model._read_tensor_input(src_name, env)
    scale_path = model._infer_param_path(node_spec, node_path=node_path, param_name="scale")
    scale = model._state[scale_path]
    if not torch.is_tensor(scale):
        raise ValueError(f"param_scale parameter is not a tensor: {scale_path}")
    env[out_name] = x * scale.to(device=x.device, dtype=x.dtype)


def compile(
    emitter: Any,
    node_spec: dict[str, Any],
    env: dict[str, str],
    *,
    node_path_var: str,
    scope_var: str,
    indent: str,
) -> list[str]:
    del scope_var
    lines: list[str] = []
    src_name = node_spec.get("_args")
    if not isinstance(src_name, str):
        raise ValueError("param_scale expects a single tensor input")
    src = emitter._read_env_var(env, src_name)
    scale = emitter._hoisted_param(
        node_spec=node_spec,
        node_path_var=node_path_var,
        param_name="scale",
        lines=lines,
        indent=indent,
    )
    out_name = str(node_spec.get("_bind"))
    out_var = emitter._assign_out_var(env, out_name)
    lines.append(f"{indent}{out_var} = {src} * {scale}.to(device={src}.device, dtype={src}.dtype)")
    return lines


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any",),
    "kwargs": dict(LOWERING_KWARG_KINDS),
    "returns": "dynamic",
}

__all__ = [
    "LOWERING_ARITY",
    "LOWERING_ALLOWED_KWARGS",
    "LOWERING_REQUIRED_KWARGS",
    "LOWERING_KWARG_KINDS",
    "LOWERING_TYPE_SIGNATURE",
    "OP_NAME",
    "compile",
    "interpret",
    "lowering_infer_metadata",
    "lowering_validate_signature",
    "uses_node_path",
]
