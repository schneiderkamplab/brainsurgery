from __future__ import annotations

from typing import Any

import torch

OP_NAME = "cache_update"
LOWERING_ARITY = (3, 3)
LOWERING_ALLOWED_KWARGS: set[str] = set()
LOWERING_REQUIRED_KWARGS: set[str] = set()
LOWERING_KWARG_KINDS: dict[str, Any] = {}


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return False


def lowering_known_output_arity(*, kwargs: dict[str, Any]) -> int:
    del kwargs
    return 3


def lowering_validate_signature(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> None:
    del args, kwargs, ctx
    if not isinstance(out, list) or len(out) != 3:
        raise ValueError("cache_update requires exactly three outputs: k_ctx, v_ctx, present")


def lowering_infer_metadata(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> bool:
    del kwargs
    if not isinstance(out, list) or len(out) != 3 or len(args) < 3:
        return False
    k_source = str(args[1]).strip()
    v_source = str(args[2]).strip()
    if k_source in ctx.tensor_last_dim:
        ctx.tensor_last_dim[out[0]] = ctx.tensor_last_dim[k_source]
    if v_source in ctx.tensor_last_dim:
        ctx.tensor_last_dim[out[1]] = ctx.tensor_last_dim[v_source]
    if k_source in ctx.tensor_shape:
        ctx.tensor_shape[out[0]] = ctx.tensor_shape[k_source]
    if v_source in ctx.tensor_shape:
        ctx.tensor_shape[out[1]] = ctx.tensor_shape[v_source]
    return True


def interpret(
    model: Any,
    node_spec: dict[str, Any],
    env: dict[str, Any],
    *,
    node_path: str,
    scope: str,
    symbols: dict[str, int],
) -> None:
    ins = node_spec.get("_args")
    outs = node_spec.get("_bind")
    if not isinstance(ins, list) or len(ins) != 3 or not isinstance(outs, list) or len(outs) != 3:
        raise ValueError("cache_update expects in=[past,k,v], out=[k_ctx,v_ctx,present]")
    past = env.get(ins[0])
    k_new = env[ins[1]]
    v_new = env[ins[2]]
    if past is None:
        k_all = k_new
        v_all = v_new
    else:
        k_all = torch.cat([past[0], k_new], dim=-2)
        v_all = torch.cat([past[1], v_new], dim=-2)
    present = (k_all, v_all)
    env[outs[0]] = k_all
    env[outs[1]] = v_all
    env[outs[2]] = present
    return


def compile(
    emitter: Any,
    node_spec: dict[str, Any],
    env: dict[str, str],
    *,
    node_path_var: str,
    scope_var: str,
    indent: str,
) -> list[str]:
    lines: list[str] = []

    def assign_out_var(out_name: str) -> str:
        return emitter._assign_out_var(env, out_name)

    def read(name: str) -> str:
        return emitter._read_env_var(env, name)

    ins = node_spec.get("_args")
    outs = node_spec.get("_bind")
    if not isinstance(ins, list) or len(ins) != 3 or not isinstance(outs, list) or len(outs) != 3:
        raise ValueError("cache_update expects in=[past,k,v], out=[k_ctx,v_ctx,present]")
    past = read(str(ins[0])) if str(ins[0]) in env else "None"
    k_new = read(str(ins[1]))
    v_new = read(str(ins[2]))
    k_ctx = assign_out_var(str(outs[0]))
    v_ctx = assign_out_var(str(outs[1]))
    present = assign_out_var(str(outs[2]))
    lines.append(f"{indent}if {past} is None:")
    lines.append(f"{indent}    {k_ctx} = {k_new}")
    lines.append(f"{indent}    {v_ctx} = {v_new}")
    lines.append(f"{indent}else:")
    lines.append(f"{indent}    {k_ctx} = torch.cat([{past}[0], {k_new}], dim=-2)")
    lines.append(f"{indent}    {v_ctx} = torch.cat([{past}[1], {v_new}], dim=-2)")
    lines.append(f"{indent}{present} = ({k_ctx}, {v_ctx})")
    return lines


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any", "Any", "Any"),
    "kwargs": dict(LOWERING_KWARG_KINDS),
    "returns": ("Tensor", "Tensor", "Tensor"),
}

__all__ = [
    "LOWERING_ARITY",
    "LOWERING_ALLOWED_KWARGS",
    "LOWERING_REQUIRED_KWARGS",
    "LOWERING_KWARG_KINDS",
    "OP_NAME",
    "lowering_known_output_arity",
    "lowering_validate_signature",
    "lowering_infer_metadata",
    "interpret",
    "compile",
    "uses_node_path",
    "LOWERING_TYPE_SIGNATURE",
]
