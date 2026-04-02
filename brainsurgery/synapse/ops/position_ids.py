from __future__ import annotations

from typing import Any

import torch

OP_NAME = "position_ids"
LOWERING_ARITY = (2, 2)
LOWERING_ALLOWED_KWARGS: set[str] = {"past_length", "pad_fill"}
LOWERING_REQUIRED_KWARGS: set[str] = set()
LOWERING_KWARG_KINDS: dict[str, Any] = {"past_length": "dim", "pad_fill": "dim"}


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return False


def lowering_validate_signature(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> None:
    del args, kwargs, ctx
    if not isinstance(out, str):
        raise ValueError("position_ids requires a single scalar output binding")


def lowering_infer_metadata(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> bool:
    del kwargs
    if not isinstance(out, str) or not args:
        return False
    source_name = str(args[0]).strip()
    source_shape = ctx.tensor_shape.get(source_name)
    if isinstance(source_shape, tuple) and len(source_shape) == 2:
        ctx.tensor_shape[out] = source_shape
        ctx.tensor_last_dim[out] = source_shape[-1]
        return True
    if source_name in ctx.tensor_last_dim:
        ctx.tensor_last_dim[out] = ctx.tensor_last_dim[source_name]
        return True
    return False


def _resolve_past_length(
    model: Any, node_spec: dict[str, Any], env: dict[str, Any], symbols: dict[str, int]
) -> int:
    raw = node_spec.get("past_length", 0)
    value = model._eval_expr(raw, env, symbols)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("position_ids.past_length must resolve to non-negative int")
    if value < 0:
        raise ValueError("position_ids.past_length must be >= 0")
    return int(value)


def _resolve_pad_fill(
    model: Any, node_spec: dict[str, Any], env: dict[str, Any], symbols: dict[str, int]
) -> int:
    raw = node_spec.get("pad_fill", 0)
    value = model._eval_expr(raw, env, symbols)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("position_ids.pad_fill must resolve to int")
    return int(value)


def interpret(
    model: Any,
    node_spec: dict[str, Any],
    env: dict[str, Any],
    *,
    node_path: str,
    scope: str,
    symbols: dict[str, int],
) -> None:
    del node_path, scope
    args = node_spec.get("_args")
    if not isinstance(args, list) or len(args) != 2:
        raise ValueError("position_ids expects _args as [input_ids, attn_mask]")
    x = model._read_tensor_input(args[0], env)
    if x.ndim != 2:
        raise ValueError("position_ids._args must resolve to rank-2 [batch, seq] tensor")
    seq_len = int(x.shape[1])
    out = model._require_name(node_spec.get("_bind"), field="position_ids._bind")
    mask_ref = args[1]
    mask_tensor = env.get(mask_ref) if isinstance(mask_ref, str) else None
    if mask_tensor is not None:
        if not torch.is_tensor(mask_tensor):
            raise ValueError("position_ids.attention_mask must resolve to tensor or null")
        if mask_tensor.ndim != 2:
            raise ValueError("position_ids.attention_mask must be rank-2 [batch, seq]")
        if int(mask_tensor.shape[0]) != int(x.shape[0]):
            raise ValueError("position_ids.attention_mask batch size must match input")
        if int(mask_tensor.shape[1]) < seq_len:
            raise ValueError("position_ids.attention_mask width must be >= input sequence length")
        full_pos = mask_tensor.to(torch.long).cumsum(dim=-1) - 1
        pad_fill = _resolve_pad_fill(model, node_spec, env, symbols)
        full_pos = full_pos.masked_fill(mask_tensor == 0, pad_fill)
        env[out] = full_pos[:, -seq_len:]
        return

    offset = _resolve_past_length(model, node_spec, env, symbols)
    env[out] = torch.arange(offset, offset + seq_len, device=x.device, dtype=torch.long).unsqueeze(
        0
    )
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
    del node_path_var, scope_var
    lines: list[str] = []

    def assign_out_var(out_name: str) -> str:
        return emitter._assign_out_var(env, out_name)

    def read(name: str) -> str:
        return emitter._read_env_var(env, name)

    args = node_spec.get("_args")
    if not isinstance(args, list) or len(args) != 2:
        raise ValueError("position_ids expects _args as [input_ids, attn_mask]")
    src = read(str(args[0]))
    mask_name = args[1]
    mask = env.get(mask_name) if isinstance(mask_name, str) and mask_name in env else None
    out_name = str(node_spec.get("_bind"))
    out_var = assign_out_var(out_name)
    past_expr = emitter._expr_code(node_spec.get("past_length", 0), env)
    pad_fill_expr = emitter._expr_code(node_spec.get("pad_fill", 0), env)
    offset = emitter._fresh("pos_offset")

    lines.append(f"{indent}if {src}.ndim != 2:")
    lines.append(
        f"{indent}    raise ValueError('position_ids._args must resolve to rank-2 [batch, seq] tensor')"
    )

    if isinstance(mask, str):
        full_pos = emitter._fresh("full_pos")
        pad_fill = emitter._fresh("pad_fill")
        lines.append(f"{indent}if {mask} is not None:")
        lines.append(f"{indent}    if {mask}.ndim != 2:")
        lines.append(
            f"{indent}        raise ValueError('position_ids.attention_mask must be rank-2 [batch, seq]')"
        )
        lines.append(f"{indent}    if int({mask}.shape[0]) != int({src}.shape[0]):")
        lines.append(
            f"{indent}        raise ValueError('position_ids.attention_mask batch size must match input')"
        )
        lines.append(f"{indent}    if int({mask}.shape[1]) < int({src}.shape[1]):")
        lines.append(
            f"{indent}        raise ValueError('position_ids.attention_mask width must be >= input sequence length')"
        )
        lines.append(f"{indent}    {full_pos} = {mask}.to(torch.long).cumsum(dim=-1) - 1")
        lines.append(f"{indent}    {pad_fill} = int({pad_fill_expr})")
        lines.append(f"{indent}    {full_pos} = {full_pos}.masked_fill({mask} == 0, {pad_fill})")
        lines.append(f"{indent}    {out_var} = {full_pos}[:, -{src}.shape[1]:]")
        lines.append(f"{indent}else:")
        lines.append(f"{indent}    {offset} = int({past_expr})")
        lines.append(f"{indent}    if {offset} < 0:")
        lines.append(
            f"{indent}        raise ValueError('position_ids.past_length must resolve to non-negative int')"
        )
        lines.append(
            f"{indent}    {out_var} = torch.arange({offset}, {offset} + {src}.shape[1], device={src}.device, dtype=torch.long).unsqueeze(0)"
        )
        return lines

    lines.append(f"{indent}{offset} = int({past_expr})")
    lines.append(f"{indent}if {offset} < 0:")
    lines.append(
        f"{indent}    raise ValueError('position_ids.past_length must resolve to non-negative int')"
    )
    lines.append(
        f"{indent}{out_var} = torch.arange({offset}, {offset} + {src}.shape[1], device={src}.device, dtype=torch.long).unsqueeze(0)"
    )
    return lines


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any", "Any"),
    "kwargs": dict(LOWERING_KWARG_KINDS),
    "returns": ("IdxTensor",),
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
