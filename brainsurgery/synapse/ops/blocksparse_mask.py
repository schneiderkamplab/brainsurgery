from __future__ import annotations

from typing import Any

import torch

OP_NAME = "blocksparse_mask"
LOWERING_ARITY = (2, 2)
LOWERING_ALLOWED_KWARGS: set[str] = {
    "padding_mask",
    "block_size",
    "local_blocks",
    "vert_stride",
    "homo_head",
}
LOWERING_REQUIRED_KWARGS: set[str] = {"block_size", "local_blocks", "vert_stride"}
LOWERING_KWARG_KINDS: dict[str, Any] = {
    "padding_mask": "str",
    "block_size": "dim",
    "local_blocks": "dim",
    "vert_stride": "dim",
    "homo_head": "bool",
}


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return False


def lowering_validate_signature(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> None:
    del args, kwargs, ctx
    if not isinstance(out, str):
        raise ValueError("blocksparse_mask requires a single scalar output binding")


def lowering_infer_metadata(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> bool:
    del kwargs
    if not isinstance(out, str) or len(args) < 2:
        return False
    query_name = str(args[0]).strip()
    key_name = str(args[1]).strip()
    query_shape = ctx.tensor_shape.get(query_name)
    key_shape = ctx.tensor_shape.get(key_name)
    if (
        isinstance(query_shape, tuple)
        and isinstance(key_shape, tuple)
        and len(query_shape) >= 4
        and len(key_shape) >= 4
    ):
        ctx.tensor_shape[out] = (query_shape[0], query_shape[1], query_shape[-2], key_shape[-2])
        ctx.tensor_last_dim[out] = key_shape[-2]
        return True
    return False


def _resolve_positive_int(
    model: Any, node_spec: dict[str, Any], env: dict[str, Any], symbols: dict[str, int], key: str
) -> int:
    value = int(model._eval_expr(node_spec.get(key), env, symbols))
    if value <= 0:
        raise ValueError(f"blocksparse_mask.{key} must be > 0")
    return value


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
    raw_args = node_spec.get("_args")
    if not isinstance(raw_args, list) or len(raw_args) != 2:
        raise ValueError("blocksparse_mask expects exactly 2 positional args: query and key")
    q = model._read_tensor_input(raw_args[0], env)
    k = model._read_tensor_input(raw_args[1], env)
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError("blocksparse_mask expects rank-4 q/k tensors [batch, heads, seq, dim]")
    if int(q.shape[0]) != int(k.shape[0]):
        raise ValueError("blocksparse_mask q and k must have the same batch size")
    if int(q.shape[1]) != int(k.shape[1]):
        raise ValueError("blocksparse_mask q and k must have the same head count")

    padding_ref = node_spec.get("padding_mask")
    padding_mask = env.get(padding_ref) if isinstance(padding_ref, str) else None
    if padding_mask is not None and not torch.is_tensor(padding_mask):
        raise ValueError("blocksparse_mask.padding_mask must resolve to tensor or null")

    block_size = _resolve_positive_int(model, node_spec, env, symbols, "block_size")
    local_blocks = _resolve_positive_int(model, node_spec, env, symbols, "local_blocks")
    vert_stride = _resolve_positive_int(model, node_spec, env, symbols, "vert_stride")
    homo_head = bool(model._eval_expr(node_spec.get("homo_head", False), env, symbols))

    q_len = int(q.shape[-2])
    kv_len = int(k.shape[-2])
    num_heads = int(q.shape[-3])

    n_blocks_k = (kv_len + block_size - 1) // block_size
    n_blocks_q = (q_len + block_size - 1) // block_size
    q_blocks = torch.arange(n_blocks_q, device=q.device) + max(n_blocks_k - n_blocks_q, 0)
    k_blocks = torch.arange(n_blocks_k, device=q.device)

    causal_blocks = k_blocks.unsqueeze(0) <= q_blocks.unsqueeze(1)
    local_blocks_mask = k_blocks.unsqueeze(0) >= (q_blocks.unsqueeze(1) - (local_blocks - 1))

    if homo_head:
        strided_blocks = ((k_blocks + 1) % vert_stride == 0).unsqueeze(0).expand(n_blocks_q, -1)
        block_keep = (
            (causal_blocks & (local_blocks_mask | strided_blocks)).unsqueeze(0).unsqueeze(0)
        )
    else:
        head_step = max(1, vert_stride // max(1, num_heads))
        head_idx = torch.arange(num_heads, device=q.device)
        strided_blocks = (
            (k_blocks.unsqueeze(0) + head_idx.unsqueeze(1) * head_step + 1) % vert_stride
        ) == 0
        block_keep = causal_blocks.unsqueeze(0) & (
            local_blocks_mask.unsqueeze(0) | strided_blocks[:, None, :]
        )
        block_keep = block_keep.unsqueeze(0)

    keep = block_keep.repeat_interleave(block_size, dim=-2).repeat_interleave(block_size, dim=-1)
    keep = keep[..., -q_len:, :kv_len]

    key_positions = torch.arange(kv_len, device=q.device)
    query_positions = torch.arange(q_len, device=q.device) + max(kv_len - q_len, 0)
    causal_token = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    keep = keep & causal_token.unsqueeze(0).unsqueeze(0)

    if int(keep.shape[0]) == 1 and int(q.shape[0]) != 1:
        keep = keep.expand(int(q.shape[0]), -1, -1, -1)
    if int(keep.shape[1]) == 1 and int(q.shape[1]) != 1:
        keep = keep.expand(-1, int(q.shape[1]), -1, -1)

    if padding_mask is not None:
        if padding_mask.ndim != 2:
            raise ValueError("blocksparse_mask.padding_mask must be rank-2 [batch, seq]")
        if int(padding_mask.shape[0]) != int(q.shape[0]):
            raise ValueError("blocksparse_mask.padding_mask batch size must match query batch")
        if int(padding_mask.shape[1]) != kv_len:
            raise ValueError("blocksparse_mask.padding_mask width must match key sequence length")
        key_valid = padding_mask.to(torch.bool).unsqueeze(1).unsqueeze(1)
        keep = keep & key_valid

    mask_value = torch.finfo(q.dtype).min
    out_name = model._require_name(node_spec.get("_bind"), field="blocksparse_mask._bind")
    env[out_name] = torch.where(
        keep,
        torch.zeros((), dtype=q.dtype, device=q.device),
        torch.full((), mask_value, dtype=q.dtype, device=q.device),
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
    lines: list[str] = []

    def assign_out_var(out_name: str) -> str:
        return emitter._assign_out_var(env, out_name)

    def read(name: str) -> str:
        return emitter._read_env_var(env, name)

    raw_args = node_spec.get("_args")
    if not isinstance(raw_args, list) or len(raw_args) != 2:
        raise ValueError("blocksparse_mask expects exactly 2 positional args: query and key")
    q = read(str(raw_args[0]))
    k = read(str(raw_args[1]))
    out_name = str(node_spec.get("_bind"))
    out_var = assign_out_var(out_name)

    padding_name = node_spec.get("padding_mask")
    padding_expr = (
        env.get(padding_name) if isinstance(padding_name, str) and padding_name in env else "None"
    )
    block_size_expr = emitter._expr_code(node_spec.get("block_size"), env)
    local_blocks_expr = emitter._expr_code(node_spec.get("local_blocks"), env)
    vert_stride_expr = emitter._expr_code(node_spec.get("vert_stride"), env)
    homo_head_expr = emitter._expr_code(node_spec.get("homo_head", False), env)

    q_len = emitter._fresh("q_len")
    kv_len = emitter._fresh("kv_len")
    num_heads = emitter._fresh("num_heads")
    block_size = emitter._fresh("block_size")
    local_blocks = emitter._fresh("local_blocks")
    vert_stride = emitter._fresh("vert_stride")
    homo_head = emitter._fresh("homo_head")
    n_blocks_k = emitter._fresh("n_blocks_k")
    n_blocks_q = emitter._fresh("n_blocks_q")
    q_blocks = emitter._fresh("q_blocks")
    k_blocks = emitter._fresh("k_blocks")
    causal_blocks = emitter._fresh("causal_blocks")
    local_blocks_mask = emitter._fresh("local_blocks_mask")
    block_keep = emitter._fresh("block_keep")
    strided_blocks = emitter._fresh("strided_blocks")
    head_step = emitter._fresh("head_step")
    head_idx = emitter._fresh("head_idx")
    keep = emitter._fresh("keep")
    key_positions = emitter._fresh("key_positions")
    query_positions = emitter._fresh("query_positions")
    causal_token = emitter._fresh("causal_token")
    key_valid = emitter._fresh("key_valid")
    mask_value = emitter._fresh("mask_value")

    lines.extend(
        [
            f"{indent}if {q}.ndim != 4 or {k}.ndim != 4:",
            f"{indent}    raise ValueError('blocksparse_mask expects rank-4 q/k tensors [batch, heads, seq, dim]')",
            f"{indent}if int({q}.shape[0]) != int({k}.shape[0]):",
            f"{indent}    raise ValueError('blocksparse_mask q and k must have the same batch size')",
            f"{indent}if int({q}.shape[1]) != int({k}.shape[1]):",
            f"{indent}    raise ValueError('blocksparse_mask q and k must have the same head count')",
            f"{indent}{q_len} = int({q}.shape[-2])",
            f"{indent}{kv_len} = int({k}.shape[-2])",
            f"{indent}{num_heads} = int({q}.shape[-3])",
            f"{indent}{block_size} = int({block_size_expr})",
            f"{indent}if {block_size} <= 0:",
            f"{indent}    raise ValueError('blocksparse_mask.block_size must be > 0')",
            f"{indent}{local_blocks} = int({local_blocks_expr})",
            f"{indent}if {local_blocks} <= 0:",
            f"{indent}    raise ValueError('blocksparse_mask.local_blocks must be > 0')",
            f"{indent}{vert_stride} = int({vert_stride_expr})",
            f"{indent}if {vert_stride} <= 0:",
            f"{indent}    raise ValueError('blocksparse_mask.vert_stride must be > 0')",
            f"{indent}{homo_head} = bool({homo_head_expr})",
            f"{indent}{n_blocks_k} = ({kv_len} + {block_size} - 1) // {block_size}",
            f"{indent}{n_blocks_q} = ({q_len} + {block_size} - 1) // {block_size}",
            f"{indent}{q_blocks} = torch.arange({n_blocks_q}, device={q}.device) + max({n_blocks_k} - {n_blocks_q}, 0)",
            f"{indent}{k_blocks} = torch.arange({n_blocks_k}, device={q}.device)",
            f"{indent}{causal_blocks} = {k_blocks}.unsqueeze(0) <= {q_blocks}.unsqueeze(1)",
            f"{indent}{local_blocks_mask} = {k_blocks}.unsqueeze(0) >= ({q_blocks}.unsqueeze(1) - ({local_blocks} - 1))",
            f"{indent}if {homo_head}:",
            f"{indent}    {strided_blocks} = (({k_blocks} + 1) % {vert_stride} == 0).unsqueeze(0).expand({n_blocks_q}, -1)",
            f"{indent}    {block_keep} = ({causal_blocks} & ({local_blocks_mask} | {strided_blocks})).unsqueeze(0).unsqueeze(0)",
            f"{indent}else:",
            f"{indent}    {head_step} = max(1, {vert_stride} // max(1, {num_heads}))",
            f"{indent}    {head_idx} = torch.arange({num_heads}, device={q}.device)",
            f"{indent}    {strided_blocks} = (({k_blocks}.unsqueeze(0) + {head_idx}.unsqueeze(1) * {head_step} + 1) % {vert_stride}) == 0",
            f"{indent}    {block_keep} = {causal_blocks}.unsqueeze(0) & ({local_blocks_mask}.unsqueeze(0) | {strided_blocks}[:, None, :])",
            f"{indent}    {block_keep} = {block_keep}.unsqueeze(0)",
            f"{indent}{keep} = {block_keep}.repeat_interleave({block_size}, dim=-2).repeat_interleave({block_size}, dim=-1)",
            f"{indent}{keep} = {keep}[..., -{q_len}:, :{kv_len}]",
            f"{indent}{key_positions} = torch.arange({kv_len}, device={q}.device)",
            f"{indent}{query_positions} = torch.arange({q_len}, device={q}.device) + max({kv_len} - {q_len}, 0)",
            f"{indent}{causal_token} = {key_positions}.unsqueeze(0) <= {query_positions}.unsqueeze(1)",
            f"{indent}{keep} = {keep} & {causal_token}.unsqueeze(0).unsqueeze(0)",
            f"{indent}if int({keep}.shape[0]) == 1 and int({q}.shape[0]) != 1:",
            f"{indent}    {keep} = {keep}.expand(int({q}.shape[0]), -1, -1, -1)",
            f"{indent}if int({keep}.shape[1]) == 1 and int({q}.shape[1]) != 1:",
            f"{indent}    {keep} = {keep}.expand(-1, int({q}.shape[1]), -1, -1)",
            f"{indent}if {padding_expr} is not None:",
            f"{indent}    if not torch.is_tensor({padding_expr}):",
            f"{indent}        raise ValueError('blocksparse_mask.padding_mask must resolve to tensor or null')",
            f"{indent}    if {padding_expr}.ndim != 2:",
            f"{indent}        raise ValueError('blocksparse_mask.padding_mask must be rank-2 [batch, seq]')",
            f"{indent}    if int({padding_expr}.shape[0]) != int({q}.shape[0]):",
            f"{indent}        raise ValueError('blocksparse_mask.padding_mask batch size must match query batch')",
            f"{indent}    if int({padding_expr}.shape[1]) != {kv_len}:",
            f"{indent}        raise ValueError('blocksparse_mask.padding_mask width must match key sequence length')",
            f"{indent}    {key_valid} = {padding_expr}.to(torch.bool).unsqueeze(1).unsqueeze(1)",
            f"{indent}    {keep} = {keep} & {key_valid}",
            f"{indent}{mask_value} = torch.finfo({q}.dtype).min",
            f"{indent}{out_var} = torch.where({keep}, torch.zeros((), dtype={q}.dtype, device={q}.device), torch.full((), {mask_value}, dtype={q}.dtype, device={q}.device))",
        ]
    )

    return lines


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any", "Any"),
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
