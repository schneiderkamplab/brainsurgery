from __future__ import annotations

import math

import torch

_FP4_VALUES = [
    +0.0,
    +0.5,
    +1.0,
    +1.5,
    +2.0,
    +3.0,
    +4.0,
    +6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def _infer_target_dtype(state_dict: dict[str, torch.Tensor]) -> torch.dtype:
    for tensor in state_dict.values():
        if torch.is_tensor(tensor) and tensor.is_floating_point():
            return tensor.dtype
    return torch.bfloat16


def _convert_moe_packed_tensors(
    blocks: torch.Tensor,
    scales: torch.Tensor,
    *,
    dtype: torch.dtype,
    rows_per_chunk: int = 32768 * 1024,
) -> torch.Tensor:
    blocks_u8 = blocks.to(torch.uint8)
    scales_i32 = scales.to(torch.int32) - 127

    if blocks_u8.ndim < 2:
        raise ValueError("MXFP4 blocks must be rank >= 2")
    if blocks_u8.shape[:-1] != scales_i32.shape:
        raise ValueError(
            f"MXFP4 shape mismatch: blocks.shape[:-1]={blocks_u8.shape[:-1]!r}, "
            f"scales.shape={scales_i32.shape!r}"
        )

    lut = torch.tensor(_FP4_VALUES, dtype=dtype, device=blocks_u8.device)

    *prefix_shape, groups, packed_cols = blocks_u8.shape
    rows_total = math.prod(prefix_shape) * groups

    blocks_2d = blocks_u8.reshape(rows_total, packed_cols)
    scales_2d = scales_i32.reshape(rows_total, 1)
    out = torch.empty(rows_total, packed_cols * 2, dtype=dtype, device=blocks_u8.device)

    for r0 in range(0, rows_total, rows_per_chunk):
        r1 = min(r0 + rows_per_chunk, rows_total)
        blk = blocks_2d[r0:r1]
        exp = scales_2d[r0:r1]
        dst = out[r0:r1]

        idx_lo = (blk & 0x0F).to(torch.int32)
        dst[:, 0::2] = lut[idx_lo]
        idx_hi = (blk >> 4).to(torch.int32)
        dst[:, 1::2] = lut[idx_hi]
        torch.ldexp(dst, exp, out=dst)

    out = out.reshape(*prefix_shape, groups, packed_cols * 2).view(
        *prefix_shape, groups * packed_cols * 2
    )
    if out.ndim == 2:
        return out.transpose(0, 1).contiguous()
    return out.transpose(1, 2).contiguous()


def materialize_mxfp4_aliases(
    state_dict: dict[str, torch.Tensor],
    *,
    dtype: torch.dtype | None = None,
    drop_packed: bool = False,
    expert_index_aliases: bool = True,
) -> None:
    target_dtype = _infer_target_dtype(state_dict) if dtype is None else dtype
    blocks_suffix = "_blocks"

    for key in list(state_dict.keys()):
        if not key.endswith(blocks_suffix):
            continue

        base = key[: -len(blocks_suffix)]
        scales_key = f"{base}_scales"
        if scales_key not in state_dict:
            continue

        weight_key = f"{base}.weight"
        if weight_key not in state_dict:
            dequantized = _convert_moe_packed_tensors(
                state_dict[key],
                state_dict[scales_key],
                dtype=target_dtype,
            )
            if dequantized.is_floating_point() and dequantized.dtype != target_dtype:
                dequantized = dequantized.to(dtype=target_dtype)
            state_dict[weight_key] = dequantized

        bias_src_key = f"{base}_bias"
        bias_dst_key = f"{base}.bias"
        if bias_src_key in state_dict and bias_dst_key not in state_dict:
            bias = state_dict[bias_src_key]
            if bias.is_floating_point() and bias.dtype != target_dtype:
                bias = bias.to(dtype=target_dtype)
            state_dict[bias_dst_key] = bias
        if drop_packed:
            state_dict.pop(key, None)
            state_dict.pop(scales_key, None)
            if bias_src_key in state_dict and bias_dst_key in state_dict:
                state_dict.pop(bias_src_key, None)

    if expert_index_aliases:
        _materialize_moe_expert_index_aliases(state_dict)


def _materialize_moe_expert_index_aliases(state_dict: dict[str, torch.Tensor]) -> None:
    expert_suffixes = (
        ".mlp.experts.gate_up_proj.weight",
        ".mlp.experts.gate_up_proj.bias",
        ".mlp.experts.down_proj.weight",
        ".mlp.experts.down_proj.bias",
    )
    for key, tensor in list(state_dict.items()):
        if not torch.is_tensor(tensor):
            continue
        matched_suffix = next((suffix for suffix in expert_suffixes if key.endswith(suffix)), None)
        if matched_suffix is None:
            continue
        if tensor.ndim < 1:
            continue
        expert_count = int(tensor.shape[0])
        base_prefix = key[: -len(matched_suffix)]
        inner_suffix = matched_suffix[len(".mlp.experts.") :]
        for expert_idx in range(expert_count):
            alias = f"{base_prefix}.mlp.experts.{expert_idx}.{inner_suffix}"
            state_dict.setdefault(alias, tensor[expert_idx])


__all__ = ["materialize_mxfp4_aliases"]
