from __future__ import annotations

from typing import Any

import torch

OP_NAME = "sinusoidal_positions"
LOWERING_ARITY = (2, 6)
LOWERING_ALLOWED_KWARGS: set[str] = set()
LOWERING_REQUIRED_KWARGS: set[str] = set()
LOWERING_KWARG_KINDS: dict[str, Any] = {}


def _raw_args(node_spec: dict[str, Any]) -> list[Any]:
    raw = node_spec.get("_args")
    if isinstance(raw, list):
        return list(raw)
    if raw is None:
        return []
    return [raw]


def _arg_or_default(args: list[Any], index: int, default: Any) -> Any:
    if index >= len(args):
        return default
    value = args[index]
    if isinstance(value, str) and value.strip().lower() == "null":
        return default
    return value


def _resolve_int_or_none(
    model: Any,
    raw: Any,
    env: dict[str, Any],
    symbols: dict[str, int],
    *,
    field: str,
) -> int | None:
    value = model._eval_expr(raw, env, symbols)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"sinusoidal_positions.{field} must resolve to int or null")
    return int(value)


def _resolve_float(
    model: Any,
    raw: Any,
    env: dict[str, Any],
    symbols: dict[str, int],
    *,
    field: str,
) -> float:
    value = model._eval_expr(raw, env, symbols)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"sinusoidal_positions.{field} must resolve to number")
    return float(value)


def _resolve_mode(
    model: Any,
    raw: Any,
    env: dict[str, Any],
    symbols: dict[str, int],
) -> str:
    value = model._eval_expr(raw, env, symbols)
    if value is None:
        return "legacy"
    if not isinstance(value, str):
        raise ValueError("sinusoidal_positions.mode must resolve to string or null")
    mode = value.strip().lower()
    if mode in {"", "legacy"}:
        return "legacy"
    if mode == "rope":
        return "rope"
    raise ValueError("sinusoidal_positions.mode must be one of: legacy, rope, or null")


def _build_sinusoidal_positions(
    ref: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    theta: float,
    offset: int,
    padding_idx: int | None,
    mode: str = "legacy",
) -> torch.Tensor:
    if ref.ndim != 3:
        raise ValueError("sinusoidal_positions.ref must be rank-3 [batch, seq, dim]")
    if position_ids.ndim != 2:
        raise ValueError("sinusoidal_positions.position_ids must be rank-2 [batch, seq]")
    if int(position_ids.shape[0]) != int(ref.shape[0]) or int(position_ids.shape[1]) != int(
        ref.shape[1]
    ):
        raise ValueError("sinusoidal_positions.position_ids shape must match ref batch/seq")
    if theta <= 0.0:
        raise ValueError("sinusoidal_positions.theta must be > 0")
    d_model = int(ref.shape[-1])
    half = d_model // 2
    if half < 1:
        raise ValueError("sinusoidal_positions requires ref last dim >= 2")

    pos = position_ids.to(torch.float32) + float(offset)
    freq_idx = torch.arange(half, dtype=torch.float32, device=ref.device)
    if mode == "rope":
        inv_freq = 1.0 / (theta ** (freq_idx / float(max(1, half))))
    else:
        inv_freq = torch.exp(
            (-torch.log(torch.tensor(theta, dtype=torch.float32, device=ref.device)) * freq_idx)
            / float(max(1, half - 1))
        )
    angles = pos.unsqueeze(-1) * inv_freq.view(1, 1, half)
    sin_part = torch.sin(angles)
    cos_part = torch.cos(angles)
    emb = torch.cat([sin_part, cos_part], dim=-1)
    if d_model % 2 == 1:
        emb = torch.cat(
            [
                emb,
                torch.zeros(
                    int(ref.shape[0]), int(ref.shape[1]), 1, device=ref.device, dtype=torch.float32
                ),
            ],
            dim=-1,
        )
    if padding_idx is not None:
        mask = (position_ids == int(padding_idx)).unsqueeze(-1)
        emb = torch.where(mask, torch.zeros_like(emb), emb)
    return emb.to(dtype=ref.dtype)


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return False


def lowering_validate_signature(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> None:
    del ctx
    if not isinstance(out, str):
        raise ValueError("sinusoidal_positions requires a single scalar output binding")
    if kwargs:
        unknown = ", ".join(sorted(str(key) for key in kwargs))
        raise ValueError(f"sinusoidal_positions unsupported kwargs: {unknown}")
    if len(args) < 2 or len(args) > 6:
        raise ValueError(
            "sinusoidal_positions requires positional args: ref position_ids [theta offset padding_idx mode]"
        )


def lowering_infer_metadata(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> bool:
    del kwargs
    if not isinstance(out, str) or len(args) < 1:
        return False
    ref_name = str(args[0]).strip()
    ref_shape = ctx.tensor_shape.get(ref_name)
    if isinstance(ref_shape, tuple):
        ctx.tensor_shape[out] = ref_shape
        if len(ref_shape) >= 1:
            ctx.tensor_last_dim[out] = ref_shape[-1]
        return True
    if ref_name in ctx.tensor_last_dim:
        ctx.tensor_last_dim[out] = ctx.tensor_last_dim[ref_name]
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
    del node_path, scope
    args = _raw_args(node_spec)
    if len(args) < 2 or len(args) > 6:
        raise ValueError(
            "sinusoidal_positions requires positional args: ref position_ids [theta offset padding_idx mode]"
        )
    ref = model._read_tensor_input(args[0], env)
    position_ids = model._read_tensor_input(args[1], env)
    theta = _resolve_float(model, _arg_or_default(args, 2, 10000.0), env, symbols, field="theta")
    offset = _resolve_int_or_none(model, _arg_or_default(args, 3, 2), env, symbols, field="offset")
    padding_idx = _resolve_int_or_none(
        model, _arg_or_default(args, 4, None), env, symbols, field="padding_idx"
    )
    mode = _resolve_mode(model, _arg_or_default(args, 5, None), env, symbols)
    out = model._require_name(node_spec.get("_bind"), field="sinusoidal_positions._bind")
    env[out] = _build_sinusoidal_positions(
        ref,
        position_ids,
        theta=theta,
        offset=(2 if offset is None else offset),
        padding_idx=padding_idx,
        mode=mode,
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
    args = _raw_args(node_spec)
    if len(args) < 2 or len(args) > 6:
        raise ValueError(
            "sinusoidal_positions requires positional args: ref position_ids [theta offset padding_idx mode]"
        )
    ref = emitter._read_env_var(env, str(args[0]))
    pos = emitter._read_env_var(env, str(args[1]))
    theta_expr = emitter._expr_code(_arg_or_default(args, 2, 10000.0), env)
    offset_expr = emitter._expr_code(_arg_or_default(args, 3, 2), env)
    padding_expr = emitter._expr_code(_arg_or_default(args, 4, None), env)
    mode_expr = emitter._expr_code(_arg_or_default(args, 5, None), env)
    out_var = emitter._assign_out_var(env, str(node_spec.get("_bind")))
    theta_var = emitter._fresh("sinpos_theta")
    offset_var = emitter._fresh("sinpos_offset")
    padding_var = emitter._fresh("sinpos_padding_idx")
    mode_var = emitter._fresh("sinpos_mode")
    d_var = emitter._fresh("sinpos_d")
    half_var = emitter._fresh("sinpos_half")
    pos_var = emitter._fresh("sinpos_pos")
    freq_idx = emitter._fresh("sinpos_freq_idx")
    inv_freq = emitter._fresh("sinpos_inv_freq")
    angles = emitter._fresh("sinpos_angles")
    sin_part = emitter._fresh("sinpos_sin")
    cos_part = emitter._fresh("sinpos_cos")
    emb = emitter._fresh("sinpos_emb")
    mask = emitter._fresh("sinpos_mask")
    return [
        f"{indent}if {ref}.ndim != 3:",
        f"{indent}    raise ValueError('sinusoidal_positions.ref must be rank-3 [batch, seq, dim]')",
        f"{indent}if {pos}.ndim != 2:",
        f"{indent}    raise ValueError('sinusoidal_positions.position_ids must be rank-2 [batch, seq]')",
        f"{indent}if int({pos}.shape[0]) != int({ref}.shape[0]) or int({pos}.shape[1]) != int({ref}.shape[1]):",
        f"{indent}    raise ValueError('sinusoidal_positions.position_ids shape must match ref batch/seq')",
        f"{indent}{theta_var} = float({theta_expr})",
        f"{indent}if {theta_var} <= 0.0:",
        f"{indent}    raise ValueError('sinusoidal_positions.theta must be > 0')",
        f"{indent}{offset_var} = {offset_expr}",
        f"{indent}{offset_var} = 2 if {offset_var} is None else int({offset_var})",
        f"{indent}{padding_var} = {padding_expr}",
        f"{indent}{padding_var} = None if {padding_var} is None else int({padding_var})",
        f"{indent}{mode_var} = {mode_expr}",
        f"{indent}{mode_var} = 'legacy' if {mode_var} is None else str({mode_var}).strip().lower()",
        f"{indent}if {mode_var} not in ('legacy', '', 'rope'):",
        f"{indent}    raise ValueError('sinusoidal_positions.mode must be one of: legacy, rope, or null')",
        f"{indent}{d_var} = int({ref}.shape[-1])",
        f"{indent}{half_var} = int({d_var} // 2)",
        f"{indent}if {half_var} < 1:",
        f"{indent}    raise ValueError('sinusoidal_positions requires ref last dim >= 2')",
        f"{indent}{pos_var} = {pos}.to(torch.float32) + float({offset_var})",
        f"{indent}{freq_idx} = torch.arange({half_var}, dtype=torch.float32, device={ref}.device)",
        f"{indent}if {mode_var} == 'rope':",
        f"{indent}    {inv_freq} = 1.0 / ({theta_var} ** ({freq_idx} / float(max(1, {half_var}))))",
        f"{indent}else:",
        f"{indent}    {inv_freq} = torch.exp((-torch.log(torch.tensor({theta_var}, dtype=torch.float32, device={ref}.device)) * {freq_idx}) / float(max(1, {half_var} - 1)))",
        f"{indent}{angles} = {pos_var}.unsqueeze(-1) * {inv_freq}.view(1, 1, {half_var})",
        f"{indent}{sin_part} = torch.sin({angles})",
        f"{indent}{cos_part} = torch.cos({angles})",
        f"{indent}{emb} = torch.cat([{sin_part}, {cos_part}], dim=-1)",
        f"{indent}if {d_var} % 2 == 1:",
        f"{indent}    {emb} = torch.cat([{emb}, torch.zeros(int({ref}.shape[0]), int({ref}.shape[1]), 1, device={ref}.device, dtype=torch.float32)], dim=-1)",
        f"{indent}if {padding_var} is not None:",
        f"{indent}    {mask} = ({pos} == int({padding_var})).unsqueeze(-1)",
        f"{indent}    {emb} = torch.where({mask}, torch.zeros_like({emb}), {emb})",
        f"{indent}{out_var} = {emb}.to(dtype={ref}.dtype)",
    ]


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any", "Any", "Any", "Any", "Any", "Any"),
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
