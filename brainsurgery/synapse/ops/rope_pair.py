from __future__ import annotations

import math
from typing import Any

import torch

OP_NAME = "rope_pair"
LOWERING_ARITY = (2, 2)
LOWERING_ALLOWED_KWARGS: set[str] = {
    "position_ids",
    "theta",
    "interleaved",
    "scale_factor",
    "beta_fast",
    "beta_slow",
    "mscale",
    "mscale_all_dim",
    "low_freq_factor",
    "high_freq_factor",
    "original_context",
    "attention_factor",
    "rope_mode",
    "truncate",
}
LOWERING_REQUIRED_KWARGS: set[str] = {"position_ids"}
LOWERING_KWARG_KINDS: dict[str, Any] = {
    "position_ids": "str",
    "theta": "number",
    "interleaved": "bool",
    "scale_factor": "number",
    "beta_fast": "number",
    "beta_slow": "number",
    "mscale": "number",
    "mscale_all_dim": "number",
    "low_freq_factor": "number",
    "high_freq_factor": "number",
    "original_context": "dim",
    "attention_factor": "number",
    "rope_mode": "str",
    "truncate": "bool",
}


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return False


def lowering_known_output_arity(*, kwargs: dict[str, Any]) -> int:
    del kwargs
    return 2


def _apply_frequency_scaling(
    *,
    inv_freq: torch.Tensor,
    scale_factor: float,
    low_freq_factor: float,
    high_freq_factor: float,
    original_context: int,
) -> torch.Tensor:
    if scale_factor <= 0.0:
        raise ValueError("rope_pair.scale_factor must be > 0")
    if low_freq_factor <= 0.0:
        raise ValueError("rope_pair.low_freq_factor must be > 0")
    if high_freq_factor <= low_freq_factor:
        raise ValueError("rope_pair.high_freq_factor must be > low_freq_factor")
    if original_context <= 0:
        raise ValueError("rope_pair.original_context must be > 0")

    low_freq_wavelen = float(original_context) / low_freq_factor
    high_freq_wavelen = float(original_context) / high_freq_factor
    wavelen = (2.0 * math.pi) / inv_freq

    inv_scaled = torch.where(wavelen > low_freq_wavelen, inv_freq / scale_factor, inv_freq)
    smooth = (float(original_context) / wavelen - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    smoothed = (1.0 - smooth) * (inv_scaled / scale_factor) + smooth * inv_scaled
    is_medium = (~(wavelen < high_freq_wavelen)) & (~(wavelen > low_freq_wavelen))
    return torch.where(is_medium, smoothed, inv_scaled)


def _apply_frequency_scaling_hf_yarn(
    *,
    inv_freq: torch.Tensor,
    scale_factor: float,
    low_freq_factor: float,
    high_freq_factor: float,
    original_context: int,
    truncate: bool,
) -> torch.Tensor:
    if scale_factor <= 0.0:
        raise ValueError("rope_pair.scale_factor must be > 0")
    if low_freq_factor <= 0.0:
        raise ValueError("rope_pair.low_freq_factor must be > 0")
    if high_freq_factor <= low_freq_factor:
        raise ValueError("rope_pair.high_freq_factor must be > low_freq_factor")
    if original_context <= 0:
        raise ValueError("rope_pair.original_context must be > 0")

    # Match HF _compute_yarn_parameters exactly on the inverse-frequency mixing.
    # We reconstruct pos_freqs from inv_freq and then blend interpolation/extrapolation.
    pos_freqs = 1.0 / inv_freq
    dim = int(inv_freq.shape[0] * 2)
    inv_freq_extrapolation = inv_freq
    inv_freq_interpolation = 1.0 / (scale_factor * pos_freqs)

    def _find_correction_dim(
        num_rotations: float, dim_value: int, base: float, max_pos: int
    ) -> float:
        return (dim_value * math.log(max_pos / (num_rotations * 2.0 * math.pi))) / (
            2.0 * math.log(base)
        )

    def _find_correction_range(
        low_rot: float,
        high_rot: float,
        dim_value: int,
        base: float,
        max_pos: int,
        truncate_value: bool,
    ) -> tuple[float, float]:
        low = _find_correction_dim(low_rot, dim_value, base, max_pos)
        high = _find_correction_dim(high_rot, dim_value, base, max_pos)
        if truncate_value:
            low = math.floor(low)
            high = math.ceil(high)
        return max(low, 0.0), min(high, float(dim_value - 1))

    def _linear_ramp_factor(min_value: float, max_value: float, ramp_dim: int) -> torch.Tensor:
        if min_value == max_value:
            max_value += 0.001
        linear = (
            torch.arange(ramp_dim, dtype=torch.float32, device=inv_freq.device) - min_value
        ) / (max_value - min_value)
        return torch.clamp(linear, 0.0, 1.0)

    # Recover theta/base robustly from inv_freq sequence relation:
    # inv_freq[i] = 1 / (theta ** (2i/dim)).
    if inv_freq.shape[0] > 1:
        ratio = float(inv_freq[1] / inv_freq[0])
        if ratio <= 0.0 or ratio == 1.0:
            theta = 10000.0
        else:
            theta = ratio ** (-dim / 2.0)
    else:
        theta = 10000.0
    low, high = _find_correction_range(
        high_freq_factor,
        low_freq_factor,
        dim,
        theta,
        original_context,
        truncate,
    )
    extrapolation_factor = 1.0 - _linear_ramp_factor(low, high, dim // 2).to(
        device=inv_freq.device, dtype=inv_freq.dtype
    )
    return (
        inv_freq_interpolation * (1.0 - extrapolation_factor)
        + inv_freq_extrapolation * extrapolation_factor
    )


def _apply_transformers_yarn_inv_freq(
    *,
    theta: float,
    head_dim: int,
    scale_factor: float,
    beta_fast: float,
    beta_slow: float,
    original_context: int,
    device: torch.device,
    truncate: bool = True,
) -> torch.Tensor:
    if scale_factor <= 0.0:
        raise ValueError("rope_pair.scale_factor must be > 0")
    if beta_fast <= 0.0:
        raise ValueError("rope_pair.beta_fast must be > 0")
    if beta_slow <= 0.0:
        raise ValueError("rope_pair.beta_slow must be > 0")
    if original_context <= 0:
        raise ValueError("rope_pair.original_context must be > 0")
    if head_dim <= 0 or head_dim % 2 != 0:
        raise ValueError("rope_pair head_dim must be positive and even")

    dim = int(head_dim)
    pos_freqs = theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / float(dim))
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (scale_factor * pos_freqs)

    def _find_correction_dim(num_rotations: float) -> float:
        return (dim * math.log(original_context / (num_rotations * 2.0 * math.pi))) / (
            2.0 * math.log(theta)
        )

    low = _find_correction_dim(beta_fast)
    high = _find_correction_dim(beta_slow)
    if truncate:
        low = math.floor(low)
        high = math.ceil(high)
    low = max(low, 0.0)
    high = min(high, float(dim - 1))

    if low == high:
        high = high + 0.001
    linear = (torch.arange(dim // 2, device=device, dtype=torch.float32) - low) / (high - low)
    ramp = torch.clamp(linear, 0.0, 1.0)
    inv_freq_extrapolation_factor = 1.0 - ramp
    return (
        inv_freq_interpolation * (1.0 - inv_freq_extrapolation_factor)
        + inv_freq_extrapolation * inv_freq_extrapolation_factor
    )


def _yarn_attention_factor(
    *, scale_factor: float, mscale: float | None, mscale_all_dim: float | None
) -> float:
    def _get_mscale(scale: float, value: float = 1.0) -> float:
        if scale <= 1.0:
            return 1.0
        return 0.1 * float(value) * math.log(scale) + 1.0

    if mscale is not None and mscale_all_dim is not None:
        return float(_get_mscale(scale_factor, mscale) / _get_mscale(scale_factor, mscale_all_dim))
    return float(_get_mscale(scale_factor))


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
    if not isinstance(ins, list) or len(ins) != 2 or not isinstance(outs, list) or len(outs) != 2:
        raise ValueError("rope_pair expects in=[q,k], out=[q_rot,k_rot]")
    q = env[ins[0]]
    k = env[ins[1]]
    if not torch.is_tensor(q) or not torch.is_tensor(k):
        raise ValueError("rope_pair expects tensor inputs for q and k")
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError("rope_pair expects q and k to be rank-4 [batch, heads, seq, head_dim]")
    if int(q.shape[0]) != int(k.shape[0]):
        raise ValueError("rope_pair expects q and k to have matching batch size")
    if int(q.shape[-2]) != int(k.shape[-2]):
        raise ValueError("rope_pair expects q and k to have matching sequence length")
    if int(q.shape[-1]) != int(k.shape[-1]):
        raise ValueError("rope_pair expects q and k to have matching head dimension")
    if int(q.shape[-1]) % 2 != 0:
        raise ValueError("rope_pair expects even head dimension")
    theta = float(model._eval_expr(node_spec.get("theta", 10000.0), env, symbols))
    attention_factor = float(model._eval_expr(node_spec.get("attention_factor", 1.0), env, symbols))
    pos_ref = node_spec.get("position_ids")
    if not isinstance(pos_ref, str):
        raise ValueError("rope_pair.position_ids must be a tensor reference")
    pos_ids = env.get(pos_ref)
    if pos_ids is None:
        raise ValueError("rope_pair.position_ids must not be null")
    if not torch.is_tensor(pos_ids):
        raise ValueError("rope_pair.position_ids must resolve to tensor")
    if pos_ids.ndim != 2:
        raise ValueError("rope_pair.position_ids must be rank-2 [batch, seq]")
    if int(pos_ids.shape[0]) != int(q.shape[0]):
        raise ValueError("rope_pair.position_ids batch size must match q/k batch")
    if int(pos_ids.shape[1]) != int(q.shape[-2]):
        raise ValueError("rope_pair.position_ids width must match q/k sequence length")
    half = q.shape[-1] // 2
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, half, device=q.device, dtype=torch.float32) / float(half))
    )
    rope_mode = str(node_spec.get("rope_mode", "")).strip().lower()
    rope_attention_factor = attention_factor
    if all(
        key in node_spec for key in ("scale_factor", "beta_fast", "beta_slow", "original_context")
    ):
        scale_factor = float(model._eval_expr(node_spec["scale_factor"], env, symbols))
        beta_fast = float(model._eval_expr(node_spec["beta_fast"], env, symbols))
        beta_slow = float(model._eval_expr(node_spec["beta_slow"], env, symbols))
        original_context = int(model._eval_expr(node_spec["original_context"], env, symbols))
        inv_freq = _apply_transformers_yarn_inv_freq(
            theta=theta,
            head_dim=int(q.shape[-1]),
            scale_factor=scale_factor,
            beta_fast=beta_fast,
            beta_slow=beta_slow,
            original_context=original_context,
            device=q.device,
            truncate=True,
        )
        mscale = (
            float(model._eval_expr(node_spec["mscale"], env, symbols))
            if "mscale" in node_spec
            else None
        )
        mscale_all_dim = (
            float(model._eval_expr(node_spec["mscale_all_dim"], env, symbols))
            if "mscale_all_dim" in node_spec
            else None
        )
        rope_attention_factor = _yarn_attention_factor(
            scale_factor=scale_factor,
            mscale=mscale,
            mscale_all_dim=mscale_all_dim,
        )
    elif all(
        key in node_spec
        for key in ("scale_factor", "low_freq_factor", "high_freq_factor", "original_context")
    ):
        scale_factor = float(model._eval_expr(node_spec["scale_factor"], env, symbols))
        low_freq_factor = float(model._eval_expr(node_spec["low_freq_factor"], env, symbols))
        high_freq_factor = float(model._eval_expr(node_spec["high_freq_factor"], env, symbols))
        original_context = int(model._eval_expr(node_spec["original_context"], env, symbols))
        if rope_mode == "hf_yarn":
            inv_freq = _apply_frequency_scaling_hf_yarn(
                inv_freq=inv_freq,
                scale_factor=scale_factor,
                low_freq_factor=low_freq_factor,
                high_freq_factor=high_freq_factor,
                original_context=original_context,
                truncate=bool(node_spec.get("truncate", True)),
            )
        else:
            inv_freq = _apply_frequency_scaling(
                inv_freq=inv_freq,
                scale_factor=scale_factor,
                low_freq_factor=low_freq_factor,
                high_freq_factor=high_freq_factor,
                original_context=original_context,
            )
    elif "scale_factor" in node_spec:
        scale_factor = float(model._eval_expr(node_spec["scale_factor"], env, symbols))
        if scale_factor <= 0.0:
            raise ValueError("rope_pair.scale_factor must be > 0")
        inv_freq = inv_freq / float(scale_factor)
    pos = pos_ids.to(device=q.device, dtype=torch.float32)
    ang = pos.unsqueeze(-1) * inv_freq.unsqueeze(0).unsqueeze(0)
    cos_half = (torch.cos(ang) * float(rope_attention_factor)).to(dtype=q.dtype)
    sin_half = (torch.sin(ang) * float(rope_attention_factor)).to(dtype=q.dtype)
    interleaved = bool(node_spec.get("interleaved", False))
    if interleaved:
        cos = cos_half.unsqueeze(1)
        sin = sin_half.unsqueeze(1)
        q_even = q[..., 0::2]
        q_odd = q[..., 1::2]
        k_even = k[..., 0::2]
        k_odd = k[..., 1::2]
        q_rot_even = q_even * cos - q_odd * sin
        q_rot_odd = q_even * sin + q_odd * cos
        k_rot_even = k_even * cos - k_odd * sin
        k_rot_odd = k_even * sin + k_odd * cos
        q_rot = torch.empty_like(q)
        k_rot = torch.empty_like(k)
        q_rot[..., 0::2] = q_rot_even
        q_rot[..., 1::2] = q_rot_odd
        k_rot[..., 0::2] = k_rot_even
        k_rot[..., 1::2] = k_rot_odd
        env[outs[0]] = q_rot
        env[outs[1]] = k_rot
    else:
        cos = cos_half.unsqueeze(1)
        sin = sin_half.unsqueeze(1)
        q1, q2 = q[..., :half], q[..., half : 2 * half]
        k1, k2 = k[..., :half], k[..., half : 2 * half]
        env[outs[0]] = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
        env[outs[1]] = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
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
    if not isinstance(ins, list) or len(ins) != 2 or not isinstance(outs, list) or len(outs) != 2:
        raise ValueError("rope_pair expects in=[q,k], out=[q_rot,k_rot]")
    q = read(str(ins[0]))
    k = read(str(ins[1]))
    q_out = assign_out_var(str(outs[0]))
    k_out = assign_out_var(str(outs[1]))
    theta = emitter._expr_code(node_spec.get("theta", 10000.0), env)
    attention_factor_expr = emitter._expr_code(node_spec.get("attention_factor", 1.0), env)
    scale_factor = emitter._expr_code(node_spec.get("scale_factor"), env)
    beta_fast = emitter._expr_code(node_spec.get("beta_fast"), env)
    beta_slow = emitter._expr_code(node_spec.get("beta_slow"), env)
    mscale_expr = emitter._expr_code(node_spec.get("mscale"), env)
    mscale_all_dim_expr = emitter._expr_code(node_spec.get("mscale_all_dim"), env)
    low_freq_factor = emitter._expr_code(node_spec.get("low_freq_factor"), env)
    high_freq_factor = emitter._expr_code(node_spec.get("high_freq_factor"), env)
    original_context = emitter._expr_code(node_spec.get("original_context"), env)
    rope_mode = str(node_spec.get("rope_mode", "")).strip().lower()
    truncate = bool(node_spec.get("truncate", True))
    pos_name = node_spec.get("position_ids")
    if not isinstance(pos_name, str) or pos_name not in env:
        raise ValueError("rope_pair.position_ids must reference an input tensor name")
    pos_ids = env[pos_name]
    half = emitter._fresh("half")
    inv_freq = emitter._fresh("inv_freq")
    pos = emitter._fresh("pos")
    ang = emitter._fresh("ang")
    cos = emitter._fresh("cos")
    sin = emitter._fresh("sin")
    cos_half = emitter._fresh("cos_half")
    sin_half = emitter._fresh("sin_half")
    q1 = emitter._fresh("q1")
    q2 = emitter._fresh("q2")
    k1 = emitter._fresh("k1")
    k2 = emitter._fresh("k2")
    rope_attention_factor = emitter._fresh("rope_attention_factor")
    interleaved = bool(node_spec.get("interleaved", False))
    lines.append(f"{indent}if {q}.ndim != 4 or {k}.ndim != 4:")
    lines.append(
        f"{indent}    raise ValueError('rope_pair expects q and k to be rank-4 [batch, heads, seq, head_dim]')"
    )
    lines.append(f"{indent}if int({q}.shape[0]) != int({k}.shape[0]):")
    lines.append(
        f"{indent}    raise ValueError('rope_pair expects q and k to have matching batch size')"
    )
    lines.append(f"{indent}if int({q}.shape[-2]) != int({k}.shape[-2]):")
    lines.append(
        f"{indent}    raise ValueError('rope_pair expects q and k to have matching sequence length')"
    )
    lines.append(f"{indent}if int({q}.shape[-1]) != int({k}.shape[-1]):")
    lines.append(
        f"{indent}    raise ValueError('rope_pair expects q and k to have matching head dimension')"
    )
    lines.append(f"{indent}{half} = {q}.shape[-1] // 2")
    lines.append(f"{indent}if int({q}.shape[-1]) % 2 != 0:")
    lines.append(f"{indent}    raise ValueError('rope_pair expects even head dimension')")
    lines.append(
        f"{indent}{inv_freq} = 1.0 / (float({theta}) ** (torch.arange(0, {half}, device={q}.device, dtype=torch.float32) / float({half})))"
    )
    lines.append(f"{indent}{rope_attention_factor} = float({attention_factor_expr})")
    if all(
        key in node_spec for key in ("scale_factor", "beta_fast", "beta_slow", "original_context")
    ):
        dim = emitter._fresh("dim")
        pos_freqs = emitter._fresh("pos_freqs")
        inv_freq_extrapolation = emitter._fresh("inv_freq_extrapolation")
        inv_freq_interpolation = emitter._fresh("inv_freq_interpolation")
        low = emitter._fresh("low")
        high = emitter._fresh("high")
        linear = emitter._fresh("linear")
        ramp = emitter._fresh("ramp")
        inv_freq_extrapolation_factor = emitter._fresh("inv_freq_extrapolation_factor")
        mscale_term = emitter._fresh("mscale_term")
        mscale_all_dim_term = emitter._fresh("mscale_all_dim_term")
        lines.append(f"{indent}if float({scale_factor}) <= 0.0:")
        lines.append(f"{indent}    raise ValueError('rope_pair.scale_factor must be > 0')")
        lines.append(f"{indent}if float({beta_fast}) <= 0.0:")
        lines.append(f"{indent}    raise ValueError('rope_pair.beta_fast must be > 0')")
        lines.append(f"{indent}if float({beta_slow}) <= 0.0:")
        lines.append(f"{indent}    raise ValueError('rope_pair.beta_slow must be > 0')")
        lines.append(f"{indent}if int({original_context}) <= 0:")
        lines.append(f"{indent}    raise ValueError('rope_pair.original_context must be > 0')")
        lines.append(f"{indent}{dim} = int({q}.shape[-1])")
        lines.append(
            f"{indent}{pos_freqs} = float({theta}) ** (torch.arange(0, {dim}, 2, device={q}.device, dtype=torch.float32) / float({dim}))"
        )
        lines.append(f"{indent}{inv_freq_extrapolation} = 1.0 / {pos_freqs}")
        lines.append(
            f"{indent}{inv_freq_interpolation} = 1.0 / (float({scale_factor}) * {pos_freqs})"
        )
        lines.append(
            f"{indent}{low} = (float({dim}) * math.log(float({original_context}) / (float({beta_fast}) * 2.0 * math.pi))) / (2.0 * math.log(float({theta})))"
        )
        lines.append(
            f"{indent}{high} = (float({dim}) * math.log(float({original_context}) / (float({beta_slow}) * 2.0 * math.pi))) / (2.0 * math.log(float({theta})))"
        )
        lines.append(f"{indent}{low} = max(math.floor({low}), 0.0)")
        lines.append(f"{indent}{high} = min(math.ceil({high}), float({dim} - 1))")
        lines.append(f"{indent}if {low} == {high}:")
        lines.append(f"{indent}    {high} = {high} + 0.001")
        lines.append(
            f"{indent}{linear} = (torch.arange({dim} // 2, device={q}.device, dtype=torch.float32) - {low}) / ({high} - {low})"
        )
        lines.append(f"{indent}{ramp} = torch.clamp({linear}, 0.0, 1.0)")
        lines.append(f"{indent}{inv_freq_extrapolation_factor} = 1.0 - {ramp}")
        lines.append(
            f"{indent}{inv_freq} = {inv_freq_interpolation} * (1.0 - {inv_freq_extrapolation_factor}) + {inv_freq_extrapolation} * {inv_freq_extrapolation_factor}"
        )
        if "mscale" in node_spec and "mscale_all_dim" in node_spec:
            lines.append(f"{indent}if float({scale_factor}) <= 1.0:")
            lines.append(f"{indent}    {rope_attention_factor} = 1.0")
            lines.append(f"{indent}else:")
            lines.append(
                f"{indent}    {mscale_term} = (0.1 * float({mscale_expr}) * math.log(float({scale_factor}))) + 1.0"
            )
            lines.append(
                f"{indent}    {mscale_all_dim_term} = (0.1 * float({mscale_all_dim_expr}) * math.log(float({scale_factor}))) + 1.0"
            )
            lines.append(
                f"{indent}    {rope_attention_factor} = float({mscale_term} / {mscale_all_dim_term})"
            )
        else:
            lines.append(f"{indent}if float({scale_factor}) <= 1.0:")
            lines.append(f"{indent}    {rope_attention_factor} = 1.0")
            lines.append(f"{indent}else:")
            lines.append(
                f"{indent}    {rope_attention_factor} = float((0.1 * math.log(float({scale_factor}))) + 1.0)"
            )
    elif all(
        key in node_spec
        for key in ("scale_factor", "low_freq_factor", "high_freq_factor", "original_context")
    ):
        low_freq_wavelen = emitter._fresh("low_freq_wavelen")
        high_freq_wavelen = emitter._fresh("high_freq_wavelen")
        wavelen = emitter._fresh("wavelen")
        inv_scaled = emitter._fresh("inv_scaled")
        smooth = emitter._fresh("smooth")
        smoothed = emitter._fresh("smoothed")
        is_medium = emitter._fresh("is_medium")
        lines.append(f"{indent}if float({scale_factor}) <= 0.0:")
        lines.append(f"{indent}    raise ValueError('rope_pair.scale_factor must be > 0')")
        lines.append(f"{indent}if float({low_freq_factor}) <= 0.0:")
        lines.append(f"{indent}    raise ValueError('rope_pair.low_freq_factor must be > 0')")
        lines.append(f"{indent}if float({high_freq_factor}) <= float({low_freq_factor}):")
        lines.append(
            f"{indent}    raise ValueError('rope_pair.high_freq_factor must be > low_freq_factor')"
        )
        lines.append(f"{indent}if int({original_context}) <= 0:")
        lines.append(f"{indent}    raise ValueError('rope_pair.original_context must be > 0')")
        lines.append(
            f"{indent}{low_freq_wavelen} = float({original_context}) / float({low_freq_factor})"
        )
        lines.append(
            f"{indent}{high_freq_wavelen} = float({original_context}) / float({high_freq_factor})"
        )
        lines.append(f"{indent}{wavelen} = (2.0 * torch.pi) / {inv_freq}")
        if rope_mode == "hf_yarn":
            theta_var = emitter._fresh("theta")
            low_var = emitter._fresh("low")
            high_var = emitter._fresh("high")
            ramp = emitter._fresh("ramp")
            extrapolation = emitter._fresh("extrapolation")
            inv_interp = emitter._fresh("inv_interp")
            inv_extra = emitter._fresh("inv_extra")
            lines.append(f"{indent}if {inv_freq}.shape[0] > 1:")
            lines.append(f"{indent}    _ratio = float({inv_freq}[1] / {inv_freq}[0])")
            lines.append(f"{indent}    if _ratio <= 0.0 or _ratio == 1.0:")
            lines.append(f"{indent}        {theta_var} = 10000.0")
            lines.append(f"{indent}    else:")
            lines.append(f"{indent}        {theta_var} = _ratio ** (-(2.0 * {half}) / 2.0)")
            lines.append(f"{indent}else:")
            lines.append(f"{indent}    {theta_var} = 10000.0")
            lines.append(
                f"{indent}{low_var} = ((2.0 * {half}) * torch.log(torch.tensor(int({original_context}) / (float({high_freq_factor}) * 2.0 * torch.pi), device={q}.device, dtype=torch.float32))) / (2.0 * torch.log(torch.tensor({theta_var}, device={q}.device, dtype=torch.float32)))"
            )
            lines.append(
                f"{indent}{high_var} = ((2.0 * {half}) * torch.log(torch.tensor(int({original_context}) / (float({low_freq_factor}) * 2.0 * torch.pi), device={q}.device, dtype=torch.float32))) / (2.0 * torch.log(torch.tensor({theta_var}, device={q}.device, dtype=torch.float32)))"
            )
            if truncate:
                lines.append(f"{indent}{low_var} = torch.floor({low_var})")
                lines.append(f"{indent}{high_var} = torch.ceil({high_var})")
            lines.append(
                f"{indent}{low_var} = torch.clamp({low_var}, min=0.0, max=float((2 * {half}) - 1))"
            )
            lines.append(
                f"{indent}{high_var} = torch.clamp({high_var}, min=0.0, max=float((2 * {half}) - 1))"
            )
            lines.append(f"{indent}if float({low_var}) == float({high_var}):")
            lines.append(f"{indent}    {high_var} = {high_var} + 0.001")
            lines.append(
                f"{indent}{ramp} = torch.clamp((torch.arange({half}, dtype=torch.float32, device={q}.device) - {low_var}) / ({high_var} - {low_var}), 0.0, 1.0).to(dtype={inv_freq}.dtype)"
            )
            lines.append(f"{indent}{extrapolation} = 1.0 - {ramp}")
            lines.append(f"{indent}{inv_extra} = {inv_freq}")
            lines.append(f"{indent}{inv_interp} = {inv_freq} / float({scale_factor})")
            lines.append(
                f"{indent}{inv_freq} = {inv_interp} * (1.0 - {extrapolation}) + {inv_extra} * {extrapolation}"
            )
        else:
            lines.append(
                f"{indent}{inv_scaled} = torch.where({wavelen} > {low_freq_wavelen}, {inv_freq} / float({scale_factor}), {inv_freq})"
            )
            lines.append(
                f"{indent}{smooth} = (float({original_context}) / {wavelen} - float({low_freq_factor})) / (float({high_freq_factor}) - float({low_freq_factor}))"
            )
            lines.append(
                f"{indent}{smoothed} = (1.0 - {smooth}) * ({inv_scaled} / float({scale_factor})) + {smooth} * {inv_scaled}"
            )
            lines.append(
                f"{indent}{is_medium} = (~({wavelen} < {high_freq_wavelen})) & (~({wavelen} > {low_freq_wavelen}))"
            )
            lines.append(f"{indent}{inv_freq} = torch.where({is_medium}, {smoothed}, {inv_scaled})")
    elif "scale_factor" in node_spec:
        lines.append(f"{indent}if float({scale_factor}) <= 0.0:")
        lines.append(f"{indent}    raise ValueError('rope_pair.scale_factor must be > 0')")
        lines.append(f"{indent}{inv_freq} = {inv_freq} / float({scale_factor})")
    lines.append(f"{indent}if {pos_ids} is None:")
    lines.append(f"{indent}    raise ValueError('rope_pair.position_ids must not be null')")
    lines.append(f"{indent}if {pos_ids}.ndim != 2:")
    lines.append(
        f"{indent}    raise ValueError('rope_pair.position_ids must be rank-2 [batch, seq]')"
    )
    lines.append(f"{indent}if int({pos_ids}.shape[0]) != int({q}.shape[0]):")
    lines.append(
        f"{indent}    raise ValueError('rope_pair.position_ids batch size must match q/k batch')"
    )
    lines.append(f"{indent}if int({pos_ids}.shape[1]) != int({q}.shape[-2]):")
    lines.append(
        f"{indent}    raise ValueError('rope_pair.position_ids width must match q/k sequence length')"
    )
    lines.append(f"{indent}{pos} = {pos_ids}.to(device={q}.device, dtype=torch.float32)")
    lines.append(f"{indent}{ang} = {pos}.unsqueeze(-1) * {inv_freq}.unsqueeze(0).unsqueeze(0)")
    lines.append(
        f"{indent}{cos_half} = (torch.cos({ang}) * float({rope_attention_factor})).to(dtype={q}.dtype)"
    )
    lines.append(
        f"{indent}{sin_half} = (torch.sin({ang}) * float({rope_attention_factor})).to(dtype={q}.dtype)"
    )
    if interleaved:
        q_even = emitter._fresh("q_even")
        q_odd = emitter._fresh("q_odd")
        k_even = emitter._fresh("k_even")
        k_odd = emitter._fresh("k_odd")
        q_rot_even = emitter._fresh("q_rot_even")
        q_rot_odd = emitter._fresh("q_rot_odd")
        k_rot_even = emitter._fresh("k_rot_even")
        k_rot_odd = emitter._fresh("k_rot_odd")
        lines.append(f"{indent}{cos} = {cos_half}.unsqueeze(1)")
        lines.append(f"{indent}{sin} = {sin_half}.unsqueeze(1)")
        lines.append(f"{indent}{q_even} = {q}[..., 0::2]")
        lines.append(f"{indent}{q_odd} = {q}[..., 1::2]")
        lines.append(f"{indent}{k_even} = {k}[..., 0::2]")
        lines.append(f"{indent}{k_odd} = {k}[..., 1::2]")
        lines.append(f"{indent}{q_rot_even} = {q_even} * {cos} - {q_odd} * {sin}")
        lines.append(f"{indent}{q_rot_odd} = {q_even} * {sin} + {q_odd} * {cos}")
        lines.append(f"{indent}{k_rot_even} = {k_even} * {cos} - {k_odd} * {sin}")
        lines.append(f"{indent}{k_rot_odd} = {k_even} * {sin} + {k_odd} * {cos}")
        lines.append(f"{indent}{q_out} = torch.empty_like({q})")
        lines.append(f"{indent}{k_out} = torch.empty_like({k})")
        lines.append(f"{indent}{q_out}[..., 0::2] = {q_rot_even}")
        lines.append(f"{indent}{q_out}[..., 1::2] = {q_rot_odd}")
        lines.append(f"{indent}{k_out}[..., 0::2] = {k_rot_even}")
        lines.append(f"{indent}{k_out}[..., 1::2] = {k_rot_odd}")
    else:
        lines.append(f"{indent}{cos} = {cos_half}.unsqueeze(1)")
        lines.append(f"{indent}{sin} = {sin_half}.unsqueeze(1)")
        lines.append(f"{indent}{q1} = {q}[..., :{half}]")
        lines.append(f"{indent}{q2} = {q}[..., {half}: 2 * {half}]")
        lines.append(f"{indent}{k1} = {k}[..., :{half}]")
        lines.append(f"{indent}{k2} = {k}[..., {half}: 2 * {half}]")
        lines.append(
            f"{indent}{q_out} = torch.cat([{q1} * {cos} - {q2} * {sin}, {q1} * {sin} + {q2} * {cos}], dim=-1)"
        )
        lines.append(
            f"{indent}{k_out} = torch.cat([{k1} * {cos} - {k2} * {sin}, {k1} * {sin} + {k2} * {cos}], dim=-1)"
        )
    return lines


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any", "Any"),
    "kwargs": dict(LOWERING_KWARG_KINDS),
    "returns": ("Tensor", "Tensor"),
}

__all__ = [
    "LOWERING_ARITY",
    "LOWERING_ALLOWED_KWARGS",
    "LOWERING_REQUIRED_KWARGS",
    "LOWERING_KWARG_KINDS",
    "OP_NAME",
    "lowering_known_output_arity",
    "interpret",
    "compile",
    "uses_node_path",
    "LOWERING_TYPE_SIGNATURE",
]
