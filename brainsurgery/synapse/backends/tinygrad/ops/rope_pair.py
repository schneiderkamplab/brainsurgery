from __future__ import annotations

import math
from typing import Any

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


def interpret(
    model: Any,
    node_spec: dict[str, Any],
    env: dict[str, Any],
    *,
    node_path: str,
    scope: str,
    symbols: dict[str, int],
) -> None:
    raise NotImplementedError(f"TinyGrad interpret for '{OP_NAME}' not yet implemented")


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
        f"{indent}{inv_freq} = 1.0 / (float({theta}) ** (Tensor.arange({half}, dtype=dtypes.float32) / float({half})))"
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
            f"{indent}{pos_freqs} = float({theta}) ** (Tensor.arange({dim} // 2, dtype=dtypes.float32) / float({dim}))"
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
            f"{indent}{linear} = (Tensor.arange({dim} // 2, dtype=dtypes.float32) - {low}) / ({high} - {low})"
        )
        lines.append(f"{indent}{ramp} = {linear}.clip(0.0, 1.0)")
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
        lines.append(f"{indent}{wavelen} = (2.0 * math.pi) / {inv_freq}")
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
                f"{indent}{low_var} = ((2.0 * {half}) * math.log(float({original_context}) / (float({high_freq_factor}) * 2.0 * math.pi))) / (2.0 * math.log({theta_var}))"
            )
            lines.append(
                f"{indent}{high_var} = ((2.0 * {half}) * math.log(float({original_context}) / (float({low_freq_factor}) * 2.0 * math.pi))) / (2.0 * math.log({theta_var}))"
            )
            if truncate:
                lines.append(f"{indent}{low_var} = math.floor({low_var})")
                lines.append(f"{indent}{high_var} = math.ceil({high_var})")
            lines.append(f"{indent}{low_var} = max({low_var}, 0.0)")
            lines.append(f"{indent}{high_var} = min({high_var}, float(2 * {half} - 1))")
            lines.append(f"{indent}if float({low_var}) == float({high_var}):")
            lines.append(f"{indent}    {high_var} = {high_var} + 0.001")
            lines.append(
                f"{indent}{ramp} = (Tensor.arange({half}, dtype=dtypes.float32) - {low_var}) / ({high_var} - {low_var})"
            )
            lines.append(f"{indent}{ramp} = {ramp}.clip(0.0, 1.0).cast({inv_freq}.dtype)")
            lines.append(f"{indent}{extrapolation} = 1.0 - {ramp}")
            lines.append(f"{indent}{inv_extra} = {inv_freq}")
            lines.append(f"{indent}{inv_interp} = {inv_freq} / float({scale_factor})")
            lines.append(
                f"{indent}{inv_freq} = {inv_interp} * (1.0 - {extrapolation}) + {inv_extra} * {extrapolation}"
            )
        else:
            lines.append(
                f"{indent}{inv_scaled} = ({wavelen} > {low_freq_wavelen}).where({inv_freq} / float({scale_factor}), {inv_freq})"
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
            lines.append(f"{indent}{inv_freq} = {is_medium}.where({smoothed}, {inv_scaled})")
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
    lines.append(f"{indent}{pos} = {pos_ids}.cast(dtypes.float32)")
    lines.append(f"{indent}{ang} = {pos}.unsqueeze(-1) * {inv_freq}.unsqueeze(0).unsqueeze(0)")
    lines.append(
        f"{indent}{cos_half} = ({ang}.cos() * float({rope_attention_factor})).cast({q}.dtype)"
    )
    lines.append(
        f"{indent}{sin_half} = ({ang}.sin() * float({rope_attention_factor})).cast({q}.dtype)"
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
        lines.append(
            f"{indent}{q_out} = Tensor.stack([{q_rot_even}, {q_rot_odd}], axis=-1).reshape({q}.shape)"
        )
        lines.append(
            f"{indent}{k_out} = Tensor.stack([{k_rot_even}, {k_rot_odd}], axis=-1).reshape({k}.shape)"
        )
    else:
        lines.append(f"{indent}{cos} = {cos_half}.unsqueeze(1)")
        lines.append(f"{indent}{sin} = {sin_half}.unsqueeze(1)")
        lines.append(f"{indent}{q1} = {q}[..., :{half}]")
        lines.append(f"{indent}{q2} = {q}[..., {half}: 2 * {half}]")
        lines.append(f"{indent}{k1} = {k}[..., :{half}]")
        lines.append(f"{indent}{k2} = {k}[..., {half}: 2 * {half}]")
        lines.append(
            f"{indent}{q_out} = ({q1} * {cos} - {q2} * {sin}).cat({q1} * {sin} + {q2} * {cos}, dim=-1)"
        )
        lines.append(
            f"{indent}{k_out} = ({k1} * {cos} - {k2} * {sin}).cat({k1} * {sin} + {k2} * {cos}, dim=-1)"
        )
    return lines


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
]
