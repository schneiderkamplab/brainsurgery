from __future__ import annotations

from typing import Any

import torch

OP_NAME = "moe_grouped_ffn"
LOWERING_ARITY = (3, 3)
LOWERING_ALLOWED_KWARGS: set[str] = {
    "gate_up_weight",
    "gate_up_bias",
    "down_weight",
    "down_bias",
    "alpha",
    "limit",
    "has_bias",
    "has_gate",
    "transpose",
}
LOWERING_REQUIRED_KWARGS: set[str] = set()
LOWERING_KWARG_KINDS: dict[str, Any] = {
    "gate_up_weight": "str",
    "gate_up_bias": "str",
    "down_weight": "str",
    "down_bias": "str",
    "alpha": "number",
    "limit": "number",
    "has_bias": "bool",
    "has_gate": "bool",
    "transpose": "bool",
}


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return True


def _resolve_inputs_and_output(
    node_spec: dict[str, Any], *, strict_out: bool
) -> tuple[list[str], str]:
    ins = node_spec.get("_args")
    if not isinstance(ins, list) or len(ins) != 3 or not all(isinstance(name, str) for name in ins):
        raise ValueError("moe_grouped_ffn expects in=[hidden,topk_scores,topk_indices]")
    out_raw = node_spec.get("_bind")
    if not isinstance(out_raw, str):
        if strict_out:
            raise ValueError("moe_grouped_ffn requires a single scalar output binding")
        out_raw = str(out_raw)
    return [str(name) for name in ins], out_raw


def _resolve_bool(node_spec: dict[str, Any], key: str, *, default: bool) -> bool:
    raw = node_spec.get(key, default)
    if isinstance(raw, bool):
        return raw
    raise ValueError(f"moe_grouped_ffn {key} must be boolean")


def _resolve_float_literal(node_spec: dict[str, Any], key: str, *, default: float) -> float:
    raw = node_spec.get(key, default)
    if isinstance(raw, bool):
        raise ValueError(f"moe_grouped_ffn {key} must be numeric")
    if isinstance(raw, (int, float)):
        return float(raw)
    raise ValueError(f"moe_grouped_ffn {key} must be numeric")


def _resolve_transpose(node_spec: dict[str, Any]) -> bool:
    return _resolve_bool(node_spec, "transpose", default=True)


def _validate_inputs(
    hidden: torch.Tensor,
    topk_scores: torch.Tensor,
    topk_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if hidden.ndim < 2:
        raise ValueError("moe_grouped_ffn hidden must be at least rank-2")
    if topk_scores.ndim < 2 or topk_indices.ndim < 2:
        raise ValueError("moe_grouped_ffn topk_scores/topk_indices must be at least rank-2")
    if topk_scores.shape != topk_indices.shape:
        raise ValueError("moe_grouped_ffn topk_scores and topk_indices must have the same shape")
    if topk_indices.dtype.is_floating_point or topk_indices.dtype.is_complex:
        raise ValueError(f"moe_grouped_ffn topk_indices must be integer, got {topk_indices.dtype}")
    hidden_flat = hidden.reshape(-1, hidden.shape[-1])
    scores_flat = topk_scores.reshape(-1, topk_scores.shape[-1])
    indices_flat = topk_indices.reshape(-1, topk_indices.shape[-1])
    if hidden_flat.shape[0] != scores_flat.shape[0]:
        raise ValueError(
            "moe_grouped_ffn hidden and topk tensors must align on flattened token count"
        )
    return hidden_flat, scores_flat, indices_flat


def _infer_path(
    model: Any, node_spec: dict[str, Any], *, node_path: str, key: str, fallback: str
) -> str:
    override = dict(node_spec)
    if key in node_spec and isinstance(node_spec[key], str):
        override[key] = str(node_spec[key])
    else:
        override[key] = fallback
    return model._infer_param_path(override, node_path=node_path, param_name=key)


def _run_grouped_moe(
    *,
    hidden_flat: torch.Tensor,
    topk_scores_flat: torch.Tensor,
    topk_indices_flat: torch.Tensor,
    gate_up_weight: torch.Tensor,
    gate_up_bias: torch.Tensor | None,
    down_weight: torch.Tensor,
    down_bias: torch.Tensor | None,
    has_gate: bool,
    has_bias: bool,
    transpose: bool,
    alpha: float,
    limit: float,
) -> torch.Tensor:
    num_tokens = int(hidden_flat.shape[0])
    hidden_dim = int(hidden_flat.shape[-1])
    num_top_k = int(topk_indices_flat.shape[-1])
    num_experts = int(gate_up_weight.shape[0])
    token_idx = (
        torch.arange(num_tokens, device=hidden_flat.device)
        .unsqueeze(1)
        .expand(-1, num_top_k)
        .reshape(-1)
    )
    sample_weights = topk_scores_flat.reshape(-1)
    expert_ids = topk_indices_flat.reshape(-1)
    if expert_ids.numel() != 0:
        if int(expert_ids.min()) < 0 or int(expert_ids.max()) >= num_experts:
            raise ValueError(
                f"moe_grouped_ffn topk_indices contains out-of-range expert ids for 0..{num_experts - 1}"
            )
    selected_hidden = hidden_flat[token_idx]

    perm = torch.argsort(expert_ids)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.size(0), device=hidden_flat.device)
    expert_ids_g = expert_ids[perm]
    sample_weights_g = sample_weights[perm]
    selected_hidden_g = selected_hidden[perm]

    # Match HF grouped_mm path: offsets from histc over sorted expert ids.
    histc_input = expert_ids_g.float() if hidden_flat.device.type == "cpu" else expert_ids_g.int()
    tokens_per_expert = torch.histc(histc_input, bins=num_experts, min=0, max=num_experts - 1)
    offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)

    def grouped_mm(input_: torch.Tensor, weight: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
        aligned_cpu = input_.device.type != "cpu" or (
            (input_.data_ptr() % 16 == 0)
            and (weight.data_ptr() % 16 == 0)
            and all(((stride * input_.element_size()) % 16 == 0) for stride in input_.stride())
            and all(((stride * weight.element_size()) % 16 == 0) for stride in weight.stride())
        )
        if hasattr(torch.nn.functional, "grouped_mm") and aligned_cpu:
            return torch.nn.functional.grouped_mm(input_.to(weight.dtype), weight, offs=offs)
        if hasattr(torch, "_grouped_mm") and aligned_cpu:
            return torch._grouped_mm(input_.to(weight.dtype), weight, offs=offs)
        out = torch.zeros(
            input_.size(0),
            weight.size(2),
            device=input_.device,
            dtype=input_.dtype,
        )
        start = 0
        for i, end in enumerate(offs.tolist()):
            if start == end:
                continue
            torch.mm(input_[start:end], weight[i], out=out[start:end])
            start = end
        return out

    def grouped_linear(
        input_: torch.Tensor,
        weight: torch.Tensor,
        offs: torch.Tensor,
        bias: torch.Tensor | None,
        *,
        is_transposed: bool,
    ) -> torch.Tensor:
        if is_transposed:
            out = grouped_mm(input_, weight, offs)
        else:
            out = grouped_mm(input_, weight.transpose(-2, -1), offs)
        if bias is not None:
            out = out + bias
        return out

    up_bias = gate_up_bias[expert_ids_g] if has_bias and gate_up_bias is not None else None
    proj = grouped_linear(
        selected_hidden_g,
        gate_up_weight,
        offsets,
        up_bias,
        is_transposed=transpose,
    )
    if has_gate:
        gate, up = proj[..., ::2], proj[..., 1::2]
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
        proj = (up + 1.0) * (gate * torch.sigmoid(gate * alpha))
    else:
        proj = torch.nn.functional.silu(proj)

    down_bias_g = down_bias[expert_ids_g] if has_bias and down_bias is not None else None
    down = grouped_linear(
        proj,
        down_weight,
        offsets,
        down_bias_g,
        is_transposed=transpose,
    )
    weighted = down * sample_weights_g.unsqueeze(-1)
    token_idx_g = token_idx[perm]
    out = torch.zeros(num_tokens, hidden_dim, device=hidden_flat.device, dtype=hidden_flat.dtype)
    out.index_add_(0, token_idx_g, weighted.to(out.dtype))
    return out


def interpret(
    model: Any,
    node_spec: dict[str, Any],
    env: dict[str, Any],
    *,
    node_path: str,
    scope: str,
    symbols: dict[str, int],
) -> None:
    del scope
    ins, out = _resolve_inputs_and_output(node_spec, strict_out=True)
    hidden = model._read_tensor_input(ins[0], env)
    topk_scores = model._read_tensor_input(ins[1], env)
    topk_indices = model._read_tensor_input(ins[2], env)
    hidden_flat, topk_scores_flat, topk_indices_flat = _validate_inputs(
        hidden, topk_scores, topk_indices
    )

    has_bias = _resolve_bool(node_spec, "has_bias", default=True)
    has_gate = _resolve_bool(node_spec, "has_gate", default=True)
    transpose = _resolve_transpose(node_spec)
    alpha_raw = node_spec.get("alpha", 1.702)
    limit_raw = node_spec.get("limit", 7.0)
    alpha_eval = model._eval_expr(alpha_raw, env, symbols)
    limit_eval = model._eval_expr(limit_raw, env, symbols)
    if isinstance(alpha_eval, bool) or not isinstance(alpha_eval, (int, float)):
        raise ValueError(f"moe_grouped_ffn alpha must evaluate to numeric, got {alpha_eval!r}")
    if isinstance(limit_eval, bool) or not isinstance(limit_eval, (int, float)):
        raise ValueError(f"moe_grouped_ffn limit must evaluate to numeric, got {limit_eval!r}")
    alpha = float(alpha_eval)
    limit = float(limit_eval)

    gate_up_weight_path = _infer_path(
        model,
        node_spec,
        node_path=node_path,
        key="gate_up_weight",
        fallback="experts.gate_up_proj.weight",
    )
    gate_up_bias_path = _infer_path(
        model,
        node_spec,
        node_path=node_path,
        key="gate_up_bias",
        fallback="experts.gate_up_proj.bias",
    )
    down_weight_path = _infer_path(
        model,
        node_spec,
        node_path=node_path,
        key="down_weight",
        fallback="experts.down_proj.weight",
    )
    down_bias_path = _infer_path(
        model,
        node_spec,
        node_path=node_path,
        key="down_bias",
        fallback="experts.down_proj.bias",
    )

    gate_up_weight = model._state[gate_up_weight_path]
    gate_up_bias = (
        model._state_tensor_from_resolved_path(
            gate_up_bias_path, field="moe_grouped_ffn.gate_up_bias"
        )
        if has_bias
        else None
    )
    down_weight = model._state[down_weight_path]
    down_bias = (
        model._state_tensor_from_resolved_path(down_bias_path, field="moe_grouped_ffn.down_bias")
        if has_bias
        else None
    )

    out_flat = _run_grouped_moe(
        hidden_flat=hidden_flat,
        topk_scores_flat=topk_scores_flat,
        topk_indices_flat=topk_indices_flat,
        gate_up_weight=gate_up_weight,
        gate_up_bias=gate_up_bias,
        down_weight=down_weight,
        down_bias=down_bias,
        has_gate=has_gate,
        has_bias=has_bias,
        transpose=transpose,
        alpha=alpha,
        limit=limit,
    )
    env[out] = out_flat.to(hidden.dtype).reshape(*hidden.shape[:-1], hidden.shape[-1])


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

    def assign_out_var(out_name: str) -> str:
        return emitter._assign_out_var(env, out_name)

    def read(name: str) -> str:
        return emitter._read_env_var(env, name)

    def infer_param(key: str, fallback: str) -> str:
        override = dict(node_spec)
        if key in node_spec and isinstance(node_spec[key], str):
            override[key] = str(node_spec[key])
        else:
            override[key] = fallback
        return emitter._infer_param_expr(override, node_path_var, key)

    ins, out_name = _resolve_inputs_and_output(node_spec, strict_out=False)
    hidden = read(ins[0])
    topk_scores = read(ins[1])
    topk_indices = read(ins[2])
    out_var = assign_out_var(out_name)
    has_bias = _resolve_bool(node_spec, "has_bias", default=True)
    has_gate = _resolve_bool(node_spec, "has_gate", default=True)
    transpose = _resolve_transpose(node_spec)
    if isinstance(node_spec.get("alpha", 1.702), bool):
        _resolve_float_literal(node_spec, "alpha", default=1.702)
    if isinstance(node_spec.get("limit", 7.0), bool):
        _resolve_float_literal(node_spec, "limit", default=7.0)
    alpha_code = emitter._expr_code(node_spec.get("alpha", 1.702), env)
    limit_code = emitter._expr_code(node_spec.get("limit", 7.0), env)

    hidden_flat = emitter._fresh("hidden_flat")
    topk_scores_flat = emitter._fresh("topk_scores_flat")
    topk_indices_flat = emitter._fresh("topk_indices_flat")
    gate_up_weight = emitter._fresh("gate_up_weight")
    gate_up_bias = emitter._fresh("gate_up_bias")
    down_weight = emitter._fresh("down_weight")
    down_bias = emitter._fresh("down_bias")
    alpha = emitter._fresh("alpha")
    limit = emitter._fresh("limit")
    final_hidden = emitter._fresh("final_hidden")

    lines.append(
        f"{indent}from brainsurgery.synapse.ops import moe_grouped_ffn as _moe_grouped_ffn_mod"
    )
    lines.append(
        f"{indent}{hidden_flat}, {topk_scores_flat}, {topk_indices_flat} = _moe_grouped_ffn_mod._validate_inputs({hidden}, {topk_scores}, {topk_indices})"
    )
    gate_up_weight_expr = infer_param("gate_up_weight", "experts.gate_up_proj.weight")
    gate_up_bias_expr = infer_param("gate_up_bias", "experts.gate_up_proj.bias")
    down_weight_expr = infer_param("down_weight", "experts.down_proj.weight")
    down_bias_expr = infer_param("down_bias", "experts.down_proj.bias")
    gate_up_weight_path = emitter._hoist_expr(
        kind="param_path",
        key=f"gate_up_weight:{gate_up_weight_expr}",
        expr=gate_up_weight_expr,
        lines=lines,
        indent=indent,
    )
    gate_up_bias_path = emitter._hoist_expr(
        kind="param_path",
        key=f"gate_up_bias:{gate_up_bias_expr}",
        expr=gate_up_bias_expr,
        lines=lines,
        indent=indent,
    )
    down_weight_path = emitter._hoist_expr(
        kind="param_path",
        key=f"down_weight:{down_weight_expr}",
        expr=down_weight_expr,
        lines=lines,
        indent=indent,
    )
    down_bias_path = emitter._hoist_expr(
        kind="param_path",
        key=f"down_bias:{down_bias_expr}",
        expr=down_bias_expr,
        lines=lines,
        indent=indent,
    )
    gate_up_weight_value = emitter._hoist_expr(
        kind="param_tensor",
        key=f"required:{gate_up_weight_path}",
        expr=f"self._param({gate_up_weight_path})",
        lines=lines,
        indent=indent,
    )
    gate_up_bias_value = emitter._hoist_expr(
        kind="param_tensor_opt",
        key=f"optional:{gate_up_bias_path}",
        expr=f"self._state.get({gate_up_bias_path})",
        lines=lines,
        indent=indent,
    )
    down_weight_value = emitter._hoist_expr(
        kind="param_tensor",
        key=f"required:{down_weight_path}",
        expr=f"self._param({down_weight_path})",
        lines=lines,
        indent=indent,
    )
    down_bias_value = emitter._hoist_expr(
        kind="param_tensor_opt",
        key=f"optional:{down_bias_path}",
        expr=f"self._state.get({down_bias_path})",
        lines=lines,
        indent=indent,
    )
    lines.append(f"{indent}{gate_up_weight} = {gate_up_weight_value}")
    lines.append(f"{indent}{gate_up_bias} = {gate_up_bias_value} if {has_bias!r} else None")
    lines.append(f"{indent}{down_weight} = {down_weight_value}")
    lines.append(f"{indent}{down_bias} = {down_bias_value} if {has_bias!r} else None")
    if has_bias:
        lines.append(f"{indent}if {gate_up_bias} is None:")
        lines.append(
            f"{indent}    raise ValueError('moe_grouped_ffn.gate_up_bias tensor not found for resolved path')"
        )
        lines.append(f"{indent}if {down_bias} is None:")
        lines.append(
            f"{indent}    raise ValueError('moe_grouped_ffn.down_bias tensor not found for resolved path')"
        )
    lines.append(f"{indent}{alpha} = float({alpha_code})")
    lines.append(f"{indent}{limit} = float({limit_code})")
    lines.append(f"{indent}{final_hidden} = _moe_grouped_ffn_mod._run_grouped_moe(")
    lines.append(f"{indent}    hidden_flat={hidden_flat},")
    lines.append(f"{indent}    topk_scores_flat={topk_scores_flat},")
    lines.append(f"{indent}    topk_indices_flat={topk_indices_flat},")
    lines.append(f"{indent}    gate_up_weight={gate_up_weight},")
    lines.append(f"{indent}    gate_up_bias={gate_up_bias},")
    lines.append(f"{indent}    down_weight={down_weight},")
    lines.append(f"{indent}    down_bias={down_bias},")
    lines.append(f"{indent}    has_gate={has_gate!r},")
    lines.append(f"{indent}    has_bias={has_bias!r},")
    lines.append(f"{indent}    transpose={transpose!r},")
    lines.append(f"{indent}    alpha={alpha},")
    lines.append(f"{indent}    limit={limit},")
    lines.append(f"{indent})")
    lines.append(
        f"{indent}{out_var} = {final_hidden}.to({hidden}.dtype).reshape(*{hidden}.shape[:-1], {hidden}.shape[-1])"
    )
    return lines


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any", "Any", "Any"),
    "kwargs": dict(LOWERING_KWARG_KINDS),
    "returns": ("Tensor",),
}

__all__ = [
    "LOWERING_ARITY",
    "LOWERING_ALLOWED_KWARGS",
    "LOWERING_REQUIRED_KWARGS",
    "LOWERING_KWARG_KINDS",
    "OP_NAME",
    "interpret",
    "compile",
    "uses_node_path",
    "LOWERING_TYPE_SIGNATURE",
]
