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
    if key in node_spec and isinstance(node_spec[key], str):
        override = dict(node_spec)
        override[key] = str(node_spec[key])
        return model._infer_param_path(override, node_path=node_path, param_name=key)
    return model._join(model._scope_of(node_path), fallback)


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
        fallback="mlp.experts.gate_up_proj.weight",
    )
    gate_up_bias_path = _infer_path(
        model,
        node_spec,
        node_path=node_path,
        key="gate_up_bias",
        fallback="mlp.experts.gate_up_proj.bias",
    )
    down_weight_path = _infer_path(
        model,
        node_spec,
        node_path=node_path,
        key="down_weight",
        fallback="mlp.experts.down_proj.weight",
    )
    down_bias_path = _infer_path(
        model,
        node_spec,
        node_path=node_path,
        key="down_bias",
        fallback="mlp.experts.down_proj.bias",
    )

    gate_up_weight = model._state[gate_up_weight_path]
    gate_up_bias = model._state.get(gate_up_bias_path) if has_bias else None
    down_weight = model._state[down_weight_path]
    down_bias = model._state.get(down_bias_path) if has_bias else None

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
        if key in node_spec and isinstance(node_spec[key], str):
            return emitter._infer_param_expr(node_spec, node_path_var, key)
        return f"self._join_scope(self._scope_of({node_path_var}), {fallback!r})"

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
    num_tokens = emitter._fresh("num_tokens")
    hidden_dim = emitter._fresh("hidden_dim")
    num_top_k = emitter._fresh("num_top_k")
    num_experts = emitter._fresh("num_experts")
    token_idx = emitter._fresh("token_idx")
    sample_weights = emitter._fresh("sample_weights")
    expert_ids = emitter._fresh("expert_ids")
    selected_hidden = emitter._fresh("selected_hidden")
    perm = emitter._fresh("perm")
    inv_perm = emitter._fresh("inv_perm")
    expert_ids_g = emitter._fresh("expert_ids_g")
    sample_weights_g = emitter._fresh("sample_weights_g")
    selected_hidden_g = emitter._fresh("selected_hidden_g")
    histc_input = emitter._fresh("histc_input")
    tokens_per_expert = emitter._fresh("tokens_per_expert")
    offsets = emitter._fresh("offsets")
    gate_up_weight = emitter._fresh("gate_up_weight")
    gate_up_bias = emitter._fresh("gate_up_bias")
    down_weight = emitter._fresh("down_weight")
    down_bias = emitter._fresh("down_bias")
    up_bias = emitter._fresh("up_bias")
    down_bias_g = emitter._fresh("down_bias_g")
    proj = emitter._fresh("proj")
    gate = emitter._fresh("gate")
    up = emitter._fresh("up")
    down = emitter._fresh("down")
    weighted = emitter._fresh("weighted")
    token_idx_g = emitter._fresh("token_idx_g")
    final_hidden = emitter._fresh("final_hidden")
    start = emitter._fresh("start")
    end = emitter._fresh("end")
    out_mm = emitter._fresh("out_mm")

    gate_up_weight_expr = infer_param("gate_up_weight", "mlp.experts.gate_up_proj.weight")
    gate_up_bias_expr = infer_param("gate_up_bias", "mlp.experts.gate_up_proj.bias")
    down_weight_expr = infer_param("down_weight", "mlp.experts.down_proj.weight")
    down_bias_expr = infer_param("down_bias", "mlp.experts.down_proj.bias")

    lines.append(f"{indent}{hidden_flat} = {hidden}.reshape(-1, {hidden}.shape[-1])")
    lines.append(f"{indent}{topk_scores_flat} = {topk_scores}.reshape(-1, {topk_scores}.shape[-1])")
    lines.append(
        f"{indent}{topk_indices_flat} = {topk_indices}.reshape(-1, {topk_indices}.shape[-1])"
    )
    lines.append(f"{indent}{num_tokens} = int({hidden_flat}.shape[0])")
    lines.append(f"{indent}{hidden_dim} = int({hidden_flat}.shape[-1])")
    lines.append(f"{indent}{num_top_k} = int({topk_indices_flat}.shape[-1])")
    lines.append(f"{indent}{gate_up_weight} = self._param({gate_up_weight_expr})")
    lines.append(
        f"{indent}{gate_up_bias} = self._state.get({gate_up_bias_expr}) if {has_bias!r} else None"
    )
    lines.append(f"{indent}{down_weight} = self._param({down_weight_expr})")
    lines.append(
        f"{indent}{down_bias} = self._state.get({down_bias_expr}) if {has_bias!r} else None"
    )
    lines.append(f"{indent}{num_experts} = int({gate_up_weight}.shape[0])")
    lines.append(
        f"{indent}{token_idx} = torch.arange({num_tokens}, device={hidden_flat}.device).unsqueeze(1).expand(-1, {num_top_k}).reshape(-1)"
    )
    lines.append(f"{indent}{sample_weights} = {topk_scores_flat}.reshape(-1)")
    lines.append(f"{indent}{expert_ids} = {topk_indices_flat}.reshape(-1)")
    lines.append(f"{indent}if {expert_ids}.numel() != 0:")
    lines.append(
        f"{indent}    if int({expert_ids}.min()) < 0 or int({expert_ids}.max()) >= {num_experts}:"
    )
    lines.append(
        f'{indent}        raise ValueError(f"moe_grouped_ffn topk_indices contains out-of-range expert ids for 0..{{{num_experts} - 1}}")'
    )
    lines.append(f"{indent}{selected_hidden} = {hidden_flat}[{token_idx}]")
    lines.append(f"{indent}{perm} = torch.argsort({expert_ids})")
    lines.append(f"{indent}{inv_perm} = torch.empty_like({perm})")
    lines.append(
        f"{indent}{inv_perm}[{perm}] = torch.arange({perm}.size(0), device={hidden_flat}.device)"
    )
    lines.append(f"{indent}{expert_ids_g} = {expert_ids}[{perm}]")
    lines.append(f"{indent}{sample_weights_g} = {sample_weights}[{perm}]")
    lines.append(f"{indent}{selected_hidden_g} = {selected_hidden}[{perm}]")
    lines.append(
        f"{indent}{histc_input} = {expert_ids_g}.float() if {hidden_flat}.device.type == 'cpu' else {expert_ids_g}.int()"
    )
    lines.append(
        f"{indent}{tokens_per_expert} = torch.histc({histc_input}, bins={num_experts}, min=0, max={num_experts} - 1)"
    )
    lines.append(f"{indent}{offsets} = torch.cumsum({tokens_per_expert}, dim=0, dtype=torch.int32)")
    lines.append(
        f"{indent}{up_bias} = ({gate_up_bias}[{expert_ids_g}] if {gate_up_bias} is not None else None)"
    )
    lines.append(
        f"{indent}if hasattr(torch.nn.functional, 'grouped_mm') and ({selected_hidden_g}.device.type != 'cpu' or ({selected_hidden_g}.data_ptr() % 16 == 0 and {gate_up_weight}.data_ptr() % 16 == 0 and all(((s * {selected_hidden_g}.element_size()) % 16 == 0) for s in {selected_hidden_g}.stride()) and all(((s * {gate_up_weight}.element_size()) % 16 == 0) for s in {gate_up_weight}.stride()))):"
    )
    if transpose:
        lines.append(
            f"{indent}    {proj} = torch.nn.functional.grouped_mm({selected_hidden_g}.to({gate_up_weight}.dtype), {gate_up_weight}, offs={offsets})"
        )
    else:
        lines.append(
            f"{indent}    {proj} = torch.nn.functional.grouped_mm({selected_hidden_g}.to({gate_up_weight}.dtype), {gate_up_weight}.transpose(-2, -1), offs={offsets})"
        )
    lines.append(
        f"{indent}elif hasattr(torch, '_grouped_mm') and ({selected_hidden_g}.device.type != 'cpu' or ({selected_hidden_g}.data_ptr() % 16 == 0 and {gate_up_weight}.data_ptr() % 16 == 0 and all(((s * {selected_hidden_g}.element_size()) % 16 == 0) for s in {selected_hidden_g}.stride()) and all(((s * {gate_up_weight}.element_size()) % 16 == 0) for s in {gate_up_weight}.stride()))):"
    )
    if transpose:
        lines.append(
            f"{indent}    {proj} = torch._grouped_mm({selected_hidden_g}.to({gate_up_weight}.dtype), {gate_up_weight}, offs={offsets})"
        )
    else:
        lines.append(
            f"{indent}    {proj} = torch._grouped_mm({selected_hidden_g}.to({gate_up_weight}.dtype), {gate_up_weight}.transpose(-2, -1), offs={offsets})"
        )
    lines.append(f"{indent}else:")
    lines.append(
        f"{indent}    {out_mm} = torch.zeros({selected_hidden_g}.size(0), {gate_up_weight}.shape[2], device={selected_hidden_g}.device, dtype={selected_hidden_g}.dtype)"
    )
    lines.append(f"{indent}    {start} = 0")
    lines.append(f"{indent}    for i, {end} in enumerate({offsets}.tolist()):")
    lines.append(f"{indent}        if {start} == {end}:")
    lines.append(f"{indent}            continue")
    if transpose:
        lines.append(
            f"{indent}        torch.mm({selected_hidden_g}[{start}:{end}], {gate_up_weight}[i], out={out_mm}[{start}:{end}])"
        )
    else:
        lines.append(
            f"{indent}        torch.mm({selected_hidden_g}[{start}:{end}], {gate_up_weight}[i].transpose(-2, -1), out={out_mm}[{start}:{end}])"
        )
    lines.append(f"{indent}        {start} = {end}")
    lines.append(f"{indent}    {proj} = {out_mm}")
    lines.append(f"{indent}if {up_bias} is not None:")
    lines.append(f"{indent}    {proj} = {proj} + {up_bias}")
    if has_gate:
        lines.append(f"{indent}{gate} = {proj}[..., ::2].clamp(max=float({limit_code}))")
        lines.append(
            f"{indent}{up} = {proj}[..., 1::2].clamp(min=-float({limit_code}), max=float({limit_code}))"
        )
        lines.append(
            f"{indent}{proj} = ({up} + 1.0) * ({gate} * torch.sigmoid({gate} * float({alpha_code})))"
        )
    else:
        lines.append(f"{indent}{proj} = torch.nn.functional.silu({proj})")
    lines.append(
        f"{indent}{down_bias_g} = ({down_bias}[{expert_ids_g}] if {down_bias} is not None else None)"
    )
    lines.append(
        f"{indent}if hasattr(torch.nn.functional, 'grouped_mm') and ({proj}.device.type != 'cpu' or ({proj}.data_ptr() % 16 == 0 and {down_weight}.data_ptr() % 16 == 0 and all(((s * {proj}.element_size()) % 16 == 0) for s in {proj}.stride()) and all(((s * {down_weight}.element_size()) % 16 == 0) for s in {down_weight}.stride()))):"
    )
    if transpose:
        lines.append(
            f"{indent}    {down} = torch.nn.functional.grouped_mm({proj}.to({down_weight}.dtype), {down_weight}, offs={offsets})"
        )
    else:
        lines.append(
            f"{indent}    {down} = torch.nn.functional.grouped_mm({proj}.to({down_weight}.dtype), {down_weight}.transpose(-2, -1), offs={offsets})"
        )
    lines.append(
        f"{indent}elif hasattr(torch, '_grouped_mm') and ({proj}.device.type != 'cpu' or ({proj}.data_ptr() % 16 == 0 and {down_weight}.data_ptr() % 16 == 0 and all(((s * {proj}.element_size()) % 16 == 0) for s in {proj}.stride()) and all(((s * {down_weight}.element_size()) % 16 == 0) for s in {down_weight}.stride()))):"
    )
    if transpose:
        lines.append(
            f"{indent}    {down} = torch._grouped_mm({proj}.to({down_weight}.dtype), {down_weight}, offs={offsets})"
        )
    else:
        lines.append(
            f"{indent}    {down} = torch._grouped_mm({proj}.to({down_weight}.dtype), {down_weight}.transpose(-2, -1), offs={offsets})"
        )
    lines.append(f"{indent}else:")
    lines.append(
        f"{indent}    {out_mm} = torch.zeros({proj}.size(0), {down_weight}.shape[2], device={proj}.device, dtype={proj}.dtype)"
    )
    lines.append(f"{indent}    {start} = 0")
    lines.append(f"{indent}    for i, {end} in enumerate({offsets}.tolist()):")
    lines.append(f"{indent}        if {start} == {end}:")
    lines.append(f"{indent}            continue")
    if transpose:
        lines.append(
            f"{indent}        torch.mm({proj}[{start}:{end}], {down_weight}[i], out={out_mm}[{start}:{end}])"
        )
    else:
        lines.append(
            f"{indent}        torch.mm({proj}[{start}:{end}], {down_weight}[i].transpose(-2, -1), out={out_mm}[{start}:{end}])"
        )
    lines.append(f"{indent}        {start} = {end}")
    lines.append(f"{indent}    {down} = {out_mm}")
    lines.append(f"{indent}if {down_bias_g} is not None:")
    lines.append(f"{indent}    {down} = {down} + {down_bias_g}")
    lines.append(f"{indent}{weighted} = {down} * {sample_weights_g}.unsqueeze(-1)")
    lines.append(f"{indent}{token_idx_g} = {token_idx}[{perm}]")
    lines.append(
        f"{indent}{final_hidden} = torch.zeros({num_tokens}, {hidden_dim}, device={hidden_flat}.device, dtype={hidden_flat}.dtype)"
    )
    lines.append(
        f"{indent}{final_hidden}.index_add_(0, {token_idx_g}, {weighted}.to({final_hidden}.dtype))"
    )
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
