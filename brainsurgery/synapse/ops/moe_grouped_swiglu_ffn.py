from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

from . import moe_grouped_ffn as _base

OP_NAME = "moe_grouped_swiglu_ffn"
LOWERING_ARITY = (3, 3)
LOWERING_ALLOWED_KWARGS: set[str] = {
    "gate_up_weight",
    "gate_up_scale",
    "down_weight",
    "out_weight",
    "down_scale",
    "transpose",
    "pre_scale_input",
}
LOWERING_REQUIRED_KWARGS: set[str] = set()
LOWERING_KWARG_KINDS: dict[str, Any] = {
    "gate_up_weight": "str",
    "gate_up_scale": "str",
    "down_weight": "str",
    "out_weight": "str",
    "down_scale": "str",
    "transpose": "bool",
    "pre_scale_input": "bool",
}


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return True


def _resolve_inputs_and_output(
    node_spec: dict[str, Any], *, strict_out: bool
) -> tuple[list[str], str]:
    return _base._resolve_inputs_and_output(node_spec, strict_out=strict_out)


def _resolve_transpose(node_spec: dict[str, Any]) -> bool:
    raw = node_spec.get("transpose", False)
    if isinstance(raw, bool):
        return raw
    raise ValueError("moe_grouped_swiglu_ffn transpose must be boolean")


def _resolve_pre_scale_input(node_spec: dict[str, Any]) -> bool:
    raw = node_spec.get("pre_scale_input", False)
    if isinstance(raw, bool):
        return raw
    raise ValueError("moe_grouped_swiglu_ffn pre_scale_input must be boolean")


def _infer_path(
    model: Any, node_spec: dict[str, Any], *, node_path: str, key: str, fallback: str
) -> str:
    return _base._infer_path(model, node_spec, node_path=node_path, key=key, fallback=fallback)


def _infer_down_weight_path(model: Any, node_spec: dict[str, Any], *, node_path: str) -> str:
    if isinstance(node_spec.get("down_weight"), str):
        return _infer_path(
            model,
            node_spec,
            node_path=node_path,
            key="down_weight",
            fallback="experts.down_proj",
        )
    if isinstance(node_spec.get("out_weight"), str):
        override = dict(node_spec)
        override["down_weight"] = str(node_spec["out_weight"])
        return _infer_path(
            model,
            override,
            node_path=node_path,
            key="down_weight",
            fallback="experts.down_proj",
        )
    return _infer_path(
        model,
        node_spec,
        node_path=node_path,
        key="down_weight",
        fallback="experts.down_proj",
    )


def _maybe_dequantize(weight: torch.Tensor, scale: torch.Tensor | None) -> torch.Tensor:
    if scale is None:
        return weight
    return weight.float() * scale.float()


def _run_grouped_swiglu_moe(
    *,
    hidden_flat: torch.Tensor,
    topk_scores_flat: torch.Tensor,
    topk_indices_flat: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    transpose: bool,
    pre_scale_input: bool,
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
                f"moe_grouped_swiglu_ffn topk_indices contains out-of-range expert ids for 0..{num_experts - 1}"
            )
    selected_hidden = hidden_flat[token_idx]
    if pre_scale_input:
        selected_hidden = selected_hidden * sample_weights.unsqueeze(-1).to(selected_hidden.dtype)

    perm = torch.argsort(expert_ids)
    expert_ids_g = expert_ids[perm]
    sample_weights_g = sample_weights[perm]
    selected_hidden_g = selected_hidden[perm]

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
        input_: torch.Tensor, weight: torch.Tensor, offs: torch.Tensor
    ) -> torch.Tensor:
        if transpose:
            return grouped_mm(input_, weight, offs)
        return grouped_mm(input_, weight.transpose(-2, -1), offs)

    proj = grouped_linear(selected_hidden_g, gate_up_weight, offsets)
    gate, up = proj.chunk(2, dim=-1)
    proj = F.silu(gate) * up
    down = grouped_linear(proj, down_weight, offsets)
    weighted = down if pre_scale_input else (down * sample_weights_g.unsqueeze(-1).to(down.dtype))
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
    del scope, symbols
    ins, out = _resolve_inputs_and_output(node_spec, strict_out=True)
    hidden = model._read_tensor_input(ins[0], env)
    topk_scores = model._read_tensor_input(ins[1], env)
    topk_indices = model._read_tensor_input(ins[2], env)
    hidden_flat, topk_scores_flat, topk_indices_flat = _base._validate_inputs(
        hidden, topk_scores, topk_indices
    )
    transpose = _resolve_transpose(node_spec)
    pre_scale_input = _resolve_pre_scale_input(node_spec)

    gate_up_weight_path = _infer_path(
        model, node_spec, node_path=node_path, key="gate_up_weight", fallback="experts.gate_up_proj"
    )
    gate_up_scale_path = _infer_path(
        model,
        node_spec,
        node_path=node_path,
        key="gate_up_scale",
        fallback="experts.gate_up_proj_scale_inv",
    )
    down_weight_path = _infer_down_weight_path(model, node_spec, node_path=node_path)
    down_scale_path = _infer_path(
        model,
        node_spec,
        node_path=node_path,
        key="down_scale",
        fallback="experts.down_proj_scale_inv",
    )

    gate_up_weight = model._param(gate_up_weight_path)
    gate_up_scale = (
        model._state_tensor_from_resolved_path(
            gate_up_scale_path, field="moe_grouped_swiglu_ffn.gate_up_scale"
        )
        if isinstance(node_spec.get("gate_up_scale"), str)
        else model._state.get(gate_up_scale_path)
    )
    down_weight = model._param(down_weight_path)
    down_scale = (
        model._state_tensor_from_resolved_path(
            down_scale_path, field="moe_grouped_swiglu_ffn.down_scale"
        )
        if isinstance(node_spec.get("down_scale"), str)
        else model._state.get(down_scale_path)
    )

    final_hidden = _run_grouped_swiglu_moe(
        hidden_flat=hidden_flat,
        topk_scores_flat=topk_scores_flat,
        topk_indices_flat=topk_indices_flat,
        gate_up_weight=_maybe_dequantize(gate_up_weight, gate_up_scale),
        down_weight=_maybe_dequantize(down_weight, down_scale),
        transpose=transpose,
        pre_scale_input=pre_scale_input,
    )
    env[out] = final_hidden.to(hidden.dtype).reshape(*hidden.shape[:-1], hidden.shape[-1])


def compile(
    emitter: Any,
    node_spec: dict[str, Any],
    env: dict[str, str],
    *,
    node_path_var: str,
    scope_var: str,
    indent: str,
) -> list[str]:
    ins, out_name = _resolve_inputs_and_output(node_spec, strict_out=True)
    hidden = emitter._read_env_var(env, ins[0])
    topk_scores = emitter._read_env_var(env, ins[1])
    topk_indices = emitter._read_env_var(env, ins[2])
    out_var = emitter._assign_out_var(env, out_name)
    transpose = _resolve_transpose(node_spec)
    pre_scale_input = _resolve_pre_scale_input(node_spec)

    def infer_param(key: str, fallback: str) -> str:
        override = dict(node_spec)
        if not isinstance(override.get(key), str):
            override[key] = fallback
        return emitter._infer_param_expr(
            override,
            node_path_var,
            key,
            env=env,
            scope_var=scope_var,
        )

    lines: list[str] = []
    hidden_flat = emitter._fresh("hidden_flat")
    topk_scores_flat = emitter._fresh("topk_scores_flat")
    topk_indices_flat = emitter._fresh("topk_indices_flat")
    gate_up_weight = emitter._fresh("gate_up_weight")
    gate_up_scale = emitter._fresh("gate_up_scale")
    down_weight = emitter._fresh("down_weight")
    down_scale = emitter._fresh("down_scale")
    final_hidden = emitter._fresh("final_hidden")

    lines.append(
        f"{indent}from brainsurgery.synapse.ops import moe_grouped_swiglu_ffn as _moe_grouped_swiglu_ffn_mod"
    )
    lines.append(
        f"{indent}{hidden_flat}, {topk_scores_flat}, {topk_indices_flat} = _moe_grouped_swiglu_ffn_mod._base._validate_inputs({hidden}, {topk_scores}, {topk_indices})"
    )

    gate_up_weight_expr = infer_param("gate_up_weight", "experts.gate_up_proj")
    gate_up_scale_expr = infer_param("gate_up_scale", "experts.gate_up_proj_scale_inv")
    down_weight_key = (
        "down_weight"
        if isinstance(node_spec.get("down_weight"), str)
        or not isinstance(node_spec.get("out_weight"), str)
        else "out_weight"
    )
    down_weight_expr = infer_param(down_weight_key, "experts.down_proj")
    down_scale_expr = infer_param("down_scale", "experts.down_proj_scale_inv")

    gate_up_weight_path = emitter._hoist_expr(
        kind="param_path",
        key=f"gate_up_weight:{gate_up_weight_expr}",
        expr=gate_up_weight_expr,
        lines=lines,
        indent=indent,
    )
    gate_up_scale_path = emitter._hoist_expr(
        kind="param_path",
        key=f"gate_up_scale:{gate_up_scale_expr}",
        expr=gate_up_scale_expr,
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
    down_scale_path = emitter._hoist_expr(
        kind="param_path",
        key=f"down_scale:{down_scale_expr}",
        expr=down_scale_expr,
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
    gate_up_scale_value = emitter._hoist_expr(
        kind="param_tensor_opt",
        key=f"optional:{gate_up_scale_path}",
        expr=f"self._state.get({gate_up_scale_path})",
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
    down_scale_value = emitter._hoist_expr(
        kind="param_tensor_opt",
        key=f"optional:{down_scale_path}",
        expr=f"self._state.get({down_scale_path})",
        lines=lines,
        indent=indent,
    )
    lines.append(
        f"{indent}{gate_up_weight} = _moe_grouped_swiglu_ffn_mod._maybe_dequantize({gate_up_weight_value}, {gate_up_scale_value})"
    )
    lines.append(f"{indent}{gate_up_scale} = {gate_up_scale_value}")
    lines.append(
        f"{indent}{down_weight} = _moe_grouped_swiglu_ffn_mod._maybe_dequantize({down_weight_value}, {down_scale_value})"
    )
    lines.append(f"{indent}{down_scale} = {down_scale_value}")
    if isinstance(node_spec.get("gate_up_scale"), str):
        lines.append(f"{indent}if {gate_up_scale} is None:")
        lines.append(
            f"{indent}    raise ValueError('moe_grouped_swiglu_ffn.gate_up_scale tensor not found for resolved path')"
        )
    if isinstance(node_spec.get("down_scale"), str):
        lines.append(f"{indent}if {down_scale} is None:")
        lines.append(
            f"{indent}    raise ValueError('moe_grouped_swiglu_ffn.down_scale tensor not found for resolved path')"
        )
    lines.append(f"{indent}{final_hidden} = _moe_grouped_swiglu_ffn_mod._run_grouped_swiglu_moe(")
    lines.append(f"{indent}    hidden_flat={hidden_flat},")
    lines.append(f"{indent}    topk_scores_flat={topk_scores_flat},")
    lines.append(f"{indent}    topk_indices_flat={topk_indices_flat},")
    lines.append(f"{indent}    gate_up_weight={gate_up_weight},")
    lines.append(f"{indent}    down_weight={down_weight},")
    lines.append(f"{indent}    transpose={transpose!r},")
    lines.append(f"{indent}    pre_scale_input={pre_scale_input!r},")
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
