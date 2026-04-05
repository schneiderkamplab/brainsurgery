from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

OP_NAME = "gemma4_moe_experts"
LOWERING_ARITY = (3, 3)
LOWERING_ALLOWED_KWARGS: set[str] = set()
LOWERING_REQUIRED_KWARGS: set[str] = set()
LOWERING_KWARG_KINDS: dict[str, Any] = {}


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter
    return True


def lowering_validate_signature(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> None:
    del args, kwargs, ctx
    if not isinstance(out, str):
        raise ValueError("gemma4_moe_experts requires a single output")


def lowering_infer_metadata(
    *,
    args: list[str],
    out: str | list[str],
    kwargs: dict[str, Any],
    ctx: Any,
) -> bool:
    del kwargs
    if not isinstance(out, str) or not args:
        return False
    src_name = args[0].strip()
    if src_name.isidentifier() and src_name in ctx.tensor_last_dim:
        ctx.tensor_last_dim[out] = ctx.tensor_last_dim[src_name]
    if src_name.isidentifier() and src_name in ctx.tensor_shape:
        ctx.tensor_shape[out] = ctx.tensor_shape[src_name]
    return True


def _infer_path(model: Any, node_spec: dict[str, Any], *, node_path: str, key: str) -> str:
    override = node_spec.get(key)
    if isinstance(override, str):
        local_spec = dict(node_spec)
        local_spec[key] = override
        return model._infer_param_path(local_spec, node_path=node_path, param_name=key)
    fallback_map = {
        "gate_up_weight": "experts.gate_up_proj",
        "down_weight": "experts.down_proj",
    }
    fallback = fallback_map[key]
    scoped = model._join(model._scope_of(node_path), fallback)
    candidates = [
        scoped,
        model._join("model", scoped),
        model._join("model.language_model", fallback),
        model._join("model", fallback),
        fallback,
    ]
    for candidate in candidates:
        if candidate in model._state:
            return candidate
    return candidates[0]


def _default_param_candidates(key: str) -> list[str]:
    fallback_map = {
        "gate_up_weight": "experts.gate_up_proj",
        "down_weight": "experts.down_proj",
    }
    fallback = fallback_map[key]
    return [
        fallback,
        f"@@model.language_model.{fallback}",
        f"@@model.{fallback}",
        f"@@{fallback}",
    ]


def _gelu_pytorch_tanh(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * (x + 0.044715 * x.pow(3))))


def interpret(
    model: Any,
    node_spec: dict[str, Any],
    env: dict[str, Any],
    *,
    node_path: str,
    scope: str,
    symbols: dict[str, int | float | bool],
) -> None:
    del scope, symbols
    inputs = node_spec.get("_args")
    if not isinstance(inputs, list) or len(inputs) != 3:
        raise ValueError("gemma4_moe_experts expects in=[hidden,topk_weights,topk_indices]")
    out_name = model._require_name(node_spec.get("_bind"), field="gemma4_moe_experts._bind")
    hidden = model._read_tensor_input(inputs[0], env)
    topk_weights = model._read_tensor_input(inputs[1], env)
    topk_indices = model._read_tensor_input(inputs[2], env)

    gate_up_weight = model._state[
        _infer_path(model, node_spec, node_path=node_path, key="gate_up_weight")
    ]
    down_weight = model._state[_infer_path(model, node_spec, node_path=node_path, key="down_weight")]

    hidden_flat = hidden.reshape(-1, hidden.shape[-1])
    weights_flat = topk_weights.reshape(-1, topk_weights.shape[-1])
    indices_flat = topk_indices.reshape(-1, topk_indices.shape[-1])
    final_hidden_states = torch.zeros_like(hidden_flat)
    expert_mask = torch.nn.functional.one_hot(indices_flat, num_classes=gate_up_weight.shape[0])
    expert_mask = expert_mask.permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx_tensor in expert_hit:
        expert_idx = int(expert_idx_tensor[0])
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_flat[token_idx]
        gate_up = F.linear(current_state, gate_up_weight[expert_idx].to(dtype=current_state.dtype))
        gate, up = gate_up.chunk(2, dim=-1)
        current_hidden_states = _gelu_pytorch_tanh(gate) * up
        current_hidden_states = F.linear(
            current_hidden_states,
            down_weight[expert_idx].to(dtype=current_hidden_states.dtype),
        )
        current_hidden_states = current_hidden_states * weights_flat[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(
            0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
        )

    env[out_name] = final_hidden_states.reshape(*hidden.shape[:-1], hidden.shape[-1])


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

    def hoisted_param(param_name: str) -> str:
        local_spec = dict(node_spec)
        override = node_spec.get(param_name)
        if isinstance(override, str):
            local_spec[param_name] = override
        elif isinstance(override, list):
            local_spec["_params"] = dict(local_spec.get("_params", {}))
            local_spec["_params"][param_name] = override
        else:
            local_spec["_params"] = dict(local_spec.get("_params", {}))
            local_spec["_params"][param_name] = _default_param_candidates(param_name)
        return emitter._hoisted_param(
            node_spec=local_spec,
            node_path_var=node_path_var,
            param_name=param_name,
            lines=lines,
            indent=indent,
        )

    inputs = node_spec.get("_args")
    if not isinstance(inputs, list) or len(inputs) != 3:
        raise ValueError("gemma4_moe_experts expects in=[hidden,topk_weights,topk_indices]")
    hidden = emitter._read_env_var(env, str(inputs[0]))
    topk_weights = emitter._read_env_var(env, str(inputs[1]))
    topk_indices = emitter._read_env_var(env, str(inputs[2]))
    out_var = emitter._assign_out_var(env, str(node_spec.get("_bind")))
    gate_up_weight = hoisted_param("gate_up_weight")
    down_weight = hoisted_param("down_weight")
    hidden_flat = emitter._fresh("hidden_flat")
    weights_flat = emitter._fresh("weights_flat")
    indices_flat = emitter._fresh("indices_flat")
    final_hidden = emitter._fresh("final_hidden")
    expert_mask = emitter._fresh("expert_mask")
    expert_hit = emitter._fresh("expert_hit")
    expert_idx = emitter._fresh("expert_idx")
    top_k_pos = emitter._fresh("top_k_pos")
    token_idx = emitter._fresh("token_idx")
    current_state = emitter._fresh("current_state")
    gate_up = emitter._fresh("gate_up")
    gate = emitter._fresh("gate")
    up = emitter._fresh("up")
    current_hidden = emitter._fresh("current_hidden")
    gelu_term = emitter._fresh("gelu_term")

    lines.append(f"{indent}{hidden_flat} = {hidden}.reshape(-1, {hidden}.shape[-1])")
    lines.append(f"{indent}{weights_flat} = {topk_weights}.reshape(-1, {topk_weights}.shape[-1])")
    lines.append(f"{indent}{indices_flat} = {topk_indices}.reshape(-1, {topk_indices}.shape[-1])")
    lines.append(f"{indent}{final_hidden} = torch.zeros_like({hidden_flat})")
    lines.append(
        f"{indent}{expert_mask} = torch.nn.functional.one_hot({indices_flat}, num_classes={gate_up_weight}.shape[0]).permute(2, 1, 0)"
    )
    lines.append(
        f"{indent}{expert_hit} = torch.greater({expert_mask}.sum(dim=(-1, -2)), 0).nonzero()"
    )
    lines.append(f"{indent}for {expert_idx} in {expert_hit}:")
    lines.append(f"{indent}    {expert_idx} = int({expert_idx}[0])")
    lines.append(
        f"{indent}    {top_k_pos}, {token_idx} = torch.where({expert_mask}[{expert_idx}])"
    )
    lines.append(f"{indent}    {current_state} = {hidden_flat}[{token_idx}]")
    lines.append(
        f"{indent}    {gate_up} = F.linear({current_state}, {gate_up_weight}[{expert_idx}].to(dtype={current_state}.dtype))"
    )
    lines.append(f"{indent}    {gate}, {up} = {gate_up}.chunk(2, dim=-1)")
    lines.append(
        f"{indent}    {gelu_term} = 0.5 * {gate} * (1.0 + torch.tanh(0.7978845608028654 * ({gate} + 0.044715 * {gate}.pow(3))))"
    )
    lines.append(f"{indent}    {current_hidden} = {gelu_term} * {up}")
    lines.append(
        f"{indent}    {current_hidden} = F.linear({current_hidden}, {down_weight}[{expert_idx}].to(dtype={current_hidden}.dtype))"
    )
    lines.append(
        f"{indent}    {current_hidden} = {current_hidden} * {weights_flat}[{token_idx}, {top_k_pos}, None]"
    )
    lines.append(
        f"{indent}    {final_hidden}.index_add_(0, {token_idx}, {current_hidden}.to({final_hidden}.dtype))"
    )
    lines.append(
        f"{indent}{out_var} = {final_hidden}.reshape(*{hidden}.shape[:-1], {hidden}.shape[-1])"
    )
    return lines


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any", "Any", "Any"),
    "kwargs": dict(LOWERING_KWARG_KINDS),
    "returns": ("Tensor",),
}

__all__ = [
    "OP_NAME",
    "LOWERING_ARITY",
    "LOWERING_ALLOWED_KWARGS",
    "LOWERING_REQUIRED_KWARGS",
    "LOWERING_KWARG_KINDS",
    "lowering_validate_signature",
    "lowering_infer_metadata",
    "interpret",
    "compile",
    "uses_node_path",
    "LOWERING_TYPE_SIGNATURE",
]
