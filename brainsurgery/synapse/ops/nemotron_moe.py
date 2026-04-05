from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

OP_NAME = "nemotron_moe"
LOWERING_ARITY = (1, 1)
LOWERING_ALLOWED_KWARGS: set[str] = {
    "top_k",
    "n_group",
    "topk_group",
    "routed_scaling_factor",
    "norm_topk_prob",
}
LOWERING_REQUIRED_KWARGS: set[str] = {
    "top_k",
    "n_group",
    "topk_group",
    "routed_scaling_factor",
    "norm_topk_prob",
}
LOWERING_KWARG_KINDS: dict[str, Any] = {
    "top_k": "dim",
    "n_group": "dim",
    "topk_group": "dim",
    "routed_scaling_factor": "number",
    "norm_topk_prob": "bool",
}


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return True


def lowering_validate_signature(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> None:
    del kwargs, ctx
    if len(args) != 1:
        raise ValueError("nemotron_moe expects a single hidden-state input")
    if not isinstance(out, str):
        raise ValueError("nemotron_moe requires a single output")


def lowering_infer_metadata(
    *,
    args: list[str],
    out: str | list[str],
    kwargs: dict[str, Any],
    ctx: Any,
) -> bool:
    del kwargs
    if not isinstance(out, str) or len(args) != 1:
        return False
    src_name = args[0].strip()
    if src_name.isidentifier() and src_name in ctx.tensor_last_dim:
        ctx.tensor_last_dim[out] = ctx.tensor_last_dim[src_name]
    if src_name.isidentifier() and src_name in ctx.tensor_shape:
        ctx.tensor_shape[out] = ctx.tensor_shape[src_name]
    return True


def _resolve_inputs(node_spec: dict[str, Any]) -> str:
    inputs = node_spec.get("_args")
    if isinstance(inputs, str):
        return inputs
    if isinstance(inputs, list) and len(inputs) == 1 and isinstance(inputs[0], str):
        return str(inputs[0])
    raise ValueError("nemotron_moe expects in=[hidden]")


def _infer_param_path(model: Any, node_spec: dict[str, Any], *, node_path: str, param_name: str) -> str:
    local_spec = dict(node_spec)
    local_spec["_params"] = dict(local_spec.get("_params", {}))
    local_spec["_params"][param_name] = [param_name, f"@@{param_name}"]
    return model._infer_param_path(local_spec, node_path=node_path, param_name=param_name)


def _expert_ffn(
    x: torch.Tensor,
    *,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    up = F.linear(x, up_weight.to(dtype=x.dtype))
    act = F.relu(up)
    return F.linear(act * act, down_weight.to(dtype=up.dtype))


def interpret(
    model: Any,
    node_spec: dict[str, Any],
    env: dict[str, Any],
    *,
    node_path: str,
    scope: str,
    symbols: dict[str, int | float | bool],
) -> None:
    del scope
    in_hidden = _resolve_inputs(node_spec)
    out_name = model._require_name(node_spec.get("_bind"), field="nemotron_moe._bind")
    hidden = model._read_tensor_input(in_hidden, env)
    top_k = int(model._eval_expr(node_spec.get("top_k"), env, symbols))
    n_group = int(model._eval_expr(node_spec.get("n_group"), env, symbols))
    topk_group = int(model._eval_expr(node_spec.get("topk_group"), env, symbols))
    routed_scaling_factor = float(model._eval_expr(node_spec.get("routed_scaling_factor"), env, symbols))
    norm_topk_prob = bool(model._eval_expr(node_spec.get("norm_topk_prob"), env, symbols))

    gate_weight = model._state[_infer_param_path(model, node_spec, node_path=node_path, param_name="gate.weight")]
    correction_bias = model._state[
        _infer_param_path(model, node_spec, node_path=node_path, param_name="gate.e_score_correction_bias")
    ]
    shared_up = model._state[
        _infer_param_path(model, node_spec, node_path=node_path, param_name="shared_experts.up_proj.weight")
    ]
    shared_down = model._state[
        _infer_param_path(model, node_spec, node_path=node_path, param_name="shared_experts.down_proj.weight")
    ]

    orig_shape = hidden.shape
    hidden_flat = hidden.view(-1, hidden.shape[-1])
    n_routed_experts = int(gate_weight.shape[0])
    router_logits = F.linear(hidden_flat.to(torch.float32), gate_weight.to(torch.float32))
    scores = router_logits.sigmoid()
    scores_for_choice = scores.view(-1, n_routed_experts) + correction_bias.to(dtype=scores.dtype).unsqueeze(0)
    group_scores = (
        scores_for_choice.view(-1, n_group, n_routed_experts // n_group).topk(2, dim=-1)[0].sum(dim=-1)
    )
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(-1, n_group, n_routed_experts // n_group)
        .reshape(-1, n_routed_experts)
    )
    scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
    topk_indices = torch.topk(scores_for_choice, k=top_k, dim=-1, sorted=False)[1]
    topk_weights = scores.gather(1, topk_indices)
    if norm_topk_prob:
        denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
        topk_weights = topk_weights / denominator
    topk_weights = topk_weights * routed_scaling_factor

    final_hidden_states = torch.zeros_like(hidden_flat, dtype=topk_weights.dtype)
    expert_mask = torch.nn.functional.one_hot(topk_indices, num_classes=n_routed_experts).permute(2, 0, 1)
    for expert_idx in range(n_routed_experts):
        up_weight = model._state[
            _infer_param_path(
                model,
                node_spec,
                node_path=node_path,
                param_name=f"experts.{expert_idx}.up_proj.weight",
            )
        ]
        down_weight = model._state[
            _infer_param_path(
                model,
                node_spec,
                node_path=node_path,
                param_name=f"experts.{expert_idx}.down_proj.weight",
            )
        ]
        mask = expert_mask[expert_idx]
        token_indices, weight_indices = torch.where(mask)
        if token_indices.numel() > 0:
            expert_weights = topk_weights[token_indices, weight_indices]
            expert_input = hidden_flat[token_indices]
            expert_output = _expert_ffn(expert_input, up_weight=up_weight, down_weight=down_weight)
            weighted_output = expert_output * expert_weights.unsqueeze(-1)
            final_hidden_states.index_add_(0, token_indices, weighted_output)
        else:
            expert_dtype = down_weight.dtype
            zero_in = torch.zeros_like(hidden_flat[0]).unsqueeze(0).to(expert_dtype)
            dummy_out = _expert_ffn(zero_in, up_weight=up_weight, down_weight=down_weight)
            final_hidden_states = final_hidden_states + dummy_out

    final_hidden_states = final_hidden_states.type(hidden_flat.dtype).view(*orig_shape)
    shared = _expert_ffn(hidden, up_weight=shared_up, down_weight=shared_down)
    env[out_name] = final_hidden_states + shared


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
        local_spec["_params"] = dict(local_spec.get("_params", {}))
        local_spec["_params"][param_name] = [param_name, f"@@{param_name}"]
        return emitter._hoisted_param(
            node_spec=local_spec,
            node_path_var=node_path_var,
            param_name=param_name,
            lines=lines,
            indent=indent,
        )

    hidden = emitter._read_env_var(env, _resolve_inputs(node_spec))
    out_var = emitter._assign_out_var(env, str(node_spec.get("_bind")))
    top_k = emitter._expr_code(node_spec.get("top_k"), env)
    n_group = emitter._expr_code(node_spec.get("n_group"), env)
    topk_group = emitter._expr_code(node_spec.get("topk_group"), env)
    routed_scaling_factor = emitter._expr_code(node_spec.get("routed_scaling_factor"), env)
    norm_topk_prob = emitter._expr_code(node_spec.get("norm_topk_prob"), env)
    gate_weight = hoisted_param("gate.weight")
    correction_bias = hoisted_param("gate.e_score_correction_bias")
    shared_up = hoisted_param("shared_experts.up_proj.weight")
    shared_down = hoisted_param("shared_experts.down_proj.weight")

    orig_shape = emitter._fresh("orig_shape")
    hidden_flat = emitter._fresh("hidden_flat")
    n_routed_experts = emitter._fresh("n_routed_experts")
    router_logits = emitter._fresh("router_logits")
    scores = emitter._fresh("scores")
    scores_for_choice = emitter._fresh("scores_for_choice")
    group_scores = emitter._fresh("group_scores")
    group_idx = emitter._fresh("group_idx")
    group_mask = emitter._fresh("group_mask")
    score_mask = emitter._fresh("score_mask")
    topk_indices = emitter._fresh("topk_indices")
    topk_weights = emitter._fresh("topk_weights")
    denominator = emitter._fresh("denominator")
    final_hidden_states = emitter._fresh("final_hidden_states")
    expert_mask = emitter._fresh("expert_mask")
    shared_up_out = emitter._fresh("shared_up_out")
    shared_act = emitter._fresh("shared_act")
    shared = emitter._fresh("shared")

    lines.append(f"{indent}{orig_shape} = {hidden}.shape")
    lines.append(f"{indent}{hidden_flat} = {hidden}.view(-1, {hidden}.shape[-1])")
    lines.append(f"{indent}{n_routed_experts} = int({gate_weight}.shape[0])")
    lines.append(
        f"{indent}{router_logits} = F.linear({hidden_flat}.to(torch.float32), {gate_weight}.to(torch.float32))"
    )
    lines.append(f"{indent}{scores} = {router_logits}.sigmoid()")
    lines.append(
        f"{indent}{scores_for_choice} = {scores}.view(-1, {n_routed_experts}) + {correction_bias}.to(dtype={scores}.dtype).unsqueeze(0)"
    )
    lines.append(
        f"{indent}{group_scores} = {scores_for_choice}.view(-1, int({n_group}), {n_routed_experts} // int({n_group})).topk(2, dim=-1)[0].sum(dim=-1)"
    )
    lines.append(
        f"{indent}{group_idx} = torch.topk({group_scores}, k=int({topk_group}), dim=-1, sorted=False)[1]"
    )
    lines.append(f"{indent}{group_mask} = torch.zeros_like({group_scores})")
    lines.append(f"{indent}{group_mask}.scatter_(1, {group_idx}, 1)")
    lines.append(
        f"{indent}{score_mask} = {group_mask}.unsqueeze(-1).expand(-1, int({n_group}), {n_routed_experts} // int({n_group})).reshape(-1, {n_routed_experts})"
    )
    lines.append(f"{indent}{scores_for_choice} = {scores_for_choice}.masked_fill(~{score_mask}.bool(), 0.0)")
    lines.append(
        f"{indent}{topk_indices} = torch.topk({scores_for_choice}, k=int({top_k}), dim=-1, sorted=False)[1]"
    )
    lines.append(f"{indent}{topk_weights} = {scores}.gather(1, {topk_indices})")
    lines.append(f"{indent}if bool({norm_topk_prob}):")
    lines.append(f"{indent}    {denominator} = {topk_weights}.sum(dim=-1, keepdim=True) + 1e-20")
    lines.append(f"{indent}    {topk_weights} = {topk_weights} / {denominator}")
    lines.append(f"{indent}{topk_weights} = {topk_weights} * float({routed_scaling_factor})")
    lines.append(
        f"{indent}{final_hidden_states} = torch.zeros_like({hidden_flat}, dtype={topk_weights}.dtype)"
    )
    lines.append(
        f"{indent}{expert_mask} = torch.nn.functional.one_hot({topk_indices}, num_classes={n_routed_experts}).permute(2, 0, 1)"
    )
    lines.append(f"{indent}for _expert_idx in range({n_routed_experts}):")
    lines.append(
        f"{indent}    _expert_up = self._pick_param_path(self._scope_of({node_path_var}), [f'experts.{{_expert_idx}}.up_proj.weight', f'@@experts.{{_expert_idx}}.up_proj.weight'])"
    )
    lines.append(
        f"{indent}    _expert_down = self._pick_param_path(self._scope_of({node_path_var}), [f'experts.{{_expert_idx}}.down_proj.weight', f'@@experts.{{_expert_idx}}.down_proj.weight'])"
    )
    lines.append(f"{indent}    _up_weight = self._state[_expert_up]")
    lines.append(f"{indent}    _down_weight = self._state[_expert_down]")
    lines.append(f"{indent}    _mask = {expert_mask}[_expert_idx]")
    lines.append(f"{indent}    _token_indices, _weight_indices = torch.where(_mask)")
    lines.append(f"{indent}    if _token_indices.numel() > 0:")
    lines.append(f"{indent}        _expert_weights = {topk_weights}[_token_indices, _weight_indices]")
    lines.append(f"{indent}        _expert_input = {hidden_flat}[_token_indices]")
    lines.append(
        f"{indent}        _up = F.linear(_expert_input, _up_weight.to(dtype=_expert_input.dtype))"
    )
    lines.append(f"{indent}        _act = F.relu(_up)")
    lines.append(
        f"{indent}        _expert_out = F.linear(_act * _act, _down_weight.to(dtype=_up.dtype))"
    )
    lines.append(f"{indent}        _weighted_output = _expert_out * _expert_weights.unsqueeze(-1)")
    lines.append(f"{indent}        {final_hidden_states}.index_add_(0, _token_indices, _weighted_output)")
    lines.append(f"{indent}    else:")
    lines.append(
        f"{indent}        _zero_in = torch.zeros_like({hidden_flat}[0]).unsqueeze(0).to(_down_weight.dtype)"
    )
    lines.append(
        f"{indent}        _up = F.linear(_zero_in, _up_weight.to(dtype=_down_weight.dtype))"
    )
    lines.append(f"{indent}        _act = F.relu(_up)")
    lines.append(
        f"{indent}        _dummy_out = F.linear(_act * _act, _down_weight.to(dtype=_down_weight.dtype))"
    )
    lines.append(f"{indent}        {final_hidden_states} = {final_hidden_states} + _dummy_out")
    lines.append(f"{indent}{final_hidden_states} = {final_hidden_states}.type({hidden_flat}.dtype).view(*{orig_shape})")
    lines.append(f"{indent}{shared_up_out} = F.linear({hidden}, {shared_up}.to(dtype={hidden}.dtype))")
    lines.append(f"{indent}{shared_act} = F.relu({shared_up_out})")
    lines.append(
        f"{indent}{shared} = F.linear({shared_act} * {shared_act}, {shared_down}.to(dtype={shared_up_out}.dtype))"
    )
    lines.append(f"{indent}{out_var} = {final_hidden_states} + {shared}")
    return lines


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any",),
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
