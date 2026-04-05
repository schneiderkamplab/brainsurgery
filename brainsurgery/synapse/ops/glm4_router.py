from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

OP_NAME = "glm4_router"
LOWERING_ARITY = (1, 1)
LOWERING_ALLOWED_KWARGS: set[str] = {
    "top_k",
    "n_group",
    "topk_group",
    "norm_topk_prob",
    "routed_scaling_factor",
    "weight",
    "e_score_correction_bias",
}
LOWERING_REQUIRED_KWARGS: set[str] = {"top_k"}
LOWERING_KWARG_KINDS: dict[str, Any] = {
    "top_k": "dim",
    "n_group": "dim",
    "topk_group": "dim",
    "norm_topk_prob": "bool",
    "routed_scaling_factor": "number",
    "weight": "str",
    "e_score_correction_bias": "str",
}


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return True


def lowering_known_output_arity(*, kwargs: dict[str, Any]) -> int:
    del kwargs
    return 2


def lowering_validate_signature(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> None:
    del args, kwargs, ctx
    if not isinstance(out, list) or len(out) != 2:
        raise ValueError("glm4_router requires exactly two outputs: weights, indices")


def _default_param_candidates(key: str) -> list[str]:
    fallback_map = {
        "weight": "gate.weight",
        "e_score_correction_bias": "gate.e_score_correction_bias",
    }
    fallback = fallback_map[key]
    return [
        fallback,
        f"@@{fallback}",
    ]


def _read_params(
    model: Any,
    node_spec: dict[str, Any],
    *,
    node_path: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    local_spec = dict(node_spec)
    params = dict(local_spec.get("_params", {}))
    for key in ("weight", "e_score_correction_bias"):
        override = local_spec.get(key)
        if isinstance(override, str):
            params[key] = override
        elif key not in params:
            params[key] = _default_param_candidates(key)
    local_spec["_params"] = params
    weight = model._param(
        model._infer_param_path(local_spec, node_path=node_path, param_name="weight")
    )
    bias = model._param(
        model._infer_param_path(
            local_spec, node_path=node_path, param_name="e_score_correction_bias"
        )
    )
    return weight, bias


def _route_tokens(
    *,
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    top_k: int,
    n_group: int,
    topk_group: int,
    norm_topk_prob: bool,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    router_probs = router_logits.sigmoid()
    router_logits_for_choice = router_probs + correction_bias.to(
        device=router_probs.device, dtype=router_probs.dtype
    )
    num_experts = int(router_probs.shape[-1])
    if n_group <= 0 or num_experts % n_group != 0:
        raise ValueError("glm4_router requires num_experts to be divisible by n_group")
    experts_per_group = num_experts // n_group
    if topk_group <= 0 or topk_group > n_group:
        raise ValueError("glm4_router topk_group must be in [1, n_group]")
    group_scores = (
        router_logits_for_choice.view(-1, n_group, experts_per_group)
        .topk(min(2, experts_per_group), dim=-1)[0]
        .sum(dim=-1)
    )
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1).expand(-1, n_group, experts_per_group).reshape(-1, num_experts)
    )
    scores_for_choice = router_logits_for_choice.masked_fill(~score_mask.bool(), 0.0)
    topk_indices = torch.topk(scores_for_choice, k=top_k, dim=-1, sorted=False)[1]
    topk_weights = router_probs.gather(1, topk_indices)
    if norm_topk_prob:
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1.0e-20)
    topk_weights = topk_weights * routed_scaling_factor
    return topk_weights, topk_indices


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
    inputs = node_spec.get("_args")
    outs = node_spec.get("_bind")
    if not isinstance(outs, list) or len(outs) != 2:
        raise ValueError("glm4_router expects out=[weights,indices]")
    hidden = model._read_tensor_input(inputs, env)
    top_k = int(model._eval_expr(node_spec.get("top_k"), env, symbols))
    n_group = int(model._eval_expr(node_spec.get("n_group", 1), env, symbols))
    topk_group = int(model._eval_expr(node_spec.get("topk_group", n_group), env, symbols))
    norm_topk_prob = bool(model._eval_expr(node_spec.get("norm_topk_prob", True), env, symbols))
    routed_scaling_factor = float(
        model._eval_expr(node_spec.get("routed_scaling_factor", 1.0), env, symbols)
    )
    weight, correction_bias = _read_params(model, node_spec, node_path=node_path)

    hidden_flat = hidden.view(-1, hidden.shape[-1])
    router_logits = F.linear(hidden_flat.float(), weight.float())
    topk_weights, topk_indices = _route_tokens(
        router_logits=router_logits,
        correction_bias=correction_bias.float(),
        top_k=top_k,
        n_group=n_group,
        topk_group=topk_group,
        norm_topk_prob=norm_topk_prob,
        routed_scaling_factor=routed_scaling_factor,
    )
    out_shape = (*hidden.shape[:-1], top_k)
    env[str(outs[0])] = topk_weights.view(out_shape).to(dtype=hidden.dtype)
    env[str(outs[1])] = topk_indices.view(out_shape)


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
        params = dict(local_spec.get("_params", {}))
        override = local_spec.get(param_name)
        if isinstance(override, str):
            params[param_name] = override
        elif param_name not in params:
            params[param_name] = _default_param_candidates(param_name)
        local_spec["_params"] = params
        return emitter._hoisted_param(
            node_spec=local_spec,
            node_path_var=node_path_var,
            param_name=param_name,
            lines=lines,
            indent=indent,
        )

    src = emitter._read_env_var(env, str(node_spec.get("_args")))
    outs = node_spec.get("_bind")
    if not isinstance(outs, list) or len(outs) != 2:
        raise ValueError("glm4_router expects out=[weights,indices]")
    weights_var = emitter._assign_out_var(env, str(outs[0]))
    indices_var = emitter._assign_out_var(env, str(outs[1]))
    hidden_flat = emitter._fresh("hidden_flat")
    router_logits = emitter._fresh("router_logits")
    router_probs = emitter._fresh("router_probs")
    router_choice = emitter._fresh("router_choice")
    group_scores = emitter._fresh("group_scores")
    group_idx = emitter._fresh("group_idx")
    group_mask = emitter._fresh("group_mask")
    score_mask = emitter._fresh("score_mask")
    scores_for_choice = emitter._fresh("scores_for_choice")
    out_shape = emitter._fresh("out_shape")
    top_k = emitter._expr_code(node_spec.get("top_k"), env)
    n_group = emitter._expr_code(node_spec.get("n_group", 1), env)
    topk_group = emitter._expr_code(node_spec.get("topk_group", node_spec.get("n_group", 1)), env)
    norm_topk_prob = emitter._expr_code(node_spec.get("norm_topk_prob", True), env)
    routed_scaling_factor = emitter._expr_code(node_spec.get("routed_scaling_factor", 1.0), env)
    weight = hoisted_param("weight")
    correction_bias = hoisted_param("e_score_correction_bias")
    num_experts = emitter._fresh("num_experts")
    experts_per_group = emitter._fresh("experts_per_group")

    lines.append(f"{indent}{hidden_flat} = {src}.view(-1, {src}.shape[-1])")
    lines.append(f"{indent}{router_logits} = F.linear({hidden_flat}.float(), {weight}.float())")
    lines.append(f"{indent}{router_probs} = torch.sigmoid({router_logits})")
    lines.append(
        f"{indent}{router_choice} = {router_probs} + {correction_bias}.to(device={router_probs}.device, dtype={router_probs}.dtype)"
    )
    lines.append(f"{indent}{num_experts} = int({router_probs}.shape[-1])")
    lines.append(f"{indent}{experts_per_group} = int({num_experts} // int({n_group}))")
    lines.append(f"{indent}if int({n_group}) <= 0 or int({num_experts}) % int({n_group}) != 0:")
    lines.append(
        f"{indent}    raise ValueError('glm4_router requires num_experts to be divisible by n_group')"
    )
    lines.append(f"{indent}if int({topk_group}) <= 0 or int({topk_group}) > int({n_group}):")
    lines.append(f"{indent}    raise ValueError('glm4_router topk_group must be in [1, n_group]')")
    lines.append(
        f"{indent}{group_scores} = {router_choice}.view(-1, int({n_group}), int({experts_per_group})).topk(min(2, int({experts_per_group})), dim=-1)[0].sum(dim=-1)"
    )
    lines.append(
        f"{indent}{group_idx} = torch.topk({group_scores}, k=int({topk_group}), dim=-1, sorted=False)[1]"
    )
    lines.append(f"{indent}{group_mask} = torch.zeros_like({group_scores})")
    lines.append(f"{indent}{group_mask}.scatter_(1, {group_idx}, 1)")
    lines.append(
        f"{indent}{score_mask} = {group_mask}.unsqueeze(-1).expand(-1, int({n_group}), int({experts_per_group})).reshape(-1, int({num_experts}))"
    )
    lines.append(
        f"{indent}{scores_for_choice} = {router_choice}.masked_fill(~{score_mask}.bool(), 0.0)"
    )
    lines.append(
        f"{indent}{indices_var} = torch.topk({scores_for_choice}, k=int({top_k}), dim=-1, sorted=False)[1]"
    )
    lines.append(f"{indent}{weights_var} = {router_probs}.gather(1, {indices_var})")
    lines.append(f"{indent}if bool({norm_topk_prob}):")
    lines.append(
        f"{indent}    {weights_var} = {weights_var} / ({weights_var}.sum(dim=-1, keepdim=True) + 1.0e-20)"
    )
    lines.append(f"{indent}{weights_var} = {weights_var} * float({routed_scaling_factor})")
    lines.append(f"{indent}{out_shape} = (*{src}.shape[:-1], int({top_k}))")
    lines.append(f"{indent}{weights_var} = {weights_var}.view({out_shape}).to(dtype={src}.dtype)")
    lines.append(f"{indent}{indices_var} = {indices_var}.view({out_shape})")
    return lines


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any",),
    "kwargs": dict(LOWERING_KWARG_KINDS),
    "returns": ("Tensor", "IdxTensor"),
}

__all__ = [
    "OP_NAME",
    "LOWERING_ARITY",
    "LOWERING_ALLOWED_KWARGS",
    "LOWERING_REQUIRED_KWARGS",
    "LOWERING_KWARG_KINDS",
    "lowering_known_output_arity",
    "lowering_validate_signature",
    "interpret",
    "compile",
    "uses_node_path",
    "LOWERING_TYPE_SIGNATURE",
]
