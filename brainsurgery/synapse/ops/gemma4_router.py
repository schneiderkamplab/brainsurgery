from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

OP_NAME = "gemma4_router"
LOWERING_ARITY = (1, 1)
LOWERING_ALLOWED_KWARGS: set[str] = {"top_k", "scalar_root_size", "rms_eps"}
LOWERING_REQUIRED_KWARGS: set[str] = {"top_k"}
LOWERING_KWARG_KINDS: dict[str, Any] = {
    "top_k": "dim",
    "scalar_root_size": "number",
    "rms_eps": "number",
}


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter
    return True


def lowering_known_output_arity(*, kwargs: dict[str, Any]) -> int:
    del kwargs
    return 2


def lowering_validate_signature(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> None:
    del args, kwargs, ctx
    if not isinstance(out, list) or len(out) != 2:
        raise ValueError("gemma4_router requires exactly two outputs: weights, indices")


def _infer_path(model: Any, node_spec: dict[str, Any], *, node_path: str, key: str) -> str:
    override = node_spec.get(key)
    if isinstance(override, str):
        local_spec = dict(node_spec)
        local_spec[key] = override
        return model._infer_param_path(local_spec, node_path=node_path, param_name=key)
    fallback_map = {
        "router_scale": "router.scale",
        "router_proj_weight": "router.proj.weight",
        "per_expert_scale": "router.per_expert_scale",
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
        "router_scale": "router.scale",
        "router_proj_weight": "router.proj.weight",
        "per_expert_scale": "router.per_expert_scale",
    }
    fallback = fallback_map[key]
    return [
        fallback,
        f"@@model.language_model.{fallback}",
        f"@@model.{fallback}",
        f"@@{fallback}",
    ]


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
        raise ValueError("gemma4_router expects out=[weights,indices]")
    hidden = model._read_tensor_input(inputs, env)
    top_k = int(model._eval_expr(node_spec.get("top_k"), env, symbols))
    scalar_root_size = float(model._eval_expr(node_spec.get("scalar_root_size", 1.0), env, symbols))
    rms_eps = float(model._eval_expr(node_spec.get("rms_eps", 1.0e-6), env, symbols))

    router_scale = model._state[
        _infer_path(model, node_spec, node_path=node_path, key="router_scale")
    ]
    router_proj_weight = model._state[
        _infer_path(model, node_spec, node_path=node_path, key="router_proj_weight")
    ]
    per_expert_scale = model._state[
        _infer_path(model, node_spec, node_path=node_path, key="per_expert_scale")
    ]

    hidden_fp32 = hidden.float()
    mean_squared = hidden_fp32.pow(2).mean(dim=-1, keepdim=True) + rms_eps
    hidden_norm = hidden_fp32 * torch.pow(mean_squared, -0.5)
    hidden_norm = hidden_norm * router_scale.float() * scalar_root_size
    expert_scores = F.linear(hidden_norm, router_proj_weight.float())
    router_probabilities = torch.softmax(expert_scores, dim=-1)
    top_k_weights, top_k_index = torch.topk(router_probabilities, k=top_k, dim=-1)
    top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
    top_k_weights = top_k_weights * per_expert_scale[top_k_index].to(dtype=top_k_weights.dtype)
    env[str(outs[0])] = top_k_weights.to(dtype=hidden.dtype)
    env[str(outs[1])] = top_k_index


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

    src = emitter._read_env_var(env, str(node_spec.get("_args")))
    outs = node_spec.get("_bind")
    if not isinstance(outs, list) or len(outs) != 2:
        raise ValueError("gemma4_router expects out=[weights,indices]")
    weights_var = emitter._assign_out_var(env, str(outs[0]))
    indices_var = emitter._assign_out_var(env, str(outs[1]))
    top_k = emitter._expr_code(node_spec.get("top_k"), env)
    scalar_root_size = emitter._expr_code(node_spec.get("scalar_root_size", 1.0), env)
    rms_eps = emitter._expr_code(node_spec.get("rms_eps", 1.0e-6), env)
    router_scale = hoisted_param("router_scale")
    router_proj_weight = hoisted_param("router_proj_weight")
    per_expert_scale = hoisted_param("per_expert_scale")
    hidden_norm = emitter._fresh("hidden_norm")
    mean_squared = emitter._fresh("mean_squared")
    expert_scores = emitter._fresh("expert_scores")
    lines.append(
        f"{indent}{mean_squared} = {src}.float().pow(2).mean(dim=-1, keepdim=True) + float({rms_eps})"
    )
    lines.append(f"{indent}{hidden_norm} = {src}.float() * torch.pow({mean_squared}, -0.5)")
    lines.append(
        f"{indent}{hidden_norm} = {hidden_norm} * {router_scale}.float() * float({scalar_root_size})"
    )
    lines.append(f"{indent}{expert_scores} = F.linear({hidden_norm}, {router_proj_weight}.float())")
    lines.append(f"{indent}{weights_var} = torch.softmax({expert_scores}, dim=-1)")
    lines.append(
        f"{indent}{weights_var}, {indices_var} = torch.topk({weights_var}, k=int({top_k}), dim=-1)"
    )
    lines.append(f"{indent}{weights_var} = {weights_var} / {weights_var}.sum(dim=-1, keepdim=True)")
    lines.append(
        f"{indent}{weights_var} = ({weights_var} * {per_expert_scale}[{indices_var}].to(dtype={weights_var}.dtype)).to(dtype={src}.dtype)"
    )
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
