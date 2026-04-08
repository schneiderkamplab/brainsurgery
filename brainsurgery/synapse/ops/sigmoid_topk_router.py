from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

OP_NAME = "sigmoid_topk_router"
LOWERING_ARITY = (1, 1)
LOWERING_ALLOWED_KWARGS: set[str] = {
    "top_k",
    "weight",
}
LOWERING_REQUIRED_KWARGS: set[str] = {"top_k"}
LOWERING_KWARG_KINDS: dict[str, Any] = {
    "top_k": "dim",
    "weight": "str",
}


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter
    explicit_weight = node_spec.get("weight")
    if isinstance(explicit_weight, str) and "." in explicit_weight:
        return False
    return True


def lowering_known_output_arity(*, kwargs: dict[str, Any]) -> int:
    del kwargs
    return 2


def lowering_validate_signature(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> None:
    del args, kwargs, ctx
    if not isinstance(out, list) or len(out) != 2:
        raise ValueError("sigmoid_topk_router requires exactly two outputs: weights, indices")


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
        raise ValueError("sigmoid_topk_router expects out=[weights,indices]")
    hidden = model._read_tensor_input(inputs, env)
    top_k = int(model._eval_expr(node_spec.get("top_k"), env, symbols))
    weight_path = model._infer_param_path(node_spec, node_path=node_path, param_name="weight")
    weight = model._state[weight_path]

    hidden_flat = hidden.view(-1, hidden.shape[-1])
    router_logits = F.linear(hidden_flat.float(), weight.float())
    topk_values, topk_indices = torch.topk(router_logits, k=top_k, dim=-1, sorted=False)
    topk_weights = torch.sigmoid(topk_values).to(dtype=hidden.dtype)
    out_shape = (*hidden.shape[:-1], top_k)
    env[str(outs[0])] = topk_weights.view(out_shape)
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
    src = emitter._read_env_var(env, str(node_spec.get("_args")))
    outs = node_spec.get("_bind")
    if not isinstance(outs, list) or len(outs) != 2:
        raise ValueError("sigmoid_topk_router expects out=[weights,indices]")
    weights_var = emitter._assign_out_var(env, str(outs[0]))
    indices_var = emitter._assign_out_var(env, str(outs[1]))
    hidden_flat = emitter._fresh("hidden_flat")
    router_logits = emitter._fresh("router_logits")
    out_shape = emitter._fresh("out_shape")
    top_k = emitter._expr_code(node_spec.get("top_k"), env)
    weight = emitter._hoisted_param(
        node_spec=node_spec,
        node_path_var=node_path_var,
        param_name="weight",
        lines=lines,
        indent=indent,
    )
    lines.append(f"{indent}{hidden_flat} = {src}.view(-1, {src}.shape[-1])")
    lines.append(f"{indent}{router_logits} = F.linear({hidden_flat}.float(), {weight}.float())")
    lines.append(
        f"{indent}{weights_var}, {indices_var} = torch.topk({router_logits}, k=int({top_k}), dim=-1, sorted=False)"
    )
    lines.append(f"{indent}{weights_var} = torch.sigmoid({weights_var}).to(dtype={src}.dtype)")
    lines.append(f"{indent}{out_shape} = (*{src}.shape[:-1], int({top_k}))")
    lines.append(f"{indent}{weights_var} = {weights_var}.view({out_shape})")
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
