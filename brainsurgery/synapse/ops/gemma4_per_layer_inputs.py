from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

OP_NAME = "gemma4_per_layer_inputs"
LOWERING_ARITY = (2, 2)
LOWERING_ALLOWED_KWARGS: set[str] = {
    "num_layers",
    "per_layer_dim",
    "embed_scale",
    "projection_scale",
    "combine_scale",
    "rms_eps",
}
LOWERING_REQUIRED_KWARGS: set[str] = {"num_layers", "per_layer_dim"}
LOWERING_KWARG_KINDS: dict[str, Any] = {
    "num_layers": "dim",
    "per_layer_dim": "dim",
    "embed_scale": "number",
    "projection_scale": "number",
    "combine_scale": "number",
    "rms_eps": "number",
}


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter
    return True


def lowering_validate_signature(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> None:
    del args, kwargs, ctx
    if not isinstance(out, str):
        raise ValueError("gemma4_per_layer_inputs requires a single output")


def _infer_path(model: Any, node_spec: dict[str, Any], *, node_path: str, key: str) -> str:
    override = node_spec.get(key)
    if isinstance(override, str):
        local_spec = dict(node_spec)
        local_spec[key] = override
        return model._infer_param_path(local_spec, node_path=node_path, param_name=key)
    fallback_map = {
        "per_layer_embed_weight": "embed_tokens_per_layer.weight",
        "projection_weight": "per_layer_model_projection.weight",
        "norm_weight": "per_layer_projection_norm.weight",
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
        "per_layer_embed_weight": "embed_tokens_per_layer.weight",
        "projection_weight": "per_layer_model_projection.weight",
        "norm_weight": "per_layer_projection_norm.weight",
    }
    fallback = fallback_map[key]
    return [
        fallback,
        f"@@model.language_model.{fallback}",
        f"@@model.{fallback}",
        f"@@{fallback}",
    ]


def _rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x_fp32 = x.float()
    mean_squared = x_fp32.pow(2).mean(dim=-1, keepdim=True) + float(eps)
    normed = x_fp32 * torch.pow(mean_squared, -0.5)
    return (normed * weight.float()).to(dtype=x.dtype)


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
    if not isinstance(inputs, list) or len(inputs) != 2:
        raise ValueError("gemma4_per_layer_inputs expects in=[input_ids,inputs_embeds]")
    out_name = model._require_name(node_spec.get("_bind"), field="gemma4_per_layer_inputs._bind")
    input_ids = model._read_tensor_input(inputs[0], env)
    inputs_embeds = model._read_tensor_input(inputs[1], env)
    num_layers = int(model._eval_expr(node_spec.get("num_layers"), env, symbols))
    per_layer_dim = int(model._eval_expr(node_spec.get("per_layer_dim"), env, symbols))
    embed_scale = float(model._eval_expr(node_spec.get("embed_scale", 1.0), env, symbols))
    projection_scale = float(
        model._eval_expr(node_spec.get("projection_scale", 1.0), env, symbols)
    )
    combine_scale = float(model._eval_expr(node_spec.get("combine_scale", 1.0), env, symbols))
    rms_eps = float(model._eval_expr(node_spec.get("rms_eps", 1.0e-6), env, symbols))

    per_layer_embed_weight = model._state[
        _infer_path(model, node_spec, node_path=node_path, key="per_layer_embed_weight")
    ]
    projection_weight = model._state[
        _infer_path(model, node_spec, node_path=node_path, key="projection_weight")
    ]
    norm_weight = model._state[_infer_path(model, node_spec, node_path=node_path, key="norm_weight")]

    per_layer_inputs = F.embedding(input_ids, per_layer_embed_weight)
    if embed_scale != 1.0:
        per_layer_inputs = per_layer_inputs * per_layer_inputs.new_tensor(embed_scale)
    per_layer_inputs = per_layer_inputs.reshape(
        *input_ids.shape,
        num_layers,
        per_layer_dim,
    )

    per_layer_projection = F.linear(inputs_embeds, projection_weight)
    if projection_scale != 1.0:
        per_layer_projection = per_layer_projection * per_layer_projection.new_tensor(
            projection_scale
        )
    per_layer_projection = per_layer_projection.reshape(
        *inputs_embeds.shape[:-1],
        num_layers,
        per_layer_dim,
    )
    per_layer_projection = _rmsnorm(per_layer_projection, norm_weight, rms_eps)
    env[out_name] = (per_layer_projection + per_layer_inputs) * per_layer_projection.new_tensor(
        combine_scale
    )


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
    if not isinstance(inputs, list) or len(inputs) != 2:
        raise ValueError("gemma4_per_layer_inputs expects in=[input_ids,inputs_embeds]")
    input_ids = read(str(inputs[0]))
    inputs_embeds = read(str(inputs[1]))
    out_var = assign_out_var(str(node_spec.get("_bind")))
    num_layers = emitter._expr_code(node_spec.get("num_layers"), env)
    per_layer_dim = emitter._expr_code(node_spec.get("per_layer_dim"), env)
    embed_scale = emitter._expr_code(node_spec.get("embed_scale", 1.0), env)
    projection_scale = emitter._expr_code(node_spec.get("projection_scale", 1.0), env)
    combine_scale = emitter._expr_code(node_spec.get("combine_scale", 1.0), env)
    rms_eps = emitter._expr_code(node_spec.get("rms_eps", 1.0e-6), env)

    per_layer_embed_weight = hoisted_param("per_layer_embed_weight")
    projection_weight = hoisted_param("projection_weight")
    norm_weight = hoisted_param("norm_weight")

    embedded = emitter._fresh("embedded")
    projected = emitter._fresh("projected")
    mean_squared = emitter._fresh("mean_squared")
    normed = emitter._fresh("normed")

    lines.append(f"{indent}{embedded} = F.embedding({input_ids}, {per_layer_embed_weight})")
    lines.append(f"{indent}if float({embed_scale}) != 1.0:")
    lines.append(
        f"{indent}    {embedded} = {embedded} * torch.tensor(float({embed_scale}), dtype={embedded}.dtype, device={embedded}.device)"
    )
    lines.append(
        f"{indent}{embedded} = {embedded}.reshape(*{input_ids}.shape, int({num_layers}), int({per_layer_dim}))"
    )
    lines.append(f"{indent}{projected} = F.linear({inputs_embeds}, {projection_weight})")
    lines.append(f"{indent}if float({projection_scale}) != 1.0:")
    lines.append(
        f"{indent}    {projected} = {projected} * torch.tensor(float({projection_scale}), dtype={projected}.dtype, device={projected}.device)"
    )
    lines.append(
        f"{indent}{projected} = {projected}.reshape(*{inputs_embeds}.shape[:-1], int({num_layers}), int({per_layer_dim}))"
    )
    lines.append(
        f"{indent}{mean_squared} = {projected}.float().pow(2).mean(dim=-1, keepdim=True) + float({rms_eps})"
    )
    lines.append(
        f"{indent}{normed} = {projected}.float() * torch.pow({mean_squared}, -0.5)"
    )
    lines.append(
        f"{indent}{projected} = ({normed} * {norm_weight}.float()).to(dtype={projected}.dtype)"
    )
    lines.append(
        f"{indent}{out_var} = ({projected} + {embedded}) * torch.tensor(float({combine_scale}), dtype={projected}.dtype, device={projected}.device)"
    )
    return lines


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any", "Any"),
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
    "interpret",
    "compile",
    "uses_node_path",
    "LOWERING_TYPE_SIGNATURE",
]
