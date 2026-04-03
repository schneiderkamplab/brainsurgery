from __future__ import annotations

import math
from typing import Any

import torch

OP_NAME = "t5_relative_position_bias"
LOWERING_ARITY = (2, 2)
LOWERING_ALLOWED_KWARGS: set[str] = {"num_buckets", "max_distance", "bidirectional"}
LOWERING_REQUIRED_KWARGS: set[str] = set()
LOWERING_KWARG_KINDS: dict[str, Any] = {
    "num_buckets": "dim",
    "max_distance": "dim",
    "bidirectional": "bool",
}


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return True


def lowering_validate_signature(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> None:
    del args, kwargs, ctx
    if isinstance(out, list):
        raise ValueError("t5_relative_position_bias requires a single scalar output binding")


def lowering_infer_metadata(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> bool:
    del kwargs
    if isinstance(out, list) or len(args) < 2:
        return False
    query_name = str(args[0]).strip()
    key_name = str(args[1]).strip()
    query_shape = ctx.tensor_shape.get(query_name)
    key_shape = ctx.tensor_shape.get(key_name)
    if not (isinstance(query_shape, tuple) and isinstance(key_shape, tuple)):
        return False
    if len(query_shape) < 4 or len(key_shape) < 4:
        return False
    heads = query_shape[1]
    q_len = query_shape[-2]
    k_len = key_shape[-2]
    ctx.tensor_shape[out] = (1, heads, q_len, k_len)
    ctx.tensor_last_dim[out] = k_len
    return True


def _relative_position_buckets(
    relative_position: torch.Tensor,
    *,
    bidirectional: bool,
    num_buckets: int,
    max_distance: int,
) -> torch.Tensor:
    if num_buckets <= 0:
        raise ValueError("t5_relative_position_bias.num_buckets must be > 0")
    if max_distance <= 0:
        raise ValueError("t5_relative_position_bias.max_distance must be > 0")

    relative_buckets = torch.zeros_like(relative_position, dtype=torch.long)
    if bidirectional:
        half_buckets = num_buckets // 2
        if half_buckets <= 0:
            raise ValueError(
                "t5_relative_position_bias.num_buckets must be >= 2 when bidirectional=true"
            )
        relative_buckets = relative_buckets + (relative_position > 0).to(torch.long) * half_buckets
        relative_position = torch.abs(relative_position)
        bucket_count = half_buckets
    else:
        relative_position = -torch.minimum(relative_position, torch.zeros_like(relative_position))
        bucket_count = num_buckets

    max_exact = max(1, bucket_count // 2)
    if max_distance <= max_exact:
        raise ValueError("t5_relative_position_bias.max_distance must be > num_buckets//2")
    is_small = relative_position < max_exact

    relative_position_clamped = torch.clamp(
        relative_position.to(torch.float32), min=float(max_exact)
    )
    log_scale = math.log(float(max_distance) / float(max_exact))
    relative_position_if_large = max_exact + (
        torch.log(relative_position_clamped / float(max_exact))
        / log_scale
        * float(bucket_count - max_exact)
    ).to(torch.long)
    relative_position_if_large = torch.minimum(
        relative_position_if_large,
        torch.full_like(relative_position_if_large, bucket_count - 1),
    )
    return relative_buckets + torch.where(is_small, relative_position, relative_position_if_large)


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
    ins = node_spec.get("_args")
    if not isinstance(ins, list) or len(ins) != 2:
        raise ValueError("t5_relative_position_bias expects [q, k]")
    q = env[ins[0]]
    k_tensor = env[ins[1]]
    if not torch.is_tensor(q) or not torch.is_tensor(k_tensor):
        raise ValueError("t5_relative_position_bias expects tensor inputs for q and k")
    if q.ndim != 4 or k_tensor.ndim != 4:
        raise ValueError("t5_relative_position_bias expects q and k to be rank-4 [B,H,S,D]")

    num_buckets = int(model._eval_expr(node_spec.get("num_buckets", 32), env, symbols))
    max_distance = int(model._eval_expr(node_spec.get("max_distance", 128), env, symbols))
    bidirectional = bool(model._eval_expr(node_spec.get("bidirectional", True), env, symbols))

    weight_path = model._infer_param_path(node_spec, node_path=node_path, param_name="weight")
    weight = model._state[weight_path]
    if weight.ndim != 2:
        raise ValueError("t5_relative_position_bias weight must be rank-2 [num_buckets, heads]")
    if int(weight.shape[0]) < num_buckets:
        raise ValueError("t5_relative_position_bias weight rows must cover num_buckets")
    if int(weight.shape[1]) != int(q.shape[1]):
        raise ValueError("t5_relative_position_bias weight heads must match q heads")

    q_len = int(q.shape[-2])
    k_len = int(k_tensor.shape[-2])
    context_position = torch.arange(q_len, device=q.device, dtype=torch.long)[:, None]
    memory_position = torch.arange(k_len, device=q.device, dtype=torch.long)[None, :]
    relative_position = memory_position - context_position
    buckets = _relative_position_buckets(
        relative_position,
        bidirectional=bidirectional,
        num_buckets=num_buckets,
        max_distance=max_distance,
    )

    target_dtype = q.dtype if q.is_floating_point() else weight.dtype
    weight_run = weight.to(device=q.device, dtype=target_dtype)
    values = weight_run[buckets].permute(2, 0, 1).unsqueeze(0)
    out_name = model._require_name(node_spec.get("_bind"), field="t5_relative_position_bias._bind")
    env[out_name] = values


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

    ins = node_spec.get("_args")
    if not isinstance(ins, list) or len(ins) != 2:
        raise ValueError("t5_relative_position_bias expects [q, k]")

    q = read(str(ins[0]))
    k = read(str(ins[1]))
    out_var = assign_out_var(str(node_spec.get("_bind")))
    weight_expr = emitter._infer_param_expr(node_spec, node_path_var, "weight")
    num_buckets_expr = emitter._expr_code(node_spec.get("num_buckets", 32), env)
    max_distance_expr = emitter._expr_code(node_spec.get("max_distance", 128), env)
    bidirectional_expr = emitter._expr_code(node_spec.get("bidirectional", True), env)

    num_buckets = emitter._fresh("num_buckets")
    max_distance = emitter._fresh("max_distance")
    bidirectional = emitter._fresh("bidirectional")
    weight = emitter._fresh("rel_bias_weight")
    q_len = emitter._fresh("q_len")
    k_len = emitter._fresh("k_len")
    context_position = emitter._fresh("context_position")
    memory_position = emitter._fresh("memory_position")
    relative_position = emitter._fresh("relative_position")
    relative_buckets = emitter._fresh("relative_buckets")
    bucket_count = emitter._fresh("bucket_count")
    half_buckets = emitter._fresh("half_buckets")
    max_exact = emitter._fresh("max_exact")
    is_small = emitter._fresh("is_small")
    relative_position_clamped = emitter._fresh("relative_position_clamped")
    log_scale = emitter._fresh("log_scale")
    relative_position_if_large = emitter._fresh("relative_position_if_large")
    target_dtype = emitter._fresh("target_dtype")
    values = emitter._fresh("rel_bias_values")

    lines.append(f"{indent}if {q}.ndim != 4 or {k}.ndim != 4:")
    lines.append(
        f"{indent}    raise ValueError('t5_relative_position_bias expects q and k to be rank-4 [B,H,S,D]')"
    )
    lines.append(f"{indent}{num_buckets} = int({num_buckets_expr})")
    lines.append(f"{indent}{max_distance} = int({max_distance_expr})")
    lines.append(f"{indent}{bidirectional} = bool({bidirectional_expr})")
    lines.append(f"{indent}if {num_buckets} <= 0:")
    lines.append(
        f"{indent}    raise ValueError('t5_relative_position_bias.num_buckets must be > 0')"
    )
    lines.append(f"{indent}if {max_distance} <= 0:")
    lines.append(
        f"{indent}    raise ValueError('t5_relative_position_bias.max_distance must be > 0')"
    )
    lines.append(f"{indent}{weight} = emitter._param({weight_expr})")
    lines.append(f"{indent}if {weight}.ndim != 2:")
    lines.append(
        f"{indent}    raise ValueError('t5_relative_position_bias weight must be rank-2 [num_buckets, heads]')"
    )
    lines.append(f"{indent}if int({weight}.shape[0]) < {num_buckets}:")
    lines.append(
        f"{indent}    raise ValueError('t5_relative_position_bias weight rows must cover num_buckets')"
    )
    lines.append(f"{indent}if int({weight}.shape[1]) != int({q}.shape[1]):")
    lines.append(
        f"{indent}    raise ValueError('t5_relative_position_bias weight heads must match q heads')"
    )
    lines.append(f"{indent}{q_len} = int({q}.shape[-2])")
    lines.append(f"{indent}{k_len} = int({k}.shape[-2])")
    lines.append(
        f"{indent}{context_position} = torch.arange({q_len}, device={q}.device, dtype=torch.long)[:, None]"
    )
    lines.append(
        f"{indent}{memory_position} = torch.arange({k_len}, device={q}.device, dtype=torch.long)[None, :]"
    )
    lines.append(f"{indent}{relative_position} = {memory_position} - {context_position}")
    lines.append(
        f"{indent}{relative_buckets} = torch.zeros_like({relative_position}, dtype=torch.long)"
    )
    lines.append(f"{indent}if {bidirectional}:")
    lines.append(f"{indent}    {half_buckets} = {num_buckets} // 2")
    lines.append(f"{indent}    if {half_buckets} <= 0:")
    lines.append(
        f"{indent}        raise ValueError('t5_relative_position_bias.num_buckets must be >= 2 when bidirectional=true')"
    )
    lines.append(
        f"{indent}    {relative_buckets} = {relative_buckets} + ({relative_position} > 0).to(torch.long) * {half_buckets}"
    )
    lines.append(f"{indent}    {relative_position} = torch.abs({relative_position})")
    lines.append(f"{indent}    {bucket_count} = {half_buckets}")
    lines.append(f"{indent}else:")
    lines.append(
        f"{indent}    {relative_position} = -torch.minimum({relative_position}, torch.zeros_like({relative_position}))"
    )
    lines.append(f"{indent}    {bucket_count} = {num_buckets}")
    lines.append(f"{indent}{max_exact} = max(1, {bucket_count} // 2)")
    lines.append(f"{indent}if {max_distance} <= {max_exact}:")
    lines.append(
        f"{indent}    raise ValueError('t5_relative_position_bias.max_distance must be > num_buckets//2')"
    )
    lines.append(f"{indent}{is_small} = {relative_position} < {max_exact}")
    lines.append(
        f"{indent}{relative_position_clamped} = torch.clamp({relative_position}.to(torch.float32), min=float({max_exact}))"
    )
    lines.append(f"{indent}{log_scale} = math.log(float({max_distance}) / float({max_exact}))")
    lines.append(
        f"{indent}{relative_position_if_large} = {max_exact} + (torch.log({relative_position_clamped} / float({max_exact})) / {log_scale} * float({bucket_count} - {max_exact})).to(torch.long)"
    )
    lines.append(
        f"{indent}{relative_position_if_large} = torch.minimum({relative_position_if_large}, torch.full_like({relative_position_if_large}, {bucket_count} - 1))"
    )
    lines.append(
        f"{indent}{relative_buckets} = {relative_buckets} + torch.where({is_small}, {relative_position}, {relative_position_if_large})"
    )
    lines.append(
        f"{indent}{target_dtype} = {q}.dtype if {q}.is_floating_point() else {weight}.dtype"
    )
    lines.append(f"{indent}{weight} = {weight}.to(device={q}.device, dtype={target_dtype})")
    lines.append(f"{indent}{values} = {weight}[{relative_buckets}].permute(2, 0, 1).unsqueeze(0)")
    lines.append(f"{indent}{out_var} = {values}")
    return lines


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any", "Any"),
    "kwargs": dict(LOWERING_KWARG_KINDS),
    "returns": ("Tensor",),
}

__all__ = [
    "LOWERING_ARITY",
    "LOWERING_ALLOWED_KWARGS",
    "LOWERING_REQUIRED_KWARGS",
    "LOWERING_KWARG_KINDS",
    "OP_NAME",
    "lowering_validate_signature",
    "lowering_infer_metadata",
    "interpret",
    "compile",
    "uses_node_path",
    "LOWERING_TYPE_SIGNATURE",
]
