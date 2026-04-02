from __future__ import annotations

import math
from typing import Any

import torch

OP_NAME = "linear_position_bias"
LOWERING_ARITY = (1, 1)
LOWERING_ALLOWED_KWARGS: set[str] = {"heads", "scale"}
LOWERING_REQUIRED_KWARGS: set[str] = {"heads"}
LOWERING_KWARG_KINDS: dict[str, Any] = {"heads": "dim", "scale": "number"}


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return False


def lowering_validate_signature(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> None:
    del args, kwargs, ctx
    if not isinstance(out, str):
        raise ValueError("linear_position_bias requires a single scalar output binding")


def lowering_infer_metadata(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> bool:
    if not isinstance(out, str) or not args:
        return False
    source_name = str(args[0]).strip()
    source_shape = ctx.tensor_shape.get(source_name)
    if isinstance(source_shape, tuple) and len(source_shape) == 2:
        seq = source_shape[-1]
        heads = kwargs.get("heads", "heads")
        ctx.tensor_shape[out] = (source_shape[0], heads, 1, seq)
        ctx.tensor_last_dim[out] = seq
        return True
    if source_name in ctx.tensor_last_dim:
        ctx.tensor_last_dim[out] = ctx.tensor_last_dim[source_name]
        return True
    return False


def _build_slopes(*, num_heads: int, device: torch.device) -> torch.Tensor:
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))),
        device=device,
        dtype=torch.float32,
    )
    powers = torch.arange(1, 1 + closest_power_of_2, device=device, dtype=torch.int32)
    slopes = torch.pow(base, powers)
    if closest_power_of_2 != num_heads:
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))),
            device=device,
            dtype=torch.float32,
        )
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = torch.arange(
            1, 1 + 2 * num_remaining_heads, 2, device=device, dtype=torch.int32
        )
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)
    return slopes


def interpret(
    model: Any,
    node_spec: dict[str, Any],
    env: dict[str, Any],
    *,
    node_path: str,
    scope: str,
    symbols: dict[str, int],
) -> None:
    del node_path, scope
    mask = model._read_tensor_input(node_spec.get("_args"), env)
    if mask.ndim != 2:
        raise ValueError("linear_position_bias expects rank-2 attention mask [batch, seq]")
    heads = int(model._eval_expr(node_spec.get("heads"), env, symbols))
    if heads <= 0:
        raise ValueError("linear_position_bias heads must be > 0")
    scale = float(model._eval_expr(node_spec.get("scale", 1.0), env, symbols))
    batch_size, seq_len = int(mask.shape[0]), int(mask.shape[1])
    slopes = _build_slopes(num_heads=heads, device=mask.device)
    arange_tensor = (mask.to(torch.float32).cumsum(dim=-1) - 1.0) * mask.to(torch.float32)
    bias = scale * (slopes.view(1, heads, 1, 1) * arange_tensor.view(batch_size, 1, 1, seq_len))
    out_name = model._require_name(node_spec.get("_bind"), field="linear_position_bias._bind")
    env[out_name] = bias
    return


def compile(
    emitter: Any,
    node_spec: dict[str, Any],
    env: dict[str, str],
    *,
    node_path_var: str,
    scope_var: str,
    indent: str,
) -> list[str]:
    del node_path_var, scope_var
    lines: list[str] = []
    src = emitter._read_env_var(env, str(node_spec.get("_args")))
    out_var = emitter._assign_out_var(env, str(node_spec.get("_bind")))
    heads_expr = emitter._expr_code(node_spec.get("heads"), env)
    scale_expr = emitter._expr_code(node_spec.get("scale", 1.0), env)
    heads_var = emitter._fresh("heads")
    cp2 = emitter._fresh("cp2")
    base = emitter._fresh("base")
    powers = emitter._fresh("powers")
    slopes = emitter._fresh("slopes")
    extra_base = emitter._fresh("extra_base")
    remain = emitter._fresh("remain")
    extra_powers = emitter._fresh("extra_powers")
    arange = emitter._fresh("arange")
    lines.append(f"{indent}if {src}.ndim != 2:")
    lines.append(
        f"{indent}    raise ValueError('linear_position_bias expects rank-2 attention mask [batch, seq]')"
    )
    lines.append(f"{indent}{heads_var} = int({heads_expr})")
    lines.append(f"{indent}if {heads_var} <= 0:")
    lines.append(f"{indent}    raise ValueError('linear_position_bias heads must be > 0')")
    lines.append(f"{indent}{cp2} = 1 << ({heads_var}.bit_length() - 1)")
    lines.append(
        f"{indent}{base} = torch.tensor(2 ** (-(2 ** -(({cp2}.bit_length() - 1) - 3))), device={src}.device, dtype=torch.float32)"
    )
    lines.append(
        f"{indent}{powers} = torch.arange(1, 1 + {cp2}, device={src}.device, dtype=torch.int32)"
    )
    lines.append(f"{indent}{slopes} = torch.pow({base}, {powers})")
    lines.append(f"{indent}if {cp2} != {heads_var}:")
    lines.append(
        f"{indent}    {extra_base} = torch.tensor(2 ** (-(2 ** -(({cp2}.bit_length()) - 3))), device={src}.device, dtype=torch.float32)"
    )
    lines.append(f"{indent}    {remain} = min({cp2}, {heads_var} - {cp2})")
    lines.append(
        f"{indent}    {extra_powers} = torch.arange(1, 1 + 2 * {remain}, 2, device={src}.device, dtype=torch.int32)"
    )
    lines.append(
        f"{indent}    {slopes} = torch.cat([{slopes}, torch.pow({extra_base}, {extra_powers})], dim=0)"
    )
    lines.append(
        f"{indent}{arange} = (({src}.to(torch.float32).cumsum(dim=-1) - 1.0) * {src}.to(torch.float32))"
    )
    lines.append(
        f"{indent}{out_var} = float({scale_expr}) * ({slopes}.view(1, {heads_var}, 1, 1) * {arange}.view({src}.shape[0], 1, 1, {src}.shape[1]))"
    )
    return lines


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any",),
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
