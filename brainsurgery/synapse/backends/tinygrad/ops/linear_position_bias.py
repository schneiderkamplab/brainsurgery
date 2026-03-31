from __future__ import annotations

import math
from typing import Any

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


def interpret(
    model: Any,
    node_spec: dict[str, Any],
    env: dict[str, Any],
    *,
    node_path: str,
    scope: str,
    symbols: dict[str, int],
) -> None:
    raise NotImplementedError(f"TinyGrad interpret for '{OP_NAME}' not yet implemented")


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
        f"{indent}{base} = Tensor(2 ** (-(2 ** -(({cp2}.bit_length() - 1) - 3))), dtype=dtypes.float32)"
    )
    lines.append(
        f"{indent}{powers} = Tensor.arange(1, 1 + {cp2}, dtype=dtypes.int32).cast(dtypes.float32)"
    )
    lines.append(f"{indent}{slopes} = {base} ** {powers}")
    lines.append(f"{indent}if {cp2} != {heads_var}:")
    extra_base = emitter._fresh("extra_base")
    remain = emitter._fresh("remain")
    extra_powers = emitter._fresh("extra_powers")
    extra_slopes = emitter._fresh("extra_slopes")
    lines.append(
        f"{indent}    {extra_base} = Tensor(2 ** (-(2 ** -(({cp2}.bit_length()) - 3))), dtype=dtypes.float32)"
    )
    lines.append(f"{indent}    {remain} = min({cp2}, {heads_var} - {cp2})")
    lines.append(
        f"{indent}    {extra_powers} = (Tensor.arange(1 + 2 * {remain}, dtype=dtypes.int32) * 2 + 1).cast(dtypes.float32)"
    )
    lines.append(f"{indent}    {extra_slopes} = {extra_base} ** {extra_powers}")
    lines.append(f"{indent}    {slopes} = {slopes}.cat({extra_slopes}, dim=0)")
    lines.append(
        f"{indent}{arange} = (({src}.cast(dtypes.float32).cumsum(axis=-1) - 1.0) * {src}.cast(dtypes.float32))"
    )
    lines.append(
        f"{indent}{out_var} = float({scale_expr}) * ({slopes}.reshape(1, {heads_var}, 1, 1) * {arange}.reshape({src}.shape[0], 1, 1, {src}.shape[1]))"
    )
    return lines


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
]
