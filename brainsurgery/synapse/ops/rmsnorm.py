from __future__ import annotations

from typing import Any

import torch

OP_NAME = "rmsnorm"
LOWERING_ARITY = (1, 1)
LOWERING_ALLOWED_KWARGS: set[str] = {"eps", "dim", "unit_offset", "cast_float", "with_scale"}
LOWERING_REQUIRED_KWARGS: set[str] = set()
LOWERING_KWARG_KINDS: dict[str, Any] = {
    "dim": "dim",
    "eps": "number",
    "cast_float": "bool",
    "unit_offset": "bool",
    "with_scale": "bool",
}


def _dims_compatible(left: Any, right: Any) -> bool:
    if isinstance(left, str) and left.strip().lstrip("-").isdigit():
        left = int(left.strip())
    if isinstance(right, str) and right.strip().lstrip("-").isdigit():
        right = int(right.strip())
    return left == right


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return True


def lowering_normalize_kwargs(
    *,
    args: list[str],
    out: str | list[str],
    kwargs: dict[str, Any],
    ctx: Any,
) -> None:
    del out
    if "dim" in kwargs or not args:
        return
    first_arg = args[0].strip()
    if not first_arg.isidentifier():
        return
    inferred = ctx.tensor_last_dim.get(first_arg)
    if inferred is not None:
        kwargs["dim"] = inferred


def lowering_infer_metadata(
    *,
    args: list[str],
    out: str | list[str],
    kwargs: dict[str, Any],
    ctx: Any,
) -> bool:
    if not isinstance(out, str):
        return False
    first_in = args[0].strip() if args else None
    first_dim = (
        ctx.tensor_last_dim.get(first_in)
        if isinstance(first_in, str) and first_in.isidentifier()
        else None
    )
    norm_dim = kwargs.get("dim")
    if norm_dim is not None:
        if first_dim is not None and not _dims_compatible(norm_dim, first_dim):
            raise ValueError(f"rmsnorm dim={norm_dim!r} mismatches input last-dim {first_dim!r}")
        if first_dim is None and isinstance(first_in, str) and first_in.isidentifier():
            ctx.tensor_last_dim[first_in] = norm_dim
    if first_dim is not None:
        ctx.tensor_last_dim[out] = first_dim
    if isinstance(first_in, str) and first_in.isidentifier() and first_in in ctx.tensor_shape:
        ctx.tensor_shape[out] = ctx.tensor_shape[first_in]
    return True


def interpret(
    model: Any,
    node_spec: dict[str, Any],
    env: dict[str, Any],
    *,
    node_path: str,
    scope: str,
    symbols: dict[str, int],
) -> None:
    x = model._read_tensor_input(node_spec.get("_args"), env)
    with_scale = bool(node_spec.get("with_scale", True))
    weight = None
    if with_scale:
        weight = model._state[
            model._infer_param_path(node_spec, node_path=node_path, param_name="weight")
        ]
    eps_value = float(model._eval_expr(node_spec.get("eps", 1e-6), env, symbols))
    cast_float = bool(node_spec.get("cast_float", False))
    align_norm_fp32 = bool(getattr(model, "_hf_align_norm_fp32", False))
    auto_cast_float = (
        align_norm_fp32 and x.is_floating_point() and x.dtype in {torch.float16, torch.bfloat16}
    )
    do_cast_float = cast_float or auto_cast_float
    unit_offset = bool(node_spec.get("unit_offset", False))
    if do_cast_float:
        # Match HF RMSNorm ordering for bf16/fp16 parity:
        # normalize in fp32, cast normalized activations back, then apply weight.
        x_norm_fp = x.float() * torch.rsqrt(
            torch.mean(x.float() * x.float(), dim=-1, keepdim=True) + eps_value
        )
        target_dtype = x.dtype if x.is_floating_point() else torch.float32
        x_norm = x_norm_fp.to(dtype=target_dtype)
        if with_scale:
            assert torch.is_tensor(weight)
            w_src = weight.to(device=x.device, dtype=target_dtype)
            y = x_norm * ((1.0 + w_src) if unit_offset else w_src)
        else:
            y = x_norm
    else:
        x_norm = x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps_value)
        if with_scale:
            assert torch.is_tensor(weight)
            y = x_norm * ((1.0 + weight) if unit_offset else weight)
        else:
            y = x_norm
    out = model._require_name(node_spec.get("_bind"), field="rmsnorm._bind")
    env[out] = y
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
    lines: list[str] = []

    def assign_out_var(out_name: str) -> str:
        return emitter._assign_out_var(env, out_name)

    def read(name: str) -> str:
        return emitter._read_env_var(env, name)

    src = read(str(node_spec.get("_args")))
    out_name = str(node_spec.get("_bind"))
    out_var = assign_out_var(out_name)
    eps = emitter._expr_code(node_spec.get("eps", 1e-6), env)
    tmp = emitter._fresh("xnorm")
    cast_float = bool(node_spec.get("cast_float", False))
    unit_offset = bool(node_spec.get("unit_offset", False))
    with_scale = bool(node_spec.get("with_scale", True))
    weight = None
    if with_scale:
        weight = emitter._hoisted_param(
            node_spec=node_spec,
            node_path_var=node_path_var,
            param_name="weight",
            lines=lines,
            indent=indent,
        )
    auto_cast_cond = f"torch.is_tensor({src}) and {src}.is_floating_point() and {src}.dtype in {{torch.float16, torch.bfloat16}}"
    lines.append(f"{indent}if getattr(self, '_hf_align_norm_fp32', False) and {auto_cast_cond}:")
    norm_cast = emitter._fresh("xnorm_cast")
    weight_cast = emitter._fresh("weight_cast")
    lines.append(
        f"{indent}    {tmp} = {src}.float() * torch.rsqrt(torch.mean({src}.float() * {src}.float(), dim=-1, keepdim=True) + float({eps}))"
    )
    lines.append(f"{indent}    {norm_cast} = {tmp}.to(dtype={src}.dtype)")
    if with_scale:
        lines.append(
            f"{indent}    {weight_cast} = {weight}.to(device={src}.device, dtype={src}.dtype)"
        )
    if with_scale and unit_offset:
        lines.append(f"{indent}    {out_var} = {norm_cast} * (1.0 + {weight_cast})")
    elif with_scale:
        lines.append(f"{indent}    {out_var} = {norm_cast} * {weight_cast}")
    else:
        lines.append(f"{indent}    {out_var} = {norm_cast}")
    lines.append(f"{indent}else:")
    if cast_float:
        norm_cast_local = emitter._fresh("xnorm_cast_local")
        weight_cast_local = emitter._fresh("weight_cast_local")
        lines.append(
            f"{indent}    {tmp} = {src}.float() * torch.rsqrt(torch.mean({src}.float() * {src}.float(), dim=-1, keepdim=True) + float({eps}))"
        )
        lines.append(f"{indent}    {norm_cast_local} = {tmp}.to(dtype={src}.dtype)")
        if with_scale:
            lines.append(
                f"{indent}    {weight_cast_local} = {weight}.to(device={src}.device, dtype={src}.dtype)"
            )
        if with_scale and unit_offset:
            lines.append(f"{indent}    {out_var} = {norm_cast_local} * (1.0 + {weight_cast_local})")
        elif with_scale:
            lines.append(f"{indent}    {out_var} = {norm_cast_local} * {weight_cast_local}")
        else:
            lines.append(f"{indent}    {out_var} = {norm_cast_local}")
    else:
        lines.append(
            f"{indent}    {tmp} = {src} * torch.rsqrt(torch.mean({src} * {src}, dim=-1, keepdim=True) + float({eps}))"
        )
        if with_scale and unit_offset:
            lines.append(f"{indent}    {out_var} = {tmp} * (1.0 + {weight})")
        elif with_scale:
            lines.append(f"{indent}    {out_var} = {tmp} * {weight}")
        else:
            lines.append(f"{indent}    {out_var} = {tmp}")
    return lines


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any",),
    "kwargs": dict(LOWERING_KWARG_KINDS),
    "returns": "dynamic",
}

__all__ = [
    "LOWERING_ARITY",
    "LOWERING_ALLOWED_KWARGS",
    "LOWERING_REQUIRED_KWARGS",
    "LOWERING_KWARG_KINDS",
    "OP_NAME",
    "lowering_normalize_kwargs",
    "lowering_infer_metadata",
    "interpret",
    "compile",
    "uses_node_path",
    "LOWERING_TYPE_SIGNATURE",
]
