from __future__ import annotations

from typing import Any

import torch

OP_NAME = "rmsnorm"
LOWERING_ARITY = (1, 1)
LOWERING_ALLOWED_KWARGS: set[str] = {"eps", "dim", "unit_offset", "cast_float"}
LOWERING_REQUIRED_KWARGS: set[str] = set()
LOWERING_KWARG_KINDS: dict[str, Any] = {
    "dim": "dim",
    "eps": "number",
    "cast_float": "bool",
    "unit_offset": "bool",
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
    x_norm_src = x.float() if do_cast_float else x
    w_src = weight.float() if do_cast_float else weight
    x_norm = x_norm_src * torch.rsqrt(
        torch.mean(x_norm_src * x_norm_src, dim=-1, keepdim=True) + eps_value
    )
    y = x_norm * ((1.0 + w_src) if unit_offset else w_src)
    out = model._require_name(node_spec.get("_bind"), field="rmsnorm._bind")
    env[out] = y.type_as(x) if do_cast_float else y
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

    def infer_param(param_name: str) -> str:
        return emitter._infer_param_expr(node_spec, node_path_var, param_name)

    def read(name: str) -> str:
        return emitter._read_env_var(env, name)

    src = read(str(node_spec.get("_args")))
    out_name = str(node_spec.get("_bind"))
    out_var = assign_out_var(out_name)
    eps = emitter._expr_code(node_spec.get("eps", 1e-6), env)
    tmp = emitter._fresh("xnorm")
    cast_float = bool(node_spec.get("cast_float", False))
    unit_offset = bool(node_spec.get("unit_offset", False))
    auto_cast_cond = f"torch.is_tensor({src}) and {src}.is_floating_point() and {src}.dtype in {{torch.float16, torch.bfloat16}}"
    lines.append(f"{indent}if getattr(self, '_hf_align_norm_fp32', False) and {auto_cast_cond}:")
    lines.append(
        f"{indent}    {tmp} = {src}.float() * torch.rsqrt(torch.mean({src}.float() * {src}.float(), dim=-1, keepdim=True) + float({eps}))"
    )
    if unit_offset:
        lines.append(
            f"{indent}    {out_var} = ({tmp} * (1.0 + emitter._param({infer_param('weight')}).float())).type_as({src})"
        )
    else:
        lines.append(
            f"{indent}    {out_var} = ({tmp} * emitter._param({infer_param('weight')}).float()).type_as({src})"
        )
    lines.append(f"{indent}else:")
    x_norm_src = f"{src}.float()" if cast_float else src
    w_src = (
        f"emitter._param({infer_param('weight')}).float()"
        if cast_float
        else f"emitter._param({infer_param('weight')})"
    )
    cast_ref = src
    if cast_float and out_var == src:
        cast_ref = emitter._fresh("rms_src")
        lines.append(f"{indent}    {cast_ref} = {src}")
    lines.append(
        f"{indent}    {tmp} = {x_norm_src} * torch.rsqrt(torch.mean({x_norm_src} * {x_norm_src}, dim=-1, keepdim=True) + float({eps}))"
    )
    if unit_offset:
        lines.append(f"{indent}    {out_var} = {tmp} * (1.0 + {w_src})")
    else:
        lines.append(f"{indent}    {out_var} = {tmp} * {w_src}")
    if cast_float:
        lines.append(f"{indent}    {out_var} = {out_var}.type_as({cast_ref})")
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
