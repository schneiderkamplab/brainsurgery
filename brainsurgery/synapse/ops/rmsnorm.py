from __future__ import annotations

from typing import Any

import torch

OP_NAME = "rmsnorm"
LOWERING_ARITY = (1, 4)
LOWERING_ALLOWED_KWARGS: set[str] = set()
LOWERING_REQUIRED_KWARGS: set[str] = set()
LOWERING_KWARG_KINDS: dict[str, Any] = {}


def _dims_compatible(left: Any, right: Any) -> bool:
    if _is_variadic_dim(left) or _is_variadic_dim(right):
        return True
    if isinstance(left, str) and left.strip().lstrip("-").isdigit():
        left = int(left.strip())
    if isinstance(right, str) and right.strip().lstrip("-").isdigit():
        right = int(right.strip())
    return left == right


def _is_variadic_dim(value: Any) -> bool:
    return isinstance(value, str) and value.strip().startswith("..")


def _bool_expr_or_default(
    *,
    emitter: Any,
    value: Any,
    env: dict[str, str],
    default: bool,
) -> str:
    if value is None:
        return "True" if default else "False"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, str):
        token = value.strip().lower()
        if token == "true":
            return "True"
        if token == "false":
            return "False"
        if token == "null":
            return "True" if default else "False"
    return f"bool({emitter._expr_code(value, env)})"


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return False


def lowering_normalize_kwargs(
    *,
    args: list[str],
    out: str | list[str],
    kwargs: dict[str, Any],
    ctx: Any,
) -> None:
    del args, out, kwargs, ctx


def lowering_infer_metadata(
    *,
    args: list[str],
    out: str | list[str],
    kwargs: dict[str, Any],
    ctx: Any,
) -> bool:
    del kwargs
    if not isinstance(out, str):
        return False
    first_in = args[0].strip() if args else None
    first_dim = (
        ctx.tensor_last_dim.get(first_in)
        if isinstance(first_in, str) and first_in.isidentifier()
        else None
    )
    if _is_variadic_dim(first_dim):
        first_dim = None
    norm_dim: Any = None
    if len(args) >= 3:
        raw = args[2].strip()
        if raw.lower() != "null":
            if raw.lstrip("-").isdigit():
                norm_dim = int(raw)
            else:
                norm_dim = raw
    if norm_dim is None:
        norm_dim = first_dim
    if norm_dim is not None:
        if first_dim is not None and not _dims_compatible(norm_dim, first_dim):
            raise ValueError(f"rmsnorm dim={norm_dim!r} mismatches input last-dim {first_dim!r}")
        if first_dim is None and isinstance(first_in, str) and first_in.isidentifier():
            ctx.tensor_last_dim[first_in] = norm_dim
    output_dim = first_dim if first_dim is not None else norm_dim
    if output_dim is not None:
        ctx.tensor_last_dim[out] = output_dim
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
    raw_args = node_spec.get("_args")
    args = raw_args if isinstance(raw_args, list) else [raw_args]
    if not isinstance(args, list) or len(args) < 1:
        raise ValueError("rmsnorm requires positional args: x [eps dim cast_float]")
    x = model._read_tensor_input(args[0], env)

    eps_raw = args[1] if len(args) >= 2 else node_spec.get("eps", 1e-6)
    eps_value = float(model._eval_expr(eps_raw, env, symbols))
    if len(args) >= 3:
        _ = model._eval_expr(args[2], env, symbols)
    cast_raw = args[3] if len(args) >= 4 else node_spec.get("cast_float", False)
    cast_float = bool(model._eval_expr(cast_raw, env, symbols))
    del node_path
    align_norm_fp32 = bool(getattr(model, "_hf_align_norm_fp32", False))
    auto_cast_float = (
        align_norm_fp32 and x.is_floating_point() and x.dtype in {torch.float16, torch.bfloat16}
    )
    do_cast_float = cast_float or auto_cast_float
    if do_cast_float:
        # Match HF RMSNorm ordering for bf16/fp16 parity:
        # normalize in fp32, cast normalized activations back, then apply weight.
        x_norm_fp = x.float() * torch.rsqrt(
            torch.mean(x.float() * x.float(), dim=-1, keepdim=True) + eps_value
        )
        target_dtype = x.dtype if x.is_floating_point() else torch.float32
        x_norm = x_norm_fp.to(dtype=target_dtype)
        y = x_norm
    else:
        x_norm = x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps_value)
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

    raw_args = node_spec.get("_args")
    args = raw_args if isinstance(raw_args, list) else [raw_args]
    if not isinstance(args, list) or len(args) < 1:
        raise ValueError("rmsnorm requires positional args: x [eps dim cast_float]")
    src = read(str(args[0]))
    out_name = str(node_spec.get("_bind"))
    out_var = assign_out_var(out_name)
    eps = (
        emitter._expr_code(args[1], env)
        if len(args) >= 2
        else emitter._expr_code(node_spec.get("eps", 1e-6), env)
    )
    tmp = emitter._fresh("xnorm")
    cast_float_expr = _bool_expr_or_default(
        emitter=emitter,
        value=args[3] if len(args) >= 4 else node_spec.get("cast_float", False),
        env=env,
        default=False,
    )
    cast_float_var = emitter._fresh("cast_float")
    lines.append(f"{indent}{cast_float_var} = {cast_float_expr}")
    del node_path_var
    auto_cast_cond = f"torch.is_tensor({src}) and {src}.is_floating_point() and {src}.dtype in {{torch.float16, torch.bfloat16}}"
    lines.append(
        f"{indent}if {cast_float_var} or (getattr(self, '_hf_align_norm_fp32', False) and {auto_cast_cond}):"
    )
    norm_cast = emitter._fresh("xnorm_cast")
    lines.append(
        f"{indent}    {tmp} = {src}.float() * torch.rsqrt(torch.mean({src}.float() * {src}.float(), dim=-1, keepdim=True) + float({eps}))"
    )
    lines.append(f"{indent}    {norm_cast} = {tmp}.to(dtype={src}.dtype)")
    lines.append(f"{indent}    {out_var} = {norm_cast}")
    lines.append(f"{indent}else:")
    lines.append(
        f"{indent}    {tmp} = {src} * torch.rsqrt(torch.mean({src} * {src}, dim=-1, keepdim=True) + float({eps}))"
    )
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
