from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

OP_NAME = "layernorm"
LOWERING_ARITY = (1, 1)
LOWERING_ALLOWED_KWARGS: set[str] = {"eps", "dim", "weight", "bias"}
LOWERING_REQUIRED_KWARGS: set[str] = set()
LOWERING_KWARG_KINDS: dict[str, Any] = {
    "dim": "dim",
    "eps": "number",
    "weight": "str",
    "bias": "str_or_bool_or_null",
}


def _dims_compatible(left: Any, right: Any) -> bool:
    if isinstance(left, str) and left.strip().lstrip("-").isdigit():
        left = int(left.strip())
    if isinstance(right, str) and right.strip().lstrip("-").isdigit():
        right = int(right.strip())
    return left == right


def _validate_layernorm_keys(node_spec: dict[str, Any]) -> None:
    if "weight" in node_spec and not isinstance(node_spec.get("weight"), str):
        raise ValueError("layernorm weight must be a string when provided")
    if "bias" in node_spec:
        bias = node_spec.get("bias")
        if isinstance(bias, str):
            return
        if bias is None:
            return
        if isinstance(bias, bool):
            return
        raise ValueError("layernorm bias must be a string, bool, or null when provided")


def _layernorm_has_bias(node_spec: dict[str, Any]) -> bool:
    if "bias" not in node_spec:
        return True
    bias = node_spec.get("bias")
    if bias is None:
        return False
    if isinstance(bias, bool):
        return bool(bias)
    return True


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
            raise ValueError(f"layernorm dim={norm_dim!r} mismatches input last-dim {first_dim!r}")
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
    _validate_layernorm_keys(node_spec)
    x = model._read_tensor_input(node_spec.get("_args"), env)
    weight = model._state[
        model._infer_param_path(node_spec, node_path=node_path, param_name="weight")
    ]
    has_bias = _layernorm_has_bias(node_spec)
    bias = (
        model._state[model._infer_param_path(node_spec, node_path=node_path, param_name="bias")]
        if has_bias
        else None
    )
    eps_value = model._eval_expr(node_spec.get("eps", 1e-5), env, symbols)
    out = model._require_name(node_spec.get("_bind"), field="layernorm._bind")
    align_norm_fp32 = bool(getattr(model, "_hf_align_norm_fp32", False))
    if align_norm_fp32 and x.is_floating_point() and x.dtype in {torch.float16, torch.bfloat16}:
        env[out] = F.layer_norm(
            x.float(),
            (x.shape[-1],),
            weight=weight.float(),
            bias=(bias.float() if bias is not None else None),
            eps=float(eps_value),
        ).to(dtype=x.dtype)
    else:
        env[out] = F.layer_norm(x, (x.shape[-1],), weight=weight, bias=bias, eps=float(eps_value))
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
    _validate_layernorm_keys(node_spec)
    lines: list[str] = []

    def assign_out_var(out_name: str) -> str:
        return emitter._assign_out_var(env, out_name)

    def read(name: str) -> str:
        return emitter._read_env_var(env, name)

    src = read(str(node_spec.get("_args")))
    out_name = str(node_spec.get("_bind"))
    out_var = assign_out_var(out_name)
    eps = emitter._expr_code(node_spec.get("eps", 1e-5), env)
    w = f"emitter._param({emitter._infer_param_expr(node_spec, node_path_var, 'weight')})"
    if _layernorm_has_bias(node_spec):
        b = f"emitter._param({emitter._infer_param_expr(node_spec, node_path_var, 'bias')})"
        b_fp32 = f"{b}.float()"
    else:
        b = "None"
        b_fp32 = "None"
    lines.append(
        f"{indent}if getattr(self, '_hf_align_norm_fp32', False) and {src}.is_floating_point() and {src}.dtype in {{torch.float16, torch.bfloat16}}:"
    )
    lines.append(
        f"{indent}    {out_var} = F.layer_norm({src}.float(), ({src}.shape[-1],), weight={w}.float(), bias={b_fp32}, eps=float({eps})).to(dtype={src}.dtype)"
    )
    lines.append(f"{indent}else:")
    lines.append(
        f"{indent}    {out_var} = F.layer_norm({src}, ({src}.shape[-1],), weight={w}, bias={b}, eps=float({eps}))"
    )
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
