from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

OP_NAME = "layernorm"
LOWERING_ARITY = (1, 6)
LOWERING_ALLOWED_KWARGS: set[str] = set()
LOWERING_REQUIRED_KWARGS: set[str] = set()
LOWERING_KWARG_KINDS: dict[str, Any] = {}


def _dims_compatible(left: Any, right: Any) -> bool:
    if isinstance(left, str) and left.strip().lstrip("-").isdigit():
        left = int(left.strip())
    if isinstance(right, str) and right.strip().lstrip("-").isdigit():
        right = int(right.strip())
    return left == right


def _raw_args(node_spec: dict[str, Any]) -> list[Any]:
    raw = node_spec.get("_args")
    if isinstance(raw, list):
        return list(raw)
    if raw is None:
        return []
    return [raw]


def _arg_or_default(args: list[Any], index: int, default: Any) -> Any:
    if index >= len(args):
        return default
    value = args[index]
    if isinstance(value, str) and value.strip().lower() == "null":
        return default
    return value


def _path_override(args: list[Any], index: int) -> str | None:
    if index >= len(args):
        return None
    value = args[index]
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() == "null":
            return None
        return stripped
    return None


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
    del out, ctx
    if kwargs:
        unknown = ", ".join(sorted(str(key) for key in kwargs))
        raise ValueError(f"layernorm unsupported kwargs: {unknown}")
    if len(args) > 6:
        raise ValueError(f"layernorm expects at most 6 positional args, got {len(args)}")


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
    norm_dim = args[2] if len(args) >= 3 else None
    if isinstance(norm_dim, str) and norm_dim.strip().lower() == "null":
        norm_dim = None
    if norm_dim is not None:
        unresolved_symbolic = (
            isinstance(norm_dim, str)
            and norm_dim.isidentifier()
            and isinstance(first_dim, str)
            and first_dim.isidentifier()
        )
        if (
            first_dim is not None
            and not unresolved_symbolic
            and not _dims_compatible(norm_dim, first_dim)
        ):
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
    args = _raw_args(node_spec)
    if not args:
        raise ValueError(
            "layernorm requires positional args: x [eps dim weight_path bias bias_path]"
        )
    x = model._read_tensor_input(args[0], env)
    eps_expr = _arg_or_default(args, 1, 1e-5)
    weight_override = _path_override(args, 3)
    bias_expr = _arg_or_default(args, 4, True)
    bias_override = _path_override(args, 5)
    if weight_override in {"weight_path", "bias_path"}:
        weight_override = None
    if bias_override in {"weight_path", "bias_path"}:
        bias_override = None
    path_spec = dict(node_spec)
    weight_param = "weight"
    direct_weight_path: str | None = None
    if weight_override is not None:
        if weight_override.isidentifier():
            resolved = env.get(weight_override)
            if isinstance(resolved, str):
                direct_weight_path = resolved
            else:
                path_spec["weight_path"] = weight_override
                weight_param = "weight_path"
        else:
            path_spec["weight_path"] = weight_override
            weight_param = "weight_path"
    bias_param = "bias"
    direct_bias_path: str | None = None
    if bias_override is not None:
        if bias_override.isidentifier():
            resolved = env.get(bias_override)
            if isinstance(resolved, str):
                direct_bias_path = resolved
            else:
                path_spec["bias_path"] = bias_override
                bias_param = "bias_path"
        else:
            path_spec["bias_path"] = bias_override
            bias_param = "bias_path"

    weight_path = (
        direct_weight_path
        if isinstance(direct_weight_path, str)
        else model._infer_param_path(
            path_spec,
            node_path=node_path,
            param_name=weight_param,
        )
    )
    weight = model._state[weight_path]
    has_bias = bool(model._eval_expr(bias_expr, env, symbols))
    bias = (
        model._state[
            (
                direct_bias_path
                if isinstance(direct_bias_path, str)
                else model._infer_param_path(
                    path_spec,
                    node_path=node_path,
                    param_name=bias_param,
                )
            )
        ]
        if has_bias
        else None
    )
    eps_value = model._eval_expr(eps_expr, env, symbols)
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
    lines: list[str] = []

    def assign_out_var(out_name: str) -> str:
        return emitter._assign_out_var(env, out_name)

    def read(name: str) -> str:
        return emitter._read_env_var(env, name)

    args = _raw_args(node_spec)
    if not args:
        raise ValueError(
            "layernorm requires positional args: x [eps dim weight_path bias bias_path]"
        )
    src = read(str(args[0]))
    eps_expr = _arg_or_default(args, 1, 1e-5)
    weight_override = _path_override(args, 3)
    bias_expr = _arg_or_default(args, 4, True)
    bias_override = _path_override(args, 5)
    if weight_override in {"weight_path", "bias_path"}:
        weight_override = None
    if bias_override in {"weight_path", "bias_path"}:
        bias_override = None
    path_spec = dict(node_spec)
    weight_param = "weight"
    weight_param_expr: str | None = None
    if weight_override is not None:
        if weight_override.isidentifier() and weight_override in env:
            weight_param_expr = f"self._param({read(weight_override)})"
        else:
            path_spec["weight_path"] = weight_override
            weight_param = "weight_path"
    bias_param = "bias"
    bias_param_expr: str | None = None
    if bias_override is not None:
        if bias_override.isidentifier() and bias_override in env:
            bias_param_expr = f"self._param({read(bias_override)})"
        else:
            path_spec["bias_path"] = bias_override
            bias_param = "bias_path"
    out_name = str(node_spec.get("_bind"))
    out_var = assign_out_var(out_name)
    eps = emitter._expr_code(eps_expr, env)
    w = (
        weight_param_expr
        if isinstance(weight_param_expr, str)
        else emitter._hoisted_param(
            node_spec=path_spec,
            node_path_var=node_path_var,
            param_name=weight_param,
            lines=lines,
            indent=indent,
        )
    )
    b = (
        bias_param_expr
        if isinstance(bias_param_expr, str)
        else emitter._hoisted_optional_param(
            node_spec=path_spec,
            node_path_var=node_path_var,
            param_name=bias_param,
            lines=lines,
            indent=indent,
        )
    )
    lines.append(f"{indent}if not bool({emitter._expr_code(bias_expr, env)}):")
    lines.append(f"{indent}    {b} = None")
    lines.append(f"{indent}elif {b} is None:")
    lines.append(
        f"{indent}    raise ValueError('layernorm.bias tensor not found for resolved path')"
    )
    b_fp32 = f"{b}.float() if {b} is not None else None"
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
