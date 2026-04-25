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
    if _is_variadic_dim(left) or _is_variadic_dim(right):
        return True
    if isinstance(left, str) and left.strip().lstrip("-").isdigit():
        left = int(left.strip())
    if isinstance(right, str) and right.strip().lstrip("-").isdigit():
        right = int(right.strip())
    return left == right


def _is_variadic_dim(value: Any) -> bool:
    return isinstance(value, str) and value.strip().startswith("..")


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


def _has_explicit_path(path_spec: dict[str, Any], key: str) -> bool:
    value = path_spec.get(key)
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    return bool(stripped) and stripped.lower() != "null"


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
    if _is_variadic_dim(first_dim):
        first_dim = None
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
    output_dim = first_dim if first_dim is not None else norm_dim
    if output_dim is not None:
        ctx.tensor_last_dim[out] = output_dim
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
    path_spec = dict(node_spec)
    weight_param = "weight_path" if _has_explicit_path(path_spec, "weight_path") else "weight"
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
    bias_param = "bias_path" if _has_explicit_path(path_spec, "bias_path") else "bias"
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

    if isinstance(direct_weight_path, str):
        resolved_direct_weight = model._resolve_state_path(
            node_path=node_path, raw_path=direct_weight_path
        )
        if resolved_direct_weight in model._state:
            weight_path = resolved_direct_weight
        else:
            weight_path = model._infer_param_path(
                path_spec,
                node_path=node_path,
                param_name=weight_param,
                env=env,
            )
    else:
        weight_path = model._infer_param_path(
            path_spec,
            node_path=node_path,
            param_name=weight_param,
            env=env,
        )
    if isinstance(weight_path, str) and weight_path.strip() in {"@weight", "weight"}:
        weight_path = model._infer_param_path(
            path_spec,
            node_path=node_path,
            param_name=weight_param,
            env=env,
        )
    weight = model._state_tensor_from_resolved_path(weight_path, field="layernorm.weight")
    has_bias = bool(model._eval_expr(bias_expr, env, symbols))
    if isinstance(direct_bias_path, str):
        resolved_direct_bias = model._resolve_state_path(
            node_path=node_path, raw_path=direct_bias_path
        )
        if resolved_direct_bias in model._state:
            bias_path = resolved_direct_bias
        else:
            bias_path = model._infer_param_path(
                path_spec,
                node_path=node_path,
                param_name=bias_param,
                env=env,
            )
    else:
        bias_path = model._infer_param_path(
            path_spec,
            node_path=node_path,
            param_name=bias_param,
            env=env,
        )
    if isinstance(bias_path, str) and bias_path.strip() in {"@bias", "bias"}:
        bias_path = model._infer_param_path(
            path_spec,
            node_path=node_path,
            param_name=bias_param,
            env=env,
        )
    bias = (
        model._state_tensor_from_resolved_path(bias_path, field="layernorm.bias")
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
    path_spec = dict(node_spec)
    weight_param = "weight_path" if _has_explicit_path(path_spec, "weight_path") else "weight"
    weight_override_name: str | None = None
    if weight_override is not None:
        if weight_override.isidentifier() and weight_override in env:
            weight_override_name = weight_override
        else:
            path_spec["weight_path"] = weight_override
            weight_param = "weight_path"
    bias_param = "bias_path" if _has_explicit_path(path_spec, "bias_path") else "bias"
    bias_override_name: str | None = None
    if bias_override is not None:
        if bias_override.isidentifier() and bias_override in env:
            bias_override_name = bias_override
        else:
            path_spec["bias_path"] = bias_override
            bias_param = "bias_path"
    out_name = str(node_spec.get("_bind"))
    out_var = assign_out_var(out_name)
    eps = emitter._expr_code(eps_expr, env)
    w = (
        emitter._hoisted_optional_param(
            node_spec=path_spec,
            node_path_var=node_path_var,
            param_name=weight_param,
            lines=lines,
            indent=indent,
        )
        if weight_override_name is not None
        else emitter._hoisted_param(
            node_spec=path_spec,
            node_path_var=node_path_var,
            param_name=weight_param,
            lines=lines,
            indent=indent,
        )
    )
    b = emitter._hoisted_optional_param(
        node_spec=path_spec,
        node_path_var=node_path_var,
        param_name=bias_param,
        lines=lines,
        indent=indent,
    )
    if weight_override_name is not None:
        raw_weight_override = emitter._fresh("raw_weight_override")
        lines.append(f"{indent}{raw_weight_override} = {read(weight_override_name)}")
        lines.append(f"{indent}if isinstance({raw_weight_override}, str):")
        lines.append(f"{indent}    if {raw_weight_override}.strip() not in ('@weight', 'weight'):")
        lines.append(f"{indent}        try:")
        lines.append(
            f"{indent}            {w} = self._state_tensor_from_path("
            f"node_path={node_path_var}, raw_path={raw_weight_override}, field='layernorm.weight')"
        )
        lines.append(f"{indent}        except ValueError:")
        lines.append(f"{indent}            pass")
    lines.append(f"{indent}if {w} is None:")
    lines.append(
        f"{indent}    raise ValueError('layernorm.weight tensor not found for resolved path')"
    )
    if bias_override_name is not None:
        raw_bias_override = emitter._fresh("raw_bias_override")
        lines.append(f"{indent}{raw_bias_override} = {read(bias_override_name)}")
        lines.append(f"{indent}if isinstance({raw_bias_override}, str):")
        lines.append(f"{indent}    if {raw_bias_override}.strip() not in ('@bias', 'bias'):")
        lines.append(f"{indent}        try:")
        lines.append(
            f"{indent}            {b} = self._state_tensor_from_path("
            f"node_path={node_path_var}, raw_path={raw_bias_override}, field='layernorm.bias')"
        )
        lines.append(f"{indent}        except ValueError:")
        lines.append(f"{indent}            pass")
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
    "args": ("Path", "Tensor[..S]", "?Float", "?Dim", "?Path", "?Bool", "?Path"),
    "kwargs": dict(LOWERING_KWARG_KINDS),
    "returns": ("Tensor[..S]",),
}


def type_rule(
    *,
    arg_types: tuple[Any, ...],
    kwarg_types: dict[str, Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    helpers: Any,
) -> Any | None:
    del kwarg_types, kwargs
    if len(arg_types) < 2:
        return None
    input_dims = helpers.type_dims(arg_types[1])
    if input_dims is None:
        return None
    dim_expr = args[3] if len(args) >= 4 else None
    dim_token = helpers.expr_to_dim_token(dim_expr)
    if dim_token is not None and input_dims:
        last = input_dims[-1]
        if last != dim_token:
            if not (isinstance(last, str) and isinstance(dim_token, str)):
                raise ValueError(
                    f"Axon typecheck failed: _layernorm dim {dim_token!r} mismatches input last dim {last!r}"
                )
    return helpers.type_tensor(dims=input_dims)


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
    "type_rule",
]
