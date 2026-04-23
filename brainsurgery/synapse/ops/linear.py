from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

OP_NAME = "linear"
LOWERING_ARITY = (1, 7)
LOWERING_ALLOWED_KWARGS: set[str] = set()
LOWERING_REQUIRED_KWARGS: set[str] = set()
LOWERING_KWARG_KINDS: dict[str, Any] = {}


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


def _path_override(args: list[Any], index: int) -> Any | None:
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
    return value


def _is_default_path_token(token: str, *, kind: str) -> bool:
    t = token.strip()
    if kind == "weight":
        return t in {"@weight", "weight"}
    if kind == "bias":
        return t in {"@bias", "bias"}
    return False


def _is_default_path_expr(value: Any, *, kind: str) -> bool:
    if isinstance(value, str):
        return _is_default_path_token(value, kind=kind)
    if not isinstance(value, dict):
        return False
    if value.get("_expr") != "path":
        return False
    if bool(value.get("absolute")):
        return False
    parts = value.get("parts")
    if isinstance(parts, tuple):
        parts = list(parts)
    if not isinstance(parts, list) or len(parts) != 1:
        return False
    leaf = parts[0]
    if not isinstance(leaf, str):
        return False
    if kind == "weight":
        return leaf == "weight"
    if kind == "bias":
        return leaf == "bias"
    return False


def _has_explicit_path(path_spec: dict[str, Any], key: str) -> bool:
    value = path_spec.get(key)
    if isinstance(value, dict):
        return True
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    return bool(stripped) and stripped.lower() != "null"


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter
    args = _raw_args(node_spec)
    bias_expr = _arg_or_default(args, 2, False)
    has_bias = bool(bias_expr) if isinstance(bias_expr, bool) else True
    explicit_weight = _path_override(args, 5)
    if isinstance(explicit_weight, str) and explicit_weight.startswith("@@"):
        return True
    has_explicit_weight = isinstance(explicit_weight, str) and "." in explicit_weight
    if not has_bias and has_explicit_weight:
        return False
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
        raise ValueError(f"linear unsupported kwargs: {unknown}")
    if len(args) > 7:
        raise ValueError(f"linear expects at most 7 positional args, got {len(args)}")


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
    dim_expr = args[1] if len(args) >= 2 else None
    if isinstance(dim_expr, str) and dim_expr.strip().lower() == "null":
        dim_expr = None
    last_dim = dim_expr if dim_expr is not None else first_dim
    if last_dim is not None:
        ctx.tensor_last_dim[out] = last_dim
    first_shape = (
        ctx.tensor_shape.get(first_in)
        if isinstance(first_in, str) and first_in.isidentifier()
        else None
    )
    if isinstance(first_shape, tuple) and len(first_shape) >= 1 and last_dim is not None:
        ctx.tensor_shape[out] = (*first_shape[:-1], last_dim)
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
    del scope
    args = _raw_args(node_spec)
    if not args:
        raise ValueError(
            "linear requires positional args: x [dim bias transpose expert weight_path bias_path]"
        )
    x = model._read_tensor_input(args[0], env)
    bias_expr = _arg_or_default(args, 2, False)
    transpose_expr = _arg_or_default(args, 3, False)
    expert_expr = _arg_or_default(args, 4, None)
    weight_override = _path_override(args, 5)
    bias_override = _path_override(args, 6)

    path_spec = dict(node_spec)
    weight_param = "weight_path" if _has_explicit_path(path_spec, "weight_path") else "weight"
    if weight_override is not None:
        override_value: Any = weight_override
        if isinstance(weight_override, str) and weight_override.isidentifier():
            resolved = env.get(weight_override)
            if resolved is not None:
                override_value = resolved
        if not _is_default_path_expr(override_value, kind="weight"):
            path_spec["weight_path"] = override_value
            weight_param = "weight_path"
    bias_param = "bias_path" if _has_explicit_path(path_spec, "bias_path") else "bias"
    if bias_override is not None:
        bias_override_value: Any = bias_override
        if isinstance(bias_override, str) and bias_override.isidentifier():
            resolved = env.get(bias_override)
            if resolved is not None:
                bias_override_value = resolved
        if not _is_default_path_expr(bias_override_value, kind="bias"):
            path_spec["bias_path"] = bias_override_value
            bias_param = "bias_path"

    linear_weight_path = model._infer_param_path(
        path_spec,
        node_path=node_path,
        param_name=weight_param,
    )
    weight = model._state[linear_weight_path]
    expert_idx: int | None = None
    if expert_expr is not None:
        expert_value = model._eval_expr(expert_expr, env, symbols)
        if expert_value is not None:
            expert_idx = int(expert_value)
    if expert_idx is not None:
        if weight.ndim < 2:
            raise ValueError("linear expert selection requires at least rank-2 weight tensor")
        if expert_idx < 0 or expert_idx >= int(weight.shape[0]):
            raise ValueError(
                f"linear expert index out of range: {expert_idx} for shape {tuple(weight.shape)}"
            )
        weight = weight[expert_idx]

    bias = None
    if bool(model._eval_expr(bias_expr, env, symbols)):
        bias_path = model._infer_param_path(
            path_spec,
            node_path=node_path,
            param_name=bias_param,
        )
        bias = model._state_tensor_from_resolved_path(bias_path, field="linear.bias")
        if bias is not None and expert_idx is not None and bias.ndim >= 2:
            bias = bias[expert_idx]

    transpose = bool(model._eval_expr(transpose_expr, env, symbols))
    out = model._require_name(node_spec.get("_bind"), field="linear._bind")
    if x.numel() == 0:
        out_dim = int(weight.shape[-1]) if transpose else int(weight.shape[0])
        env[out] = x.new_empty((*x.shape[:-1], out_dim))
        return
    weight_run = weight
    bias_run = bias
    if x.is_floating_point() and weight_run.is_floating_point() and x.dtype != weight_run.dtype:
        weight_run = weight_run.to(dtype=x.dtype)
        if bias_run is not None and bias_run.is_floating_point() and bias_run.dtype != x.dtype:
            bias_run = bias_run.to(dtype=x.dtype)

    align_linear_fp32 = bool(getattr(model, "_hf_align_linear_fp32_accum", False))
    if align_linear_fp32 and x.is_floating_point() and x.dtype in {torch.float16, torch.bfloat16}:
        if transpose:
            y_fp32 = torch.matmul(x.float(), weight_run.float())
            if bias_run is not None:
                y_fp32 = y_fp32 + bias_run.float()
            env[out] = y_fp32.to(dtype=x.dtype)
        else:
            env[out] = F.linear(
                x.float(),
                weight_run.float(),
                bias_run.float() if bias_run is not None else None,
            ).to(dtype=x.dtype)
    else:
        if transpose:
            y = torch.matmul(x, weight_run)
            env[out] = y + bias_run if bias_run is not None else y
        else:
            env[out] = F.linear(x, weight_run, bias_run)


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

    args = _raw_args(node_spec)
    if not args:
        raise ValueError(
            "linear requires positional args: x [dim bias transpose expert weight_path bias_path]"
        )
    src = read(str(args[0]))
    bias_expr = _arg_or_default(args, 2, False)
    transpose_expr = _arg_or_default(args, 3, False)
    expert_expr = _arg_or_default(args, 4, None)
    weight_override = _path_override(args, 5)
    bias_override = _path_override(args, 6)

    path_spec = dict(node_spec)
    weight_param = "weight_path" if _has_explicit_path(path_spec, "weight_path") else "weight"
    weight_param_expr: str | None = None
    weight_override_name: str | None = None
    if weight_override is not None:
        if (
            isinstance(weight_override, str)
            and weight_override.isidentifier()
            and weight_override in env
        ):
            weight_override_name = weight_override
        else:
            path_spec["weight_path"] = weight_override
            weight_param = "weight_path"
    bias_param = "bias_path" if _has_explicit_path(path_spec, "bias_path") else "bias"
    bias_param_expr: str | None = None
    bias_override_name: str | None = None
    if bias_override is not None:
        if isinstance(bias_override, str) and bias_override.isidentifier() and bias_override in env:
            bias_override_name = bias_override
        else:
            path_spec["bias_path"] = bias_override
            bias_param = "bias_path"

    out_name = str(node_spec.get("_bind"))
    out_var = assign_out_var(out_name)
    expert_code = emitter._expr_code(expert_expr, env) if expert_expr is not None else None
    has_bias_code = emitter._expr_code(bias_expr, env)
    transpose_code = emitter._expr_code(transpose_expr, env)
    selected_weight = (
        weight_param_expr
        if isinstance(weight_param_expr, str)
        else (
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
    )
    selected_bias = (
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
    weight_var = emitter._fresh("weight")
    bias_var = emitter._fresh("bias")
    resolved_weight_var = emitter._fresh("resolved_weight")
    resolved_bias_var = emitter._fresh("resolved_bias")
    weight_run_var = emitter._fresh("weight_run")
    bias_run_var = emitter._fresh("bias_run")
    out_dim_expr = (
        f"{weight_var}.shape[-1]"
        if str(transpose_expr).lower() in {"true", "1"}
        else f"{weight_var}.shape[0]"
    )

    lines.append(f"{indent}{resolved_weight_var} = {selected_weight}")
    lines.append(f"{indent}{resolved_bias_var} = {selected_bias}")
    if weight_override_name is not None:
        raw_weight_override = emitter._fresh("raw_weight_override")
        lines.append(f"{indent}{raw_weight_override} = {read(weight_override_name)}")
        lines.append(
            f"{indent}if not ("
            f"(isinstance({raw_weight_override}, str) and {raw_weight_override}.strip() in ('@weight', 'weight')) "
            f"or "
            f"(isinstance({raw_weight_override}, dict) and {raw_weight_override}.get('_expr') == 'path' "
            f"and not bool({raw_weight_override}.get('absolute')) and {raw_weight_override}.get('parts') == ['weight'])"
            f"):"
        )
        lines.append(
            f"{indent}    {resolved_weight_var} = emitter._param(self._resolve_state_path("
            f"node_path={node_path_var}, raw_path={raw_weight_override}))"
        )
    if bias_override_name is not None:
        raw_bias_override = emitter._fresh("raw_bias_override")
        lines.append(f"{indent}{raw_bias_override} = {read(bias_override_name)}")
        lines.append(
            f"{indent}if not ("
            f"(isinstance({raw_bias_override}, str) and {raw_bias_override}.strip() in ('@bias', 'bias')) "
            f"or "
            f"(isinstance({raw_bias_override}, dict) and {raw_bias_override}.get('_expr') == 'path' "
            f"and not bool({raw_bias_override}.get('absolute')) and {raw_bias_override}.get('parts') == ['bias'])"
            f"):"
        )
        lines.append(
            f"{indent}    {resolved_bias_var} = self._state_tensor_from_path("
            f"node_path={node_path_var}, raw_path={raw_bias_override}, field='linear.bias')"
        )
    lines.append(f"{indent}if {resolved_weight_var} is None:")
    lines.append(
        f"{indent}    raise ValueError('linear.weight tensor not found for resolved path')"
    )
    if expert_code is not None:
        expert_value_var = emitter._fresh("expert_value")
        expert_idx_var = emitter._fresh("expert_idx")
        lines.append(f"{indent}{expert_value_var} = {expert_code}")
        lines.append(f"{indent}if {expert_value_var} is None:")
        lines.append(f"{indent}    {weight_var} = {resolved_weight_var}")
        lines.append(f"{indent}    {bias_var} = {resolved_bias_var}")
        lines.append(f"{indent}else:")
        lines.append(f"{indent}    {expert_idx_var} = int({expert_value_var})")
        lines.append(f"{indent}    {weight_var} = ({resolved_weight_var})[{expert_idx_var}]")
        lines.append(
            f"{indent}    {bias_var} = (({resolved_bias_var})[{expert_idx_var}] "
            f"if ({resolved_bias_var}) is not None and ({resolved_bias_var}).ndim >= 2 "
            f"else ({resolved_bias_var}))"
        )
    else:
        lines.append(f"{indent}{weight_var} = {resolved_weight_var}")
        lines.append(f"{indent}{bias_var} = {resolved_bias_var}")
    lines.append(f"{indent}if not bool({has_bias_code}):")
    lines.append(f"{indent}    {bias_var} = None")
    lines.append(f"{indent}elif {bias_var} is None:")
    lines.append(f"{indent}    raise ValueError('linear.bias tensor not found for resolved path')")
    lines.append(f"{indent}{weight_run_var} = {weight_var}")
    lines.append(f"{indent}{bias_run_var} = {bias_var}")
    lines.append(
        f"{indent}if torch.is_tensor({src}) and torch.is_tensor({weight_run_var}) and {src}.is_floating_point() and {weight_run_var}.is_floating_point() and {src}.dtype != {weight_run_var}.dtype:"
    )
    lines.append(f"{indent}    {weight_run_var} = {weight_run_var}.to(dtype={src}.dtype)")
    lines.append(
        f"{indent}    if {bias_run_var} is not None and torch.is_tensor({bias_run_var}) and {bias_run_var}.is_floating_point() and {bias_run_var}.dtype != {src}.dtype:"
    )
    lines.append(f"{indent}        {bias_run_var} = {bias_run_var}.to(dtype={src}.dtype)")
    lines.append(f"{indent}if {src}.numel() == 0:")
    lines.append(
        f"{indent}    {out_var} = {src}.new_empty((*{src}.shape[:-1], int({out_dim_expr})))"
    )
    lines.append(f"{indent}else:")
    lines.append(
        f"{indent}    if getattr(self, '_hf_align_linear_fp32_accum', False) and {src}.is_floating_point() and {src}.dtype in {{torch.float16, torch.bfloat16}}:"
    )
    lines.append(f"{indent}        if bool({transpose_code}):")
    lines.append(
        f"{indent}            {out_var} = torch.matmul({src}.float(), {weight_run_var}.float())"
    )
    lines.append(f"{indent}            if {bias_run_var} is not None:")
    lines.append(f"{indent}                {out_var} = {out_var} + {bias_run_var}.float()")
    lines.append(f"{indent}            {out_var} = {out_var}.to(dtype={src}.dtype)")
    lines.append(f"{indent}        else:")
    lines.append(
        f"{indent}            {out_var} = F.linear({src}.float(), {weight_run_var}.float(), {bias_run_var}.float() if {bias_run_var} is not None else None).to(dtype={src}.dtype)"
    )
    lines.append(f"{indent}    else:")
    lines.append(f"{indent}        if bool({transpose_code}):")
    lines.append(f"{indent}            {out_var} = torch.matmul({src}, {weight_run_var})")
    lines.append(f"{indent}            if {bias_run_var} is not None:")
    lines.append(f"{indent}                {out_var} = {out_var} + {bias_run_var}")
    lines.append(f"{indent}        else:")
    lines.append(
        f"{indent}            {out_var} = F.linear({src}, {weight_run_var}, {bias_run_var})"
    )

    return lines


LOWERING_TYPE_SIGNATURE = {
    "args": ("Path", "Tensor[..S,Din]", "?Dim", "?Bool", "?Bool", "?Int", "?Path", "?Path"),
    "kwargs": dict(LOWERING_KWARG_KINDS),
    "returns": ("Tensor[..S,dim]",),
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
    if len(arg_types) < 2 or len(args) < 2:
        return None
    input_dims = helpers.type_dims(arg_types[1])
    if input_dims is None or len(input_dims) < 1:
        return None
    out_dim = helpers.expr_to_dim_token(args[2]) if len(args) >= 3 else None
    if out_dim is None:
        out_dim = input_dims[-1]
    return helpers.type_tensor(dims=(*input_dims[:-1], out_dim))


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
