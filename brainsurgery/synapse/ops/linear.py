from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

OP_NAME = "linear"
LOWERING_ARITY = (1, 1)
LOWERING_ALLOWED_KWARGS: set[str] = {
    "dim",
    "transpose",
    "bias",
    "expert",
    "weight",
    "bias_path",
}
LOWERING_REQUIRED_KWARGS: set[str] = set()
LOWERING_KWARG_KINDS: dict[str, Any] = {
    "dim": "dim",
    "bias": "bool",
    "transpose": "bool",
    "expert": "dim",
    "weight": "str",
    "bias_path": "str",
}


def _validate_linear_keys(node_spec: dict[str, Any]) -> None:
    if "weight_layout" in node_spec:
        raise ValueError("linear does not support weight_layout; use transpose=true/false")
    if "tie_weight" in node_spec:
        raise ValueError("linear does not support tie_weight; use linear@<path> or weight=<path>")
    if "share" in node_spec:
        raise ValueError("linear does not support share; use linear@<path> or weight=<path>")


def _resolve_transpose(node_spec: dict[str, Any]) -> bool:
    _validate_linear_keys(node_spec)
    transpose = node_spec.get("transpose", False)
    if isinstance(transpose, bool):
        return transpose
    raise ValueError("linear transpose must be boolean when provided")


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter
    has_bias = bool(node_spec["bias"]) if "bias" in node_spec else False
    explicit_weight = node_spec.get("weight")
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
    del args
    if "weight_layout" in kwargs:
        raise ValueError("linear does not support weight_layout; use transpose=true/false")
    if "tie_weight" in kwargs:
        raise ValueError("linear does not support tie_weight; use linear@<path>")
    if "out_features" in kwargs:
        raise ValueError("linear does not support out_features; use dim")
    if "out_dim" in kwargs:
        raise ValueError("linear does not support out_dim; use dim")
    if "dim" not in kwargs and isinstance(out, str):
        inferred = ctx.tensor_last_dim.get(out)
        if inferred is not None:
            kwargs["dim"] = inferred
    if "transpose" not in kwargs:
        return
    raw_transpose = kwargs["transpose"]
    if isinstance(raw_transpose, bool):
        return
    if isinstance(raw_transpose, str) and raw_transpose.lower() in {"true", "false"}:
        kwargs["transpose"] = raw_transpose.lower() == "true"
        return
    raise ValueError("linear transpose must be true/false")


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
    last_dim = kwargs.get("dim", first_dim)
    if last_dim is not None:
        ctx.tensor_last_dim[out] = last_dim
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
    x = model._read_tensor_input(node_spec.get("_args"), env)
    linear_weight_path = model._infer_param_path(
        node_spec, node_path=node_path, param_name="weight"
    )
    weight = model._state[linear_weight_path]
    expert_expr = node_spec.get("expert")
    if expert_expr is not None:
        expert_idx = int(model._eval_expr(expert_expr, env, symbols))
        if weight.ndim < 2:
            raise ValueError("linear expert selection requires at least rank-2 weight tensor")
        if expert_idx < 0 or expert_idx >= int(weight.shape[0]):
            raise ValueError(
                f"linear expert index out of range: {expert_idx} for shape {tuple(weight.shape)}"
            )
        weight = weight[expert_idx]

    bias = None
    if node_spec.get("bias", False):
        bias_path = model._infer_param_path(
            node_spec,
            node_path=node_path,
            param_name=("bias_path" if isinstance(node_spec.get("bias_path"), str) else "bias"),
        )
        bias = model._state.get(bias_path)
        if bias is not None and expert_expr is not None and bias.ndim >= 2:
            bias = bias[expert_idx]

    transpose = _resolve_transpose(node_spec)
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

    def infer_param(param_name: str) -> str:
        return emitter._infer_param_expr(node_spec, node_path_var, param_name)

    def read(name: str) -> str:
        return emitter._read_env_var(env, name)

    src = read(str(node_spec.get("_args")))
    out_name = str(node_spec.get("_bind"))
    out_var = assign_out_var(out_name)
    weight_expr = infer_param("weight")
    has_bias = bool(node_spec["bias"]) if "bias" in node_spec else False
    bias_expr = "None"
    if has_bias:
        bias_param = "bias_path" if isinstance(node_spec.get("bias_path"), str) else "bias"
        bias_expr = f"self._state.get({infer_param(bias_param)})"
    expert_expr = node_spec.get("expert")
    expert_code = emitter._expr_code(expert_expr, env) if expert_expr is not None else None
    transpose = _resolve_transpose(node_spec)
    selected_weight = f"self._param({weight_expr})"
    selected_bias = bias_expr
    if expert_code is not None:
        selected_weight = f"{selected_weight}[int({expert_code})]"
        if has_bias:
            selected_bias = (
                f"(({bias_expr})[int({expert_code})] "
                f"if ({bias_expr}) is not None and ({bias_expr}).ndim >= 2 "
                f"else ({bias_expr}))"
            )
    out_dim_expr = f"{selected_weight}.shape[-1]" if transpose else f"{selected_weight}.shape[0]"
    weight_var = emitter._fresh("weight")
    bias_var = emitter._fresh("bias")
    weight_run_var = emitter._fresh("weight_run")
    bias_run_var = emitter._fresh("bias_run")

    lines.append(f"{indent}{weight_var} = {selected_weight}")
    lines.append(f"{indent}{bias_var} = {selected_bias}")
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
    if transpose:
        lines.append(
            f"{indent}        {out_var} = torch.matmul({src}.float(), {weight_run_var}.float())"
        )
        lines.append(f"{indent}        if {bias_run_var} is not None:")
        lines.append(f"{indent}            {out_var} = {out_var} + {bias_run_var}.float()")
        lines.append(f"{indent}        {out_var} = {out_var}.to(dtype={src}.dtype)")
    else:
        lines.append(
            f"{indent}        {out_var} = F.linear({src}.float(), {weight_run_var}.float(), {bias_run_var}.float() if {bias_run_var} is not None else None).to(dtype={src}.dtype)"
        )
    lines.append(f"{indent}    else:")
    if transpose:
        lines.append(f"{indent}        {out_var} = torch.matmul({src}, {weight_run_var})")
        lines.append(f"{indent}        if {bias_run_var} is not None:")
        lines.append(f"{indent}            {out_var} = {out_var} + {bias_run_var}")
    else:
        lines.append(
            f"{indent}        {out_var} = F.linear({src}, {weight_run_var}, {bias_run_var})"
        )

    return lines


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any",),
    "kwargs": dict(LOWERING_KWARG_KINDS),
    "returns": ("Any",),
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
