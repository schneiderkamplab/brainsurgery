from __future__ import annotations

from typing import Any

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
        bias_expr = f"self._param({infer_param(bias_param)})"
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
                f"if ({bias_expr}) is not None and len(({bias_expr}).shape) >= 2 "
                f"else ({bias_expr}))"
            )
    weight_var = emitter._fresh("weight")

    lines.append(f"{indent}{weight_var} = {selected_weight}")
    if expert_code is not None and has_bias:
        bias_var = emitter._fresh("bias")
        lines.append(f"{indent}{bias_var} = {selected_bias}")

    if transpose:
        if has_bias:
            if expert_code is not None:
                lines.append(f"{indent}{out_var} = {src}.matmul({weight_var}) + {bias_var}")
            else:
                lines.append(
                    f"{indent}{out_var} = {src}.matmul({weight_var}) "
                    f"if {bias_expr} is None else {src}.matmul({weight_var}) + {bias_expr}"
                )
        else:
            lines.append(f"{indent}{out_var} = {src}.matmul({weight_var})")
    else:
        bias_arg = bias_var if (expert_code is not None and has_bias) else bias_expr
        if has_bias:
            lines.append(
                f"{indent}{out_var} = {src}.linear({weight_var}.transpose(), {bias_arg})"
            )
        else:
            lines.append(f"{indent}{out_var} = {src}.linear({weight_var}.transpose(), None)")

    return lines


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
]
