from __future__ import annotations

from typing import Any

import torch

OP_NAME = "clamp"
LOWERING_ARITY = (1, 1)
LOWERING_ALLOWED_KWARGS: set[str] = {"min", "max"}
LOWERING_REQUIRED_KWARGS: set[str] = set()
LOWERING_KWARG_KINDS: dict[str, Any] = {"min": "number", "max": "number"}


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return False


def lowering_validate_signature(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> None:
    del args, ctx
    if not isinstance(out, str):
        raise ValueError("clamp requires a single output binding")
    if "min" not in kwargs and "max" not in kwargs:
        raise ValueError("clamp requires at least one of min/max")


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
    if isinstance(first_in, str) and first_in.isidentifier():
        first_dim = ctx.tensor_last_dim.get(first_in)
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
    del node_path, scope
    x = model._read_tensor_input(node_spec.get("_args"), env)
    out = model._require_name(node_spec.get("_bind"), field="clamp._bind")
    has_min = "min" in node_spec
    has_max = "max" in node_spec
    if not has_min and not has_max:
        raise ValueError("clamp requires at least one of min/max")
    min_value = float(model._eval_expr(node_spec.get("min"), env, symbols)) if has_min else None
    max_value = float(model._eval_expr(node_spec.get("max"), env, symbols)) if has_max else None
    env[out] = torch.clamp(x, min=min_value, max=max_value)


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
    has_min = "min" in node_spec
    has_max = "max" in node_spec
    if not has_min and not has_max:
        raise ValueError("clamp requires at least one of min/max")
    min_code = emitter._expr_code(node_spec.get("min"), env) if has_min else "None"
    max_code = emitter._expr_code(node_spec.get("max"), env) if has_max else "None"
    lines.append(
        f"{indent}{out_var} = torch.clamp({src}, min=({min_code} if {str(has_min)} else None), max=({max_code} if {str(has_max)} else None))"
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
    "lowering_validate_signature",
    "lowering_infer_metadata",
    "interpret",
    "compile",
    "uses_node_path",
    "LOWERING_TYPE_SIGNATURE",
]
