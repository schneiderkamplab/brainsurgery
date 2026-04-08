from __future__ import annotations

import math
from typing import Any

import torch

OP_NAME = "log"
LOWERING_ARITY = (1, 1)
LOWERING_ALLOWED_KWARGS: set[str] = set()
LOWERING_REQUIRED_KWARGS: set[str] = set()
LOWERING_KWARG_KINDS: dict[str, Any] = {}


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return False


def lowering_validate_signature(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> None:
    del args, kwargs, ctx
    if not isinstance(out, str):
        raise ValueError("log requires a single scalar output binding")


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
    src_value = model._eval_expr(node_spec.get("_args"), env, symbols)
    out = model._require_name(node_spec.get("_bind"), field="log._bind")
    if torch.is_tensor(src_value):
        tensor = src_value
        if not tensor.is_floating_point():
            tensor = tensor.to(dtype=torch.float32)
        env[out] = torch.log(tensor)
        return
    env[out] = math.log(float(src_value))


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
    src = emitter._expr_code(node_spec.get("_args"), env)
    out_var = emitter._assign_out_var(env, str(node_spec.get("_bind")))
    src_val = emitter._fresh("log_src")
    tensor_val = emitter._fresh("log_tensor")
    lines.append(f"{indent}{src_val} = {src}")
    lines.append(f"{indent}if torch.is_tensor({src_val}):")
    lines.append(f"{indent}    {tensor_val} = {src_val}")
    lines.append(f"{indent}    if not {tensor_val}.is_floating_point():")
    lines.append(f"{indent}        {tensor_val} = {tensor_val}.to(dtype=torch.float32)")
    lines.append(f"{indent}    {out_var} = torch.log({tensor_val})")
    lines.append(f"{indent}else:")
    lines.append(f"{indent}    {out_var} = math.log(float({src_val}))")
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
    "interpret",
    "compile",
    "uses_node_path",
    "LOWERING_TYPE_SIGNATURE",
]
