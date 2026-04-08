from __future__ import annotations

from typing import Any

import torch

OP_NAME = "arange"
LOWERING_ARITY = (1, 1)
LOWERING_ALLOWED_KWARGS: set[str] = {"start", "end"}
LOWERING_REQUIRED_KWARGS: set[str] = {"start", "end"}
LOWERING_KWARG_KINDS: dict[str, Any] = {
    "start": "dim",
    "end": "dim",
}


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return False


def lowering_validate_signature(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> None:
    del args, ctx
    if not isinstance(out, str):
        raise ValueError("arange requires a single scalar output binding")
    if "start" not in kwargs or "end" not in kwargs:
        raise ValueError("arange requires start and end")


def lowering_infer_metadata(
    *,
    args: list[str],
    out: str | list[str],
    kwargs: dict[str, Any],
    ctx: Any,
) -> bool:
    del args, kwargs, ctx
    return isinstance(out, str)


def _resolve_bound(
    model: Any,
    raw: Any,
    env: dict[str, Any],
    symbols: dict[str, int],
    *,
    field: str,
) -> int:
    value = model._eval_expr(raw, env, symbols)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"arange.{field} must resolve to int")
    return int(value)


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
    src = model._read_tensor_input(node_spec.get("_args"), env)
    start = _resolve_bound(model, node_spec.get("start"), env, symbols, field="start")
    end = _resolve_bound(model, node_spec.get("end"), env, symbols, field="end")
    out = model._require_name(node_spec.get("_bind"), field="arange._bind")
    env[out] = torch.arange(start, end, device=src.device, dtype=torch.long)


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
    src = emitter._read_env_var(env, str(node_spec.get("_args")))
    out_var = emitter._assign_out_var(env, str(node_spec.get("_bind")))
    start_expr = emitter._expr_code(node_spec.get("start"), env)
    end_expr = emitter._expr_code(node_spec.get("end"), env)
    start_var = emitter._fresh("arange_start")
    end_var = emitter._fresh("arange_end")
    return [
        f"{indent}{start_var} = int({start_expr})",
        f"{indent}{end_var} = int({end_expr})",
        f"{indent}{out_var} = torch.arange({start_var}, {end_var}, device={src}.device, dtype=torch.long)",
    ]


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any",),
    "kwargs": dict(LOWERING_KWARG_KINDS),
    "returns": ("IdxTensor",),
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
