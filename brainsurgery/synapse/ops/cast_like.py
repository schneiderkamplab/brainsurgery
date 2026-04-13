from __future__ import annotations

from typing import Any

import torch

OP_NAME = "cast_like"
LOWERING_ARITY = (2, 2)
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


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return False


def lowering_validate_signature(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> None:
    del ctx
    if not isinstance(out, str):
        raise ValueError("cast_like requires a single scalar output binding")
    if kwargs:
        unknown = ", ".join(sorted(str(key) for key in kwargs))
        raise ValueError(f"cast_like unsupported kwargs: {unknown}")
    if len(args) != 2:
        raise ValueError("cast_like requires positional args: x ref")


def lowering_infer_metadata(
    *,
    args: list[str],
    out: str | list[str],
    kwargs: dict[str, Any],
    ctx: Any,
) -> bool:
    del kwargs
    if not isinstance(out, str) or not args:
        return False
    source_name = str(args[0]).strip()
    source_shape = ctx.tensor_shape.get(source_name)
    if isinstance(source_shape, tuple):
        ctx.tensor_shape[out] = source_shape
        ctx.tensor_last_dim[out] = source_shape[-1]
        return True
    if source_name in ctx.tensor_last_dim:
        ctx.tensor_last_dim[out] = ctx.tensor_last_dim[source_name]
        return True
    return False


def interpret(
    model: Any,
    node_spec: dict[str, Any],
    env: dict[str, Any],
    *,
    node_path: str,
    scope: str,
    symbols: dict[str, int],
) -> None:
    del node_path, scope, symbols
    args = _raw_args(node_spec)
    if len(args) != 2:
        raise ValueError("cast_like requires positional args: x ref")
    src = model._read_tensor_input(args[0], env)
    ref = model._read_tensor_input(args[1], env)
    if not torch.is_tensor(ref):
        raise ValueError("cast_like second argument must resolve to tensor")
    out = model._require_name(node_spec.get("_bind"), field="cast_like._bind")
    env[out] = src.to(dtype=ref.dtype)


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
    args = _raw_args(node_spec)
    if len(args) != 2:
        raise ValueError("cast_like requires positional args: x ref")
    src = emitter._read_env_var(env, str(args[0]))
    ref = emitter._read_env_var(env, str(args[1]))
    out_var = emitter._assign_out_var(env, str(node_spec.get("_bind")))
    return [
        f"{indent}{out_var} = {src}.to(dtype={ref}.dtype)",
    ]


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any", "Any"),
    "kwargs": {},
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
