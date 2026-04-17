from __future__ import annotations

from typing import Any

import torch

OP_NAME = "where_indices"
LOWERING_ARITY = (1, 1)
LOWERING_ALLOWED_KWARGS: set[str] = set()
LOWERING_REQUIRED_KWARGS: set[str] = set()
LOWERING_KWARG_KINDS: dict[str, Any] = {}


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return False


def lowering_known_output_arity(*, kwargs: dict[str, Any]) -> int:
    del kwargs
    return 2


def lowering_validate_signature(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> None:
    del args, kwargs, ctx
    if not isinstance(out, list) or len(out) != 2:
        raise ValueError("where_indices requires exactly two outputs")


def interpret(
    model: Any,
    node_spec: dict[str, Any],
    env: dict[str, Any],
    *,
    node_path: str,
    scope: str,
    symbols: dict[str, int | float | bool],
) -> None:
    del node_path, scope
    raw_args = node_spec.get("_args")
    args = (
        list(raw_args)
        if isinstance(raw_args, list)
        else ([raw_args] if raw_args is not None else [])
    )
    if len(args) != 1:
        raise ValueError("where_indices expects one positional argument")
    out = node_spec.get("_bind")
    if not isinstance(out, list) or len(out) != 2:
        raise ValueError("where_indices requires exactly two outputs")
    mask = model._read_tensor_input(args[0], env)
    if not torch.is_tensor(mask) or mask.ndim != 2:
        raise ValueError("where_indices expects a rank-2 tensor mask")
    idx0, idx1 = torch.where(mask)
    env[str(out[0])] = idx0
    env[str(out[1])] = idx1


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
    raw_args = node_spec.get("_args")
    args = (
        list(raw_args)
        if isinstance(raw_args, list)
        else ([raw_args] if raw_args is not None else [])
    )
    if len(args) != 1:
        raise ValueError("where_indices expects one positional argument")
    out = node_spec.get("_bind")
    if not isinstance(out, list) or len(out) != 2:
        raise ValueError("where_indices requires exactly two outputs")
    mask = emitter._read_env_var(env, str(args[0]))
    out0 = emitter._assign_out_var(env, str(out[0]))
    out1 = emitter._assign_out_var(env, str(out[1]))
    return [f"{indent}{out0}, {out1} = torch.where({mask})"]


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any",),
    "kwargs": {},
    "returns": ("IdxTensor", "IdxTensor"),
}

__all__ = [
    "OP_NAME",
    "LOWERING_ARITY",
    "LOWERING_ALLOWED_KWARGS",
    "LOWERING_REQUIRED_KWARGS",
    "LOWERING_KWARG_KINDS",
    "lowering_known_output_arity",
    "lowering_validate_signature",
    "interpret",
    "compile",
    "uses_node_path",
    "LOWERING_TYPE_SIGNATURE",
]
