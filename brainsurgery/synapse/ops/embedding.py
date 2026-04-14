from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

OP_NAME = "embedding"
LOWERING_ARITY = (1, 2)
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


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return True


def lowering_infer_metadata(
    *,
    args: list[str],
    out: str | list[str],
    kwargs: dict[str, Any],
    ctx: Any,
) -> bool:
    if not isinstance(out, str):
        return False
    last_dim: Any = None
    if len(args) >= 2:
        last_dim = args[1]
    else:
        inferred = ctx.tensor_last_dim.get(out)
        if inferred is not None:
            last_dim = inferred
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
    args = _raw_args(node_spec)
    if not args:
        raise ValueError("embedding requires positional args: x [dim]")
    x = model._read_tensor_input(args[0], env)
    weight_path = model._infer_param_path(node_spec, node_path=node_path, param_name="weight")
    weight = model._state_tensor_from_resolved_path(weight_path, field="embedding.weight")
    if torch.is_tensor(weight) and torch.is_tensor(x) and weight.device != x.device:
        weight = weight.to(device=x.device)
    out = model._require_name(node_spec.get("_bind"), field="embedding._bind")
    env[out] = F.embedding(x, weight)
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
        raise ValueError("embedding requires positional args: x [dim]")
    src = read(str(args[0]))
    out_name = str(node_spec.get("_bind"))
    out_var = assign_out_var(out_name)
    weight = emitter._hoisted_param(
        node_spec=node_spec,
        node_path_var=node_path_var,
        param_name="weight",
        lines=lines,
        indent=indent,
    )
    lines.append(f"{indent}{out_var} = F.embedding({src}, {weight})")
    return lines


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any",),
    "kwargs": dict(LOWERING_KWARG_KINDS),
    "returns": ("Tensor",),
}

__all__ = [
    "LOWERING_ARITY",
    "LOWERING_ALLOWED_KWARGS",
    "LOWERING_REQUIRED_KWARGS",
    "LOWERING_KWARG_KINDS",
    "OP_NAME",
    "lowering_infer_metadata",
    "interpret",
    "compile",
    "uses_node_path",
    "LOWERING_TYPE_SIGNATURE",
]
