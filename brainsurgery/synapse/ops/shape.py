from __future__ import annotations

from typing import Any

import torch

OP_NAME = "shape"
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
        raise ValueError("shape requires a single output binding")


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
    raw_args = node_spec.get("_args")
    args = raw_args if isinstance(raw_args, list) else [raw_args]
    if len(args) != 1:
        raise ValueError("shape expects exactly one positional arg")
    x = model._read_tensor_input(args[0], env)
    if not isinstance(x, torch.Tensor):
        raise ValueError("shape input must be a tensor")
    out_name = model._require_name(node_spec.get("_bind"), field="shape._bind")
    env[out_name] = [int(v) for v in x.shape]


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

    def assign_out_var(out_name: str) -> str:
        return emitter._assign_out_var(env, out_name)

    def read(name: str) -> str:
        return emitter._read_env_var(env, name)

    raw_args = node_spec.get("_args")
    args = raw_args if isinstance(raw_args, list) else [raw_args]
    if len(args) != 1:
        raise ValueError("shape expects exactly one positional arg")
    src = read(str(args[0]))
    out_name = str(node_spec.get("_bind"))
    out_var = assign_out_var(out_name)
    lines.append(f"{indent}if not isinstance({src}, torch.Tensor):")
    lines.append(f"{indent}    raise ValueError('shape input must be a tensor')")
    lines.append(f"{indent}{out_var} = [int(v) for v in {src}.shape]")
    return lines


LOWERING_TYPE_SIGNATURE = {
    "args": ("Tensor",),
    "kwargs": dict(LOWERING_KWARG_KINDS),
    "returns": ("List[Int]",),
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
