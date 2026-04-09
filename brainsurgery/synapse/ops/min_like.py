from __future__ import annotations

from typing import Any

import torch

OP_NAME = "min_like"
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
        raise ValueError("min_like requires a single scalar output binding")


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
    x = model._read_tensor_input(node_spec.get("_args"), env)
    if not torch.is_floating_point(x):
        raise ValueError("min_like expects floating-point tensor input")
    out = model._require_name(node_spec.get("_bind"), field="min_like._bind")
    env[out] = torch.full_like(x, torch.finfo(x.dtype).min)


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
    return [
        f"{indent}if not torch.is_floating_point({src}):",
        f"{indent}    raise ValueError('min_like expects floating-point tensor input')",
        f"{indent}{out_var} = torch.full_like({src}, torch.finfo({src}.dtype).min)",
    ]


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
