from __future__ import annotations

from typing import Any

import torch

OP_NAME = "index_add"
LOWERING_ARITY = (3, 4)
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
    return False


def lowering_validate_signature(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> None:
    del kwargs, ctx
    if not isinstance(out, str):
        raise ValueError("index_add requires a single output")
    if len(args) < 3 or len(args) > 4:
        raise ValueError(f"index_add expects 3..4 positional args, got {len(args)}")


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
    args = _raw_args(node_spec)
    if len(args) < 3:
        raise ValueError("index_add requires positional args: x index src [dim]")
    x = model._read_tensor_input(args[0], env)
    index = model._read_tensor_input(args[1], env)
    src = model._read_tensor_input(args[2], env)
    dim = int(model._eval_expr(_arg_or_default(args, 3, 0), env, symbols))
    out = model._require_name(node_spec.get("_bind"), field="index_add._bind")
    env[out] = torch.index_add(x, dim=dim, index=index, source=src)


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
    if len(args) < 3:
        raise ValueError("index_add requires positional args: x index src [dim]")
    x = emitter._read_env_var(env, str(args[0]))
    index = emitter._read_env_var(env, str(args[1]))
    src = emitter._read_env_var(env, str(args[2]))
    dim = emitter._expr_code(_arg_or_default(args, 3, 0), env)
    out = emitter._assign_out_var(env, str(node_spec.get("_bind")))
    return [f"{indent}{out} = torch.index_add({x}, dim=int({dim}), index={index}, source={src})"]


LOWERING_TYPE_SIGNATURE = {
    "args": ("Tensor[..S]", "IdxTensor[..I]", "Tensor[..T]", "Any"),
    "kwargs": {},
    "returns": ("Tensor[..S]",),
}

__all__ = [
    "OP_NAME",
    "LOWERING_ARITY",
    "LOWERING_ALLOWED_KWARGS",
    "LOWERING_REQUIRED_KWARGS",
    "LOWERING_KWARG_KINDS",
    "lowering_validate_signature",
    "interpret",
    "compile",
    "uses_node_path",
    "LOWERING_TYPE_SIGNATURE",
]
