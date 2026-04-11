from __future__ import annotations

from typing import Any

import torch

OP_NAME = "gather"
LOWERING_ARITY = (2, 3)
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


def _name_expr(value: str) -> dict[str, Any]:
    return {"_expr": "name", "id": value}


def _expr_payload(value: Any) -> Any:
    if isinstance(value, str):
        token = value.strip()
        if token.isidentifier():
            return _name_expr(token)
    return value


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return False


def lowering_validate_signature(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> None:
    del ctx
    if not isinstance(out, str):
        raise ValueError("gather requires a single scalar output binding")
    if kwargs:
        unknown = ", ".join(sorted(str(key) for key in kwargs))
        raise ValueError(f"gather unsupported kwargs: {unknown}")
    if len(args) < 2 or len(args) > 3:
        raise ValueError(f"gather expects 2..3 positional args, got {len(args)}")


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
    args = _raw_args(node_spec)
    if len(args) < 2:
        raise ValueError("gather requires positional args: x index [dim]")
    x = model._read_tensor_input(args[0], env)
    index = model._read_tensor_input(args[1], env)
    if not torch.is_tensor(index):
        raise ValueError("gather index must be a tensor")
    dim = int(model._eval_expr(_arg_or_default(args, 2, -1), env, symbols))
    out = model._require_name(node_spec.get("_bind"), field="gather._bind")
    env[out] = torch.gather(x, dim=dim, index=index)


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
    if len(args) < 2:
        raise ValueError("gather requires positional args: x index [dim]")
    src = emitter._read_env_var(env, str(args[0]))
    index = emitter._read_env_var(env, str(args[1]))
    out_var = emitter._assign_out_var(env, str(node_spec.get("_bind")))
    dim = emitter._expr_code(_expr_payload(_arg_or_default(args, 2, -1)), env)
    return [f"{indent}{out_var} = torch.gather({src}, dim=int({dim}), index={index})"]


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any", "Any", "Any"),
    "kwargs": {},
    "returns": ("Tensor",),
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
