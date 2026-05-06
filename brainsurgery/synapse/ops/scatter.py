from __future__ import annotations

from typing import Any

import torch

OP_NAME = "scatter"
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
        raise ValueError("scatter requires a single scalar output binding")
    if kwargs:
        unknown = ", ".join(sorted(str(key) for key in kwargs))
        raise ValueError(f"scatter unsupported kwargs: {unknown}")
    if len(args) < 3 or len(args) > 4:
        raise ValueError(f"scatter expects 3..4 positional args, got {len(args)}")


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
        raise ValueError("scatter requires positional args: x index src [dim]")
    x = model._read_tensor_input(args[0], env)
    index = model._read_tensor_input(args[1], env)
    src_ref = args[2]
    src = (
        env[src_ref]
        if isinstance(src_ref, str) and src_ref in env
        else model._eval_expr(src_ref, env, symbols)
    )
    if not torch.is_tensor(index):
        raise ValueError("scatter index must be a tensor")
    dim = int(model._eval_expr(_arg_or_default(args, 3, -1), env, symbols))
    out = model._require_name(node_spec.get("_bind"), field="scatter._bind")
    if torch.is_tensor(src):
        env[out] = torch.scatter(x, dim=dim, index=index, src=src)
    else:
        env[out] = torch.scatter(x, dim=dim, index=index, value=src)


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
        raise ValueError("scatter requires positional args: x index src [dim]")
    x = emitter._read_env_var(env, str(args[0]))
    index = emitter._read_env_var(env, str(args[1]))
    src_ref = args[2]
    src = (
        emitter._read_env_var(env, str(src_ref))
        if isinstance(src_ref, str) and str(src_ref) in env
        else emitter._expr_code(_expr_payload(src_ref), env)
    )
    dim = emitter._expr_code(_expr_payload(_arg_or_default(args, 3, -1)), env)
    out_var = emitter._assign_out_var(env, str(node_spec.get("_bind")))
    src_is_tensor = emitter._fresh("src_is_tensor")
    lines = [f"{indent}{src_is_tensor} = torch.is_tensor({src})"]
    lines.append(f"{indent}if {src_is_tensor}:")
    lines.append(
        f"{indent}    {out_var} = torch.scatter({x}, dim=int({dim}), index={index}, src={src})"
    )
    lines.append(f"{indent}else:")
    lines.append(
        f"{indent}    {out_var} = torch.scatter({x}, dim=int({dim}), index={index}, value={src})"
    )
    return lines


LOWERING_TYPE_SIGNATURE = {
    "args": ("Tensor[..S]", "IdxTensor[..I]", "Any", "Any"),
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
