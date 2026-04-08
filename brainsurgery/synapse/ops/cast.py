from __future__ import annotations

from typing import Any

import torch

OP_NAME = "cast"
LOWERING_ARITY = (1, 1)
LOWERING_ALLOWED_KWARGS: set[str] = {"dtype"}
LOWERING_REQUIRED_KWARGS: set[str] = {"dtype"}
LOWERING_KWARG_KINDS: dict[str, Any] = {"dtype": "str"}


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return False


def lowering_validate_signature(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> None:
    del args, ctx
    if not isinstance(out, str):
        raise ValueError("cast requires a single scalar output binding")
    if "dtype" not in kwargs:
        raise ValueError("cast requires dtype")


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


def _resolve_dtype(raw: Any) -> torch.dtype:
    if not isinstance(raw, str):
        raise ValueError("cast.dtype must be a string")
    value = raw.strip().lower()
    if value in {"long", "int64"}:
        return torch.long
    if value in {"bool"}:
        return torch.bool
    if value in {"float", "float32", "fp32"}:
        return torch.float32
    raise ValueError("cast.dtype must be one of: long, int64, bool, float32")


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
    src = model._read_tensor_input(node_spec.get("_args"), env)
    dtype = _resolve_dtype(node_spec.get("dtype"))
    out = model._require_name(node_spec.get("_bind"), field="cast._bind")
    env[out] = src.to(dtype=dtype)


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
    dtype = _resolve_dtype(node_spec.get("dtype"))
    return [
        f"{indent}{out_var} = {src}.to(dtype={dtype!r})".replace(
            "'" + str(dtype) + "'", f"torch.{str(dtype).split('.')[-1]}"
        )
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
    "lowering_infer_metadata",
    "interpret",
    "compile",
    "uses_node_path",
    "LOWERING_TYPE_SIGNATURE",
]
