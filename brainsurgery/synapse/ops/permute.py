from __future__ import annotations

from typing import Any

import torch

OP_NAME = "permute"
LOWERING_ARITY = (1, 1)
LOWERING_ALLOWED_KWARGS: set[str] = {"dims"}
LOWERING_REQUIRED_KWARGS: set[str] = {"dims"}
LOWERING_KWARG_KINDS: dict[str, Any] = {"dims": "list_dim"}


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return False


def lowering_validate_signature(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> None:
    del args, ctx
    if not isinstance(out, str):
        raise ValueError("permute requires a single scalar output binding")
    if "dims" not in kwargs:
        raise ValueError("permute requires dims")


def lowering_infer_metadata(
    *,
    args: list[str],
    out: str | list[str],
    kwargs: dict[str, Any],
    ctx: Any,
) -> bool:
    if not isinstance(out, str) or not args:
        return False
    dims = kwargs.get("dims")
    if not isinstance(dims, list) or not all(
        isinstance(v, int) and not isinstance(v, bool) for v in dims
    ):
        return False
    source_name = str(args[0]).strip()
    source_shape = ctx.tensor_shape.get(source_name)
    if not isinstance(source_shape, tuple) or len(source_shape) != len(dims):
        return False
    rank = len(source_shape)
    normalized = [dim if dim >= 0 else rank + dim for dim in dims]
    if sorted(normalized) != list(range(rank)):
        return False
    new_shape = tuple(source_shape[idx] for idx in normalized)
    ctx.tensor_shape[out] = new_shape
    ctx.tensor_last_dim[out] = new_shape[-1]
    return True


def _resolve_dims(
    model: Any,
    raw_dims: Any,
    env: dict[str, Any],
    symbols: dict[str, int],
) -> tuple[int, ...]:
    if isinstance(raw_dims, list):
        raw_items = raw_dims
    else:
        evaluated = model._eval_expr(raw_dims, env, symbols)
        if not isinstance(evaluated, list | tuple) or not evaluated:
            raise ValueError("permute.dims must be a non-empty list")
        raw_items = list(evaluated)
    resolved: list[int] = []
    for item in raw_items:
        value = model._eval_expr(item, env, symbols)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("permute.dims entries must resolve to ints")
        resolved.append(int(value))
    return tuple(resolved)


def _validate_permutation(dims: tuple[int, ...], rank: int) -> tuple[int, ...]:
    normalized = tuple(dim if dim >= 0 else rank + dim for dim in dims)
    if any(dim < 0 or dim >= rank for dim in normalized):
        raise ValueError("permute.dims entries out of range for tensor rank")
    if len(set(normalized)) != rank or sorted(normalized) != list(range(rank)):
        raise ValueError("permute.dims must be a permutation of tensor dimensions")
    return normalized


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
    dims = _resolve_dims(model, node_spec.get("dims"), env, symbols)
    normalized = _validate_permutation(dims, src.dim())
    out = model._require_name(node_spec.get("_bind"), field="permute._bind")
    env[out] = torch.permute(src, normalized)


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
    src = emitter._read_env_var(env, str(node_spec.get("_args")))
    out_var = emitter._assign_out_var(env, str(node_spec.get("_bind")))
    raw_dims = node_spec.get("dims")
    if isinstance(raw_dims, list) and raw_dims:
        dims_expr = f"({', '.join(f'int({emitter._expr_code(item, env)})' for item in raw_dims)},)"
    else:
        dims_expr = emitter._expr_code(raw_dims, env)
    rank_var = emitter._fresh("rank")
    dims_var = emitter._fresh("dims")
    norm_var = emitter._fresh("normalized_dims")
    lines.append(f"{indent}{rank_var} = {src}.dim()")
    lines.append(f"{indent}{dims_var} = tuple(int(v) for v in {dims_expr})")
    lines.append(f"{indent}if len({dims_var}) == 0:")
    lines.append(f"{indent}    raise ValueError('permute.dims must be a non-empty list')")
    lines.append(
        f"{indent}{norm_var} = tuple(dim if dim >= 0 else {rank_var} + dim for dim in {dims_var})"
    )
    lines.append(f"{indent}if any(dim < 0 or dim >= {rank_var} for dim in {norm_var}):")
    lines.append(
        f"{indent}    raise ValueError('permute.dims entries out of range for tensor rank')"
    )
    lines.append(
        f"{indent}if len(set({norm_var})) != {rank_var} or sorted({norm_var}) != list(range({rank_var})):"
    )
    lines.append(
        f"{indent}    raise ValueError('permute.dims must be a permutation of tensor dimensions')"
    )
    lines.append(f"{indent}{out_var} = torch.permute({src}, {norm_var})")
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
    "lowering_validate_signature",
    "lowering_infer_metadata",
    "interpret",
    "compile",
    "uses_node_path",
    "LOWERING_TYPE_SIGNATURE",
]
