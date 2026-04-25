from __future__ import annotations

from typing import Any

import torch

from ..axon.ast import AxonExprAscribe, AxonExprInt, AxonExprParen, DimExprBinary

OP_NAME = "repeat"
LOWERING_ARITY = (3, 3)
LOWERING_ALLOWED_KWARGS: set[str] = set()
LOWERING_REQUIRED_KWARGS: set[str] = set()
LOWERING_KWARG_KINDS: dict[str, Any] = {}


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return False


def lowering_normalize_kwargs(
    *,
    args: list[str],
    out: str | list[str],
    kwargs: dict[str, Any],
    ctx: Any,
) -> None:
    del kwargs, out, ctx, args


def lowering_infer_metadata(
    *,
    args: list[str],
    out: str | list[str],
    kwargs: dict[str, Any],
    ctx: Any,
) -> bool:
    del kwargs
    if not isinstance(out, str):
        return False
    src_name: str | None = None
    # Preserve known heads metadata through repeat on head axis.
    if args:
        src_name = args[0].strip()
    dim_expr = args[2] if len(args) >= 3 else None
    dim_literal: int | None = None
    if isinstance(dim_expr, str):
        raw = dim_expr.strip()
        try:
            dim_literal = int(raw)
        except ValueError:
            dim_literal = None
    elif isinstance(dim_expr, int):
        dim_literal = int(dim_expr)
    if (
        isinstance(src_name, str)
        and src_name.isidentifier()
        and src_name in ctx.tensor_heads
        and dim_literal == 1
    ):
        src_heads = ctx.tensor_heads[src_name]
        repeats = args[1] if len(args) >= 2 else None
        ctx.tensor_heads[out] = f"({src_heads} * {repeats})" if repeats is not None else src_heads
    if isinstance(src_name, str) and src_name.isidentifier():
        if src_name in ctx.tensor_last_dim:
            ctx.tensor_last_dim[out] = ctx.tensor_last_dim[src_name]
        src_shape = ctx.tensor_shape.get(src_name)
        if isinstance(src_shape, tuple):
            rank = len(src_shape)
            if rank <= 0:
                return True
            dim_norm = dim_literal
            if isinstance(dim_norm, int) and dim_norm < 0:
                dim_norm += rank
            if not isinstance(dim_norm, int) or dim_norm < 0 or dim_norm >= rank:
                return True
            repeats = args[1] if len(args) >= 2 else None
            if repeats is not None:
                out_shape = list(src_shape)
                out_shape[dim_norm] = f"({out_shape[dim_norm]} * {repeats})"
                ctx.tensor_shape[out] = tuple(out_shape)
            else:
                ctx.tensor_shape[out] = src_shape
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
    raw_args = node_spec.get("_args")
    if not isinstance(raw_args, list) or len(raw_args) != 3:
        raise ValueError("repeat requires positional args: x repeats dim")
    src = model._read_tensor_input(raw_args[0], env)
    n_rep = int(model._eval_expr(raw_args[1], env, symbols))
    dim = int(model._eval_expr(raw_args[2], env, symbols))
    rank = int(src.dim())
    if dim < 0:
        dim += rank
    if dim < 0 or dim >= rank:
        raise ValueError(f"repeat dim out of range for rank {rank}: {dim}")
    out = model._require_name(node_spec.get("_bind"), field="repeat._bind")
    if n_rep == 1:
        env[out] = src
    else:
        env[out] = torch.repeat_interleave(src, repeats=n_rep, dim=dim)
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

    raw_args = node_spec.get("_args")
    if not isinstance(raw_args, list) or len(raw_args) != 3:
        raise ValueError("repeat requires positional args: x repeats dim")
    src = read(str(raw_args[0]))
    out_name = str(node_spec.get("_bind"))
    out_var = assign_out_var(out_name)
    repeats_code = emitter._expr_code(raw_args[1], env)
    dim_code = emitter._expr_code(raw_args[2], env)
    dim_var = emitter._fresh("dim")
    n_rep = emitter._fresh("n_rep")
    rank_var = emitter._fresh("rank")
    lines.append(f"{indent}{dim_var} = int({dim_code})")
    lines.append(f"{indent}{rank_var} = int({src}.dim())")
    lines.append(f"{indent}if {dim_var} < 0:")
    lines.append(f"{indent}    {dim_var} += {rank_var}")
    lines.append(f"{indent}if {dim_var} < 0 or {dim_var} >= {rank_var}:")
    lines.append(
        f"{indent}    raise ValueError(f'repeat dim out of range for rank {{{rank_var}}}: {{{dim_var}}}')"
    )
    lines.append(f"{indent}{n_rep} = int({repeats_code})")
    lines.append(f"{indent}if {n_rep} == 1:")
    lines.append(f"{indent}    {out_var} = {src}")
    lines.append(f"{indent}else:")
    lines.append(
        f"{indent}    {out_var} = torch.repeat_interleave({src}, repeats={n_rep}, dim={dim_var})"
    )
    return lines


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any", "Any", "Any"),
    "kwargs": dict(LOWERING_KWARG_KINDS),
    "returns": ("Tensor",),
}


def _unwrap_expr(expr: Any) -> Any:
    while isinstance(expr, AxonExprAscribe | AxonExprParen):
        expr = expr.expr if isinstance(expr, AxonExprAscribe) else expr.inner
    return expr


def _dim_mul(left: Any, right: Any) -> Any:
    if isinstance(left, int) and isinstance(right, int):
        return left * right
    if isinstance(right, DimExprBinary) and right.op == "/" and right.right == left:
        return right.left
    if isinstance(left, DimExprBinary) and left.op == "/" and left.right == right:
        return left.left
    if left == 1:
        return right
    if right == 1:
        return left
    return DimExprBinary(op="*", left=left, right=right)


def _expr_to_dim_token(expr: Any, helpers: Any, seen: frozenset[str] = frozenset()) -> Any | None:
    token = helpers.expr_to_dim_token(expr)
    if isinstance(token, str) and token not in seen:
        resolved = helpers.resolve_name_expr(token)
        if resolved is not None:
            return _expr_to_dim_token(resolved, helpers, seen | {token})
    return token


def type_rule(
    *,
    arg_types: tuple[Any, ...],
    kwarg_types: dict[str, Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    helpers: Any,
) -> Any | None:
    del kwarg_types, kwargs
    if len(arg_types) != 3 or len(args) != 3:
        return None
    dims = helpers.type_dims(arg_types[0])
    if dims is None:
        return None
    repeats = _expr_to_dim_token(args[1], helpers)
    if repeats is None:
        return None
    raw_dim = _unwrap_expr(args[2])
    if isinstance(raw_dim, AxonExprInt):
        dim_value = raw_dim.value
    elif isinstance(raw_dim, int):
        dim_value = raw_dim
    else:
        return None
    rank = len(dims)
    dim = dim_value if dim_value >= 0 else rank + dim_value
    if dim < 0 or dim >= rank:
        return None
    out_dims = list(dims)
    out_dims[dim] = _dim_mul(out_dims[dim], repeats)
    return helpers.type_tensor(dims=tuple(out_dims))

__all__ = [
    "OP_NAME",
    "LOWERING_ARITY",
    "LOWERING_ALLOWED_KWARGS",
    "LOWERING_REQUIRED_KWARGS",
    "LOWERING_KWARG_KINDS",
    "lowering_normalize_kwargs",
    "lowering_infer_metadata",
    "interpret",
    "compile",
    "uses_node_path",
    "LOWERING_TYPE_SIGNATURE",
    "type_rule",
]
