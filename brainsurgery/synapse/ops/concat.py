from __future__ import annotations

from typing import Any

import torch

from ..axon.ast import AxonExprAscribe, AxonExprInt, AxonExprParen, DimExprBinary

OP_NAME = "concat"
LOWERING_ARITY = (2, 2)
LOWERING_ALLOWED_KWARGS: set[str] = {"dim"}
LOWERING_REQUIRED_KWARGS: set[str] = set()
LOWERING_KWARG_KINDS: dict[str, Any] = {"dim": "int"}


def _dim_add(left: Any, right: Any) -> Any:
    if isinstance(left, int) and isinstance(right, int):
        return left + right
    if (
        isinstance(left, DimExprBinary)
        and left.op == "/"
        and isinstance(right, DimExprBinary)
        and right.op == "/"
        and left.left == right.left
        and left.right == right.right == 2
    ):
        return left.left
    if isinstance(right, DimExprBinary) and right.op == "-" and right.right == left:
        return right.left
    if isinstance(left, DimExprBinary) and left.op == "-" and left.right == right:
        return left.left
    return DimExprBinary(op="+", left=left, right=right)


def _resolve_dim_alias(dim: Any, helpers: Any) -> Any:
    if not isinstance(dim, str):
        return dim
    resolved = helpers.resolve_name_expr(dim)
    if resolved is None:
        return dim
    token = helpers.expr_to_dim_token(resolved)
    return dim if token is None else token


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return False


def lowering_validate_signature(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> None:
    del args, kwargs, ctx
    if isinstance(out, list):
        raise ValueError("concat requires a single scalar output binding")


def lowering_infer_metadata(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> bool:
    if isinstance(out, list):
        return False
    if len(args) != 2:
        return False
    dim = kwargs.get("dim", -1)
    if dim not in (-1, 2):
        return False
    lhs = str(args[0]).strip()
    rhs = str(args[1]).strip()
    lhs_last = ctx.tensor_last_dim.get(lhs)
    rhs_last = ctx.tensor_last_dim.get(rhs)
    if lhs_last is not None and rhs_last is not None:
        ctx.tensor_last_dim[out] = f"({lhs_last} + {rhs_last})"
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
    del node_path, scope
    ins = node_spec.get("_args")
    if not isinstance(ins, list) or len(ins) != 2:
        raise ValueError("concat expects two inputs [x, y]")
    x_ref = ins[0]
    y_ref = ins[1]
    x = env[x_ref] if isinstance(x_ref, str) and x_ref in env else model._eval_expr(x_ref, env, symbols)
    y = env[y_ref] if isinstance(y_ref, str) and y_ref in env else model._eval_expr(y_ref, env, symbols)
    dim = int(model._eval_expr(node_spec.get("dim", -1), env, symbols))
    out = model._require_name(node_spec.get("_bind"), field="concat._bind")
    env[out] = torch.cat([x, y], dim=dim)
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
    del node_path_var, scope_var
    ins = node_spec.get("_args")
    if not isinstance(ins, list) or len(ins) != 2:
        raise ValueError("concat expects two inputs [x, y]")
    x_ref = ins[0]
    y_ref = ins[1]
    x = emitter._read_env_var(env, str(x_ref)) if isinstance(x_ref, str) and str(x_ref) in env else emitter._expr_code(x_ref, env)
    y = emitter._read_env_var(env, str(y_ref)) if isinstance(y_ref, str) and str(y_ref) in env else emitter._expr_code(y_ref, env)
    out = emitter._assign_out_var(env, str(node_spec.get("_bind")))
    dim_expr = emitter._expr_code(node_spec.get("dim", -1), env)
    return [f"{indent}{out} = torch.cat([{x}, {y}], dim=int({dim_expr}))"]


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any", "Any"),
    "kwargs": dict(LOWERING_KWARG_KINDS),
    "returns": "dynamic",
}


def type_rule(
    *,
    arg_types: tuple[Any, ...],
    kwarg_types: dict[str, Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    helpers: Any,
) -> Any | None:
    del kwarg_types, args
    if len(arg_types) != 2:
        return None
    left_dims = helpers.type_dims(arg_types[0])
    right_dims = helpers.type_dims(arg_types[1])
    if left_dims is None or right_dims is None:
        return None
    if len(left_dims) != len(right_dims):
        if any(isinstance(dim, str) and dim.startswith("..") for dim in left_dims):
            return helpers.type_tensor(dims=left_dims)
        if any(isinstance(dim, str) and dim.startswith("..") for dim in right_dims):
            return helpers.type_tensor(dims=right_dims)
        return None
    raw_dim = kwargs.get("dim", -1)
    while isinstance(raw_dim, AxonExprAscribe | AxonExprParen):
        raw_dim = raw_dim.expr if isinstance(raw_dim, AxonExprAscribe) else raw_dim.inner
    if isinstance(raw_dim, AxonExprInt):
        raw_dim = raw_dim.value
    else:
        resolved_dim = helpers.expr_to_dim_token(raw_dim)
        if isinstance(resolved_dim, int):
            raw_dim = resolved_dim
    if isinstance(raw_dim, bool) or not isinstance(raw_dim, int):
        return None
    rank = len(left_dims)
    dim = raw_dim if raw_dim >= 0 else rank + raw_dim
    if dim < 0 or dim >= rank:
        return None
    out_dims: list[Any] = []
    for idx, (left_dim, right_dim) in enumerate(zip(left_dims, right_dims, strict=True)):
        if idx == dim:
            out_dims.append(
                _dim_add(
                    _resolve_dim_alias(left_dim, helpers),
                    _resolve_dim_alias(right_dim, helpers),
                )
            )
            continue
        if left_dim != right_dim:
            return None
        out_dims.append(left_dim)
    return helpers.type_tensor(dims=tuple(out_dims))


__all__ = [
    "OP_NAME",
    "LOWERING_ARITY",
    "LOWERING_ALLOWED_KWARGS",
    "LOWERING_REQUIRED_KWARGS",
    "LOWERING_KWARG_KINDS",
    "lowering_validate_signature",
    "lowering_infer_metadata",
    "interpret",
    "compile",
    "uses_node_path",
    "LOWERING_TYPE_SIGNATURE",
    "type_rule",
]
