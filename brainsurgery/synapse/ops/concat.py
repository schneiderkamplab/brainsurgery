from __future__ import annotations

from typing import Any

import torch

OP_NAME = "concat"
LOWERING_ARITY = (2, 2)
LOWERING_ALLOWED_KWARGS: set[str] = {"dim"}
LOWERING_REQUIRED_KWARGS: set[str] = set()
LOWERING_KWARG_KINDS: dict[str, Any] = {"dim": "int"}


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
    x = env[ins[0]]
    y = env[ins[1]]
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
    x = emitter._read_env_var(env, str(ins[0]))
    y = emitter._read_env_var(env, str(ins[1]))
    out = emitter._assign_out_var(env, str(node_spec.get("_bind")))
    dim_expr = emitter._expr_code(node_spec.get("dim", -1), env)
    return [f"{indent}{out} = torch.cat([{x}, {y}], dim=int({dim_expr}))"]


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any", "Any"),
    "kwargs": dict(LOWERING_KWARG_KINDS),
    "returns": "dynamic",
}

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
]
