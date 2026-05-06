from __future__ import annotations

from typing import Any

import torch

from ..axon.ast import AxonExprAscribe, AxonExprInt, AxonExprParen, TypeTuple, TypeTensor

OP_NAME = "topk"
LOWERING_ARITY = (5, 5)
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


def _name_expr(value: str) -> dict[str, Any]:
    return {"_expr": "name", "id": value}


def _expr_payload(value: Any) -> Any:
    if isinstance(value, str):
        token = value.strip()
        lowered = token.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if token.isidentifier():
            return _name_expr(token)
    return value


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return False


def lowering_known_output_arity(*, kwargs: dict[str, Any]) -> int:
    del kwargs
    return 2


def lowering_validate_signature(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> None:
    del ctx
    if isinstance(out, list):
        if len(out) != 2:
            raise ValueError("topk requires exactly two outputs: values, indices")
    elif not isinstance(out, str):
        raise ValueError("topk requires exactly two outputs: values, indices")
    if kwargs:
        unknown = ", ".join(sorted(str(key) for key in kwargs))
        raise ValueError(f"topk unsupported kwargs: {unknown}")
    if len(args) != 5:
        raise ValueError(f"topk expects exactly 5 positional args, got {len(args)}")


def interpret(
    model: Any,
    node_spec: dict[str, Any],
    env: dict[str, Any],
    *,
    node_path: str,
    scope: str,
    symbols: dict[str, int],
) -> None:
    args = _raw_args(node_spec)
    if len(args) != 5:
        raise ValueError("topk requires positional args: x k dim largest sorted")
    x = model._read_tensor_input(args[0], env)
    outs = node_spec.get("_bind")
    if not isinstance(outs, list) or len(outs) != 2:
        raise ValueError("topk expects out=[values,indices]")
    k = int(model._eval_expr(args[1], env, symbols))
    dim = int(model._eval_expr(args[2], env, symbols))
    largest = bool(model._eval_expr(args[3], env, symbols))
    sorted_flag = bool(model._eval_expr(args[4], env, symbols))
    values, indices = torch.topk(x, k, dim=dim, largest=largest, sorted=sorted_flag)
    env[outs[0]] = values
    env[outs[1]] = indices
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

    def infer_param(param_name: str) -> str:
        return emitter._infer_param_expr(node_spec, node_path_var, param_name)

    def read(name: str) -> str:
        return emitter._read_env_var(env, name)

    args = _raw_args(node_spec)
    if len(args) != 5:
        raise ValueError("topk requires positional args: x k dim largest sorted")
    src = read(str(args[0]))
    outs = node_spec.get("_bind")
    if not isinstance(outs, list) or len(outs) != 2:
        raise ValueError("topk expects out=[values,indices]")
    values_var = assign_out_var(str(outs[0]))
    indices_var = assign_out_var(str(outs[1]))
    k = emitter._expr_code(_expr_payload(args[1]), env)
    dim = emitter._expr_code(_expr_payload(args[2]), env)
    largest = emitter._expr_code(_expr_payload(args[3]), env)
    sorted_flag = emitter._expr_code(_expr_payload(args[4]), env)
    lines.append(
        f"{indent}{values_var}, {indices_var} = torch.topk({src}, int({k}), dim=int({dim}), largest=bool({largest}), sorted=bool({sorted_flag}))"
    )
    return lines


LOWERING_TYPE_SIGNATURE = {
    "args": ("Tensor[..S]", "Any", "Any", "Any", "Any"),
    "kwargs": {},
    "returns": ("Tensor[..S]", "IdxTensor[..S]"),
}


def _unwrap_expr(expr: Any) -> Any:
    while isinstance(expr, AxonExprAscribe | AxonExprParen):
        expr = expr.expr if isinstance(expr, AxonExprAscribe) else expr.inner
    return expr


def type_rule(
    *,
    arg_types: tuple[Any, ...],
    kwarg_types: dict[str, Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    helpers: Any,
) -> Any | None:
    del kwarg_types, kwargs
    if len(arg_types) != 5 or len(args) != 5:
        return None
    input_dims = helpers.type_dims(arg_types[0])
    if input_dims is None:
        return None
    k_dim = helpers.expr_to_dim_token(args[1])
    if k_dim is None:
        return None
    raw_axis = _unwrap_expr(args[2])
    if isinstance(raw_axis, AxonExprInt):
        axis_value = raw_axis.value
    elif isinstance(raw_axis, int):
        axis_value = raw_axis
    else:
        return None
    rank = len(input_dims)
    axis = axis_value if axis_value >= 0 else rank + axis_value
    if axis < 0 or axis >= rank:
        return None
    out_dims = list(input_dims)
    out_dims[axis] = k_dim
    dims = tuple(out_dims)
    return TypeTuple(
        items=(
            TypeTensor(base="Tensor", dims=dims),
            TypeTensor(base="IdxTensor", dims=dims),
        )
    )

__all__ = [
    "OP_NAME",
    "LOWERING_ARITY",
    "LOWERING_ALLOWED_KWARGS",
    "LOWERING_REQUIRED_KWARGS",
    "LOWERING_KWARG_KINDS",
    "lowering_known_output_arity",
    "lowering_validate_signature",
    "interpret",
    "compile",
    "uses_node_path",
    "LOWERING_TYPE_SIGNATURE",
    "type_rule",
]
