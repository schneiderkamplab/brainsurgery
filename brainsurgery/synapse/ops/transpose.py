from __future__ import annotations

from typing import Any

import torch

from ._broadcast import _normalize_dim_token

OP_NAME = "transpose"
LOWERING_ARITY = (1, 3)
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


def _int_literal(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip()
        if token and (token.isdigit() or (token[0] in {"+", "-"} and token[1:].isdigit())):
            return int(token)
    return None


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
        raise ValueError("transpose requires a single scalar output binding")
    if kwargs:
        unknown = ", ".join(sorted(str(key) for key in kwargs))
        raise ValueError(f"transpose unsupported kwargs: {unknown}")
    if len(args) < 1 or len(args) > 3:
        raise ValueError(f"transpose expects 1..3 positional args, got {len(args)}")


def lowering_infer_metadata(
    *,
    args: list[str],
    out: str | list[str],
    kwargs: dict[str, Any],
    ctx: Any,
) -> bool:
    if not isinstance(out, str) or not args:
        return False
    source_name = str(args[0]).strip()
    source_shape = ctx.tensor_shape.get(source_name)
    raw_dim1 = _int_literal(_arg_or_default(args, 1, 1))
    raw_dim2 = _int_literal(_arg_or_default(args, 2, 2))
    if not isinstance(source_shape, tuple) or raw_dim1 is None or raw_dim2 is None:
        return False
    rank = len(source_shape)
    dim1 = raw_dim1 if raw_dim1 >= 0 else rank + raw_dim1
    dim2 = raw_dim2 if raw_dim2 >= 0 else rank + raw_dim2
    if not (0 <= dim1 < rank and 0 <= dim2 < rank):
        return False
    new_shape = list(source_shape)
    new_shape[dim1], new_shape[dim2] = new_shape[dim2], new_shape[dim1]
    out_shape = tuple(_normalize_dim_token(v) for v in new_shape)
    ctx.tensor_shape[out] = out_shape
    ctx.tensor_last_dim[out] = out_shape[-1]
    return True


def _resolve_int(
    model: Any, raw: Any, env: dict[str, Any], symbols: dict[str, int], name: str
) -> int:
    value = model._eval_expr(raw, env, symbols)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"transpose.{name} must resolve to int")
    return int(value)


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
    if not args:
        raise ValueError("transpose requires positional args: x [dim1 dim2]")
    src = model._read_tensor_input(args[0], env)
    dim1 = _resolve_int(model, _arg_or_default(args, 1, 1), env, symbols, "dim1")
    dim2 = _resolve_int(model, _arg_or_default(args, 2, 2), env, symbols, "dim2")
    out = model._require_name(node_spec.get("_bind"), field="transpose._bind")
    env[out] = torch.transpose(src, dim1, dim2)


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
    if not args:
        raise ValueError("transpose requires positional args: x [dim1 dim2]")
    src = emitter._read_env_var(env, str(args[0]))
    out_var = emitter._assign_out_var(env, str(node_spec.get("_bind")))
    dim1_expr = emitter._expr_code(_expr_payload(_arg_or_default(args, 1, 1)), env)
    dim2_expr = emitter._expr_code(_expr_payload(_arg_or_default(args, 2, 2)), env)
    return [f"{indent}{out_var} = torch.transpose({src}, int({dim1_expr}), int({dim2_expr}))"]


LOWERING_TYPE_SIGNATURE = {
    "args": ("Tensor[..S]", "Any", "Any"),
    "kwargs": {},
    "returns": ("Tensor[..R]",),
}


def type_rule(
    *,
    arg_types: tuple[Any, ...],
    kwarg_types: dict[str, Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    helpers: Any,
) -> Any | None:
    del kwarg_types, kwargs
    if not arg_types or not args:
        return None
    input_dims = helpers.type_dims(arg_types[0])
    if input_dims is None:
        return None
    rank = len(input_dims)
    if rank == 0:
        return None
    dim1_token = helpers.expr_to_dim_token(args[1]) if len(args) >= 2 else 1
    dim2_token = helpers.expr_to_dim_token(args[2]) if len(args) >= 3 else 2
    if not isinstance(dim1_token, int) or not isinstance(dim2_token, int):
        return None
    dim1 = dim1_token if dim1_token >= 0 else rank + dim1_token
    dim2 = dim2_token if dim2_token >= 0 else rank + dim2_token
    if not (0 <= dim1 < rank and 0 <= dim2 < rank):
        return None
    out_dims = list(input_dims)
    out_dims[dim1], out_dims[dim2] = out_dims[dim2], out_dims[dim1]
    return helpers.type_tensor(dims=tuple(out_dims))


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
    "type_rule",
]
