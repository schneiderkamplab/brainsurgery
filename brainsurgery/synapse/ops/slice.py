from __future__ import annotations

from typing import Any

from ..axon.ast import AxonExprAscribe, AxonExprName, AxonExprParen, DimExprBinary

OP_NAME = "slice"
LOWERING_ARITY = (4, 4)
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


def _dim_sub(left: Any, right: Any) -> Any:
    if right == 0:
        return left
    if isinstance(left, int) and isinstance(right, int):
        return left - right
    if isinstance(right, DimExprBinary) and right.op == "-" and right.left == left:
        return right.right
    return DimExprBinary(op="-", left=left, right=right)


def _arg_dim_token(value: Any, helpers: Any) -> Any | None:
    while isinstance(value, AxonExprAscribe | AxonExprParen):
        value = value.expr if isinstance(value, AxonExprAscribe) else value.inner
    token = helpers.expr_to_dim_token(value)
    if isinstance(value, AxonExprName):
        resolved = helpers.resolve_name_expr(value.name)
        if resolved is not None:
            resolved_token = helpers.expr_to_dim_token(resolved)
            if resolved_token is not None:
                return resolved_token
    return token


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return False


def lowering_validate_signature(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> None:
    del ctx
    if not isinstance(out, str):
        raise ValueError("slice requires a single scalar output binding")
    if kwargs:
        unknown = ", ".join(sorted(str(key) for key in kwargs))
        raise ValueError(f"slice unsupported kwargs: {unknown}")
    if len(args) != 4:
        raise ValueError(f"slice expects 4 positional args, got {len(args)}")


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
    if len(args) != 4:
        return False
    raw_dim = _int_literal(args[1])
    raw_start = _int_literal(args[2])
    raw_end = _int_literal(args[3])
    if (
        isinstance(source_shape, tuple)
        and raw_dim is not None
        and raw_start is not None
        and raw_end is not None
    ):
        rank = len(source_shape)
        dim = raw_dim if raw_dim >= 0 else raw_dim + rank
        if 0 <= dim < rank:
            out_shape = list(source_shape)
            out_shape[dim] = raw_end - raw_start
            ctx.tensor_shape[out] = tuple(out_shape)
            ctx.tensor_last_dim[out] = out_shape[-1]
            return True
    if source_name in ctx.tensor_last_dim:
        ctx.tensor_last_dim[out] = ctx.tensor_last_dim[source_name]
        return True
    return False


def _resolve_int(
    model: Any, raw: Any, env: dict[str, Any], symbols: dict[str, int], name: str
) -> int:
    value = model._eval_expr(raw, env, symbols)
    if isinstance(value, bool):
        raise ValueError(
            f"slice.{name} must resolve to int (raw={raw!r}, value={value!r}, type={type(value).__name__})"
        )
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and float(value).is_integer():
        return int(value)
    raise ValueError(
        f"slice.{name} must resolve to int (raw={raw!r}, value={value!r}, type={type(value).__name__})"
    )


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
    if len(args) != 4:
        raise ValueError("slice requires positional args: x dim start end")
    src = model._read_tensor_input(args[0], env)
    dim = _resolve_int(model, args[1], env, symbols, "dim")
    start = _resolve_int(model, args[2], env, symbols, "start")
    end = _resolve_int(model, args[3], env, symbols, "end")
    rank = src.ndim
    axis = dim if dim >= 0 else dim + rank
    if axis < 0 or axis >= rank:
        raise ValueError("slice.dim out of range")
    pieces: list[slice] = [slice(None)] * rank
    pieces[axis] = slice(start, end)
    out = model._require_name(node_spec.get("_bind"), field="slice._bind")
    env[out] = src[tuple(pieces)]


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
    if len(args) != 4:
        raise ValueError("slice requires positional args: x dim start end")
    src = emitter._read_env_var(env, str(args[0]))
    out_var = emitter._assign_out_var(env, str(node_spec.get("_bind")))
    dim_expr = emitter._expr_code(_expr_payload(args[1]), env)
    start_expr = emitter._expr_code(_expr_payload(args[2]), env)
    end_expr = emitter._expr_code(_expr_payload(args[3]), env)
    rank_var = emitter._fresh("rank")
    axis_var = emitter._fresh("axis")
    slices_var = emitter._fresh("slices")
    return [
        f"{indent}{rank_var} = {src}.ndim",
        f"{indent}{axis_var} = int({dim_expr})",
        f"{indent}if {axis_var} < 0:",
        f"{indent}    {axis_var} += {rank_var}",
        f"{indent}if {axis_var} < 0 or {axis_var} >= {rank_var}:",
        f"{indent}    raise ValueError('slice.dim out of range')",
        f"{indent}{slices_var} = [slice(None)] * {rank_var}",
        f"{indent}{slices_var}[{axis_var}] = slice(int({start_expr}), int({end_expr}))",
        f"{indent}{out_var} = {src}[tuple({slices_var})]",
    ]


LOWERING_TYPE_SIGNATURE = {
    "args": ("Tensor", "Int", "Int", "Int"),
    "kwargs": {},
    "returns": ("Tensor",),
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
    if len(arg_types) != 4 or len(args) != 4:
        return None
    input_dims = helpers.type_dims(arg_types[0])
    if input_dims is None:
        return None
    dim_token = _arg_dim_token(args[1], helpers)
    if not isinstance(dim_token, int):
        return None
    rank = len(input_dims)
    dim = dim_token if dim_token >= 0 else rank + dim_token
    if dim < 0 or dim >= rank:
        return None
    start = _arg_dim_token(args[2], helpers)
    end = _arg_dim_token(args[3], helpers)
    if start is None or end is None:
        return None
    out_dims = list(input_dims)
    out_dims[dim] = _dim_sub(end, start)
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
