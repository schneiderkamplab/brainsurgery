from __future__ import annotations

from typing import Any

OP_NAME = "slice"
LOWERING_ARITY = (1, 1)
LOWERING_ALLOWED_KWARGS: set[str] = {"dim", "start", "end"}
LOWERING_REQUIRED_KWARGS: set[str] = {"dim", "start", "end"}
LOWERING_KWARG_KINDS: dict[str, Any] = {
    "dim": "int",
    "start": "dim",
    "end": "dim",
}


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return False


def lowering_validate_signature(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> None:
    del args, ctx
    if not isinstance(out, str):
        raise ValueError("slice requires a single scalar output binding")
    for key in ("dim", "start", "end"):
        if key not in kwargs:
            raise ValueError(f"slice requires {key}")


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
    raw_dim = kwargs.get("dim")
    raw_start = kwargs.get("start")
    raw_end = kwargs.get("end")
    if (
        isinstance(source_shape, tuple)
        and isinstance(raw_dim, int)
        and isinstance(raw_start, int)
        and isinstance(raw_end, int)
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
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"slice.{name} must resolve to int")
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
    src = model._read_tensor_input(node_spec.get("_args"), env)
    dim = _resolve_int(model, node_spec.get("dim"), env, symbols, "dim")
    start = _resolve_int(model, node_spec.get("start"), env, symbols, "start")
    end = _resolve_int(model, node_spec.get("end"), env, symbols, "end")
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
    src = emitter._read_env_var(env, str(node_spec.get("_args")))
    out_var = emitter._assign_out_var(env, str(node_spec.get("_bind")))
    dim_expr = emitter._expr_code(node_spec.get("dim"), env)
    start_expr = emitter._expr_code(node_spec.get("start"), env)
    end_expr = emitter._expr_code(node_spec.get("end"), env)
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
