from __future__ import annotations

from typing import Any

from ._broadcast import broadcast_last_dim, broadcast_shape

OP_NAME = "masked_fill"
LOWERING_ARITY = (2, 2)
LOWERING_ALLOWED_KWARGS: set[str] = {"value"}
LOWERING_REQUIRED_KWARGS: set[str] = {"value"}
LOWERING_KWARG_KINDS: dict[str, Any] = {"value": "number"}


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return False


def _parse_scalar_token(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if text and text[0] == "-" and text[1:].isdigit():
            return int(text)
        if text.isdigit():
            return int(text)
        try:
            return float(text)
        except ValueError:
            return value
    return value


def lowering_validate_signature(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> None:
    del args, ctx
    if not isinstance(out, str):
        raise ValueError("masked_fill requires a single scalar output binding")
    if "value" not in kwargs:
        raise ValueError("masked_fill requires value")


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
    src_name = args[0].strip() if args else None
    mask_name = args[1].strip() if len(args) > 1 else None
    src_shape = ctx.tensor_shape.get(src_name) if isinstance(src_name, str) else None
    mask_shape = ctx.tensor_shape.get(mask_name) if isinstance(mask_name, str) else None
    broadcasted_shape = broadcast_shape(src_shape, mask_shape)
    if src_shape is not None and mask_shape is not None and broadcasted_shape is None:
        raise ValueError(
            f"masked_fill requires broadcastable shapes; got {src_shape!r} and {mask_shape!r}"
        )
    src_last = ctx.tensor_last_dim.get(src_name) if isinstance(src_name, str) else None
    mask_last = ctx.tensor_last_dim.get(mask_name) if isinstance(mask_name, str) else None
    unified = broadcast_last_dim(src_last, mask_last)
    if src_last is not None and mask_last is not None and unified is None:
        raise ValueError(
            f"masked_fill requires broadcastable last-dim; got {src_last!r} and {mask_last!r}"
        )
    if src_shape is not None:
        ctx.tensor_shape[out] = src_shape
    elif broadcasted_shape is not None:
        ctx.tensor_shape[out] = broadcasted_shape
    if src_last is not None:
        ctx.tensor_last_dim[out] = src_last
    elif unified is not None:
        ctx.tensor_last_dim[out] = unified
    return True


def interpret(
    model: Any,
    node_spec: dict[str, Any],
    env: dict[str, Any],
    *,
    node_path: str,
    scope: str,
    symbols: dict[str, int | float],
) -> None:
    del node_path, scope
    inputs = node_spec.get("_args")
    if not isinstance(inputs, list) or len(inputs) != 2:
        raise ValueError("masked_fill expects two inputs")
    src = env[inputs[0]]
    mask = env[inputs[1]]
    raw_value = node_spec.get("value")
    value = (
        env[raw_value]
        if isinstance(raw_value, str) and raw_value in env
        else _parse_scalar_token(model._eval_expr(raw_value, env, symbols))
    )
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("masked_fill.value must resolve to number")
    out = model._require_name(node_spec.get("_bind"), field="masked_fill._bind")
    env[out] = src.masked_fill(mask.bool(), value)


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
    inputs = node_spec.get("_args")
    if not isinstance(inputs, list) or len(inputs) != 2:
        raise ValueError("masked_fill expects two inputs")
    src = emitter._read_env_var(env, str(inputs[0]))
    mask = emitter._read_env_var(env, str(inputs[1]))
    out_var = emitter._assign_out_var(env, str(node_spec.get("_bind")))
    raw_value = node_spec.get("value")
    value_expr = (
        emitter._read_env_var(env, str(raw_value))
        if isinstance(raw_value, str) and str(raw_value) in env
        else emitter._expr_code(raw_value, env)
    )
    return [f"{indent}{out_var} = {src}.masked_fill({mask}.bool(), {value_expr})"]


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any", "Any"),
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
