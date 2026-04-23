from __future__ import annotations

from typing import Any

from ._broadcast import broadcast_last_dim, broadcast_shape

OP_NAME = "where"
LOWERING_ARITY = (3, 3)
LOWERING_ALLOWED_KWARGS: set[str] = set()
LOWERING_REQUIRED_KWARGS: set[str] = set()
LOWERING_KWARG_KINDS: dict[str, Any] = {}


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return False


def _parse_scalar_token(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if text.lower() == "true":
            return True
        if text.lower() == "false":
            return False
        if text.lower() == "null":
            return None
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
    del args, kwargs, ctx
    if not isinstance(out, str):
        raise ValueError("where requires a single scalar output binding")


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
    shapes = [
        ctx.tensor_shape.get(name.strip())
        for name in args
        if isinstance(name, str) and name.strip()
    ]
    dims = [
        ctx.tensor_last_dim.get(name.strip())
        for name in args
        if isinstance(name, str) and name.strip()
    ]
    out_shape = None
    for shape in shapes:
        if shape is None:
            continue
        out_shape = shape if out_shape is None else broadcast_shape(out_shape, shape)
        if out_shape is None:
            raise ValueError(f"where requires broadcastable shapes; got {shapes!r}")
    out_dim = None
    for dim in dims:
        if dim is None:
            continue
        out_dim = dim if out_dim is None else broadcast_last_dim(out_dim, dim)
        if out_dim is None:
            raise ValueError(f"where requires broadcastable last-dim; got {dims!r}")
    if out_shape is not None:
        ctx.tensor_shape[out] = out_shape
    if out_dim is not None:
        ctx.tensor_last_dim[out] = out_dim
    return True


def _resolve_value(
    model: Any, raw: Any, env: dict[str, Any], symbols: dict[str, int | float]
) -> Any:
    return (
        env[raw]
        if isinstance(raw, str) and raw in env
        else _parse_scalar_token(model._eval_expr(raw, env, symbols))
    )


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
    if not isinstance(inputs, list) or len(inputs) != 3:
        raise ValueError("where expects three inputs")
    out = model._require_name(node_spec.get("_bind"), field="where._bind")
    cond = _resolve_value(model, inputs[0], env, symbols)
    true_value = _resolve_value(model, inputs[1], env, symbols)
    false_value = _resolve_value(model, inputs[2], env, symbols)
    env[out] = __import__("torch").where(cond, true_value, false_value)


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
    if not isinstance(inputs, list) or len(inputs) != 3:
        raise ValueError("where expects three inputs")

    def render(raw: Any) -> str:
        if isinstance(raw, str) and str(raw) in env:
            return emitter._read_env_var(env, str(raw))
        return emitter._expr_code(_parse_scalar_token(raw), env)

    cond = render(inputs[0])
    true_value = render(inputs[1])
    false_value = render(inputs[2])
    out_var = emitter._assign_out_var(env, str(node_spec.get("_bind")))
    return [f"{indent}{out_var} = torch.where({cond}, {true_value}, {false_value})"]


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any", "Any", "Any"),
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
    del kwarg_types, args, kwargs
    dims: tuple[Any, ...] | None = None
    for arg_type in arg_types:
        arg_dims = helpers.type_dims(arg_type)
        if arg_dims is None:
            continue
        if dims is None:
            dims = arg_dims
            continue
        dims = broadcast_shape(dims, arg_dims)
        if dims is None:
            return None
    if dims is None:
        return None
    return helpers.type_tensor(dims=dims)


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
