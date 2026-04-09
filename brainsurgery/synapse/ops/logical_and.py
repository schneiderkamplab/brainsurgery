from __future__ import annotations

from typing import Any

from ._broadcast import broadcast_last_dim, broadcast_shape

OP_NAME = "and"
LOWERING_ARITY = (2, 2)
LOWERING_ALLOWED_KWARGS: set[str] = set()
LOWERING_REQUIRED_KWARGS: set[str] = set()
LOWERING_KWARG_KINDS: dict[str, Any] = {}


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return False


def lowering_validate_signature(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> None:
    del args, kwargs, ctx
    if not isinstance(out, str):
        raise ValueError("and requires a single scalar output binding")


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
    first_in = args[0].strip() if args else None
    second_in = args[1].strip() if len(args) > 1 else None
    first_dim = ctx.tensor_last_dim.get(first_in) if isinstance(first_in, str) else None
    second_dim = ctx.tensor_last_dim.get(second_in) if isinstance(second_in, str) else None
    first_shape = ctx.tensor_shape.get(first_in) if isinstance(first_in, str) else None
    second_shape = ctx.tensor_shape.get(second_in) if isinstance(second_in, str) else None
    broadcasted_shape = broadcast_shape(first_shape, second_shape)
    if first_shape is not None and second_shape is not None and broadcasted_shape is None:
        raise ValueError(
            f"and requires broadcastable shapes; got {first_shape!r} and {second_shape!r}"
        )
    unified = broadcast_last_dim(first_dim, second_dim)
    if first_dim is not None and second_dim is not None and unified is None:
        raise ValueError(
            f"and requires broadcastable last-dim; got {first_dim!r} and {second_dim!r}"
        )
    if unified is not None:
        ctx.tensor_last_dim[out] = unified
    if broadcasted_shape is not None:
        ctx.tensor_shape[out] = broadcasted_shape
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
    del node_path, scope, symbols
    inputs = node_spec.get("_args")
    if not isinstance(inputs, list) or len(inputs) != 2:
        raise ValueError("and expects two inputs")
    out = model._require_name(node_spec.get("_bind"), field="and._bind")
    left = model._read_tensor_input(inputs[0], env)
    right = model._read_tensor_input(inputs[1], env)
    env[out] = left.to(bool) & right.to(bool)


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
        raise ValueError("and expects two inputs")
    left = emitter._read_env_var(env, str(inputs[0]))
    right = emitter._read_env_var(env, str(inputs[1]))
    out_var = emitter._assign_out_var(env, str(node_spec.get("_bind")))
    return [f"{indent}{out_var} = {left}.to(torch.bool) & {right}.to(torch.bool)"]


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
