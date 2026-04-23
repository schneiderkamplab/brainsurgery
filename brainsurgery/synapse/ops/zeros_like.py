from __future__ import annotations

from typing import Any

import torch

OP_NAME = "zeros_like"
LOWERING_ARITY = (1, 1)
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
        raise ValueError("zeros_like requires a single scalar output binding")


def interpret(
    model: Any,
    node_spec: dict[str, Any],
    env: dict[str, Any],
    *,
    node_path: str,
    scope: str,
    symbols: dict[str, int],
) -> None:
    x = model._read_tensor_input(node_spec.get("_args"), env)
    out = model._require_name(node_spec.get("_bind"), field="zeros_like._bind")
    env[out] = torch.zeros_like(x)
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

    src = read(str(node_spec.get("_args")))
    out_name = str(node_spec.get("_bind"))
    out_var = assign_out_var(out_name)
    lines.append(f"{indent}{out_var} = torch.zeros_like({src})")
    return lines


LOWERING_TYPE_SIGNATURE = {
    "args": ("Tensor[..S]",),
    "kwargs": dict(LOWERING_KWARG_KINDS),
    "returns": ("Tensor[..S]",),
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
    if not arg_types:
        return None
    input_dims = helpers.type_dims(arg_types[0])
    if input_dims is None:
        return None
    return helpers.type_tensor(dims=input_dims)


__all__ = [
    "LOWERING_ARITY",
    "LOWERING_ALLOWED_KWARGS",
    "LOWERING_REQUIRED_KWARGS",
    "LOWERING_KWARG_KINDS",
    "OP_NAME",
    "lowering_validate_signature",
    "interpret",
    "compile",
    "uses_node_path",
    "LOWERING_TYPE_SIGNATURE",
    "type_rule",
]
