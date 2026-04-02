from __future__ import annotations

from typing import Any

import torch

OP_NAME = "topk"
LOWERING_ARITY = (1, 1)
LOWERING_ALLOWED_KWARGS: set[str] = {"k", "dim", "largest", "sorted"}
LOWERING_REQUIRED_KWARGS: set[str] = {"k"}
LOWERING_KWARG_KINDS: dict[str, Any] = {
    "k": "dim",
    "dim": "int",
    "largest": "bool",
    "sorted": "bool",
}


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return False


def lowering_known_output_arity(*, kwargs: dict[str, Any]) -> int:
    del kwargs
    return 2


def lowering_validate_signature(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> None:
    del args, kwargs, ctx
    if not isinstance(out, list) or len(out) != 2:
        raise ValueError("topk requires exactly two outputs: values, indices")


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
    outs = node_spec.get("_bind")
    if not isinstance(outs, list) or len(outs) != 2:
        raise ValueError("topk expects out=[values,indices]")
    k = int(model._eval_expr(node_spec.get("k"), env, symbols))
    dim = int(model._eval_expr(node_spec.get("dim", -1), env, symbols))
    largest = bool(node_spec.get("largest", True))
    sorted_flag = bool(node_spec.get("sorted", True))
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

    src = read(str(node_spec.get("_args")))
    outs = node_spec.get("_bind")
    if not isinstance(outs, list) or len(outs) != 2:
        raise ValueError("topk expects out=[values,indices]")
    values_var = assign_out_var(str(outs[0]))
    indices_var = assign_out_var(str(outs[1]))
    k = emitter._expr_code(node_spec.get("k"), env)
    dim = emitter._expr_code(node_spec.get("dim", -1), env)
    largest = bool(node_spec.get("largest", True))
    sorted_flag = bool(node_spec.get("sorted", True))
    lines.append(
        f"{indent}{values_var}, {indices_var} = torch.topk({src}, int({k}), dim=int({dim}), largest={largest}, sorted={sorted_flag})"
    )
    return lines


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any",),
    "kwargs": dict(LOWERING_KWARG_KINDS),
    "returns": ("Tensor", "IdxTensor"),
}

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
]
