from __future__ import annotations

from typing import Any

OP_NAME = "embedding"
LOWERING_ARITY = (1, 1)
LOWERING_ALLOWED_KWARGS: set[str] = {"scale", "dim"}
LOWERING_REQUIRED_KWARGS: set[str] = set()
LOWERING_KWARG_KINDS: dict[str, Any] = {"dim": "dim", "scale": "number"}


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return True


def lowering_normalize_kwargs(
    *,
    args: list[str],
    out: str | list[str],
    kwargs: dict[str, Any],
    ctx: Any,
) -> None:
    del args
    if "embedding_dim" in kwargs:
        raise ValueError("embedding does not support embedding_dim; use dim")
    if "dim" in kwargs or not isinstance(out, str):
        return
    inferred = ctx.tensor_last_dim.get(out)
    if inferred is not None:
        kwargs["dim"] = inferred


def lowering_infer_metadata(
    *,
    args: list[str],
    out: str | list[str],
    kwargs: dict[str, Any],
    ctx: Any,
) -> bool:
    del args
    if not isinstance(out, str):
        return False
    last_dim = kwargs.get("dim")
    if last_dim is not None:
        ctx.tensor_last_dim[out] = last_dim
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
    raise NotImplementedError(f"TinyGrad interpret for '{OP_NAME}' not yet implemented")


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
    weight_expr = f"self._param({infer_param('weight')})"
    scale_expr = node_spec.get("scale")
    if scale_expr is None:
        lines.append(f"{indent}{out_var} = {weight_expr}[{src}]")
    else:
        scale = emitter._expr_code(scale_expr, env)
        lines.append(f"{indent}{out_var} = {weight_expr}[{src}]")
        lines.append(f"{indent}{out_var} = {out_var} * {scale}")
    return lines


__all__ = [
    "LOWERING_ARITY",
    "LOWERING_ALLOWED_KWARGS",
    "LOWERING_REQUIRED_KWARGS",
    "LOWERING_KWARG_KINDS",
    "OP_NAME",
    "lowering_normalize_kwargs",
    "lowering_infer_metadata",
    "interpret",
    "compile",
    "uses_node_path",
]
