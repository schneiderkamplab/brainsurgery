from __future__ import annotations

from typing import Any

from torch.nn import functional as F

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
    x = model._read_tensor_input(node_spec.get("_args"), env)
    weight_path = model._infer_param_path(node_spec, node_path=node_path, param_name="weight")
    weight = model._state[weight_path]
    out = model._require_name(node_spec.get("_bind"), field="embedding._bind")
    y = F.embedding(x, weight)
    if node_spec.get("scale") is not None:
        scale = float(model._eval_expr(node_spec.get("scale"), env, symbols))
        y = y * y.new_tensor(scale)
    env[out] = y
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
    scale_expr = node_spec.get("scale")
    if scale_expr is None:
        lines.append(
            f"{indent}{out_var} = F.embedding({src}, emitter._param({infer_param('weight')}))"
        )
    else:
        scale = emitter._expr_code(scale_expr, env)
        lines.append(
            f"{indent}{out_var} = F.embedding({src}, emitter._param({infer_param('weight')}))"
        )
        lines.append(
            f"{indent}{out_var} = {out_var} * torch.tensor(float({scale}), dtype={out_var}.dtype, device={out_var}.device)"
        )
    return lines


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
    "lowering_normalize_kwargs",
    "lowering_infer_metadata",
    "interpret",
    "compile",
    "uses_node_path",
    "LOWERING_TYPE_SIGNATURE",
]
