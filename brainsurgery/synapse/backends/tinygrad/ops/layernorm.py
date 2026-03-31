from __future__ import annotations

from typing import Any

OP_NAME = "layernorm"
LOWERING_ARITY = (1, 1)
LOWERING_ALLOWED_KWARGS: set[str] = {"eps", "dim"}
LOWERING_REQUIRED_KWARGS: set[str] = set()
LOWERING_KWARG_KINDS: dict[str, Any] = {"dim": "dim", "eps": "number"}


def _dims_compatible(left: Any, right: Any) -> bool:
    if isinstance(left, str) and left.strip().lstrip("-").isdigit():
        left = int(left.strip())
    if isinstance(right, str) and right.strip().lstrip("-").isdigit():
        right = int(right.strip())
    return left == right


def _validate_layernorm_keys(node_spec: dict[str, Any]) -> None:
    if "weight" in node_spec:
        raise ValueError("layernorm does not support explicit weight binding")
    if "bias" in node_spec:
        raise ValueError("layernorm does not support explicit bias binding")


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
    del out
    if "dim" in kwargs or not args:
        return
    first_arg = args[0].strip()
    if not first_arg.isidentifier():
        return
    inferred = ctx.tensor_last_dim.get(first_arg)
    if inferred is not None:
        kwargs["dim"] = inferred


def lowering_infer_metadata(
    *,
    args: list[str],
    out: str | list[str],
    kwargs: dict[str, Any],
    ctx: Any,
) -> bool:
    if not isinstance(out, str):
        return False
    first_in = args[0].strip() if args else None
    first_dim = (
        ctx.tensor_last_dim.get(first_in)
        if isinstance(first_in, str) and first_in.isidentifier()
        else None
    )
    norm_dim = kwargs.get("dim")
    if norm_dim is not None:
        if first_dim is not None and not _dims_compatible(norm_dim, first_dim):
            raise ValueError(f"layernorm dim={norm_dim!r} mismatches input last-dim {first_dim!r}")
        if first_dim is None and isinstance(first_in, str) and first_in.isidentifier():
            ctx.tensor_last_dim[first_in] = norm_dim
    if first_dim is not None:
        ctx.tensor_last_dim[out] = first_dim
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
    _validate_layernorm_keys(node_spec)
    lines: list[str] = []

    def assign_out_var(out_name: str) -> str:
        return emitter._assign_out_var(env, out_name)

    def read(name: str) -> str:
        return emitter._read_env_var(env, name)

    src = read(str(node_spec.get("_args")))
    out_name = str(node_spec.get("_bind"))
    out_var = assign_out_var(out_name)
    eps = emitter._expr_code(node_spec.get("eps", 1e-5), env)
    w = f"self._param({emitter._infer_param_expr(node_spec, node_path_var, 'weight')})"
    b = f"self._param({emitter._infer_param_expr(node_spec, node_path_var, 'bias')})"
    norm_var = emitter._fresh("ln")
    lines.append(
        f"{indent}{norm_var} = {src}.layernorm(axis=-1, eps=float({eps}))"
    )
    lines.append(f"{indent}{out_var} = {norm_var} * {w} + {b}")
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
