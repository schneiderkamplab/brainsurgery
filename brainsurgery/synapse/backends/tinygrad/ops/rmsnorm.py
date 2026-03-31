from __future__ import annotations

from typing import Any

OP_NAME = "rmsnorm"
LOWERING_ARITY = (1, 1)
LOWERING_ALLOWED_KWARGS: set[str] = {"eps", "dim", "unit_offset", "cast_float"}
LOWERING_REQUIRED_KWARGS: set[str] = set()
LOWERING_KWARG_KINDS: dict[str, Any] = {
    "dim": "dim",
    "eps": "number",
    "cast_float": "bool",
    "unit_offset": "bool",
}


def _dims_compatible(left: Any, right: Any) -> bool:
    if isinstance(left, str) and left.strip().lstrip("-").isdigit():
        left = int(left.strip())
    if isinstance(right, str) and right.strip().lstrip("-").isdigit():
        right = int(right.strip())
    return left == right


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
            raise ValueError(f"rmsnorm dim={norm_dim!r} mismatches input last-dim {first_dim!r}")
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
    eps = emitter._expr_code(node_spec.get("eps", 1e-6), env)
    unit_offset = bool(node_spec.get("unit_offset", False))
    cast_float = bool(node_spec.get("cast_float", False))
    w = f"self._param({infer_param('weight')})"

    x_norm_src = f"{src}.cast(dtypes.float32)" if cast_float else src
    w_src = f"{w}.cast(dtypes.float32)" if cast_float else w
    tmp = emitter._fresh("xnorm")

    lines.append(
        f"{indent}{tmp} = {x_norm_src} * ({x_norm_src}.square().mean(axis=-1, keepdim=True) + float({eps})).rsqrt()"
    )
    if unit_offset:
        lines.append(f"{indent}{out_var} = {tmp} * (1.0 + {w_src})")
    else:
        lines.append(f"{indent}{out_var} = {tmp} * {w_src}")
    if cast_float:
        lines.append(f"{indent}{out_var} = {out_var}.cast({src}.dtype)")
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
