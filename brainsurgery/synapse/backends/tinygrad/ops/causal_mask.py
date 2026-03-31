from __future__ import annotations

from typing import Any

OP_NAME = "causal_mask"
LOWERING_ARITY = (2, 2)
LOWERING_ALLOWED_KWARGS: set[str] = {"window", "padding_mask"}
LOWERING_REQUIRED_KWARGS: set[str] = set()
LOWERING_KWARG_KINDS: dict[str, Any] = {"window": "dim", "padding_mask": "str"}


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return False


def lowering_validate_signature(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> None:
    del args, kwargs, ctx
    if not isinstance(out, str):
        raise ValueError("causal_mask requires a single scalar output binding")


def lowering_infer_metadata(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> bool:
    del kwargs
    if not isinstance(out, str) or len(args) < 2:
        return False
    query_name = str(args[0]).strip()
    key_name = str(args[1]).strip()
    query_shape = ctx.tensor_shape.get(query_name)
    key_shape = ctx.tensor_shape.get(key_name)
    if isinstance(query_shape, tuple) and len(query_shape) >= 2 and isinstance(key_shape, tuple):
        if len(key_shape) >= 2:
            q_len = query_shape[-2]
            k_len = key_shape[-2]
            ctx.tensor_shape[out] = (1, 1, q_len, k_len)
            ctx.tensor_last_dim[out] = k_len
            return True
    return False


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

    def read(name: str) -> str:
        return emitter._read_env_var(env, name)

    raw_args = node_spec.get("_args")
    if not isinstance(raw_args, list) or len(raw_args) != 2:
        raise ValueError("causal_mask expects exactly 2 positional args: query and key")
    q = read(str(raw_args[0]))
    k = read(str(raw_args[1]))
    out_name = str(node_spec.get("_bind"))
    out_var = assign_out_var(out_name)
    q_len = emitter._fresh("q_len")
    k_len = emitter._fresh("k_len")
    i_idx = emitter._fresh("i_idx")
    j_idx = emitter._fresh("j_idx")
    keep = emitter._fresh("keep")
    window_expr = node_spec.get("window")
    padding_name = node_spec.get("padding_mask")
    padding_expr = (
        env.get(padding_name) if isinstance(padding_name, str) and padding_name in env else None
    )
    if window_expr is None and padding_expr is None:
        lines.append(f"{indent}{out_var} = None")
        return lines
    lines.append(f"{indent}{q_len} = {q}.shape[-2]")
    lines.append(f"{indent}{k_len} = {k}.shape[-2]")
    if window_expr is not None:
        win = emitter._fresh("window")
        window_code = emitter._expr_code(window_expr, env)
        lines.append(f"{indent}{win} = int({window_code})")
    else:
        win = None

    lines.append(f"{indent}{j_idx} = Tensor.arange({k_len}, dtype=dtypes.float32).reshape(1, {k_len})")
    if window_expr is None:
        lines.append(
            f"{indent}{keep} = Tensor.ones({q_len}, {k_len}, dtype=dtypes.bool)"
        )
    else:
        lines.append(f"{indent}if {q_len} == 1:")
        lines.append(f"{indent}    {keep} = ({j_idx} >= ({k_len} - {win}))")
        lines.append(f"{indent}else:")
        lines.append(
            f"{indent}    {i_idx} = Tensor.arange({q_len}, dtype=dtypes.float32).reshape({q_len}, 1)"
        )
        lines.append(f"{indent}    {keep} = ({j_idx} <= {i_idx})")
        lines.append(f"{indent}    {keep} = {keep} & ({j_idx} >= ({i_idx} - {win} + 1))")

    if padding_expr is not None:
        pad_keep = emitter._fresh("pad_keep")
        lines.append(f"{indent}if {padding_expr} is not None:")
        lines.append(f"{indent}    if {padding_expr}.ndim != 2:")
        lines.append(
            f"{indent}        raise ValueError('causal_mask.padding_mask must be rank-2 [batch, seq]')"
        )
        lines.append(f"{indent}    if int({padding_expr}.shape[-1]) != {k_len}:")
        lines.append(
            f"{indent}        raise ValueError('causal_mask.padding_mask width must match key sequence length')"
        )
        lines.append(
            f"{indent}    {pad_keep} = {padding_expr}.cast(dtypes.bool).unsqueeze(1).unsqueeze(1)"
        )
        lines.append(f"{indent}    {keep} = {keep}.unsqueeze(0).unsqueeze(0) & {pad_keep}")
        lines.append(f"{indent}else:")
        lines.append(f"{indent}    {keep} = {keep}.reshape(1, 1, {q_len}, {k_len})")
    else:
        lines.append(f"{indent}{keep} = {keep}.reshape(1, 1, {q_len}, {k_len})")

    mask_val = "-1e9"
    lines.append(
        f"{indent}{out_var} = {keep}.where(Tensor.zeros(1, 1, {q_len}, {k_len}, dtype={q}.dtype), Tensor.full((1, 1, {q_len}, {k_len}), {mask_val}, dtype={q}.dtype))"
    )
    return lines


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
]
