from __future__ import annotations

from typing import Any

OP_NAME = "repeat"
LOWERING_ARITY = (2, 3)
LOWERING_ALLOWED_KWARGS: set[str] = set()
LOWERING_REQUIRED_KWARGS: set[str] = set()
LOWERING_KWARG_KINDS: dict[str, Any] = {}


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return False


def lowering_normalize_kwargs(
    *,
    args: list[str],
    out: str | list[str],
    kwargs: dict[str, Any],
    ctx: Any,
) -> None:
    del kwargs, out, ctx
    dim_literal: int | None = 1
    if len(args) >= 3:
        dim_expr = args[2]
        if isinstance(dim_expr, str):
            raw = dim_expr.strip()
            try:
                dim_literal = int(raw)
            except ValueError:
                dim_literal = None
        elif isinstance(dim_expr, int):
            dim_literal = int(dim_expr)
        del args[2:]
    if dim_literal is not None and dim_literal != 1:
        raise ValueError("repeat currently supports only dim=1 (head axis)")


def lowering_infer_metadata(
    *,
    args: list[str],
    out: str | list[str],
    kwargs: dict[str, Any],
    ctx: Any,
) -> bool:
    if not isinstance(out, str):
        return False
    src_name: str | None = None
    # Preserve known heads metadata through repeat.
    if args:
        src_name = args[0].strip()
    if isinstance(src_name, str) and src_name.isidentifier() and src_name in ctx.tensor_heads:
        src_heads = ctx.tensor_heads[src_name]
        repeats = kwargs.get("repeats")
        ctx.tensor_heads[out] = f"({src_heads} * {repeats})" if repeats is not None else src_heads
    if isinstance(src_name, str) and src_name.isidentifier():
        if src_name in ctx.tensor_last_dim:
            ctx.tensor_last_dim[out] = ctx.tensor_last_dim[src_name]
        src_shape = ctx.tensor_shape.get(src_name)
        if isinstance(src_shape, tuple) and len(src_shape) == 4:
            batch, heads, seq_len, head_dim = src_shape
            repeats = kwargs.get("repeats")
            if repeats is not None:
                ctx.tensor_shape[out] = (batch, f"({heads} * {repeats})", seq_len, head_dim)
            else:
                ctx.tensor_shape[out] = src_shape
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
    raw_args = node_spec.get("_args")
    if not isinstance(raw_args, list) or len(raw_args) < 2:
        raise ValueError("repeat requires positional args: x repeats [dim]")
    src = model._read_tensor_input(raw_args[0], env)
    n_rep = int(model._eval_expr(raw_args[1], env, symbols))
    dim = int(model._eval_expr(raw_args[2], env, symbols)) if len(raw_args) >= 3 else 1
    if dim != 1:
        raise ValueError("repeat currently supports only dim=1 (head axis)")
    out = model._require_name(node_spec.get("_bind"), field="repeat._bind")
    if n_rep == 1:
        env[out] = src
    else:
        bsz, kvh, seq_len, hd = src.shape
        expanded = src[:, :, None, :, :].expand(bsz, kvh, n_rep, seq_len, hd)
        env[out] = expanded.reshape(bsz, kvh * n_rep, seq_len, hd)
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

    def read(name: str) -> str:
        return emitter._read_env_var(env, name)

    raw_args = node_spec.get("_args")
    if not isinstance(raw_args, list) or len(raw_args) < 2:
        raise ValueError("repeat requires positional args: x repeats [dim]")
    src = read(str(raw_args[0]))
    out_name = str(node_spec.get("_bind"))
    out_var = assign_out_var(out_name)
    repeats_code = emitter._expr_code(raw_args[1], env)
    dim_code = emitter._expr_code(raw_args[2], env) if len(raw_args) >= 3 else "1"
    dim_var = emitter._fresh("dim")
    n_rep = emitter._fresh("n_rep")
    lines.append(f"{indent}{dim_var} = int({dim_code})")
    lines.append(f"{indent}if {dim_var} != 1:")
    lines.append(
        f"{indent}    raise ValueError('repeat currently supports only dim=1 (head axis)')"
    )
    lines.append(f"{indent}{n_rep} = int({repeats_code})")
    lines.append(f"{indent}if {n_rep} == 1:")
    lines.append(f"{indent}    {out_var} = {src}")
    lines.append(f"{indent}else:")
    lines.append(
        f"{indent}    {out_var} = {src}[:, :, None, :, :].expand({src}.shape[0], {src}.shape[1], {n_rep}, {src}.shape[2], {src}.shape[3]).reshape({src}.shape[0], {src}.shape[1] * {n_rep}, {src}.shape[2], {src}.shape[3])"
    )
    return lines


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any", "Any", "Any"),
    "kwargs": dict(LOWERING_KWARG_KINDS),
    "returns": ("Tensor",),
}

__all__ = [
    "OP_NAME",
    "LOWERING_ARITY",
    "LOWERING_ALLOWED_KWARGS",
    "LOWERING_REQUIRED_KWARGS",
    "LOWERING_KWARG_KINDS",
    "lowering_normalize_kwargs",
    "lowering_infer_metadata",
    "interpret",
    "compile",
    "uses_node_path",
    "LOWERING_TYPE_SIGNATURE",
]
