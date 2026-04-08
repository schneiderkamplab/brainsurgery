from __future__ import annotations

from typing import Any

OP_NAME = "reshape_heads"
LOWERING_ARITY = (1, 1)
LOWERING_ALLOWED_KWARGS: set[str] = {"heads", "head_dim"}
LOWERING_REQUIRED_KWARGS: set[str] = set()
LOWERING_KWARG_KINDS: dict[str, Any] = {"heads": "dim", "head_dim": "dim"}


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return False


def lowering_infer_metadata(
    *,
    args: list[str],
    out: str | list[str],
    kwargs: dict[str, Any],
    ctx: Any,
) -> bool:
    if not isinstance(out, str):
        return False
    src_name = args[0].strip() if args else None
    head_dim = kwargs.get("head_dim")
    if head_dim is not None:
        ctx.tensor_last_dim[out] = head_dim
    heads = kwargs.get("heads")
    if heads is not None:
        ctx.tensor_heads[out] = heads
    src_shape = (
        ctx.tensor_shape.get(src_name)
        if isinstance(src_name, str) and src_name.isidentifier()
        else None
    )
    if isinstance(src_shape, tuple) and len(src_shape) == 3:
        batch, seq_len, hidden = src_shape
        out_heads = heads
        out_head_dim = head_dim
        if out_heads is None and out_head_dim is not None:
            out_heads = f"({hidden} / {out_head_dim})"
        if out_head_dim is None and out_heads is not None:
            out_head_dim = f"({hidden} / {out_heads})"
        if out_heads is not None and out_head_dim is not None:
            ctx.tensor_shape[out] = (batch, out_heads, seq_len, out_head_dim)
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
    src = model._read_tensor_input(node_spec.get("_args"), env)
    hidden = int(src.shape[-1])
    raw_heads = node_spec.get("heads")
    raw_head_dim = node_spec.get("head_dim")
    if raw_heads is None and raw_head_dim is None:
        raise ValueError("reshape_heads requires heads or head_dim")
    heads = int(model._eval_expr(raw_heads, env, symbols)) if raw_heads is not None else None
    head_dim = (
        int(model._eval_expr(raw_head_dim, env, symbols)) if raw_head_dim is not None else None
    )
    if heads is None:
        assert head_dim is not None
        if hidden % head_dim != 0:
            raise ValueError("reshape_heads could not infer heads from head_dim")
        heads = hidden // head_dim
    if head_dim is None:
        assert heads is not None
        if hidden % heads != 0:
            raise ValueError("reshape_heads could not infer head_dim from heads")
        head_dim = hidden // heads
    if hidden != heads * head_dim:
        raise ValueError("reshape_heads heads*head_dim must equal input width")
    bsz, seq_len, _ = src.shape
    out = model._require_name(node_spec.get("_bind"), field="reshape_heads._bind")
    env[out] = src.view(bsz, seq_len, heads, head_dim).transpose(1, 2)
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
    raw_heads = node_spec.get("heads")
    raw_head_dim = node_spec.get("head_dim")
    if raw_heads is None and raw_head_dim is None:
        raise ValueError("reshape_heads requires heads or head_dim")
    heads_code = emitter._expr_code(raw_heads, env) if raw_heads is not None else None
    head_dim_code = emitter._expr_code(raw_head_dim, env) if raw_head_dim is not None else None
    heads_var = emitter._fresh("heads")
    head_dim_var = emitter._fresh("head_dim")
    expected_var = emitter._fresh("expected_hidden")
    if heads_code is None:
        lines.append(f"{indent}{heads_var} = None")
    else:
        lines.append(f"{indent}{heads_var} = int({heads_code})")
    if head_dim_code is None:
        lines.append(f"{indent}{head_dim_var} = None")
    else:
        lines.append(f"{indent}{head_dim_var} = int({head_dim_code})")
    lines.append(f"{indent}if {heads_var} is None and {head_dim_var} is None:")
    lines.append(f"{indent}    raise ValueError('reshape_heads requires heads or head_dim')")
    lines.append(f"{indent}if {heads_var} is None:")
    lines.append(f"{indent}    if {src}.shape[-1] % int({head_dim_var}) != 0:")
    lines.append(
        f"{indent}        raise ValueError('reshape_heads could not infer heads from head_dim')"
    )
    lines.append(f"{indent}    {heads_var} = {src}.shape[-1] // int({head_dim_var})")
    lines.append(f"{indent}if {head_dim_var} is None:")
    lines.append(f"{indent}    if {src}.shape[-1] % int({heads_var}) != 0:")
    lines.append(
        f"{indent}        raise ValueError('reshape_heads could not infer head_dim from heads')"
    )
    lines.append(f"{indent}    {head_dim_var} = {src}.shape[-1] // int({heads_var})")
    lines.append(f"{indent}{expected_var} = int({heads_var}) * int({head_dim_var})")
    lines.append(f"{indent}if {src}.shape[-1] != {expected_var}:")
    lines.append(
        f"{indent}    raise ValueError('reshape_heads heads*head_dim must equal input width')"
    )
    out_var = assign_out_var(out_name)
    lines.append(
        f"{indent}{out_var} = {src}.view({src}.shape[0], {src}.shape[1], int({heads_var}), int({head_dim_var})).transpose(1, 2)"
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
    "lowering_infer_metadata",
    "interpret",
    "compile",
    "uses_node_path",
    "LOWERING_TYPE_SIGNATURE",
]
