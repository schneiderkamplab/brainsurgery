from __future__ import annotations

from typing import Any

OP_NAME = "split_qkv_heads"
LOWERING_ARITY = (1, 1)
LOWERING_ALLOWED_KWARGS: set[str] = {"heads", "layout"}
LOWERING_REQUIRED_KWARGS: set[str] = {"heads"}
LOWERING_KWARG_KINDS: dict[str, Any] = {"heads": "dim", "layout": "str"}


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return False


def lowering_known_output_arity(*, kwargs: dict[str, Any]) -> int:
    del kwargs
    return 3


def _layout(node_spec: dict[str, Any]) -> str:
    layout = node_spec.get("layout", "packed")
    if not isinstance(layout, str):
        raise ValueError("split_qkv_heads.layout must be string")
    if layout not in {"packed", "interleaved"}:
        raise ValueError("split_qkv_heads.layout must be one of: packed, interleaved")
    return layout


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
    del node_path_var, scope_var
    src = emitter._read_env_var(env, str(node_spec.get("_args")))
    outs = node_spec.get("_bind")
    if not isinstance(outs, list) or len(outs) != 3:
        raise ValueError("split_qkv_heads expects three outputs [q,k,v]")
    q_out = emitter._assign_out_var(env, str(outs[0]))
    k_out = emitter._assign_out_var(env, str(outs[1]))
    v_out = emitter._assign_out_var(env, str(outs[2]))
    heads_expr = emitter._expr_code(node_spec.get("heads"), env)
    heads = emitter._fresh("heads")
    hidden = emitter._fresh("hidden")
    hd = emitter._fresh("hd")
    bsz = emitter._fresh("bsz")
    seq_len = emitter._fresh("seq_len")
    layout = _layout(node_spec)
    lines: list[str] = [
        f"{indent}if {src}.ndim != 3:",
        f"{indent}    raise ValueError('split_qkv_heads expects rank-3 input [batch, seq, hidden]')",
        f"{indent}{heads} = int({heads_expr})",
        f"{indent}if {heads} <= 0:",
        f"{indent}    raise ValueError('split_qkv_heads heads must be > 0')",
        f"{indent}{bsz} = int({src}.shape[0])",
        f"{indent}{seq_len} = int({src}.shape[1])",
        f"{indent}{hidden} = int({src}.shape[2])",
        f"{indent}if {hidden} % (3 * {heads}) != 0:",
        f"{indent}    raise ValueError('split_qkv_heads {layout} layout requires hidden divisible by 3*heads')",
        f"{indent}{hd} = {hidden} // (3 * {heads})",
    ]
    if layout == "packed":
        q_lin = emitter._fresh("q_lin")
        k_lin = emitter._fresh("k_lin")
        v_lin = emitter._fresh("v_lin")
        lines.extend(
            [
                # TinyGrad: use .chunk() method on tensor, and .reshape() instead of .view()
                f"{indent}{q_lin}, {k_lin}, {v_lin} = {src}.chunk(3, dim=-1)",
                f"{indent}{q_out} = {q_lin}.reshape({bsz}, {seq_len}, {heads}, {hd}).permute(0, 2, 1, 3)",
                f"{indent}{k_out} = {k_lin}.reshape({bsz}, {seq_len}, {heads}, {hd}).permute(0, 2, 1, 3)",
                f"{indent}{v_out} = {v_lin}.reshape({bsz}, {seq_len}, {heads}, {hd}).permute(0, 2, 1, 3)",
            ]
        )
    else:
        qkv = emitter._fresh("qkv")
        lines.extend(
            [
                # TinyGrad: use .reshape() instead of .view()
                f"{indent}{qkv} = {src}.reshape({bsz}, {seq_len}, {heads}, 3, {hd})",
                f"{indent}{q_out} = {qkv}[..., 0, :].permute(0, 2, 1, 3)",
                f"{indent}{k_out} = {qkv}[..., 1, :].permute(0, 2, 1, 3)",
                f"{indent}{v_out} = {qkv}[..., 2, :].permute(0, 2, 1, 3)",
            ]
        )
    return lines


__all__ = [
    "LOWERING_ARITY",
    "LOWERING_ALLOWED_KWARGS",
    "LOWERING_REQUIRED_KWARGS",
    "LOWERING_KWARG_KINDS",
    "OP_NAME",
    "lowering_known_output_arity",
    "interpret",
    "compile",
    "uses_node_path",
]
