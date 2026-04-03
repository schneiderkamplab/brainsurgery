from __future__ import annotations

from typing import Any

OP_NAME = "split_qkv_grouped"
LOWERING_ARITY = (1, 1)
LOWERING_ALLOWED_KWARGS: set[str] = {"heads", "kv_heads", "head_dim"}
LOWERING_REQUIRED_KWARGS: set[str] = {"heads", "kv_heads"}
LOWERING_KWARG_KINDS: dict[str, Any] = {
    "heads": "dim",
    "kv_heads": "dim",
    "head_dim": "dim",
}


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return False


def lowering_known_output_arity(*, kwargs: dict[str, Any]) -> int:
    del kwargs
    return 3


def _resolve_positive_int(
    model: Any, expr: Any, env: dict[str, Any], symbols: dict[str, int], name: str
) -> int:
    value = int(model._eval_expr(expr, env, symbols))
    if value <= 0:
        raise ValueError(f"split_qkv_grouped {name} must be > 0")
    return value


def interpret(
    model: Any,
    node_spec: dict[str, Any],
    env: dict[str, Any],
    *,
    node_path: str,
    scope: str,
    symbols: dict[str, int],
) -> None:
    del node_path, scope
    src = model._read_tensor_input(node_spec.get("_args"), env)
    outs = node_spec.get("_bind")
    if not isinstance(outs, list) or len(outs) != 3:
        raise ValueError("split_qkv_grouped expects three outputs [q,k,v]")
    if src.ndim != 3:
        raise ValueError("split_qkv_grouped expects rank-3 input [batch, seq, hidden]")

    heads = _resolve_positive_int(model, node_spec.get("heads"), env, symbols, "heads")
    kv_heads = _resolve_positive_int(model, node_spec.get("kv_heads"), env, symbols, "kv_heads")
    if heads % kv_heads != 0:
        raise ValueError("split_qkv_grouped requires heads to be divisible by kv_heads")
    q_per_kv = heads // kv_heads

    bsz, seq_len, hidden = int(src.shape[0]), int(src.shape[1]), int(src.shape[2])
    expected_parts = kv_heads * (q_per_kv + 2)
    if expected_parts <= 0 or hidden % expected_parts != 0:
        raise ValueError(
            "split_qkv_grouped hidden must be divisible by kv_heads * (heads/kv_heads + 2)"
        )
    inferred_head_dim = hidden // expected_parts
    head_dim_expr = node_spec.get("head_dim")
    if head_dim_expr is not None:
        head_dim = _resolve_positive_int(model, head_dim_expr, env, symbols, "head_dim")
        if head_dim != inferred_head_dim:
            raise ValueError(
                "split_qkv_grouped head_dim does not match hidden/(kv_heads*(heads/kv_heads+2))"
            )
    else:
        head_dim = inferred_head_dim

    grouped = src.view(bsz, seq_len, kv_heads, q_per_kv + 2, head_dim)
    q = grouped[:, :, :, :q_per_kv, :].reshape(bsz, seq_len, heads, head_dim).permute(0, 2, 1, 3)
    k = grouped[:, :, :, -2, :].reshape(bsz, seq_len, kv_heads, head_dim).permute(0, 2, 1, 3)
    v = grouped[:, :, :, -1, :].reshape(bsz, seq_len, kv_heads, head_dim).permute(0, 2, 1, 3)

    env[str(outs[0])] = q
    env[str(outs[1])] = k
    env[str(outs[2])] = v


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
        raise ValueError("split_qkv_grouped expects three outputs [q,k,v]")
    q_out = emitter._assign_out_var(env, str(outs[0]))
    k_out = emitter._assign_out_var(env, str(outs[1]))
    v_out = emitter._assign_out_var(env, str(outs[2]))

    heads_expr = emitter._expr_code(node_spec.get("heads"), env)
    kv_heads_expr = emitter._expr_code(node_spec.get("kv_heads"), env)
    head_dim_expr = node_spec.get("head_dim")
    head_dim_code = emitter._expr_code(head_dim_expr, env) if head_dim_expr is not None else None

    heads = emitter._fresh("heads")
    kv_heads = emitter._fresh("kv_heads")
    q_per_kv = emitter._fresh("q_per_kv")
    hidden = emitter._fresh("hidden")
    expected_parts = emitter._fresh("expected_parts")
    inferred_hd = emitter._fresh("inferred_hd")
    head_dim = emitter._fresh("head_dim")
    bsz = emitter._fresh("bsz")
    seq_len = emitter._fresh("seq_len")
    grouped = emitter._fresh("grouped_qkv")

    lines: list[str] = [
        f"{indent}if {src}.ndim != 3:",
        f"{indent}    raise ValueError('split_qkv_grouped expects rank-3 input [batch, seq, hidden]')",
        f"{indent}{heads} = int({heads_expr})",
        f"{indent}if {heads} <= 0:",
        f"{indent}    raise ValueError('split_qkv_grouped heads must be > 0')",
        f"{indent}{kv_heads} = int({kv_heads_expr})",
        f"{indent}if {kv_heads} <= 0:",
        f"{indent}    raise ValueError('split_qkv_grouped kv_heads must be > 0')",
        f"{indent}if ({heads} % {kv_heads}) != 0:",
        f"{indent}    raise ValueError('split_qkv_grouped requires heads to be divisible by kv_heads')",
        f"{indent}{q_per_kv} = {heads} // {kv_heads}",
        f"{indent}{hidden} = int({src}.shape[2])",
        f"{indent}{expected_parts} = {kv_heads} * ({q_per_kv} + 2)",
        f"{indent}if {expected_parts} <= 0 or ({hidden} % {expected_parts}) != 0:",
        f"{indent}    raise ValueError('split_qkv_grouped hidden must be divisible by kv_heads * (heads/kv_heads + 2)')",
        f"{indent}{inferred_hd} = {hidden} // {expected_parts}",
    ]

    if head_dim_code is not None:
        lines.extend(
            [
                f"{indent}{head_dim} = int({head_dim_code})",
                f"{indent}if {head_dim} <= 0:",
                f"{indent}    raise ValueError('split_qkv_grouped head_dim must be > 0')",
                f"{indent}if {head_dim} != {inferred_hd}:",
                f"{indent}    raise ValueError('split_qkv_grouped head_dim does not match hidden/(kv_heads*(heads/kv_heads+2))')",
            ]
        )
    else:
        lines.append(f"{indent}{head_dim} = {inferred_hd}")

    lines.extend(
        [
            f"{indent}{bsz} = int({src}.shape[0])",
            f"{indent}{seq_len} = int({src}.shape[1])",
            f"{indent}{grouped} = {src}.view({bsz}, {seq_len}, {kv_heads}, {q_per_kv} + 2, {head_dim})",
            f"{indent}{q_out} = {grouped}[:, :, :, :{q_per_kv}, :].reshape({bsz}, {seq_len}, {heads}, {head_dim}).permute(0, 2, 1, 3)",
            f"{indent}{k_out} = {grouped}[:, :, :, -2, :].reshape({bsz}, {seq_len}, {kv_heads}, {head_dim}).permute(0, 2, 1, 3)",
            f"{indent}{v_out} = {grouped}[:, :, :, -1, :].reshape({bsz}, {seq_len}, {kv_heads}, {head_dim}).permute(0, 2, 1, 3)",
        ]
    )
    return lines


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any",),
    "kwargs": dict(LOWERING_KWARG_KINDS),
    "returns": ("Tensor", "Tensor", "Tensor"),
}

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
    "LOWERING_TYPE_SIGNATURE",
]
