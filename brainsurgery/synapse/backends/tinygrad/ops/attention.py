from __future__ import annotations

from typing import Any

OP_NAME = "attention"
LOWERING_ARITY = (3, 3)
LOWERING_ALLOWED_KWARGS: set[str] = {"scale", "mask", "causal", "eager", "sink", "sink_path"}
LOWERING_REQUIRED_KWARGS: set[str] = set()
LOWERING_KWARG_KINDS: dict[str, Any] = {
    "causal": "bool",
    "mask": "str",
    "scale": "number",
    "eager": "bool",
    "sink": "str",
    "sink_path": "str",
}


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return True


def lowering_validate_signature(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> None:
    del args, kwargs, ctx
    if isinstance(out, list):
        raise ValueError("attention requires a single scalar output binding")


def lowering_infer_metadata(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> bool:
    del kwargs
    if isinstance(out, list) or not args:
        return False
    source_name = str(args[0]).strip()
    if source_name in ctx.tensor_last_dim:
        ctx.tensor_last_dim[out] = ctx.tensor_last_dim[source_name]
    if source_name in ctx.tensor_shape:
        ctx.tensor_shape[out] = ctx.tensor_shape[source_name]
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
    del scope_var
    lines: list[str] = []

    def assign_out_var(out_name: str) -> str:
        return emitter._assign_out_var(env, out_name)

    def read(name: str) -> str:
        return emitter._read_env_var(env, name)

    ins = node_spec.get("_args")
    if not isinstance(ins, list) or len(ins) != 3:
        raise ValueError("attention expects 3 inputs")
    q = read(str(ins[0]))
    k = read(str(ins[1]))
    v = read(str(ins[2]))
    out_name = str(node_spec.get("_bind"))
    out_var = assign_out_var(out_name)

    mask_name = node_spec.get("mask")
    mask_expr = "None"
    if isinstance(mask_name, str) and mask_name in env:
        mask_expr = env[mask_name]
    mask_for_sdpa = mask_expr

    sink_name = node_spec.get("sink")
    sink_expr = env[sink_name] if isinstance(sink_name, str) and sink_name in env else None
    if sink_expr is None and isinstance(node_spec.get("sink_path"), str):
        sink_scope_expr = f"self._scope_of({node_path_var})"
        sink_path_expr = f"self._join_scope({sink_scope_expr}, {node_spec['sink_path']!r})"
        sink_expr = f"self._state.get({sink_path_expr})"

    scale_value = node_spec.get("scale")
    scale_expr = "None" if scale_value is None else emitter._expr_code(scale_value, env)

    if bool(node_spec.get("causal", True)):
        is_causal = f"({q}.shape[-2] > 1 and {mask_for_sdpa} is None)"
    else:
        is_causal = "False"

    if sink_expr is None:
        # No sink: use SDPA or eager fallback
        use_eager = emitter._fresh("use_eager_attn")
        eager_kw = node_spec.get("eager")
        if eager_kw is None:
            lines.append(
                f"{indent}{use_eager} = bool(getattr(self, '_hf_align_attention_eager', False))"
            )
        else:
            eager_expr = emitter._expr_code(eager_kw, env)
            lines.append(f"{indent}{use_eager} = bool({eager_expr})")

        attn_scores = emitter._fresh("attn_scores")
        attn_probs = emitter._fresh("attn_probs")

        lines.append(f"{indent}if {use_eager}:")
        lines.append(f"{indent}    _scale = {scale_expr}")
        lines.append(f"{indent}    if _scale is None:")
        lines.append(f"{indent}        _scale = float({q}.shape[-1]) ** -0.5")
        lines.append(
            f"{indent}    {attn_scores} = {q}.matmul({k}.transpose(-2, -1)) * float(_scale)"
        )
        if mask_for_sdpa != "None":
            lines.append(f"{indent}    if {mask_for_sdpa} is not None:")
            lines.append(f"{indent}        {attn_scores} = {attn_scores} + {mask_for_sdpa}")
        lines.append(
            f"{indent}    {attn_probs} = {attn_scores}.softmax(axis=-1)"
        )
        lines.append(f"{indent}    {out_var} = {attn_probs}.matmul({v})")
        lines.append(f"{indent}else:")
        lines.append(
            f"{indent}    {out_var} = {q}.scaled_dot_product_attention({k}, {v}, attn_mask={mask_for_sdpa}, is_causal={is_causal}, scale={scale_expr})"
        )
        return lines

    # Sink path
    sink_var = assign_out_var(f"{out_name}_sink")
    attn_logits_var = assign_out_var(f"{out_name}_logits")
    sinks_var = assign_out_var(f"{out_name}_sinks")
    combined_var = assign_out_var(f"{out_name}_combined")
    probs_var = assign_out_var(f"{out_name}_probs")
    scores_var = assign_out_var(f"{out_name}_scores")

    lines.append(f"{indent}{sink_var} = {sink_expr}")
    lines.append(f"{indent}if {sink_var} is None:")
    lines.append(
        f"{indent}    {out_var} = {q}.scaled_dot_product_attention({k}, {v}, attn_mask={mask_for_sdpa}, is_causal={is_causal}, scale={scale_expr})"
    )
    lines.append(f"{indent}else:")
    lines.append(f"{indent}    _scale = {scale_expr}")
    lines.append(f"{indent}    if _scale is None:")
    lines.append(f"{indent}        _scale = float({q}.shape[-1]) ** -0.5")
    lines.append(f"{indent}    {attn_logits_var} = ({q} @ {k}.transpose(-2, -1)) * float(_scale)")
    lines.append(f"{indent}    if {mask_for_sdpa} is not None:")
    lines.append(f"{indent}        {attn_logits_var} = {attn_logits_var} + {mask_for_sdpa}")
    lines.append(
        f"{indent}    {sinks_var} = {sink_var}.reshape(1, -1, 1, 1).expand({q}.shape[0], -1, {q}.shape[-2], -1)"
    )
    lines.append(
        f"{indent}    {combined_var} = {attn_logits_var}.cat({sinks_var}, dim=-1)"
    )
    lines.append(
        f"{indent}    {combined_var} = {combined_var} - {combined_var}.max(axis=-1, keepdims=True)"
    )
    lines.append(
        f"{indent}    {probs_var} = {combined_var}.softmax(axis=-1)"
    )
    lines.append(f"{indent}    {scores_var} = {probs_var}[..., :-1]")
    lines.append(f"{indent}    {out_var} = {scores_var}.cast({v}.dtype) @ {v}")
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
