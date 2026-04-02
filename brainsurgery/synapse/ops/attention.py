from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

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


def _normalize_mask_contract(model: Any, mask: torch.Tensor | None) -> torch.Tensor | None:
    if not torch.is_tensor(mask):
        return mask
    if not bool(getattr(model, "_hf_align_mask_contract", False)):
        return mask
    if not (mask.is_floating_point() and mask.numel() > 0):
        return mask
    mask_max = float(mask.max())
    mask_min = float(mask.min())
    mask_floor = float(torch.finfo(mask.dtype).min)
    if mask_max == 0.0 and mask_min <= (0.5 * mask_floor):
        return mask == 0
    return mask


def interpret(
    model: Any,
    node_spec: dict[str, Any],
    env: dict[str, Any],
    *,
    node_path: str,
    scope: str,
    symbols: dict[str, int],
) -> None:
    del scope
    ins = node_spec.get("_args")
    if not isinstance(ins, list) or len(ins) != 3:
        raise ValueError("attention expects [q, k, v]")
    q = env[ins[0]]
    k_tensor = env[ins[1]]
    v = env[ins[2]]

    mask = None
    mask_name = node_spec.get("mask")
    if isinstance(mask_name, str):
        mask = env.get(mask_name)
    if torch.is_tensor(mask):
        mask = _normalize_mask_contract(model, mask)

    sink_name = node_spec.get("sink")
    sink = env.get(sink_name) if isinstance(sink_name, str) else None
    if sink is None and isinstance(node_spec.get("sink_path"), str):
        sink_scope = model._scope_of(node_path)
        sink_path = model._join_scope(sink_scope, str(node_spec["sink_path"]))
        sink = model._state.get(sink_path)

    scale_expr = node_spec.get("scale")
    scale_value = None if scale_expr is None else float(model._eval_expr(scale_expr, env, symbols))
    if scale_value is None:
        scale_value = float(q.shape[-1]) ** -0.5

    causal_flag = bool(node_spec.get("causal", True))
    eager_kw = node_spec.get("eager")
    use_eager = (
        bool(eager_kw)
        if eager_kw is not None
        else bool(getattr(model, "_hf_align_attention_eager", False))
    )

    if sink is not None:
        attn_logits = (q @ k_tensor.transpose(-2, -1)) * float(scale_value)
        if mask is not None:
            attn_logits = attn_logits + mask
        elif causal_flag:
            q_len = int(q.shape[-2])
            k_len = int(k_tensor.shape[-2])
            causal = torch.ones((q_len, k_len), dtype=torch.bool, device=q.device).tril(
                diagonal=k_len - q_len
            )
            floor = torch.finfo(attn_logits.dtype).min
            attn_logits = attn_logits.masked_fill(~causal, floor)
        sinks = sink.reshape(1, -1, 1, 1).expand(q.shape[0], -1, q.shape[-2], -1)
        combined_logits = torch.cat([attn_logits, sinks], dim=-1)
        combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
        probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
        scores = probs[..., :-1]
        attn_out = scores.to(v.dtype) @ v
    elif use_eager:
        attn_scores = torch.matmul(q, k_tensor.transpose(-2, -1)) * float(scale_value)
        keep_mask: torch.Tensor | None = None
        if causal_flag:
            q_len = int(q.shape[-2])
            k_len = int(k_tensor.shape[-2])
            causal_mask = torch.ones((q_len, k_len), dtype=torch.bool, device=q.device).tril(
                diagonal=k_len - q_len
            )
            keep_mask = causal_mask
            floor = torch.finfo(attn_scores.dtype).min
            attn_scores = attn_scores.masked_fill(~causal_mask, floor)
        if mask is not None:
            if torch.is_tensor(mask):
                if mask.dtype == torch.bool:
                    keep_from_mask = mask
                elif mask.is_floating_point():
                    keep_from_mask = mask == 0
                else:
                    keep_from_mask = None
                if torch.is_tensor(keep_from_mask):
                    keep_mask = (
                        keep_from_mask if keep_mask is None else (keep_mask & keep_from_mask)
                    )
            attn_scores = attn_scores + mask
        attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)
        if torch.is_tensor(keep_mask):
            has_valid = keep_mask.any(dim=-1, keepdim=True)
            attn_probs = torch.where(has_valid, attn_probs, torch.zeros_like(attn_probs))
        attn_out = torch.matmul(attn_probs, v)
    else:
        is_causal_flag = causal_flag and q.shape[2] > 1 and mask is None
        attn_out = F.scaled_dot_product_attention(
            q,
            k_tensor,
            v,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=is_causal_flag,
            scale=scale_value,
        )

    out_name = model._require_name(node_spec.get("_bind"), field="attention._bind")
    env[out_name] = attn_out


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
    if mask_expr != "None":
        normalized_mask = emitter._fresh("mask_for_sdpa")
        mask_max = emitter._fresh("mask_max")
        mask_min = emitter._fresh("mask_min")
        mask_floor = emitter._fresh("mask_floor")
        lines.append(f"{indent}{normalized_mask} = {mask_expr}")
        lines.append(f"{indent}if getattr(self, '_hf_align_mask_contract', False):")
        lines.append(
            f"{indent}    if torch.is_tensor({normalized_mask}) and {normalized_mask}.is_floating_point() and {normalized_mask}.numel() > 0:"
        )
        lines.append(f"{indent}        {mask_max} = float({normalized_mask}.max())")
        lines.append(f"{indent}        {mask_min} = float({normalized_mask}.min())")
        lines.append(
            f"{indent}        {mask_floor} = float(torch.finfo({normalized_mask}.dtype).min)"
        )
        lines.append(
            f"{indent}        if {mask_max} == 0.0 and {mask_min} <= (0.5 * {mask_floor}):"
        )
        lines.append(f"{indent}            {normalized_mask} = ({normalized_mask} == 0)")
        mask_for_sdpa = normalized_mask

    sink_name = node_spec.get("sink")
    sink_expr = env[sink_name] if isinstance(sink_name, str) and sink_name in env else None
    if sink_expr is None and isinstance(node_spec.get("sink_path"), str):
        sink_scope_expr = f"self._scope_of({node_path_var})"
        sink_path_expr = f"self._join_scope({sink_scope_expr}, {node_spec['sink_path']!r})"
        sink_expr = f"self._state.get({sink_path_expr})"

    scale_value = node_spec.get("scale")
    scale_expr = "None" if scale_value is None else emitter._expr_code(scale_value, env)
    use_eager = emitter._fresh("use_eager_attn")
    eager_kw = node_spec.get("eager")
    if eager_kw is None:
        lines.append(
            f"{indent}{use_eager} = bool(getattr(self, '_hf_align_attention_eager', False))"
        )
    else:
        eager_expr = emitter._expr_code(eager_kw, env)
        lines.append(f"{indent}{use_eager} = bool({eager_expr})")

    if bool(node_spec.get("causal", True)):
        is_causal = f"({q}.shape[-2] > 1 and {mask_for_sdpa} is None)"
    else:
        is_causal = "False"

    if sink_expr is None:
        attn_scores = emitter._fresh("attn_scores")
        attn_probs = emitter._fresh("attn_probs")
        keep_mask = emitter._fresh("keep_mask")
        keep_from_mask = emitter._fresh("keep_from_mask")
        has_valid = emitter._fresh("has_valid")
        lines.append(f"{indent}if {use_eager}:")
        lines.append(f"{indent}    _scale = {scale_expr}")
        lines.append(f"{indent}    if _scale is None:")
        lines.append(f"{indent}        _scale = float({q}.shape[-1]) ** -0.5")
        lines.append(
            f"{indent}    {attn_scores} = torch.matmul({q}, {k}.transpose(-2, -1)) * float(_scale)"
        )
        lines.append(f"{indent}    {keep_mask} = None")
        if bool(node_spec.get("causal", True)):
            q_len = emitter._fresh("q_len")
            k_len = emitter._fresh("k_len")
            causal_mask = emitter._fresh("causal_mask")
            floor = emitter._fresh("score_floor")
            lines.append(f"{indent}    {q_len} = int({q}.shape[-2])")
            lines.append(f"{indent}    {k_len} = int({k}.shape[-2])")
            lines.append(
                f"{indent}    {causal_mask} = torch.ones(({q_len}, {k_len}), dtype=torch.bool, device={q}.device).tril(diagonal={k_len} - {q_len})"
            )
            lines.append(f"{indent}    {keep_mask} = {causal_mask}")
            lines.append(f"{indent}    {floor} = torch.finfo({attn_scores}.dtype).min")
            lines.append(
                f"{indent}    {attn_scores} = {attn_scores}.masked_fill(~{causal_mask}, {floor})"
            )
        lines.append(f"{indent}    if {mask_for_sdpa} is not None:")
        lines.append(f"{indent}        if torch.is_tensor({mask_for_sdpa}):")
        lines.append(f"{indent}            if {mask_for_sdpa}.dtype == torch.bool:")
        lines.append(f"{indent}                {keep_from_mask} = {mask_for_sdpa}")
        lines.append(f"{indent}            elif {mask_for_sdpa}.is_floating_point():")
        lines.append(f"{indent}                {keep_from_mask} = ({mask_for_sdpa} == 0)")
        lines.append(f"{indent}            else:")
        lines.append(f"{indent}                {keep_from_mask} = None")
        lines.append(f"{indent}            if torch.is_tensor({keep_from_mask}):")
        lines.append(
            f"{indent}                {keep_mask} = {keep_from_mask} if {keep_mask} is None else ({keep_mask} & {keep_from_mask})"
        )
        lines.append(f"{indent}        {attn_scores} = {attn_scores} + {mask_for_sdpa}")
        lines.append(
            f"{indent}    {attn_probs} = F.softmax({attn_scores}, dim=-1, dtype=torch.float32).to({q}.dtype)"
        )
        lines.append(f"{indent}    if torch.is_tensor({keep_mask}):")
        lines.append(f"{indent}        {has_valid} = {keep_mask}.any(dim=-1, keepdim=True)")
        lines.append(
            f"{indent}        {attn_probs} = torch.where({has_valid}, {attn_probs}, torch.zeros_like({attn_probs}))"
        )
        lines.append(f"{indent}    {out_var} = torch.matmul({attn_probs}, {v})")
        lines.append(f"{indent}else:")
        lines.append(
            f"{indent}    {out_var} = F.scaled_dot_product_attention({q}, {k}, {v}, attn_mask={mask_for_sdpa}, dropout_p=0.0, is_causal={is_causal}, scale={scale_expr})"
        )
        return lines

    sink_var = assign_out_var(f"{out_name}_sink")
    attn_logits_var = assign_out_var(f"{out_name}_logits")
    sinks_var = assign_out_var(f"{out_name}_sinks")
    combined_var = assign_out_var(f"{out_name}_combined")
    probs_var = assign_out_var(f"{out_name}_probs")
    scores_var = assign_out_var(f"{out_name}_scores")

    lines.append(f"{indent}{sink_var} = {sink_expr}")
    lines.append(f"{indent}if {sink_var} is None:")
    lines.append(
        f"{indent}    {out_var} = F.scaled_dot_product_attention({q}, {k}, {v}, attn_mask={mask_for_sdpa}, dropout_p=0.0, is_causal={is_causal}, scale={scale_expr})"
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
        f"{indent}    {combined_var} = torch.cat([{attn_logits_var}, {sinks_var}], dim=-1)"
    )
    lines.append(
        f"{indent}    {combined_var} = {combined_var} - {combined_var}.max(dim=-1, keepdim=True).values"
    )
    lines.append(
        f"{indent}    {probs_var} = F.softmax({combined_var}, dim=-1, dtype={combined_var}.dtype)"
    )
    lines.append(f"{indent}    {scores_var} = {probs_var}[..., :-1]")
    lines.append(f"{indent}    {out_var} = {scores_var}.to({v}.dtype) @ {v}")
    return lines


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any", "Any", "Any"),
    "kwargs": dict(LOWERING_KWARG_KINDS),
    "returns": "dynamic",
}

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
    "LOWERING_TYPE_SIGNATURE",
]
