from __future__ import annotations

from typing import Any

import torch

OP_NAME = "bidirectional_mask"
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
        raise ValueError("bidirectional_mask requires a single scalar output binding")


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


def _to_keep_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.dtype == torch.bool:
        return mask
    if mask.is_floating_point():
        if mask.numel() == 0:
            return mask.to(torch.bool)
        mask_max = float(mask.max())
        mask_min = float(mask.min())
        mask_floor = float(torch.finfo(mask.dtype).min)
        if mask_max == 0.0 and mask_min <= (0.5 * mask_floor):
            return mask == 0
        return mask != 0
    return mask.to(torch.bool)


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
    raw_args = node_spec.get("_args")
    if not isinstance(raw_args, list) or len(raw_args) != 2:
        raise ValueError("bidirectional_mask expects exactly 2 positional args: query and key")
    q = model._read_tensor_input(raw_args[0], env)
    key_tensor = model._read_tensor_input(raw_args[1], env)
    if not torch.is_tensor(q) or not torch.is_tensor(key_tensor):
        raise ValueError("bidirectional_mask expects tensor inputs for query and key")
    if q.ndim != 4 or key_tensor.ndim != 4:
        raise ValueError("bidirectional_mask expects rank-4 q/k tensors [B,H,S,HD]")
    if int(q.shape[0]) != int(key_tensor.shape[0]):
        raise ValueError("bidirectional_mask query/key batch size must match")

    padding_ref = node_spec.get("padding_mask")
    padding_mask = env.get(padding_ref) if isinstance(padding_ref, str) else None
    if padding_mask is not None and not torch.is_tensor(padding_mask):
        raise ValueError("bidirectional_mask.padding_mask must resolve to tensor or null")

    window_expr = node_spec.get("window")
    if window_expr is None and padding_mask is None:
        out_name = model._require_name(node_spec.get("_bind"), field="bidirectional_mask._bind")
        env[out_name] = None
        return

    q_len = int(q.shape[-2])
    k_len = int(key_tensor.shape[-2])
    q_idx = torch.arange(q_len, device=q.device).unsqueeze(1)
    k_idx = torch.arange(k_len, device=q.device).unsqueeze(0)

    if window_expr is None:
        keep = torch.ones((q_len, k_len), dtype=torch.bool, device=q.device)
    else:
        window_value = int(model._eval_expr(window_expr, env, symbols))
        if window_value <= 0:
            raise ValueError("bidirectional_mask.window must be > 0")
        if q_len == k_len:
            keep = torch.abs(q_idx - k_idx) <= window_value
        else:
            q_aligned = q_idx + (k_len - q_len)
            keep = torch.abs(q_aligned - k_idx) <= window_value

    if padding_mask is not None:
        if padding_mask.ndim != 2:
            raise ValueError("bidirectional_mask.padding_mask must be rank-2 [batch, seq]")
        if int(padding_mask.shape[0]) != int(q.shape[0]):
            raise ValueError("bidirectional_mask.padding_mask batch size must match query batch")
        if int(padding_mask.shape[1]) < k_len:
            raise ValueError("bidirectional_mask.padding_mask width must be >= key sequence length")
        padding_keep = _to_keep_mask(padding_mask[:, -k_len:]).to(device=q.device)
        keep = keep.unsqueeze(0).unsqueeze(0) & padding_keep.unsqueeze(1).unsqueeze(1)
    else:
        keep = keep.unsqueeze(0).unsqueeze(0)

    out_name = model._require_name(node_spec.get("_bind"), field="bidirectional_mask._bind")
    mask_value = torch.finfo(q.dtype).min
    env[out_name] = torch.where(
        keep,
        torch.zeros((), dtype=q.dtype, device=q.device),
        torch.full((), mask_value, dtype=q.dtype, device=q.device),
    )
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
    del node_path_var, scope_var
    lines: list[str] = []

    def assign_out_var(out_name: str) -> str:
        return emitter._assign_out_var(env, out_name)

    def read(name: str) -> str:
        return emitter._read_env_var(env, name)

    raw_args = node_spec.get("_args")
    if not isinstance(raw_args, list) or len(raw_args) != 2:
        raise ValueError("bidirectional_mask expects exactly 2 positional args: query and key")

    q = read(str(raw_args[0]))
    k = read(str(raw_args[1]))
    out_var = assign_out_var(str(node_spec.get("_bind")))
    window_expr = node_spec.get("window")
    padding_name = node_spec.get("padding_mask")
    padding_expr = (
        env.get(padding_name) if isinstance(padding_name, str) and padding_name in env else None
    )

    if window_expr is None and padding_expr is None:
        lines.append(f"{indent}{out_var} = None")
        return lines

    q_len = emitter._fresh("q_len")
    k_len = emitter._fresh("k_len")
    q_idx = emitter._fresh("q_idx")
    k_idx = emitter._fresh("k_idx")
    keep = emitter._fresh("keep")
    fill = emitter._fresh("mask_fill")
    lines.append(f"{indent}if {q}.ndim != 4 or {k}.ndim != 4:")
    lines.append(
        f"{indent}    raise ValueError('bidirectional_mask expects rank-4 q/k tensors [B,H,S,HD]')"
    )
    lines.append(f"{indent}if int({q}.shape[0]) != int({k}.shape[0]):")
    lines.append(
        f"{indent}    raise ValueError('bidirectional_mask query/key batch size must match')"
    )
    lines.append(f"{indent}{q_len} = int({q}.shape[-2])")
    lines.append(f"{indent}{k_len} = int({k}.shape[-2])")
    lines.append(f"{indent}{q_idx} = torch.arange({q_len}, device={q}.device).unsqueeze(1)")
    lines.append(f"{indent}{k_idx} = torch.arange({k_len}, device={q}.device).unsqueeze(0)")

    if window_expr is None:
        lines.append(
            f"{indent}{keep} = torch.ones(({q_len}, {k_len}), dtype=torch.bool, device={q}.device)"
        )
    else:
        window = emitter._fresh("window")
        window_code = emitter._expr_code(window_expr, env)
        q_aligned = emitter._fresh("q_aligned")
        lines.append(f"{indent}{window} = int({window_code})")
        lines.append(f"{indent}if {window} <= 0:")
        lines.append(f"{indent}    raise ValueError('bidirectional_mask.window must be > 0')")
        lines.append(f"{indent}if {q_len} == {k_len}:")
        lines.append(f"{indent}    {keep} = torch.abs({q_idx} - {k_idx}) <= {window}")
        lines.append(f"{indent}else:")
        lines.append(f"{indent}    {q_aligned} = {q_idx} + ({k_len} - {q_len})")
        lines.append(f"{indent}    {keep} = torch.abs({q_aligned} - {k_idx}) <= {window}")

    if padding_expr is not None:
        padding_keep = emitter._fresh("padding_keep")
        padding_max = emitter._fresh("padding_max")
        padding_min = emitter._fresh("padding_min")
        padding_floor = emitter._fresh("padding_floor")
        lines.append(f"{indent}if {padding_expr} is not None:")
        lines.append(f"{indent}    if {padding_expr}.ndim != 2:")
        lines.append(
            f"{indent}        raise ValueError('bidirectional_mask.padding_mask must be rank-2 [batch, seq]')"
        )
        lines.append(f"{indent}    if int({padding_expr}.shape[0]) != int({q}.shape[0]):")
        lines.append(
            f"{indent}        raise ValueError('bidirectional_mask.padding_mask batch size must match query batch')"
        )
        lines.append(f"{indent}    if int({padding_expr}.shape[1]) < {k_len}:")
        lines.append(
            f"{indent}        raise ValueError('bidirectional_mask.padding_mask width must be >= key sequence length')"
        )
        lines.append(f"{indent}    {padding_keep} = {padding_expr}[:, -{k_len}:]")
        lines.append(f"{indent}    if {padding_keep}.dtype == torch.bool:")
        lines.append(f"{indent}        {padding_keep} = {padding_keep}")
        lines.append(f"{indent}    elif {padding_keep}.is_floating_point():")
        lines.append(f"{indent}        if {padding_keep}.numel() == 0:")
        lines.append(f"{indent}            {padding_keep} = {padding_keep}.to(torch.bool)")
        lines.append(f"{indent}        else:")
        lines.append(f"{indent}            {padding_max} = float({padding_keep}.max())")
        lines.append(f"{indent}            {padding_min} = float({padding_keep}.min())")
        lines.append(
            f"{indent}            {padding_floor} = float(torch.finfo({padding_keep}.dtype).min)"
        )
        lines.append(
            f"{indent}            if {padding_max} == 0.0 and {padding_min} <= (0.5 * {padding_floor}):"
        )
        lines.append(f"{indent}                {padding_keep} = ({padding_keep} == 0)")
        lines.append(f"{indent}            else:")
        lines.append(f"{indent}                {padding_keep} = ({padding_keep} != 0)")
        lines.append(f"{indent}    else:")
        lines.append(f"{indent}        {padding_keep} = {padding_keep}.to(torch.bool)")
        lines.append(
            f"{indent}    {keep} = {keep}.unsqueeze(0).unsqueeze(0) & {padding_keep}.to({q}.device).unsqueeze(1).unsqueeze(1)"
        )
        lines.append(f"{indent}else:")
        lines.append(f"{indent}    {keep} = {keep}.unsqueeze(0).unsqueeze(0)")
    else:
        lines.append(f"{indent}{keep} = {keep}.unsqueeze(0).unsqueeze(0)")

    lines.append(f"{indent}{fill} = torch.finfo({q}.dtype).min")
    lines.append(
        f"{indent}{out_var} = torch.where({keep}, torch.zeros((), dtype={q}.dtype, device={q}.device), torch.full((), {fill}, dtype={q}.dtype, device={q}.device))"
    )
    return lines


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any", "Any"),
    "kwargs": dict(LOWERING_KWARG_KINDS),
    "returns": ("Tensor",),
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
