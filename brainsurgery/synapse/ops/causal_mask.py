from __future__ import annotations

from typing import Any

import torch

OP_NAME = "causal_mask"
LOWERING_ARITY = (2, 2)
LOWERING_ALLOWED_KWARGS: set[str] = {"window", "padding_mask", "early_exit"}
LOWERING_REQUIRED_KWARGS: set[str] = set()
LOWERING_KWARG_KINDS: dict[str, Any] = {
    "window": "dim",
    "padding_mask": "str",
    "early_exit": "any",
}


def _padding_mask_is_trivial(mask: torch.Tensor | None) -> bool:
    if mask is None:
        return True
    if not torch.is_tensor(mask):
        raise ValueError("causal_mask.padding_mask must resolve to tensor or null")
    if mask.numel() == 0:
        return True
    if mask.dtype == torch.bool:
        return bool(mask.all())
    if mask.is_floating_point():
        return bool((mask != 0).all())
    return bool(mask.to(torch.bool).all())


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
    raw_args = node_spec.get("_args")
    if not isinstance(raw_args, list) or len(raw_args) != 2:
        raise ValueError("causal_mask expects exactly 2 positional args: query and key")
    q = model._read_tensor_input(raw_args[0], env)
    key_tensor = model._read_tensor_input(raw_args[1], env)
    padding_ref = node_spec.get("padding_mask")
    padding_mask = env.get(padding_ref) if isinstance(padding_ref, str) else None
    out_name = model._require_name(node_spec.get("_bind"), field="causal_mask._bind")
    window_expr = node_spec.get("window")

    q_len = q.shape[-2]
    k_len = key_tensor.shape[-2]
    window_value = (
        int(model._eval_expr(window_expr, env, symbols)) if window_expr is not None else None
    )
    early_exit = bool(model._eval_expr(node_spec.get("early_exit", False), env, symbols))
    if early_exit:
        full_causal = window_value is None or int(window_value) >= int(k_len)
        if full_causal and _padding_mask_is_trivial(padding_mask):
            env[out_name] = None
            return

    if window_expr is None and padding_mask is None:
        env[out_name] = None
        return
    padding_key: tuple[int, int, tuple[int, ...]] | None = None
    if padding_mask is not None:
        padding_key = (
            int(padding_mask.data_ptr()),
            int(padding_mask.storage_offset()),
            tuple(int(x) for x in padding_mask.shape),
        )
    cache_key = (int(q_len), int(k_len), window_value, q.dtype, q.device, padding_key)
    cache = getattr(model, "_causal_mask_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        setattr(model, "_causal_mask_cache", cache)
    cached = cache.get(cache_key)
    if torch.is_tensor(cached):
        env[out_name] = cached
        return

    i_idx = torch.arange(q_len, device=q.device).unsqueeze(1)
    j_idx = torch.arange(k_len, device=q.device).unsqueeze(0)

    if q_len == 1:
        if window_value is None:
            keep = torch.ones((q_len, k_len), dtype=torch.bool, device=q.device)
            if padding_mask is None:
                env[out_name] = None
                return
        else:
            win = window_value
            if win >= k_len and padding_mask is None:
                env[out_name] = None
                return
            keep = j_idx >= (k_len - win)
    else:
        if window_value is None:
            keep = j_idx <= i_idx
        else:
            keep = j_idx <= i_idx
            win = window_value
            if win >= k_len and q_len == k_len and padding_mask is None:
                env[out_name] = None
                return
            keep = keep & (j_idx >= (i_idx - win + 1))

    if padding_mask is not None:
        if padding_mask.ndim != 2:
            raise ValueError("causal_mask.padding_mask must be rank-2 [batch, seq]")
        if int(padding_mask.shape[-1]) != k_len:
            raise ValueError("causal_mask.padding_mask width must match key sequence length")
        pad_keep = padding_mask.to(torch.bool).unsqueeze(1).unsqueeze(1)
        keep = keep.unsqueeze(0).unsqueeze(0) & pad_keep
    else:
        keep = keep.view(1, 1, q_len, k_len)

    mask_value = torch.finfo(q.dtype).min
    env[out_name] = torch.where(
        keep,
        torch.zeros((), dtype=q.dtype, device=q.device),
        torch.full((), mask_value, dtype=q.dtype, device=q.device),
    )
    cache[cache_key] = env[out_name]
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
    if not isinstance(raw_args, list) or len(raw_args) != 2:
        raise ValueError("causal_mask expects exactly 2 positional args: query and key")
    q = read(str(raw_args[0]))
    k = read(str(raw_args[1]))
    out_name = str(node_spec.get("_bind"))
    out_var = assign_out_var(out_name)
    q_len = emitter._fresh("q_len")
    k_len = emitter._fresh("k_len")
    cache_key = emitter._fresh("cache_key")
    cached = emitter._fresh("cached_mask")
    i_idx = emitter._fresh("i_idx")
    j_idx = emitter._fresh("j_idx")
    keep = emitter._fresh("keep")
    mask_val = emitter._fresh("mask_val")
    window_expr = node_spec.get("window")
    padding_name = node_spec.get("padding_mask")
    padding_expr = (
        env.get(padding_name) if isinstance(padding_name, str) and padding_name in env else None
    )
    early_exit_expr = emitter._expr_code(node_spec.get("early_exit", False), env)
    if window_expr is None and padding_expr is None:
        lines.append(f"{indent}{out_var} = None")
        return lines
    lines.append(f"{indent}{q_len} = {q}.shape[-2]")
    lines.append(f"{indent}{k_len} = {k}.shape[-2]")
    lines.append(f"{indent}if not hasattr(self, '_causal_mask_cache'):")
    lines.append(f"{indent}    self._causal_mask_cache = {{}}")
    if window_expr is not None:
        win = emitter._fresh("window")
        window_code = emitter._expr_code(window_expr, env)
        lines.append(f"{indent}{win} = int({window_code})")
        window_key_expr = win
    else:
        win = None
        window_key_expr = "None"
    if padding_expr is not None:
        pad_key = emitter._fresh("pad_key")
        lines.append(f"{indent}if {padding_expr} is None:")
        lines.append(f"{indent}    {pad_key} = None")
        lines.append(f"{indent}else:")
        lines.append(
            f"{indent}    {pad_key} = (int({padding_expr}.data_ptr()), int({padding_expr}.storage_offset()), tuple(int(x) for x in {padding_expr}.shape))"
        )
        pad_key_expr = pad_key
    else:
        pad_key_expr = "None"
    early_exit = emitter._fresh("early_exit")
    full_causal = emitter._fresh("full_causal")
    trivial_padding = emitter._fresh("trivial_padding")
    lines.append(f"{indent}{early_exit} = bool({early_exit_expr})")
    lines.append(
        f"{indent}{full_causal} = ({window_key_expr} is None) or (int({window_key_expr}) >= int({k_len}))"
    )
    if padding_expr is not None:
        lines.append(f"{indent}if {padding_expr} is None:")
        lines.append(f"{indent}    {trivial_padding} = True")
        lines.append(f"{indent}else:")
        lines.append(f"{indent}    if not torch.is_tensor({padding_expr}):")
        lines.append(
            f"{indent}        raise ValueError('causal_mask.padding_mask must resolve to tensor or null')"
        )
        lines.append(f"{indent}    if {padding_expr}.numel() == 0:")
        lines.append(f"{indent}        {trivial_padding} = True")
        lines.append(f"{indent}    elif {padding_expr}.dtype == torch.bool:")
        lines.append(f"{indent}        {trivial_padding} = bool({padding_expr}.all())")
        lines.append(f"{indent}    elif {padding_expr}.is_floating_point():")
        lines.append(f"{indent}        {trivial_padding} = bool(({padding_expr} != 0).all())")
        lines.append(f"{indent}    else:")
        lines.append(
            f"{indent}        {trivial_padding} = bool({padding_expr}.to(torch.bool).all())"
        )
    else:
        lines.append(f"{indent}{trivial_padding} = True")
    lines.append(f"{indent}if {early_exit} and {full_causal} and {trivial_padding}:")
    lines.append(f"{indent}    {out_var} = None")
    lines.append(f"{indent}else:")
    body_indent = indent + "    "
    lines.append(
        f"{body_indent}{cache_key} = (int({q_len}), int({k_len}), {window_key_expr}, {q}.dtype, {q}.device, {pad_key_expr})"
    )
    lines.append(f"{body_indent}{cached} = self._causal_mask_cache.get({cache_key})")
    lines.append(f"{body_indent}if torch.is_tensor({cached}):")
    lines.append(f"{body_indent}    {out_var} = {cached}")
    lines.append(f"{body_indent}else:")
    body_indent = body_indent + "    "
    lines.append(f"{body_indent}{j_idx} = torch.arange({k_len}, device={q}.device).unsqueeze(0)")
    if window_expr is None:
        lines.append(
            f"{body_indent}{i_idx} = torch.arange({q_len}, device={q}.device).unsqueeze(1)"
        )
        lines.append(f"{body_indent}{keep} = ({j_idx} <= {i_idx})")
    else:
        lines.append(f"{body_indent}if {q_len} == 1:")
        lines.append(f"{body_indent}    {keep} = ({j_idx} >= ({k_len} - {win}))")
        lines.append(f"{body_indent}else:")
        lines.append(
            f"{body_indent}    {i_idx} = torch.arange({q_len}, device={q}.device).unsqueeze(1)"
        )
        lines.append(f"{body_indent}    {keep} = ({j_idx} <= {i_idx})")
        lines.append(f"{body_indent}    {keep} = {keep} & ({j_idx} >= ({i_idx} - {win} + 1))")

    if padding_expr is not None:
        pad_keep = emitter._fresh("pad_keep")
        lines.append(f"{body_indent}if {padding_expr} is not None:")
        lines.append(f"{body_indent}    if {padding_expr}.ndim != 2:")
        lines.append(
            f"{body_indent}        raise ValueError('causal_mask.padding_mask must be rank-2 [batch, seq]')"
        )
        lines.append(f"{body_indent}    if int({padding_expr}.shape[-1]) != {k_len}:")
        lines.append(
            f"{body_indent}        raise ValueError('causal_mask.padding_mask width must match key sequence length')"
        )
        lines.append(
            f"{body_indent}    {pad_keep} = {padding_expr}.to(torch.bool).unsqueeze(1).unsqueeze(1)"
        )
        lines.append(f"{body_indent}    {keep} = {keep}.unsqueeze(0).unsqueeze(0) & {pad_keep}")
        lines.append(f"{body_indent}else:")
        lines.append(f"{body_indent}    {keep} = {keep}.view(1, 1, {q_len}, {k_len})")
    else:
        lines.append(f"{body_indent}{keep} = {keep}.view(1, 1, {q_len}, {k_len})")

    lines.append(f"{body_indent}{mask_val} = torch.finfo({q}.dtype).min")
    lines.append(
        f"{body_indent}{out_var} = torch.where({keep}, torch.zeros((), dtype={q}.dtype, device={q}.device), torch.full((), {mask_val}, dtype={q}.dtype, device={q}.device))"
    )
    lines.append(f"{body_indent}self._causal_mask_cache[{cache_key}] = {out_var}")
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
