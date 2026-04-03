from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

OP_NAME = "disentangled_relative_bias"
LOWERING_ARITY = (2, 2)
LOWERING_ALLOWED_KWARGS: set[str] = {
    "rel_embeddings",
    "position_buckets",
    "max_relative_positions",
    "share_att_key",
    "c2p",
    "p2c",
    "apply_rel_layernorm",
    "rel_norm_weight",
    "rel_norm_bias",
    "rel_norm_eps",
    "query_weight",
    "query_bias",
    "key_weight",
    "key_bias",
    "pos_key_weight",
    "pos_key_bias",
    "pos_query_weight",
    "pos_query_bias",
}
LOWERING_REQUIRED_KWARGS: set[str] = set()
LOWERING_KWARG_KINDS: dict[str, Any] = {
    "rel_embeddings": "str",
    "position_buckets": "dim",
    "max_relative_positions": "dim",
    "share_att_key": "bool",
    "c2p": "bool",
    "p2c": "bool",
    "apply_rel_layernorm": "bool",
    "rel_norm_weight": "str",
    "rel_norm_bias": "str_or_bool_or_null",
    "rel_norm_eps": "number",
    "query_weight": "str",
    "query_bias": "str_or_bool_or_null",
    "key_weight": "str",
    "key_bias": "str_or_bool_or_null",
    "pos_key_weight": "str",
    "pos_key_bias": "str_or_bool_or_null",
    "pos_query_weight": "str",
    "pos_query_bias": "str_or_bool_or_null",
}


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return True


def lowering_validate_signature(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> None:
    del args, kwargs, ctx
    if isinstance(out, list):
        raise ValueError("disentangled_relative_bias requires a single scalar output binding")


def lowering_infer_metadata(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> bool:
    del kwargs
    if isinstance(out, list) or len(args) < 2:
        return False
    q_name = str(args[0]).strip()
    k_name = str(args[1]).strip()
    q_shape = ctx.tensor_shape.get(q_name)
    k_shape = ctx.tensor_shape.get(k_name)
    if not (isinstance(q_shape, tuple) and isinstance(k_shape, tuple)):
        return False
    if len(q_shape) < 4 or len(k_shape) < 4:
        return False
    batch = q_shape[0]
    heads = q_shape[1]
    q_len = q_shape[-2]
    k_len = k_shape[-2]
    ctx.tensor_shape[out] = (batch, heads, q_len, k_len)
    ctx.tensor_last_dim[out] = k_len
    return True


def _make_log_bucket_position(
    relative_pos: torch.Tensor,
    *,
    bucket_size: int,
    max_position: int,
) -> torch.Tensor:
    sign = torch.sign(relative_pos)
    mid = int(bucket_size) // 2
    abs_pos = torch.where(
        (relative_pos < mid) & (relative_pos > -mid),
        torch.tensor(mid - 1, device=relative_pos.device, dtype=relative_pos.dtype),
        torch.abs(relative_pos),
    )
    log_base = torch.log(
        torch.tensor(
            (float(max_position) - 1.0) / float(mid),
            device=relative_pos.device,
            dtype=torch.float32,
        )
    )
    log_pos = torch.ceil(
        torch.log(abs_pos.to(torch.float32) / float(mid)) / log_base * float(mid - 1)
    ) + float(mid)
    bucket_pos = torch.where(
        abs_pos <= mid,
        relative_pos.to(dtype=log_pos.dtype),
        log_pos * sign.to(dtype=log_pos.dtype),
    )
    return bucket_pos


def _build_relative_position(
    q_layer: torch.Tensor,
    k_layer: torch.Tensor,
    *,
    bucket_size: int,
    max_position: int,
) -> torch.Tensor:
    q_size = int(q_layer.size(-2))
    k_size = int(k_layer.size(-2))
    q_ids = torch.arange(q_size, dtype=torch.long, device=q_layer.device)
    k_ids = torch.arange(k_size, dtype=torch.long, device=k_layer.device)
    rel_pos = q_ids[:, None] - k_ids[None, :]
    if bucket_size > 0 and max_position > 0:
        rel_pos = _make_log_bucket_position(
            rel_pos,
            bucket_size=int(bucket_size),
            max_position=int(max_position),
        )
    return rel_pos.to(torch.long).unsqueeze(0)


def _build_rpos(
    q_layer: torch.Tensor,
    k_layer: torch.Tensor,
    relative_pos: torch.Tensor,
    *,
    position_buckets: int,
    max_relative_positions: int,
) -> torch.Tensor:
    if int(k_layer.size(-2)) != int(q_layer.size(-2)):
        return _build_relative_position(
            k_layer,
            k_layer,
            bucket_size=position_buckets,
            max_position=max_relative_positions,
        )
    return relative_pos


def _scaled_size_sqrt(layer: torch.Tensor, scale_factor: int) -> torch.Tensor:
    return torch.sqrt(
        torch.tensor(float(layer.size(-1)), device=layer.device, dtype=torch.float32)
        * float(scale_factor)
    )


def _transpose_for_scores(x: torch.Tensor, *, heads: int) -> torch.Tensor:
    new_shape = x.size()[:-1] + (heads, -1)
    x = x.view(new_shape)
    return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))


def _resolve_param_path(
    *,
    model: Any,
    node_spec: dict[str, Any],
    node_path: str,
    param_name: str,
    default_candidate: str,
) -> str:
    candidate = node_spec.get(param_name)
    if not isinstance(candidate, str):
        candidate = default_candidate
    node_scope = str(model._scope_of(node_path)) if hasattr(model, "_scope_of") else ""
    scope_hint_raw = node_spec.get("_scope")
    scope_hint = str(scope_hint_raw).strip() if isinstance(scope_hint_raw, str) else ""
    if (
        scope_hint
        and not node_scope
        and isinstance(candidate, str)
        and not candidate.startswith("@@")
    ):
        candidate = f"{scope_hint}.{candidate}"
    if hasattr(model, "_infer_param_path"):
        local_spec = dict(node_spec)
        local_spec[param_name] = candidate
        return str(model._infer_param_path(local_spec, node_path=node_path, param_name=param_name))
    scope = str(model._scope_of(node_path))
    return str(model._pick_param_from_single(scope, candidate))


def _resolve_optional_bias_path(
    *,
    model: Any,
    node_spec: dict[str, Any],
    node_path: str,
    kwarg_name: str,
    default_candidate: str,
) -> str | None:
    value = node_spec.get(kwarg_name)
    if value is None:
        candidate: str | None = default_candidate
    elif isinstance(value, bool):
        candidate = default_candidate if value else None
    elif isinstance(value, str):
        candidate = value
    else:
        raise ValueError(
            f"disentangled_relative_bias.{kwarg_name} must be string, bool, or null when provided"
        )
    if candidate is None:
        return None
    return _resolve_param_path(
        model=model,
        node_spec=node_spec,
        node_path=node_path,
        param_name=kwarg_name,
        default_candidate=candidate,
    )


def _eval_value(
    *,
    model: Any,
    expr: Any,
    env: dict[str, Any],
    symbols: dict[str, int | float | bool],
) -> Any:
    if hasattr(model, "_eval_expr"):
        return model._eval_expr(expr, env, symbols)
    if isinstance(expr, list):
        return [_eval_value(model=model, expr=item, env=env, symbols=symbols) for item in expr]
    if isinstance(expr, tuple):
        return tuple(_eval_value(model=model, expr=item, env=env, symbols=symbols) for item in expr)
    if isinstance(expr, (bool, int, float)) or expr is None:
        return expr
    if isinstance(expr, dict):
        kind = expr.get("_expr")
        if kind == "name":
            ident = expr.get("id")
            if not isinstance(ident, str) or not ident:
                raise ValueError(f"Invalid name expression payload: {expr!r}")
            if ident in env:
                return env[ident]
            if ident in symbols:
                return symbols[ident]
            raise ValueError(f"Unknown symbol in expression: {ident}")
        if kind == "tuple":
            items = expr.get("items")
            if not isinstance(items, list):
                raise ValueError(f"Invalid tuple expression payload: {expr!r}")
            return tuple(
                _eval_value(model=model, expr=item, env=env, symbols=symbols) for item in items
            )
        if kind == "if":
            cond = bool(_eval_value(model=model, expr=expr.get("cond"), env=env, symbols=symbols))
            branch = expr.get("then") if cond else expr.get("else")
            return _eval_value(model=model, expr=branch, env=env, symbols=symbols)
        if kind == "binary":
            op = expr.get("op")
            left = _eval_value(model=model, expr=expr.get("left"), env=env, symbols=symbols)
            right = _eval_value(model=model, expr=expr.get("right"), env=env, symbols=symbols)
            if op == "+":
                return left + right
            if op == "-":
                return left - right
            if op == "*":
                return left * right
            if op == "/":
                return left / right
            if op == "%":
                return left % right
            if op == "==":
                return left == right
            if op == "!=":
                return left != right
            if op == "<":
                return left < right
            if op == "<=":
                return left <= right
            if op == ">":
                return left > right
            if op == ">=":
                return left >= right
            if op == "and":
                return bool(left) and bool(right)
            if op == "or":
                return bool(left) or bool(right)
            raise ValueError(f"Unsupported binary operator in expression: {op!r}")
        if kind == "string":
            value = expr.get("value")
            if not isinstance(value, str):
                raise ValueError(f"Invalid string expression payload: {expr!r}")
            return value
        return {
            key: _eval_value(model=model, expr=value, env=env, symbols=symbols)
            for key, value in expr.items()
        }
    if isinstance(expr, str):
        token = expr.strip()
        if token in env:
            return env[token]
        if token in symbols:
            return symbols[token]
        lowered = token.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered == "null":
            return None
        if token and token.lstrip("-").isdigit():
            return int(token)
        try:
            return float(token)
        except ValueError:
            return expr
    return expr


def _compute_disentangled_relative_bias(
    *,
    model: Any,
    node_spec: dict[str, Any],
    env: dict[str, Any],
    node_path: str,
    q: torch.Tensor,
    k_tensor: torch.Tensor,
    symbols: dict[str, int | float | bool],
) -> torch.Tensor:
    if q.ndim != 4 or k_tensor.ndim != 4:
        raise ValueError("disentangled_relative_bias expects rank-4 q/k tensors [B,H,S,HD]")
    if int(q.shape[0]) != int(k_tensor.shape[0]):
        raise ValueError("disentangled_relative_bias requires matching q/k batch size")
    if int(q.shape[1]) != int(k_tensor.shape[1]):
        raise ValueError("disentangled_relative_bias requires matching q/k head count")
    if int(q.shape[-1]) != int(k_tensor.shape[-1]):
        raise ValueError("disentangled_relative_bias requires matching q/k head dim")

    batch = int(q.shape[0])
    heads = int(q.shape[1])
    q_len = int(q.shape[-2])
    k_len = int(k_tensor.shape[-2])
    head_dim = int(q.shape[-1])

    c2p = bool(_eval_value(model=model, expr=node_spec.get("c2p", True), env=env, symbols=symbols))
    p2c = bool(_eval_value(model=model, expr=node_spec.get("p2c", True), env=env, symbols=symbols))
    if not c2p and not p2c:
        return torch.zeros((batch, heads, q_len, k_len), dtype=q.dtype, device=q.device)

    share_att_key = bool(
        _eval_value(
            model=model, expr=node_spec.get("share_att_key", False), env=env, symbols=symbols
        )
    )
    position_buckets = int(
        _eval_value(
            model=model, expr=node_spec.get("position_buckets", -1), env=env, symbols=symbols
        )
    )
    max_relative_positions = int(
        _eval_value(
            model=model,
            expr=node_spec.get("max_relative_positions", -1),
            env=env,
            symbols=symbols,
        )
    )
    if max_relative_positions < 1:
        max_relative_positions = max(q_len, k_len)
    att_span = position_buckets if position_buckets > 0 else max_relative_positions
    if att_span <= 0:
        raise ValueError("disentangled_relative_bias requires positive attention span")

    rel_path = _resolve_param_path(
        model=model,
        node_spec=node_spec,
        node_path=node_path,
        param_name="rel_embeddings",
        default_candidate="rel_embeddings.weight",
    )
    rel_embeddings = model._state[rel_path]
    if rel_embeddings.ndim != 2:
        raise ValueError("disentangled_relative_bias rel_embeddings must be rank-2 [N,D]")
    if int(rel_embeddings.shape[0]) < int(att_span * 2):
        raise ValueError("disentangled_relative_bias rel_embeddings rows must cover 2*att_span")

    rel_embeddings = rel_embeddings[0 : att_span * 2, :]
    if bool(
        _eval_value(
            model=model,
            expr=node_spec.get("apply_rel_layernorm", False),
            env=env,
            symbols=symbols,
        )
    ):
        rel_norm_weight_path = _resolve_param_path(
            model=model,
            node_spec=node_spec,
            node_path=node_path,
            param_name="rel_norm_weight",
            default_candidate="LayerNorm.weight",
        )
        rel_norm_weight = model._state[rel_norm_weight_path]
        rel_norm_bias_path = _resolve_optional_bias_path(
            model=model,
            node_spec=node_spec,
            node_path=node_path,
            kwarg_name="rel_norm_bias",
            default_candidate="LayerNorm.bias",
        )
        rel_norm_bias = model._state[rel_norm_bias_path] if rel_norm_bias_path is not None else None
        rel_norm_eps = float(
            _eval_value(
                model=model, expr=node_spec.get("rel_norm_eps", 1e-7), env=env, symbols=symbols
            )
        )
        rel_embeddings = F.layer_norm(
            rel_embeddings,
            (int(rel_embeddings.shape[-1]),),
            weight=rel_norm_weight.to(device=rel_embeddings.device, dtype=rel_embeddings.dtype),
            bias=(
                rel_norm_bias.to(device=rel_embeddings.device, dtype=rel_embeddings.dtype)
                if rel_norm_bias is not None
                else None
            ),
            eps=rel_norm_eps,
        )

    rel_embeddings = rel_embeddings.to(
        device=q.device,
        dtype=(q.dtype if q.is_floating_point() else rel_embeddings.dtype),
    ).unsqueeze(0)

    query_layer = q.contiguous().view(batch * heads, q_len, head_dim)
    key_layer = k_tensor.contiguous().view(batch * heads, k_len, head_dim)

    def _project_rel(
        *,
        weight_name: str,
        weight_default: str,
        bias_name: str,
        bias_default: str,
    ) -> torch.Tensor:
        weight_path = _resolve_param_path(
            model=model,
            node_spec=node_spec,
            node_path=node_path,
            param_name=weight_name,
            default_candidate=weight_default,
        )
        weight = model._state[weight_path]
        bias_path = _resolve_optional_bias_path(
            model=model,
            node_spec=node_spec,
            node_path=node_path,
            kwarg_name=bias_name,
            default_candidate=bias_default,
        )
        bias = model._state[bias_path] if bias_path is not None else None
        projected = F.linear(
            rel_embeddings,
            weight.to(device=rel_embeddings.device, dtype=rel_embeddings.dtype),
            (
                bias.to(device=rel_embeddings.device, dtype=rel_embeddings.dtype)
                if bias is not None
                else None
            ),
        )
        return _transpose_for_scores(projected, heads=heads).repeat(batch, 1, 1)

    pos_key_layer: torch.Tensor | None = None
    pos_query_layer: torch.Tensor | None = None
    if share_att_key:
        pos_query_layer = _project_rel(
            weight_name="query_weight",
            weight_default="query_proj.weight",
            bias_name="query_bias",
            bias_default="query_proj.bias",
        )
        pos_key_layer = _project_rel(
            weight_name="key_weight",
            weight_default="key_proj.weight",
            bias_name="key_bias",
            bias_default="key_proj.bias",
        )
    else:
        if c2p:
            pos_key_layer = _project_rel(
                weight_name="pos_key_weight",
                weight_default="pos_key_proj.weight",
                bias_name="pos_key_bias",
                bias_default="pos_key_proj.bias",
            )
        if p2c:
            pos_query_layer = _project_rel(
                weight_name="pos_query_weight",
                weight_default="pos_query_proj.weight",
                bias_name="pos_query_bias",
                bias_default="pos_query_proj.bias",
            )

    relative_pos = _build_relative_position(
        query_layer,
        key_layer,
        bucket_size=position_buckets,
        max_position=max_relative_positions,
    )
    if relative_pos.dim() == 2:
        relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
    elif relative_pos.dim() == 3:
        relative_pos = relative_pos.unsqueeze(1)
    elif relative_pos.dim() != 4:
        raise ValueError(
            "disentangled_relative_bias relative position ids must have dim 2, 3, or 4"
        )
    relative_pos = relative_pos.to(device=q.device, dtype=torch.long)

    scale_factor = 1 + (1 if c2p else 0) + (1 if p2c else 0)
    score = torch.zeros((batch * heads, q_len, k_len), dtype=q.dtype, device=q.device)

    if c2p:
        if pos_key_layer is None:
            raise ValueError(
                "disentangled_relative_bias c2p enabled but pos_key projection missing"
            )
        scale = _scaled_size_sqrt(pos_key_layer, scale_factor)
        c2p_att = torch.bmm(query_layer, pos_key_layer.transpose(-1, -2))
        c2p_pos = torch.clamp(relative_pos + att_span, 0, att_span * 2 - 1)
        c2p_index = c2p_pos.squeeze(0).expand(
            [query_layer.size(0), query_layer.size(1), relative_pos.size(-1)]
        )
        c2p_att = torch.gather(c2p_att, dim=-1, index=c2p_index)
        score = score + (c2p_att / scale.to(dtype=c2p_att.dtype))

    if p2c:
        if pos_query_layer is None:
            raise ValueError(
                "disentangled_relative_bias p2c enabled but pos_query projection missing"
            )
        scale = _scaled_size_sqrt(pos_query_layer, scale_factor)
        # Keep argument order aligned with HF implementation.
        r_pos = _build_rpos(
            query_layer,
            key_layer,
            relative_pos,
            position_buckets=max_relative_positions,
            max_relative_positions=position_buckets,
        )
        p2c_pos = torch.clamp(-r_pos + att_span, 0, att_span * 2 - 1)
        p2c_att = torch.bmm(key_layer, pos_query_layer.transpose(-1, -2))
        p2c_index = p2c_pos.squeeze(0).expand(
            [query_layer.size(0), key_layer.size(-2), key_layer.size(-2)]
        )
        p2c_att = torch.gather(p2c_att, dim=-1, index=p2c_index).transpose(-1, -2)
        score = score + (p2c_att / scale.to(dtype=p2c_att.dtype))

    return score.view(batch, heads, q_len, k_len)


def interpret(
    model: Any,
    node_spec: dict[str, Any],
    env: dict[str, Any],
    *,
    node_path: str,
    scope: str,
    symbols: dict[str, int | float | bool],
) -> None:
    del scope
    ins = node_spec.get("_args")
    if not isinstance(ins, list) or len(ins) != 2:
        raise ValueError("disentangled_relative_bias expects [q, k]")
    q = model._read_tensor_input(ins[0], env)
    k_tensor = model._read_tensor_input(ins[1], env)
    if not torch.is_tensor(q) or not torch.is_tensor(k_tensor):
        raise ValueError("disentangled_relative_bias expects tensor inputs for q and k")

    out_name = model._require_name(node_spec.get("_bind"), field="disentangled_relative_bias._bind")
    env[out_name] = _compute_disentangled_relative_bias(
        model=model,
        node_spec=node_spec,
        env=env,
        node_path=node_path,
        q=q,
        k_tensor=k_tensor,
        symbols=symbols,
    )


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
    if not isinstance(ins, list) or len(ins) != 2:
        raise ValueError("disentangled_relative_bias expects [q, k]")
    q = read(str(ins[0]))
    k = read(str(ins[1]))
    out_var = assign_out_var(str(node_spec.get("_bind")))

    lines.append(
        f"{indent}from brainsurgery.synapse.ops.disentangled_relative_bias import _compute_disentangled_relative_bias as _op_disentangled_relative_bias"
    )
    lines.append(
        f"{indent}{out_var} = _op_disentangled_relative_bias(model=emitter, node_spec={repr(node_spec)}, env=env, node_path={node_path_var}, q={q}, k_tensor={k}, symbols=self._symbols)"
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
