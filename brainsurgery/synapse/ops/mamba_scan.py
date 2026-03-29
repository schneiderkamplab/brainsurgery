from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

OP_NAME = "mamba_scan"
LOWERING_ARITY = (4, 7)
LOWERING_ALLOWED_KWARGS: set[str] = {"a_is_log", "A", "D"}
LOWERING_REQUIRED_KWARGS: set[str] = set()
LOWERING_KWARG_KINDS: dict[str, Any] = {"a_is_log": "bool", "A": "str", "D": "str"}


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter
    args = node_spec.get("_args")
    if isinstance(args, list) and len(args) in {6, 7}:
        return False
    return True


def lowering_validate_signature(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> None:
    del kwargs, ctx
    if len(args) not in {4, 5, 6, 7}:
        raise ValueError(
            "mamba_scan expects 4/5 inputs [u,delta,B,C,?state] or 6/7 [u,delta,A,B,C,D,?state]"
        )
    if isinstance(out, str):
        return
    if isinstance(out, list) and len(out) == 2:
        return
    raise ValueError("mamba_scan requires _bind as a single output or [y, final_state]")


def lowering_normalize_kwargs(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> None:
    del args, out
    scope_stack = getattr(ctx, "scope_stack", None)
    if not isinstance(scope_stack, list) or not scope_stack:
        return
    scope_prefix = ".".join(part for part in scope_stack if isinstance(part, str) and part)
    if not scope_prefix:
        return
    for key in ("A", "D"):
        value = kwargs.get(key)
        if not isinstance(value, str) or "." in value:
            continue
        kwargs[key] = f"{scope_prefix}.{value}"


def _validate_shapes(
    *,
    u: torch.Tensor,
    delta: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    state: torch.Tensor | None,
) -> tuple[int, int, int, int]:
    if u.ndim != 3:
        raise ValueError("mamba_scan expects u shape [batch, seq, dim]")
    if delta.ndim != 3:
        raise ValueError("mamba_scan expects delta shape [batch, seq, dim]")
    if u.shape != delta.shape:
        raise ValueError("mamba_scan requires u and delta to share shape [batch, seq, dim]")
    if a.ndim != 2:
        raise ValueError("mamba_scan expects A shape [dim, state_dim]")
    if b.ndim != 3:
        raise ValueError("mamba_scan expects B shape [batch, seq, state_dim]")
    if c.ndim != 3:
        raise ValueError("mamba_scan expects C shape [batch, seq, state_dim]")
    if b.shape != c.shape:
        raise ValueError("mamba_scan requires B and C to share shape [batch, seq, state_dim]")
    if d.ndim != 1:
        raise ValueError("mamba_scan expects D shape [dim]")

    batch, seq, dim = int(u.shape[0]), int(u.shape[1]), int(u.shape[2])
    state_dim = int(a.shape[1])
    if int(a.shape[0]) != dim:
        raise ValueError("mamba_scan A first dim must match input dim")
    if int(b.shape[0]) != batch or int(b.shape[1]) != seq:
        raise ValueError("mamba_scan B first two dims must match [batch, seq]")
    if int(b.shape[2]) != state_dim:
        raise ValueError("mamba_scan B/C last dim must match A state_dim")
    if int(d.shape[0]) != dim:
        raise ValueError("mamba_scan D dim must match input dim")
    if state is not None:
        if state.ndim != 3:
            raise ValueError("mamba_scan state must be rank-3 [batch, dim, state_dim]")
        if tuple(state.shape) != (batch, dim, state_dim):
            raise ValueError("mamba_scan state shape must match [batch, dim, state_dim]")
    return batch, seq, dim, state_dim


def _resolve_bind(node_spec: dict[str, Any], model: Any) -> tuple[str, str | None]:
    out = node_spec.get("_bind")
    if isinstance(out, str):
        return model._require_name(out, field="mamba_scan._bind"), None
    if isinstance(out, list) and len(out) == 2:
        return (
            model._require_name(out[0], field="mamba_scan._bind[0]"),
            model._require_name(out[1], field="mamba_scan._bind[1]"),
        )
    raise ValueError("mamba_scan requires _bind as a single output or [y, final_state]")


def _compute_scan(
    *,
    u: torch.Tensor,
    delta: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, seq, dim, state_dim = _validate_shapes(
        u=u,
        delta=delta,
        a=a,
        b=b,
        c=c,
        d=d,
        state=state,
    )
    dtype = u.dtype
    work_dtype = torch.float32 if dtype in {torch.float16, torch.bfloat16} else dtype

    u_work = u.to(dtype=work_dtype)
    delta_work = F.softplus(delta.to(dtype=work_dtype))
    a_work = a.to(dtype=work_dtype)
    b_work = b.to(dtype=work_dtype)
    c_work = c.to(dtype=work_dtype)
    d_work = d.to(dtype=work_dtype)
    state_work = (
        torch.zeros((batch, dim, state_dim), device=u.device, dtype=work_dtype)
        if state is None
        else state.to(dtype=work_dtype)
    )

    outputs = []
    for t in range(seq):
        u_t = u_work[:, t, :]
        delta_t = delta_work[:, t, :]
        b_t = b_work[:, t, :]
        c_t = c_work[:, t, :]
        a_t = torch.exp(delta_t.unsqueeze(-1) * a_work.unsqueeze(0))
        bu_t = (delta_t * u_t).unsqueeze(-1) * b_t.unsqueeze(1)
        state_work = a_t * state_work + bu_t
        y_t = (state_work * c_t.unsqueeze(1)).sum(dim=-1) + u_t * d_work.unsqueeze(0)
        outputs.append(y_t)

    y = torch.stack(outputs, dim=1)
    return y.to(dtype=dtype), state_work.to(dtype=dtype)


def _resolve_interpret_inputs(
    model: Any,
    node_spec: dict[str, Any],
    env: dict[str, Any],
    *,
    node_path: str,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    args = node_spec.get("_args")
    if not isinstance(args, list) or len(args) not in {4, 5, 6, 7}:
        raise ValueError(
            "mamba_scan expects 4/5 inputs [u,delta,B,C,?state] or 6/7 [u,delta,A,B,C,D,?state]"
        )

    u = env[str(args[0])]
    delta = env[str(args[1])]
    if len(args) in {6, 7}:
        a = env[str(args[2])]
        b = env[str(args[3])]
        c = env[str(args[4])]
        d = env[str(args[5])]
        state = env.get(str(args[6])) if len(args) == 7 else None
        return u, delta, a, b, c, d, state

    explicit_a = isinstance(node_spec.get("A"), str)
    a_param = "A" if explicit_a else ("A_log" if bool(node_spec.get("a_is_log", False)) else "A")
    a_path = model._infer_param_path(node_spec, node_path=node_path, param_name=a_param)
    d_path = model._infer_param_path(node_spec, node_path=node_path, param_name="D")
    a = model._state[a_path]
    if bool(node_spec.get("a_is_log", False)):
        a = -torch.exp(a.float())
    b = env[str(args[2])]
    c = env[str(args[3])]
    d = model._state[d_path]
    state = env.get(str(args[4])) if len(args) == 5 else None
    return u, delta, a, b, c, d, state


def interpret(
    model: Any,
    node_spec: dict[str, Any],
    env: dict[str, Any],
    *,
    node_path: str,
    scope: str,
    symbols: dict[str, int],
) -> None:
    del scope, symbols
    u, delta, a, b, c, d, state = _resolve_interpret_inputs(
        model,
        node_spec,
        env,
        node_path=node_path,
    )
    y, final_state = _compute_scan(u=u, delta=delta, a=a, b=b, c=c, d=d, state=state)

    y_name, state_name = _resolve_bind(node_spec, model)
    env[y_name] = y
    if state_name is not None:
        env[state_name] = final_state


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
    args = node_spec.get("_args")
    if not isinstance(args, list) or len(args) not in {4, 5, 6, 7}:
        raise ValueError(
            "mamba_scan expects 4/5 inputs [u,delta,B,C,?state] or 6/7 [u,delta,A,B,C,D,?state]"
        )
    out = node_spec.get("_bind")
    if isinstance(out, str):
        y_out = emitter._assign_out_var(env, out)
        state_out = None
    elif isinstance(out, list) and len(out) == 2:
        y_out = emitter._assign_out_var(env, str(out[0]))
        state_out = emitter._assign_out_var(env, str(out[1]))
    else:
        raise ValueError("mamba_scan requires _bind as a single output or [y, final_state]")

    u = emitter._read_env_var(env, str(args[0]))
    delta = emitter._read_env_var(env, str(args[1]))
    if len(args) in {6, 7}:
        a = emitter._read_env_var(env, str(args[2]))
        b = emitter._read_env_var(env, str(args[3]))
        c = emitter._read_env_var(env, str(args[4]))
        d = emitter._read_env_var(env, str(args[5]))
        state_in = emitter._read_env_var(env, str(args[6])) if len(args) == 7 else "None"
    else:
        explicit_a = isinstance(node_spec.get("A"), str)
        a_param_name = (
            "A" if explicit_a else ("A_log" if bool(node_spec.get("a_is_log", False)) else "A")
        )
        a_expr = emitter._infer_param_expr(node_spec, node_path_var, a_param_name)
        d_expr = emitter._infer_param_expr(node_spec, node_path_var, "D")
        a = emitter._fresh("a")
        b = emitter._read_env_var(env, str(args[2]))
        c = emitter._read_env_var(env, str(args[3]))
        d = emitter._fresh("d")
        state_in = emitter._read_env_var(env, str(args[4])) if len(args) == 5 else "None"

    batch = emitter._fresh("batch")
    seq = emitter._fresh("seq")
    dim = emitter._fresh("dim")
    state_dim = emitter._fresh("state_dim")
    dtype = emitter._fresh("dtype")
    work_dtype = emitter._fresh("work_dtype")
    u_work = emitter._fresh("u_work")
    delta_work = emitter._fresh("delta_work")
    a_work = emitter._fresh("a_work")
    b_work = emitter._fresh("b_work")
    c_work = emitter._fresh("c_work")
    d_work = emitter._fresh("d_work")
    state_work = emitter._fresh("state_work")
    outputs = emitter._fresh("outputs")
    t = emitter._fresh("t")
    u_t = emitter._fresh("u_t")
    delta_t = emitter._fresh("delta_t")
    b_t = emitter._fresh("b_t")
    c_t = emitter._fresh("c_t")
    a_t = emitter._fresh("a_t")
    bu_t = emitter._fresh("bu_t")
    y_t = emitter._fresh("y_t")

    lines: list[str] = []
    if len(args) in {4, 5}:
        lines.append(f"{indent}{a} = emitter._param({a_expr})")
        if bool(node_spec.get("a_is_log", False)):
            lines.append(f"{indent}{a} = -torch.exp({a}.float())")
        lines.append(f"{indent}{d} = emitter._param({d_expr})")
    lines.extend(
        [
            f"{indent}if {u}.ndim != 3:",
            f"{indent}    raise ValueError('mamba_scan expects u shape [batch, seq, dim]')",
            f"{indent}if {delta}.ndim != 3:",
            f"{indent}    raise ValueError('mamba_scan expects delta shape [batch, seq, dim]')",
            f"{indent}if tuple({u}.shape) != tuple({delta}.shape):",
            f"{indent}    raise ValueError('mamba_scan requires u and delta to share shape [batch, seq, dim]')",
            f"{indent}if {a}.ndim != 2:",
            f"{indent}    raise ValueError('mamba_scan expects A shape [dim, state_dim]')",
            f"{indent}if {b}.ndim != 3:",
            f"{indent}    raise ValueError('mamba_scan expects B shape [batch, seq, state_dim]')",
            f"{indent}if {c}.ndim != 3:",
            f"{indent}    raise ValueError('mamba_scan expects C shape [batch, seq, state_dim]')",
            f"{indent}if tuple({b}.shape) != tuple({c}.shape):",
            f"{indent}    raise ValueError('mamba_scan requires B and C to share shape [batch, seq, state_dim]')",
            f"{indent}if {d}.ndim != 1:",
            f"{indent}    raise ValueError('mamba_scan expects D shape [dim]')",
            f"{indent}{batch} = int({u}.shape[0])",
            f"{indent}{seq} = int({u}.shape[1])",
            f"{indent}{dim} = int({u}.shape[2])",
            f"{indent}{state_dim} = int({a}.shape[1])",
            f"{indent}if int({a}.shape[0]) != {dim}:",
            f"{indent}    raise ValueError('mamba_scan A first dim must match input dim')",
            f"{indent}if int({b}.shape[0]) != {batch} or int({b}.shape[1]) != {seq}:",
            f"{indent}    raise ValueError('mamba_scan B first two dims must match [batch, seq]')",
            f"{indent}if int({b}.shape[2]) != {state_dim}:",
            f"{indent}    raise ValueError('mamba_scan B/C last dim must match A state_dim')",
            f"{indent}if int({d}.shape[0]) != {dim}:",
            f"{indent}    raise ValueError('mamba_scan D dim must match input dim')",
            f"{indent}if {state_in} is not None:",
            f"{indent}    if {state_in}.ndim != 3:",
            f"{indent}        raise ValueError('mamba_scan state must be rank-3 [batch, dim, state_dim]')",
            f"{indent}    if tuple({state_in}.shape) != ({batch}, {dim}, {state_dim}):",
            f"{indent}        raise ValueError('mamba_scan state shape must match [batch, dim, state_dim]')",
            f"{indent}{dtype} = {u}.dtype",
            f"{indent}{work_dtype} = torch.float32 if {dtype} in {{torch.float16, torch.bfloat16}} else {dtype}",
            f"{indent}{u_work} = {u}.to(dtype={work_dtype})",
            f"{indent}{delta_work} = F.softplus({delta}.to(dtype={work_dtype}))",
            f"{indent}{a_work} = {a}.to(dtype={work_dtype})",
            f"{indent}{b_work} = {b}.to(dtype={work_dtype})",
            f"{indent}{c_work} = {c}.to(dtype={work_dtype})",
            f"{indent}{d_work} = {d}.to(dtype={work_dtype})",
            f"{indent}if {state_in} is None:",
            f"{indent}    {state_work} = torch.zeros(({batch}, {dim}, {state_dim}), device={u}.device, dtype={work_dtype})",
            f"{indent}else:",
            f"{indent}    {state_work} = {state_in}.to(dtype={work_dtype})",
            f"{indent}{outputs} = []",
            f"{indent}for {t} in range({seq}):",
            f"{indent}    {u_t} = {u_work}[:, {t}, :]",
            f"{indent}    {delta_t} = {delta_work}[:, {t}, :]",
            f"{indent}    {b_t} = {b_work}[:, {t}, :]",
            f"{indent}    {c_t} = {c_work}[:, {t}, :]",
            f"{indent}    {a_t} = torch.exp({delta_t}.unsqueeze(-1) * {a_work}.unsqueeze(0))",
            f"{indent}    {bu_t} = ({delta_t} * {u_t}).unsqueeze(-1) * {b_t}.unsqueeze(1)",
            f"{indent}    {state_work} = {a_t} * {state_work} + {bu_t}",
            f"{indent}    {y_t} = ({state_work} * {c_t}.unsqueeze(1)).sum(dim=-1) + {u_t} * {d_work}.unsqueeze(0)",
            f"{indent}    {outputs}.append({y_t})",
            f"{indent}{y_out} = torch.stack({outputs}, dim=1).to(dtype={dtype})",
        ]
    )
    if state_out is not None:
        lines.append(f"{indent}{state_out} = {state_work}.to(dtype={dtype})")
    return lines


__all__ = [
    "LOWERING_ARITY",
    "LOWERING_ALLOWED_KWARGS",
    "LOWERING_REQUIRED_KWARGS",
    "LOWERING_KWARG_KINDS",
    "OP_NAME",
    "lowering_normalize_kwargs",
    "lowering_validate_signature",
    "interpret",
    "compile",
    "uses_node_path",
]
