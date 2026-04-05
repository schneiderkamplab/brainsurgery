from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

OP_NAME = "mamba2_scan"
LOWERING_ARITY = (5, 6)
LOWERING_ALLOWED_KWARGS: set[str] = {
    "A",
    "D",
    "dt_bias",
    "norm_weight",
    "n_groups",
    "head_dim",
    "time_step_min",
    "time_step_max",
}
LOWERING_REQUIRED_KWARGS: set[str] = {
    "A",
    "D",
    "dt_bias",
    "norm_weight",
    "n_groups",
    "head_dim",
}
LOWERING_KWARG_KINDS: dict[str, Any] = {
    "A": "str",
    "D": "str",
    "dt_bias": "str",
    "norm_weight": "str",
    "n_groups": "dim",
    "head_dim": "dim",
    "time_step_min": "number",
    "time_step_max": "number",
}


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter
    args = node_spec.get("_args")
    if isinstance(args, list) and len(args) == 6:
        return False
    return True


def lowering_validate_signature(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> None:
    del ctx
    if len(args) not in {5, 6}:
        raise ValueError("mamba2_scan expects 5/6 inputs [u,gate,dt,B,C,?state]")
    if isinstance(out, str):
        return
    if isinstance(out, list) and len(out) == 2:
        return
    raise ValueError("mamba2_scan requires _bind as a single output or [y, final_state]")
    # kwargs validated by lowering core from LOWERING_REQUIRED_KWARGS/KINDS


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
    for key in ("A", "D", "dt_bias", "norm_weight"):
        value = kwargs.get(key)
        if not isinstance(value, str) or "." in value:
            continue
        kwargs[key] = f"{scope_prefix}.{value}"


def _resolve_bind(node_spec: dict[str, Any], model: Any) -> tuple[str, str | None]:
    out = node_spec.get("_bind")
    if isinstance(out, str):
        return model._require_name(out, field="mamba2_scan._bind"), None
    if isinstance(out, list) and len(out) == 2:
        return (
            model._require_name(out[0], field="mamba2_scan._bind[0]"),
            model._require_name(out[1], field="mamba2_scan._bind[1]"),
        )
    raise ValueError("mamba2_scan requires _bind as a single output or [y, final_state]")


def _gated_group_rmsnorm(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    *,
    n_groups: int,
) -> torch.Tensor:
    gated = x * F.silu(gate.to(dtype=x.dtype))
    work = gated.to(torch.float32)
    group_size = int(work.shape[-1]) // int(n_groups)
    if group_size <= 0 or work.shape[-1] % int(n_groups) != 0:
        raise ValueError("mamba2_scan requires hidden dim divisible by n_groups")
    work_view = work.view(*work.shape[:-1], int(n_groups), group_size)
    variance = work_view.pow(2).mean(dim=-1, keepdim=True)
    normed = (work_view * torch.rsqrt(variance + 1e-5)).view_as(work)
    return (normed * weight.to(dtype=normed.dtype)).to(dtype=x.dtype)


def _compute_scan(
    *,
    u: torch.Tensor,
    gate: torch.Tensor,
    dt: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    a_log: torch.Tensor,
    d: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    n_groups: int,
    head_dim: int,
    time_step_min: float,
    time_step_max: float,
    state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if u.ndim != 3 or gate.ndim != 3:
        raise ValueError("mamba2_scan expects u/gate shape [batch, seq, dim]")
    if dt.ndim != 3:
        raise ValueError("mamba2_scan expects dt shape [batch, seq, num_heads]")
    if b.ndim != 3 or c.ndim != 3:
        raise ValueError("mamba2_scan expects B/C shape [batch, seq, groups*state_dim]")
    if u.shape != gate.shape:
        raise ValueError("mamba2_scan requires u and gate to share shape [batch, seq, dim]")
    if b.shape != c.shape:
        raise ValueError("mamba2_scan requires B and C to share shape [batch, seq, groups*state_dim]")
    if a_log.ndim != 1 or d.ndim != 1 or dt_bias.ndim != 1:
        raise ValueError("mamba2_scan expects A/D/dt_bias rank-1 tensors")
    if norm_weight.ndim != 1:
        raise ValueError("mamba2_scan expects norm_weight rank-1 tensor")

    batch, seq, dim = map(int, u.shape)
    num_heads = int(dt.shape[-1])
    if dim != num_heads * int(head_dim):
        raise ValueError("mamba2_scan hidden dim must equal num_heads * head_dim")
    if int(a_log.shape[0]) != num_heads:
        raise ValueError("mamba2_scan A must have shape [num_heads]")
    if int(d.shape[0]) != num_heads:
        raise ValueError("mamba2_scan D must have shape [num_heads]")
    if int(dt_bias.shape[0]) != num_heads:
        raise ValueError("mamba2_scan dt_bias must have shape [num_heads]")
    if int(norm_weight.shape[0]) != dim:
        raise ValueError("mamba2_scan norm_weight must have shape [dim]")
    if b.shape[0] != batch or b.shape[1] != seq:
        raise ValueError("mamba2_scan B/C first dims must match [batch, seq]")
    if int(n_groups) <= 0 or num_heads % int(n_groups) != 0:
        raise ValueError("mamba2_scan requires num_heads divisible by n_groups")
    if int(b.shape[-1]) % int(n_groups) != 0:
        raise ValueError("mamba2_scan B/C last dim must be divisible by n_groups")

    state_dim = int(b.shape[-1]) // int(n_groups)
    work_dtype = torch.float32 if u.dtype in {torch.float16, torch.bfloat16} else u.dtype
    u_work = u.to(dtype=work_dtype)
    gate_work = gate.to(dtype=work_dtype)
    dt_work = F.softplus(dt.to(dtype=work_dtype) + dt_bias.to(dtype=work_dtype).view(1, 1, -1))
    if time_step_min != 0.0 or time_step_max != float("inf"):
        dt_work = torch.clamp(dt_work, min=float(time_step_min), max=float(time_step_max))
    b_work = b.to(dtype=work_dtype).view(batch, seq, int(n_groups), state_dim)
    c_work = c.to(dtype=work_dtype).view(batch, seq, int(n_groups), state_dim)
    if num_heads != int(n_groups):
        rep = num_heads // int(n_groups)
        b_work = b_work.repeat_interleave(rep, dim=2, output_size=num_heads)
        c_work = c_work.repeat_interleave(rep, dim=2, output_size=num_heads)
    hidden_work = u_work.view(batch, seq, num_heads, int(head_dim))
    a_work = -torch.exp(a_log.to(dtype=torch.float32)).to(dtype=work_dtype)
    d_work = d.to(dtype=work_dtype).view(1, num_heads, 1)

    if state is None:
        state_work = torch.zeros(
            (batch, num_heads, int(head_dim), state_dim),
            device=u.device,
            dtype=work_dtype,
        )
    else:
        if state.ndim != 4:
            raise ValueError("mamba2_scan state must be rank-4 [batch, heads, head_dim, state_dim]")
        if tuple(state.shape) != (batch, num_heads, int(head_dim), state_dim):
            raise ValueError("mamba2_scan state shape mismatch")
        state_work = state.to(dtype=work_dtype)

    outputs: list[torch.Tensor] = []
    for t in range(seq):
        dt_t = dt_work[:, t, :]
        b_t = b_work[:, t, :, :]
        c_t = c_work[:, t, :, :]
        hidden_t = hidden_work[:, t, :, :]
        d_a = torch.exp(dt_t[..., None, None] * a_work.view(1, num_heads, 1, 1))
        d_b = dt_t[..., None, None] * b_t[:, :, None, :]
        d_bx = d_b * hidden_t[..., None]
        state_work = state_work * d_a + d_bx
        y_t = torch.matmul(state_work, c_t[..., None]).squeeze(-1)
        y_t = y_t + hidden_t * d_work
        outputs.append(y_t.reshape(batch, dim))

    y = torch.stack(outputs, dim=1)
    y = _gated_group_rmsnorm(y, gate_work, norm_weight, n_groups=int(n_groups))
    return y.to(dtype=u.dtype), state_work.to(dtype=u.dtype)


def _resolve_interpret_inputs(
    model: Any,
    node_spec: dict[str, Any],
    env: dict[str, Any],
    *,
    node_path: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    args = node_spec.get("_args")
    if not isinstance(args, list) or len(args) not in {5, 6}:
        raise ValueError("mamba2_scan expects 5/6 inputs [u,gate,dt,B,C,?state]")
    state = env.get(str(args[5])) if len(args) == 6 else None
    return env[str(args[0])], env[str(args[1])], env[str(args[2])], env[str(args[3])], env[str(args[4])], state


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
    u, gate, dt, b, c, state = _resolve_interpret_inputs(model, node_spec, env, node_path=node_path)
    a_log = model._state[model._infer_param_path(node_spec, node_path=node_path, param_name="A")]
    d = model._state[model._infer_param_path(node_spec, node_path=node_path, param_name="D")]
    dt_bias = model._state[model._infer_param_path(node_spec, node_path=node_path, param_name="dt_bias")]
    norm_weight = model._state[
        model._infer_param_path(node_spec, node_path=node_path, param_name="norm_weight")
    ]
    n_groups = int(model._eval_expr(node_spec.get("n_groups"), env, symbols))
    head_dim = int(model._eval_expr(node_spec.get("head_dim"), env, symbols))
    time_step_min = float(model._eval_expr(node_spec.get("time_step_min", 0.0), env, symbols))
    time_step_max = float(
        model._eval_expr(node_spec.get("time_step_max", float("inf")), env, symbols)
    )
    y, final_state = _compute_scan(
        u=u,
        gate=gate,
        dt=dt,
        b=b,
        c=c,
        a_log=a_log,
        d=d,
        dt_bias=dt_bias,
        norm_weight=norm_weight,
        n_groups=n_groups,
        head_dim=head_dim,
        time_step_min=time_step_min,
        time_step_max=time_step_max,
        state=state,
    )
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
    if not isinstance(args, list) or len(args) not in {5, 6}:
        raise ValueError("mamba2_scan expects 5/6 inputs [u,gate,dt,B,C,?state]")
    out = node_spec.get("_bind")
    if isinstance(out, str):
        y_out = emitter._assign_out_var(env, out)
        state_out = None
    elif isinstance(out, list) and len(out) == 2:
        y_out = emitter._assign_out_var(env, str(out[0]))
        state_out = emitter._assign_out_var(env, str(out[1]))
    else:
        raise ValueError("mamba2_scan requires _bind as a single output or [y, final_state]")

    u = emitter._read_env_var(env, str(args[0]))
    gate = emitter._read_env_var(env, str(args[1]))
    dt = emitter._read_env_var(env, str(args[2]))
    b = emitter._read_env_var(env, str(args[3]))
    c = emitter._read_env_var(env, str(args[4]))
    state_in = emitter._read_env_var(env, str(args[5])) if len(args) == 6 else "None"

    lines: list[str] = []
    a_log = emitter._hoisted_param(node_spec=node_spec, node_path_var=node_path_var, param_name="A", lines=lines, indent=indent)
    d = emitter._hoisted_param(node_spec=node_spec, node_path_var=node_path_var, param_name="D", lines=lines, indent=indent)
    dt_bias = emitter._hoisted_param(node_spec=node_spec, node_path_var=node_path_var, param_name="dt_bias", lines=lines, indent=indent)
    norm_weight = emitter._hoisted_param(node_spec=node_spec, node_path_var=node_path_var, param_name="norm_weight", lines=lines, indent=indent)
    n_groups = emitter._expr_code(node_spec.get("n_groups"), env)
    head_dim = emitter._expr_code(node_spec.get("head_dim"), env)
    time_step_min = emitter._expr_code(node_spec.get("time_step_min", 0.0), env)
    time_step_max = emitter._expr_code(node_spec.get("time_step_max", float("inf")), env)

    batch = emitter._fresh("batch")
    seq = emitter._fresh("seq")
    dim = emitter._fresh("dim")
    num_heads = emitter._fresh("num_heads")
    state_dim = emitter._fresh("state_dim")
    work_dtype = emitter._fresh("work_dtype")
    gate_work = emitter._fresh("gate_work")
    u_work = emitter._fresh("u_work")
    dt_work = emitter._fresh("dt_work")
    b_work = emitter._fresh("b_work")
    c_work = emitter._fresh("c_work")
    hidden_work = emitter._fresh("hidden_work")
    a_work = emitter._fresh("a_work")
    d_work = emitter._fresh("d_work")
    state_work = emitter._fresh("state_work")
    outputs = emitter._fresh("outputs")
    t = emitter._fresh("t")
    dt_t = emitter._fresh("dt_t")
    b_t = emitter._fresh("b_t")
    c_t = emitter._fresh("c_t")
    hidden_t = emitter._fresh("hidden_t")
    d_a = emitter._fresh("d_a")
    d_b = emitter._fresh("d_b")
    d_bx = emitter._fresh("d_bx")
    y_t = emitter._fresh("y_t")
    group_size = emitter._fresh("group_size")
    gated = emitter._fresh("gated")
    work = emitter._fresh("work")
    work_view = emitter._fresh("work_view")
    variance = emitter._fresh("variance")

    lines.extend(
        [
            f"{indent}if {u}.ndim != 3 or {gate}.ndim != 3:",
            f"{indent}    raise ValueError('mamba2_scan expects u/gate shape [batch, seq, dim]')",
            f"{indent}if {dt}.ndim != 3:",
            f"{indent}    raise ValueError('mamba2_scan expects dt shape [batch, seq, num_heads]')",
            f"{indent}if {b}.ndim != 3 or {c}.ndim != 3:",
            f"{indent}    raise ValueError('mamba2_scan expects B/C shape [batch, seq, groups*state_dim]')",
            f"{indent}if tuple({u}.shape) != tuple({gate}.shape):",
            f"{indent}    raise ValueError('mamba2_scan requires u and gate to share shape [batch, seq, dim]')",
            f"{indent}if tuple({b}.shape) != tuple({c}.shape):",
            f"{indent}    raise ValueError('mamba2_scan requires B and C to share shape [batch, seq, groups*state_dim]')",
            f"{indent}if {a_log}.ndim != 1 or {d}.ndim != 1 or {dt_bias}.ndim != 1:",
            f"{indent}    raise ValueError('mamba2_scan expects A/D/dt_bias rank-1 tensors')",
            f"{indent}if {norm_weight}.ndim != 1:",
            f"{indent}    raise ValueError('mamba2_scan expects norm_weight rank-1 tensor')",
            f"{indent}{batch} = int({u}.shape[0])",
            f"{indent}{seq} = int({u}.shape[1])",
            f"{indent}{dim} = int({u}.shape[2])",
            f"{indent}{num_heads} = int({dt}.shape[2])",
            f"{indent}if {dim} != {num_heads} * int({head_dim}):",
            f"{indent}    raise ValueError('mamba2_scan hidden dim must equal num_heads * head_dim')",
            f"{indent}if int({a_log}.shape[0]) != {num_heads}:",
            f"{indent}    raise ValueError('mamba2_scan A must have shape [num_heads]')",
            f"{indent}if int({d}.shape[0]) != {num_heads}:",
            f"{indent}    raise ValueError('mamba2_scan D must have shape [num_heads]')",
            f"{indent}if int({dt_bias}.shape[0]) != {num_heads}:",
            f"{indent}    raise ValueError('mamba2_scan dt_bias must have shape [num_heads]')",
            f"{indent}if int({norm_weight}.shape[0]) != {dim}:",
            f"{indent}    raise ValueError('mamba2_scan norm_weight must have shape [dim]')",
            f"{indent}if int({b}.shape[0]) != {batch} or int({b}.shape[1]) != {seq}:",
            f"{indent}    raise ValueError('mamba2_scan B/C first dims must match [batch, seq]')",
            f"{indent}if int({n_groups}) <= 0 or {num_heads} % int({n_groups}) != 0:",
            f"{indent}    raise ValueError('mamba2_scan requires num_heads divisible by n_groups')",
            f"{indent}if int({b}.shape[2]) % int({n_groups}) != 0:",
            f"{indent}    raise ValueError('mamba2_scan B/C last dim must be divisible by n_groups')",
            f"{indent}{state_dim} = int({b}.shape[2]) // int({n_groups})",
            f"{indent}{work_dtype} = torch.float32 if {u}.dtype in {{torch.float16, torch.bfloat16}} else {u}.dtype",
            f"{indent}{u_work} = {u}.to(dtype={work_dtype})",
            f"{indent}{gate_work} = {gate}.to(dtype={work_dtype})",
            f"{indent}{dt_work} = F.softplus({dt}.to(dtype={work_dtype}) + {dt_bias}.to(dtype={work_dtype}).view(1, 1, -1))",
            f"{indent}if float({time_step_min}) != 0.0 or float({time_step_max}) != float('inf'):",
            f"{indent}    {dt_work} = torch.clamp({dt_work}, min=float({time_step_min}), max=float({time_step_max}))",
            f"{indent}{b_work} = {b}.to(dtype={work_dtype}).view({batch}, {seq}, int({n_groups}), {state_dim})",
            f"{indent}{c_work} = {c}.to(dtype={work_dtype}).view({batch}, {seq}, int({n_groups}), {state_dim})",
            f"{indent}if {num_heads} != int({n_groups}):",
            f"{indent}    _rep = {num_heads} // int({n_groups})",
            f"{indent}    {b_work} = {b_work}.repeat_interleave(_rep, dim=2, output_size={num_heads})",
            f"{indent}    {c_work} = {c_work}.repeat_interleave(_rep, dim=2, output_size={num_heads})",
            f"{indent}{hidden_work} = {u_work}.view({batch}, {seq}, {num_heads}, int({head_dim}))",
            f"{indent}{a_work} = (-torch.exp({a_log}.to(dtype=torch.float32))).to(dtype={work_dtype})",
            f"{indent}{d_work} = {d}.to(dtype={work_dtype}).view(1, {num_heads}, 1)",
            f"{indent}if {state_in} is None:",
            f"{indent}    {state_work} = torch.zeros(({batch}, {num_heads}, int({head_dim}), {state_dim}), device={u}.device, dtype={work_dtype})",
            f"{indent}else:",
            f"{indent}    if {state_in}.ndim != 4:",
            f"{indent}        raise ValueError('mamba2_scan state must be rank-4 [batch, heads, head_dim, state_dim]')",
            f"{indent}    if tuple({state_in}.shape) != ({batch}, {num_heads}, int({head_dim}), {state_dim}):",
            f"{indent}        raise ValueError('mamba2_scan state shape mismatch')",
            f"{indent}    {state_work} = {state_in}.to(dtype={work_dtype})",
            f"{indent}{outputs} = []",
            f"{indent}for {t} in range({seq}):",
            f"{indent}    {dt_t} = {dt_work}[:, {t}, :]",
            f"{indent}    {b_t} = {b_work}[:, {t}, :, :]",
            f"{indent}    {c_t} = {c_work}[:, {t}, :, :]",
            f"{indent}    {hidden_t} = {hidden_work}[:, {t}, :, :]",
            f"{indent}    {d_a} = torch.exp({dt_t}[..., None, None] * {a_work}.view(1, {num_heads}, 1, 1))",
            f"{indent}    {d_b} = {dt_t}[..., None, None] * {b_t}[:, :, None, :]",
            f"{indent}    {d_bx} = {d_b} * {hidden_t}[..., None]",
            f"{indent}    {state_work} = {state_work} * {d_a} + {d_bx}",
            f"{indent}    {y_t} = torch.matmul({state_work}, {c_t}[..., None]).squeeze(-1)",
            f"{indent}    {y_t} = {y_t} + {hidden_t} * {d_work}",
            f"{indent}    {outputs}.append({y_t}.reshape({batch}, {dim}))",
            f"{indent}{gated} = torch.stack({outputs}, dim=1)",
            f"{indent}{gated} = {gated} * F.silu({gate_work}.to(dtype={gated}.dtype))",
            f"{indent}{work} = {gated}.to(torch.float32)",
            f"{indent}{group_size} = int({work}.shape[-1]) // int({n_groups})",
            f"{indent}if {group_size} <= 0 or int({work}.shape[-1]) % int({n_groups}) != 0:",
            f"{indent}    raise ValueError('mamba2_scan requires hidden dim divisible by n_groups')",
            f"{indent}{work_view} = {work}.view(*{work}.shape[:-1], int({n_groups}), {group_size})",
            f"{indent}{variance} = {work_view}.pow(2).mean(dim=-1, keepdim=True)",
            f"{indent}{y_out} = (({work_view} * torch.rsqrt({variance} + 1e-5)).view_as({work}) * {norm_weight}.to(dtype={work}.dtype)).to(dtype={u}.dtype)",
        ]
    )
    if state_out is not None:
        lines.append(f"{indent}{state_out} = {state_work}.to(dtype={u}.dtype)")
    return lines


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any", "Any", "Any", "Any", "Any", "Any"),
    "kwargs": dict(LOWERING_KWARG_KINDS),
    "returns": "dynamic",
}

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
    "LOWERING_TYPE_SIGNATURE",
]
