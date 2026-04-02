from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

OP_NAME = "causal_conv1d"
LOWERING_ARITY = (1, 2)
LOWERING_ALLOWED_KWARGS: set[str] = {"activation"}
LOWERING_REQUIRED_KWARGS: set[str] = set()
LOWERING_KWARG_KINDS: dict[str, Any] = {"activation": "str"}


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return True


def lowering_known_output_arity(*, kwargs: dict[str, Any]) -> int | None:
    del kwargs
    return None


def _activation(name: str, x: torch.Tensor) -> torch.Tensor:
    if name == "silu":
        return F.silu(x)
    if name == "relu":
        return F.relu(x)
    if name == "gelu":
        return F.gelu(x)
    raise ValueError(f"Unsupported causal_conv1d activation: {name!r}")


def _compute_new_state(x_t: torch.Tensor, kernel_size: int) -> torch.Tensor:
    seq = int(x_t.shape[-1])
    if seq >= kernel_size:
        return x_t[..., seq - kernel_size :]
    return F.pad(x_t, (kernel_size - seq, 0))


def _decode_step(
    x_t: torch.Tensor,
    state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    new_state = torch.roll(state, shifts=-1, dims=-1)
    new_state[:, :, -1] = x_t[:, :, 0]
    y_t = torch.sum(new_state * weight[:, 0, :], dim=-1)
    if bias is not None:
        y_t = y_t + bias
    y = _activation(activation, y_t).unsqueeze(1)
    return y, new_state


def _full_conv(
    x_t: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    kernel_size = int(weight.shape[-1])
    y_t = F.conv1d(
        x_t,
        weight,
        bias=bias,
        stride=1,
        padding=kernel_size - 1,
        groups=int(x_t.shape[1]),
    )
    y_t = y_t[..., : int(x_t.shape[-1])]
    y_t = _activation(activation, y_t)
    y = y_t.transpose(1, 2).contiguous()
    return y, _compute_new_state(x_t, kernel_size)


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
    args = node_spec.get("_args")
    if isinstance(args, list):
        if len(args) not in {1, 2}:
            raise ValueError("causal_conv1d expects args [x] or [x, state]")
        x = env[str(args[0])]
        state = env.get(str(args[1])) if len(args) == 2 else None
    else:
        x = env[str(args)]
        state = None

    if not torch.is_tensor(x) or x.ndim != 3:
        raise ValueError("causal_conv1d expects input x shape [batch, seq, channels]")

    weight_path = model._infer_param_path(node_spec, node_path=node_path, param_name="weight")
    weight = model._state[weight_path]
    bias = None
    bias_path = model._infer_param_path(node_spec, node_path=node_path, param_name="bias")
    if bias_path in model._state:
        bias = model._state[bias_path]

    if int(weight.ndim) != 3:
        raise ValueError("causal_conv1d weight must have shape [channels, 1, kernel]")
    channels = int(x.shape[2])
    if int(weight.shape[0]) != channels or int(weight.shape[1]) != 1:
        raise ValueError("causal_conv1d weight shape must be [channels, 1, kernel] matching input")
    if bias is not None and (bias.ndim != 1 or int(bias.shape[0]) != channels):
        raise ValueError("causal_conv1d bias must have shape [channels]")

    activation = str(node_spec.get("activation", "silu"))
    x_t = x.transpose(1, 2).contiguous()
    kernel_size = int(weight.shape[-1])
    use_decode = state is not None and int(x.shape[1]) == 1
    if use_decode:
        if not torch.is_tensor(state) or state.ndim != 3:
            raise ValueError("causal_conv1d state must be [batch, channels, kernel]")
        if tuple(state.shape) != (int(x.shape[0]), channels, kernel_size):
            raise ValueError("causal_conv1d state shape mismatch")
        y, new_state = _decode_step(x_t, state, weight, bias, activation)
    else:
        y, new_state = _full_conv(x_t, weight, bias, activation)

    out = node_spec.get("_bind")
    if isinstance(out, str):
        env[model._require_name(out, field="causal_conv1d._bind")] = y
        return
    if isinstance(out, list) and len(out) == 2:
        env[model._require_name(out[0], field="causal_conv1d._bind[0]")] = y
        env[model._require_name(out[1], field="causal_conv1d._bind[1]")] = new_state
        return
    raise ValueError("causal_conv1d requires _bind as y or [y, state]")


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
    if isinstance(args, list):
        if len(args) not in {1, 2}:
            raise ValueError("causal_conv1d expects args [x] or [x, state]")
        x = emitter._read_env_var(env, str(args[0]))
        state_in = emitter._read_env_var(env, str(args[1])) if len(args) == 2 else "None"
    else:
        x = emitter._read_env_var(env, str(args))
        state_in = "None"

    out = node_spec.get("_bind")
    if isinstance(out, str):
        y_out = emitter._assign_out_var(env, out)
        state_out = None
    elif isinstance(out, list) and len(out) == 2:
        y_out = emitter._assign_out_var(env, str(out[0]))
        state_out = emitter._assign_out_var(env, str(out[1]))
    else:
        raise ValueError("causal_conv1d requires _bind as y or [y, state]")

    act_name = str(node_spec.get("activation", "silu"))
    if act_name == "silu":
        act_fn = "F.silu"
    elif act_name == "relu":
        act_fn = "F.relu"
    elif act_name == "gelu":
        act_fn = "F.gelu"
    else:
        raise ValueError(f"Unsupported causal_conv1d activation: {act_name!r}")

    weight_expr = emitter._infer_param_expr(node_spec, node_path_var, "weight")
    bias_expr = emitter._infer_param_expr(node_spec, node_path_var, "bias")
    weight = emitter._fresh("w")
    bias = emitter._fresh("b")
    x_t = emitter._fresh("x_t")
    channels = emitter._fresh("ch")
    kernel = emitter._fresh("k")
    use_decode = emitter._fresh("use_decode")
    state = emitter._fresh("state")
    y_t = emitter._fresh("y_t")
    new_state = emitter._fresh("new_state")

    lines = [
        f"{indent}if not torch.is_tensor({x}) or {x}.ndim != 3:",
        f"{indent}    raise ValueError('causal_conv1d expects input x shape [batch, seq, channels]')",
        f"{indent}{weight} = emitter._param({weight_expr})",
        f"{indent}{bias} = self._state.get({bias_expr})",
        f"{indent}if int({weight}.ndim) != 3:",
        f"{indent}    raise ValueError('causal_conv1d weight must have shape [channels, 1, kernel]')",
        f"{indent}{channels} = int({x}.shape[2])",
        f"{indent}if int({weight}.shape[0]) != {channels} or int({weight}.shape[1]) != 1:",
        f"{indent}    raise ValueError('causal_conv1d weight shape must be [channels, 1, kernel] matching input')",
        f"{indent}if {bias} is not None and ({bias}.ndim != 1 or int({bias}.shape[0]) != {channels}):",
        f"{indent}    raise ValueError('causal_conv1d bias must have shape [channels]')",
        f"{indent}{x_t} = {x}.transpose(1, 2).contiguous()",
        f"{indent}{kernel} = int({weight}.shape[-1])",
        f"{indent}{use_decode} = ({state_in} is not None and int({x}.shape[1]) == 1)",
        f"{indent}if {use_decode}:",
        f"{indent}    if (not torch.is_tensor({state_in})) or {state_in}.ndim != 3:",
        f"{indent}        raise ValueError('causal_conv1d state must be [batch, channels, kernel]')",
        f"{indent}    if tuple({state_in}.shape) != (int({x}.shape[0]), {channels}, {kernel}):",
        f"{indent}        raise ValueError('causal_conv1d state shape mismatch')",
        f"{indent}    {state} = torch.roll({state_in}, shifts=-1, dims=-1)",
        f"{indent}    {state}[:, :, -1] = {x_t}[:, :, 0]",
        f"{indent}    {y_t} = torch.sum({state} * {weight}[:, 0, :], dim=-1)",
        f"{indent}    if {bias} is not None:",
        f"{indent}        {y_t} = {y_t} + {bias}",
        f"{indent}    {y_out} = {act_fn}({y_t}).unsqueeze(1)",
        f"{indent}    {new_state} = {state}",
        f"{indent}else:",
        f"{indent}    {y_t} = F.conv1d({x_t}, {weight}, bias={bias}, stride=1, padding={kernel} - 1, groups={channels})",
        f"{indent}    {y_t} = {y_t}[..., : int({x_t}.shape[-1])]",
        f"{indent}    {y_t} = {act_fn}({y_t})",
        f"{indent}    {y_out} = {y_t}.transpose(1, 2).contiguous()",
        f"{indent}    if int({x_t}.shape[-1]) >= {kernel}:",
        f"{indent}        {new_state} = {x_t}[..., int({x_t}.shape[-1]) - {kernel}:]",
        f"{indent}    else:",
        f"{indent}        {new_state} = F.pad({x_t}, ({kernel} - int({x_t}.shape[-1]), 0))",
    ]
    if state_out is not None:
        lines.append(f"{indent}{state_out} = {new_state}")
    return lines


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any", "Any"),
    "kwargs": dict(LOWERING_KWARG_KINDS),
    "returns": "dynamic",
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
