from __future__ import annotations

import math
from typing import Any

import torch
from torch.nn import functional as F

OP_NAME = "activation"
LOWERING_ARITY = (1, 1)
LOWERING_ALLOWED_KWARGS: set[str] = {"fp32_accum", "limit"}
LOWERING_REQUIRED_KWARGS: set[str] = set()
LOWERING_KWARG_KINDS: dict[str, Any] = {"fp32_accum": "bool", "limit": "number"}

_XIELU_ALPHA_P_INIT = 0.8
_XIELU_ALPHA_N_INIT = 0.8
_XIELU_BETA_INIT = 0.5
_XIELU_EPS_INIT = -1.0e-6
_XIELU_ALPHA_P_PARAM_DEFAULT = math.log(math.expm1(_XIELU_ALPHA_P_INIT))
_XIELU_ALPHA_N_PARAM_DEFAULT = math.log(math.expm1(_XIELU_ALPHA_N_INIT - _XIELU_BETA_INIT))


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter
    op_name = node_spec.get("_op")
    return op_name == "activations_xielu"


def _resolve_xielu_param(
    *,
    model: Any,
    node_spec: dict[str, Any],
    node_path: str,
    param_name: str,
    default: float,
    x: torch.Tensor,
) -> torch.Tensor:
    target_dtype = x.dtype if x.is_floating_point() else torch.float32
    param_path = model._infer_param_path(node_spec, node_path=node_path, param_name=param_name)
    param_value = model._state.get(param_path)
    if torch.is_tensor(param_value):
        return param_value.to(device=x.device, dtype=target_dtype)
    return torch.tensor(default, device=x.device, dtype=target_dtype)


def lowering_infer_metadata(
    *,
    args: list[str],
    out: str | list[str],
    kwargs: dict[str, Any],
    ctx: Any,
) -> bool:
    del kwargs
    if not isinstance(out, str):
        return False
    first_in = args[0].strip() if args else None
    if isinstance(first_in, str) and first_in.isidentifier():
        first_dim = ctx.tensor_last_dim.get(first_in)
        if first_dim is not None:
            ctx.tensor_last_dim[out] = first_dim
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
    x = model._read_tensor_input(node_spec.get("_args"), env)
    op_name = node_spec.get("_op")
    if not isinstance(op_name, str) or not op_name.startswith("activations_"):
        raise ValueError("legacy activation node name; use _op: activations_<kind>")
    kind = op_name[len("activations_") :]
    out = model._require_name(node_spec.get("_bind"), field="activation._bind")
    align_activation_fp32 = bool(node_spec.get("fp32_accum", False))
    if kind == "gelu_new" or kind == "gelu_pytorch_tanh":
        if (
            align_activation_fp32
            and x.is_floating_point()
            and x.dtype in {torch.float16, torch.bfloat16}
        ):
            x_fp32 = x.float()
            y_fp32 = (
                0.5
                * x_fp32
                * (
                    1.0
                    + torch.tanh(
                        0.7978845608028654 * (x_fp32 + 0.044715 * x_fp32 * x_fp32 * x_fp32)
                    )
                )
            )
            env[out] = y_fp32.to(dtype=x.dtype)
        else:
            env[out] = 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
    elif kind == "gegelu":
        if x.shape[-1] % 2 != 0:
            raise ValueError("gegelu requires even last dimension")
        x_gelu = x[..., ::2]
        x_linear = x[..., 1::2]
        limit_expr = node_spec.get("limit")
        limit_value = (
            None if limit_expr is None else float(model._eval_expr(limit_expr, env, symbols))
        )
        if limit_value is not None:
            x_gelu = torch.where(
                torch.isinf(x_gelu),
                x_gelu,
                x_gelu.clamp(min=None, max=limit_value),
            )
            x_linear = torch.where(
                torch.isinf(x_linear),
                x_linear,
                x_linear.clamp(min=-limit_value, max=limit_value),
            )
        if (
            align_activation_fp32
            and x.is_floating_point()
            and x.dtype in {torch.float16, torch.bfloat16}
        ):
            x_gelu_fp32 = x_gelu.float()
            x_linear_fp32 = x_linear.float()
            y_fp32 = x_gelu_fp32 * torch.sigmoid(1.702 * x_gelu_fp32) * (x_linear_fp32 + 1.0)
            env[out] = y_fp32.to(dtype=x.dtype)
        else:
            env[out] = x_gelu * torch.sigmoid(1.702 * x_gelu) * (x_linear + 1.0)
    elif kind == "gelu":
        if (
            align_activation_fp32
            and x.is_floating_point()
            and x.dtype in {torch.float16, torch.bfloat16}
        ):
            env[out] = F.gelu(x.float()).to(dtype=x.dtype)
        else:
            env[out] = F.gelu(x)
    elif kind == "relu":
        env[out] = F.relu(x)
    elif kind == "relu2":
        y = F.relu(x)
        env[out] = y * y
    elif kind == "silu":
        env[out] = F.silu(x)
    elif kind == "swiglu":
        env[out] = F.silu(x) * x
    elif kind == "sigmoid":
        env[out] = torch.sigmoid(x)
    elif kind == "tanh":
        env[out] = torch.tanh(x)
    elif kind == "xielu":
        alpha_p_param = _resolve_xielu_param(
            model=model,
            node_spec=node_spec,
            node_path=node_path,
            param_name="alpha_p",
            default=_XIELU_ALPHA_P_PARAM_DEFAULT,
            x=x,
        )
        alpha_n_param = _resolve_xielu_param(
            model=model,
            node_spec=node_spec,
            node_path=node_path,
            param_name="alpha_n",
            default=_XIELU_ALPHA_N_PARAM_DEFAULT,
            x=x,
        )
        beta = _resolve_xielu_param(
            model=model,
            node_spec=node_spec,
            node_path=node_path,
            param_name="beta",
            default=_XIELU_BETA_INIT,
            x=x,
        )
        eps = _resolve_xielu_param(
            model=model,
            node_spec=node_spec,
            node_path=node_path,
            param_name="eps",
            default=_XIELU_EPS_INIT,
            x=x,
        )
        alpha_p = F.softplus(alpha_p_param)
        alpha_n = beta + F.softplus(alpha_n_param)
        env[out] = torch.where(
            x > 0,
            alpha_p * x * x + beta * x,
            (torch.expm1(torch.minimum(x, eps)) - x) * alpha_n + beta * x,
        )
    else:
        raise ValueError(f"Unsupported activation kind: {kind}")
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

    src = read(str(node_spec.get("_args")))
    out_name = str(node_spec.get("_bind"))
    op_name = node_spec.get("_op")
    if not isinstance(op_name, str) or not op_name.startswith("activations_"):
        raise ValueError("legacy activation node name; use _op: activations_<kind>")
    kind = op_name[len("activations_") :]
    out_var = assign_out_var(out_name)
    relu2_tmp = emitter._fresh("relu2_tmp")
    if kind in {"gelu_new", "gelu_pytorch_tanh"}:
        if bool(node_spec.get("fp32_accum", False)):
            lines.append(
                f"{indent}if {src}.is_floating_point() and {src}.dtype in {{torch.float16, torch.bfloat16}}:"
            )
            lines.append(f"{indent}    _x_fp32 = {src}.float()")
            lines.append(
                f"{indent}    {out_var} = (0.5 * _x_fp32 * (1.0 + torch.tanh(0.7978845608028654 * (_x_fp32 + 0.044715 * _x_fp32 * _x_fp32 * _x_fp32)))).to(dtype={src}.dtype)"
            )
            lines.append(f"{indent}else:")
            lines.append(
                f"{indent}    {out_var} = 0.5 * {src} * (1.0 + torch.tanh(0.7978845608028654 * ({src} + 0.044715 * {src} * {src} * {src})))"
            )
        else:
            lines.append(
                f"{indent}{out_var} = 0.5 * {src} * (1.0 + torch.tanh(0.7978845608028654 * ({src} + 0.044715 * {src} * {src} * {src})))"
            )
    elif kind == "gegelu":
        x_gelu_var = emitter._fresh("gegelu_x_gelu")
        x_linear_var = emitter._fresh("gegelu_x_linear")
        limit_expr = emitter._expr_code(node_spec.get("limit"), env)
        limit_var = emitter._fresh("gegelu_limit")
        lines.append(f"{indent}if ({src}.shape[-1] % 2) != 0:")
        lines.append(f"{indent}    raise ValueError('gegelu requires even last dimension')")
        lines.append(f"{indent}{x_gelu_var} = {src}[..., ::2]")
        lines.append(f"{indent}{x_linear_var} = {src}[..., 1::2]")
        lines.append(f"{indent}{limit_var} = {limit_expr}")
        lines.append(f"{indent}if {limit_var} is not None:")
        lines.append(
            f"{indent}    {x_gelu_var} = torch.where(torch.isinf({x_gelu_var}), {x_gelu_var}, {x_gelu_var}.clamp(min=None, max=float({limit_var})))"
        )
        lines.append(
            f"{indent}    {x_linear_var} = torch.where(torch.isinf({x_linear_var}), {x_linear_var}, {x_linear_var}.clamp(min=-float({limit_var}), max=float({limit_var})))"
        )
        if bool(node_spec.get("fp32_accum", False)):
            lines.append(
                f"{indent}if {src}.is_floating_point() and {src}.dtype in {{torch.float16, torch.bfloat16}}:"
            )
            lines.append(f"{indent}    _x_gelu_fp32 = {x_gelu_var}.float()")
            lines.append(f"{indent}    _x_linear_fp32 = {x_linear_var}.float()")
            lines.append(
                f"{indent}    {out_var} = (_x_gelu_fp32 * torch.sigmoid(1.702 * _x_gelu_fp32) * (_x_linear_fp32 + 1.0)).to(dtype={src}.dtype)"
            )
            lines.append(f"{indent}else:")
            lines.append(
                f"{indent}    {out_var} = {x_gelu_var} * torch.sigmoid(1.702 * {x_gelu_var}) * ({x_linear_var} + 1.0)"
            )
        else:
            lines.append(
                f"{indent}{out_var} = {x_gelu_var} * torch.sigmoid(1.702 * {x_gelu_var}) * ({x_linear_var} + 1.0)"
            )
    elif kind == "gelu":
        if bool(node_spec.get("fp32_accum", False)):
            lines.append(
                f"{indent}if {src}.is_floating_point() and {src}.dtype in {{torch.float16, torch.bfloat16}}:"
            )
            lines.append(f"{indent}    {out_var} = F.gelu({src}.float()).to(dtype={src}.dtype)")
            lines.append(f"{indent}else:")
            lines.append(f"{indent}    {out_var} = F.gelu({src})")
        else:
            lines.append(f"{indent}{out_var} = F.gelu({src})")
    elif kind == "relu2":
        lines.append(f"{indent}{relu2_tmp} = F.relu({src})")
        lines.append(f"{indent}{out_var} = {relu2_tmp} * {relu2_tmp}")
    elif kind == "relu":
        lines.append(f"{indent}{out_var} = F.relu({src})")
    elif kind == "silu":
        lines.append(f"{indent}{out_var} = F.silu({src})")
    elif kind == "swiglu":
        lines.append(f"{indent}{out_var} = F.silu({src}) * {src}")
    elif kind == "sigmoid":
        lines.append(f"{indent}{out_var} = torch.sigmoid({src})")
    elif kind == "tanh":
        lines.append(f"{indent}{out_var} = torch.tanh({src})")
    elif kind == "xielu":
        target_dtype = f"({src}.dtype if {src}.is_floating_point() else torch.float32)"
        alpha_p_path_var = emitter._fresh("xielu_alpha_p_path")
        alpha_n_path_var = emitter._fresh("xielu_alpha_n_path")
        beta_path_var = emitter._fresh("xielu_beta_path")
        eps_path_var = emitter._fresh("xielu_eps_path")
        alpha_p_param_var = emitter._fresh("xielu_alpha_p_param")
        alpha_n_param_var = emitter._fresh("xielu_alpha_n_param")
        beta_var = emitter._fresh("xielu_beta")
        eps_var = emitter._fresh("xielu_eps")
        alpha_p_var = emitter._fresh("xielu_alpha_p")
        alpha_n_var = emitter._fresh("xielu_alpha_n")

        alpha_p_path = emitter._hoisted_param_path(
            node_spec=node_spec,
            node_path_var=node_path_var,
            param_name="alpha_p",
            lines=lines,
            indent=indent,
        )
        alpha_n_path = emitter._hoisted_param_path(
            node_spec=node_spec,
            node_path_var=node_path_var,
            param_name="alpha_n",
            lines=lines,
            indent=indent,
        )
        beta_path = emitter._hoisted_param_path(
            node_spec=node_spec,
            node_path_var=node_path_var,
            param_name="beta",
            lines=lines,
            indent=indent,
        )
        eps_path = emitter._hoisted_param_path(
            node_spec=node_spec,
            node_path_var=node_path_var,
            param_name="eps",
            lines=lines,
            indent=indent,
        )

        lines.append(f"{indent}{alpha_p_path_var} = {alpha_p_path}")
        lines.append(f"{indent}{alpha_n_path_var} = {alpha_n_path}")
        lines.append(f"{indent}{beta_path_var} = {beta_path}")
        lines.append(f"{indent}{eps_path_var} = {eps_path}")

        lines.append(f"{indent}{alpha_p_param_var} = self._state.get({alpha_p_path})")
        lines.append(f"{indent}if {alpha_p_param_var} is None:")
        lines.append(
            f"{indent}    {alpha_p_param_var} = torch.tensor({_XIELU_ALPHA_P_PARAM_DEFAULT!r}, dtype={target_dtype}, device={src}.device)"
        )
        lines.append(f"{indent}else:")
        lines.append(
            f"{indent}    {alpha_p_param_var} = {alpha_p_param_var}.to(device={src}.device, dtype={target_dtype})"
        )

        lines.append(f"{indent}{alpha_n_param_var} = self._state.get({alpha_n_path})")
        lines.append(f"{indent}if {alpha_n_param_var} is None:")
        lines.append(
            f"{indent}    {alpha_n_param_var} = torch.tensor({_XIELU_ALPHA_N_PARAM_DEFAULT!r}, dtype={target_dtype}, device={src}.device)"
        )
        lines.append(f"{indent}else:")
        lines.append(
            f"{indent}    {alpha_n_param_var} = {alpha_n_param_var}.to(device={src}.device, dtype={target_dtype})"
        )

        lines.append(f"{indent}{beta_var} = self._state.get({beta_path})")
        lines.append(f"{indent}if {beta_var} is None:")
        lines.append(
            f"{indent}    {beta_var} = torch.tensor({_XIELU_BETA_INIT!r}, dtype={target_dtype}, device={src}.device)"
        )
        lines.append(f"{indent}else:")
        lines.append(
            f"{indent}    {beta_var} = {beta_var}.to(device={src}.device, dtype={target_dtype})"
        )

        lines.append(f"{indent}{eps_var} = self._state.get({eps_path})")
        lines.append(f"{indent}if {eps_var} is None:")
        lines.append(
            f"{indent}    {eps_var} = torch.tensor({_XIELU_EPS_INIT!r}, dtype={target_dtype}, device={src}.device)"
        )
        lines.append(f"{indent}else:")
        lines.append(
            f"{indent}    {eps_var} = {eps_var}.to(device={src}.device, dtype={target_dtype})"
        )

        lines.append(f"{indent}{alpha_p_var} = F.softplus({alpha_p_param_var})")
        lines.append(f"{indent}{alpha_n_var} = {beta_var} + F.softplus({alpha_n_param_var})")
        lines.append(
            f"{indent}{out_var} = torch.where({src} > 0, {alpha_p_var} * {src} * {src} + {beta_var} * {src}, (torch.expm1(torch.minimum({src}, {eps_var})) - {src}) * {alpha_n_var} + {beta_var} * {src})"
        )
    else:
        raise ValueError(f"Unsupported activation kind: {kind}")
    return lines


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any",),
    "kwargs": dict(LOWERING_KWARG_KINDS),
    "returns": "dynamic",
}

__all__ = [
    "LOWERING_ARITY",
    "LOWERING_ALLOWED_KWARGS",
    "LOWERING_REQUIRED_KWARGS",
    "LOWERING_KWARG_KINDS",
    "OP_NAME",
    "lowering_infer_metadata",
    "interpret",
    "compile",
    "uses_node_path",
    "LOWERING_TYPE_SIGNATURE",
]
