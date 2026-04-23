from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

OP_NAME = "activation"
LOWERING_ARITY = (1, 5)
LOWERING_ALLOWED_KWARGS: set[str] = set()
LOWERING_REQUIRED_KWARGS: set[str] = set()
LOWERING_KWARG_KINDS: dict[str, Any] = {}


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return False


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
    del node_path, scope
    raw_args = node_spec.get("_args")
    arg_list = raw_args if isinstance(raw_args, list) else [raw_args]
    if not arg_list:
        raise ValueError("activation requires at least one positional arg")
    x = model._read_tensor_input(arg_list[0], env)
    op_name = node_spec.get("_op")
    if not isinstance(op_name, str) or not op_name.startswith("activations_"):
        raise ValueError("legacy activation node name; use _op: activations_<kind>")
    kind = op_name[len("activations_") :]
    out = model._require_name(node_spec.get("_bind"), field="activation._bind")
    align_activation_fp32 = False
    if "fp32_accum" in node_spec:
        raise ValueError("activation primitive does not accept kwargs; use positional args only")

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
        limit_expr = arg_list[1] if len(arg_list) >= 2 else None
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
        if len(arg_list) != 5:
            raise ValueError("activations_xielu expects exactly 5 positional args")

        def _as_xielu_value(value: Any, *, name: str) -> torch.Tensor:
            target_dtype = x.dtype if x.is_floating_point() else torch.float32
            if torch.is_tensor(value):
                return value.to(device=x.device, dtype=target_dtype)
            if isinstance(value, (int, float)):
                return torch.tensor(value, device=x.device, dtype=target_dtype)
            raise ValueError(
                f"activations_xielu arg {name} must resolve to tensor/int/float, got {type(value).__name__}"
            )

        alpha_p_value = model._eval_expr(arg_list[1], env, symbols)
        alpha_n_value = model._eval_expr(arg_list[2], env, symbols)
        beta_value = model._eval_expr(arg_list[3], env, symbols)
        eps_value = model._eval_expr(arg_list[4], env, symbols)
        alpha_p = F.softplus(_as_xielu_value(alpha_p_value, name="alpha_p"))
        alpha_n_raw = _as_xielu_value(alpha_n_value, name="alpha_n")
        beta = _as_xielu_value(beta_value, name="beta")
        eps = _as_xielu_value(eps_value, name="eps")
        alpha_n = beta + F.softplus(alpha_n_raw)
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
    del node_path_var
    lines: list[str] = []

    def assign_out_var(out_name: str) -> str:
        return emitter._assign_out_var(env, out_name)

    def read(name: str) -> str:
        return emitter._read_env_var(env, name)

    raw_args = node_spec.get("_args")
    arg_list = raw_args if isinstance(raw_args, list) else [raw_args]
    if not arg_list:
        raise ValueError("activation requires at least one positional arg")
    src = read(str(arg_list[0]))
    out_name = str(node_spec.get("_bind"))
    op_name = node_spec.get("_op")
    if not isinstance(op_name, str) or not op_name.startswith("activations_"):
        raise ValueError("legacy activation node name; use _op: activations_<kind>")
    kind = op_name[len("activations_") :]
    out_var = assign_out_var(out_name)
    relu2_tmp = emitter._fresh("relu2_tmp")
    if "fp32_accum" in node_spec:
        raise ValueError("activation primitive does not accept kwargs; use positional args only")
    if kind in {"gelu_new", "gelu_pytorch_tanh"}:
        lines.append(
            f"{indent}{out_var} = 0.5 * {src} * (1.0 + torch.tanh(0.7978845608028654 * ({src} + 0.044715 * {src} * {src} * {src})))"
        )
    elif kind == "gegelu":
        x_gelu_var = emitter._fresh("gegelu_x_gelu")
        x_linear_var = emitter._fresh("gegelu_x_linear")
        limit_expr = emitter._expr_code(arg_list[1] if len(arg_list) >= 2 else None, env)
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
        lines.append(
            f"{indent}{out_var} = {x_gelu_var} * torch.sigmoid(1.702 * {x_gelu_var}) * ({x_linear_var} + 1.0)"
        )
    elif kind == "gelu":
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
        if len(arg_list) != 5:
            raise ValueError("activations_xielu expects exactly 5 positional args")
        target_dtype = f"({src}.dtype if {src}.is_floating_point() else torch.float32)"
        alpha_p_input_var = emitter._fresh("xielu_alpha_p_input")
        alpha_n_input_var = emitter._fresh("xielu_alpha_n_input")
        beta_input_var = emitter._fresh("xielu_beta_input")
        eps_input_var = emitter._fresh("xielu_eps_input")
        alpha_p_value_var = emitter._fresh("xielu_alpha_p")
        alpha_n_value_var = emitter._fresh("xielu_alpha_n")
        beta_var = emitter._fresh("xielu_beta")
        eps_var = emitter._fresh("xielu_eps")
        alpha_p_expr = emitter._expr_code(arg_list[1] if len(arg_list) >= 2 else None, env)
        alpha_n_expr = emitter._expr_code(arg_list[2] if len(arg_list) >= 3 else None, env)
        beta_expr = emitter._expr_code(arg_list[3] if len(arg_list) >= 4 else None, env)
        eps_expr = emitter._expr_code(arg_list[4] if len(arg_list) >= 5 else None, env)
        lines.append(f"{indent}{alpha_p_input_var} = {alpha_p_expr}")
        lines.append(f"{indent}{alpha_n_input_var} = {alpha_n_expr}")
        lines.append(f"{indent}{beta_input_var} = {beta_expr}")
        lines.append(f"{indent}{eps_input_var} = {eps_expr}")
        lines.append(f"{indent}if torch.is_tensor({alpha_p_input_var}):")
        lines.append(
            f"{indent}    {alpha_p_value_var} = {alpha_p_input_var}.to(device={src}.device, dtype={target_dtype})"
        )
        lines.append(f"{indent}else:")
        lines.append(
            f"{indent}    {alpha_p_value_var} = torch.tensor({alpha_p_input_var}, dtype={target_dtype}, device={src}.device)"
        )
        lines.append(f"{indent}if torch.is_tensor({alpha_n_input_var}):")
        lines.append(
            f"{indent}    {alpha_n_value_var} = {alpha_n_input_var}.to(device={src}.device, dtype={target_dtype})"
        )
        lines.append(f"{indent}else:")
        lines.append(
            f"{indent}    {alpha_n_value_var} = torch.tensor({alpha_n_input_var}, dtype={target_dtype}, device={src}.device)"
        )
        lines.append(f"{indent}if torch.is_tensor({beta_input_var}):")
        lines.append(
            f"{indent}    {beta_var} = {beta_input_var}.to(device={src}.device, dtype={target_dtype})"
        )
        lines.append(f"{indent}else:")
        lines.append(
            f"{indent}    {beta_var} = torch.tensor({beta_input_var}, dtype={target_dtype}, device={src}.device)"
        )
        lines.append(f"{indent}if torch.is_tensor({eps_input_var}):")
        lines.append(
            f"{indent}    {eps_var} = {eps_input_var}.to(device={src}.device, dtype={target_dtype})"
        )
        lines.append(f"{indent}else:")
        lines.append(
            f"{indent}    {eps_var} = torch.tensor({eps_input_var}, dtype={target_dtype}, device={src}.device)"
        )

        lines.append(f"{indent}{alpha_p_value_var} = F.softplus({alpha_p_value_var})")
        lines.append(f"{indent}{alpha_n_value_var} = {beta_var} + F.softplus({alpha_n_value_var})")
        lines.append(
            f"{indent}{out_var} = torch.where({src} > 0, {alpha_p_value_var} * {src} * {src} + {beta_var} * {src}, (torch.expm1(torch.minimum({src}, {eps_var})) - {src}) * {alpha_n_value_var} + {beta_var} * {src})"
        )
    else:
        raise ValueError(f"Unsupported activation kind: {kind}")
    return lines


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any",),
    "kwargs": dict(LOWERING_KWARG_KINDS),
    "returns": "dynamic",
}


def type_rule(
    *,
    arg_types: tuple[Any, ...],
    kwarg_types: dict[str, Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    helpers: Any,
) -> Any | None:
    del kwarg_types, args, kwargs
    if len(arg_types) < 1:
        return None
    input_dims = helpers.type_dims(arg_types[0])
    if input_dims is None:
        return None
    return helpers.type_tensor(dims=input_dims)


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
    "type_rule",
]
