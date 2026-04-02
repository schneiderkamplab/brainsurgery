from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

OP_NAME = "activation"
LOWERING_ARITY = (1, 1)
LOWERING_ALLOWED_KWARGS: set[str] = {"fp32_accum"}
LOWERING_REQUIRED_KWARGS: set[str] = set()
LOWERING_KWARG_KINDS: dict[str, Any] = {"fp32_accum": "bool"}


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
    elif kind == "silu":
        env[out] = F.silu(x)
    elif kind == "swiglu":
        env[out] = F.silu(x) * x
    elif kind == "sigmoid":
        env[out] = torch.sigmoid(x)
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
    elif kind == "relu":
        lines.append(f"{indent}{out_var} = F.relu({src})")
    elif kind == "silu":
        lines.append(f"{indent}{out_var} = F.silu({src})")
    elif kind == "swiglu":
        lines.append(f"{indent}{out_var} = F.silu({src}) * {src}")
    elif kind == "sigmoid":
        lines.append(f"{indent}{out_var} = torch.sigmoid({src})")
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
