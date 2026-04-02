from __future__ import annotations

from typing import Any

import torch

OP_NAME = "add"
LOWERING_ARITY = (2, 2)
LOWERING_ALLOWED_KWARGS: set[str] = set()
LOWERING_REQUIRED_KWARGS: set[str] = set()
LOWERING_KWARG_KINDS: dict[str, Any] = {}


def _normalize_dim_token(value: Any) -> Any:
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return value


def _dims_compatible(left: Any, right: Any) -> bool:
    return _normalize_dim_token(left) == _normalize_dim_token(right)


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return False


def lowering_validate_signature(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> None:
    del args, kwargs, ctx
    if not isinstance(out, str):
        raise ValueError("add requires a single scalar output binding")


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
    second_in = args[1].strip() if len(args) > 1 else None
    first_dim = (
        ctx.tensor_last_dim.get(first_in)
        if isinstance(first_in, str) and first_in.isidentifier()
        else None
    )
    second_dim = (
        ctx.tensor_last_dim.get(second_in)
        if isinstance(second_in, str) and second_in.isidentifier()
        else None
    )
    if (
        first_dim is not None
        and second_dim is not None
        and not _dims_compatible(first_dim, second_dim)
    ):
        raise ValueError(f"add requires matching last-dim; got {first_dim!r} and {second_dim!r}")
    unified = first_dim if first_dim is not None else second_dim
    if unified is not None:
        if isinstance(first_in, str) and first_in.isidentifier() and first_dim is None:
            ctx.tensor_last_dim[first_in] = unified
        if isinstance(second_in, str) and second_in.isidentifier() and second_dim is None:
            ctx.tensor_last_dim[second_in] = unified
        ctx.tensor_last_dim[out] = unified
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
    inputs = node_spec.get("_args")
    if not isinstance(inputs, list) or len(inputs) != 2:
        raise ValueError("add expects two inputs")
    out = model._require_name(node_spec.get("_bind"), field="add._bind")
    left = env[inputs[0]]
    right = env[inputs[1]]
    align_add_fp32 = bool(getattr(model, "_hf_align_add_fp32_accum", False))
    if align_add_fp32 and (
        torch.is_tensor(left)
        and torch.is_tensor(right)
        and left.is_floating_point()
        and right.is_floating_point()
        and left.dtype == right.dtype
        and left.dtype in {torch.float16, torch.bfloat16}
    ):
        env[out] = (left.float() + right.float()).to(dtype=left.dtype)
    else:
        env[out] = left + right
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

    def infer_param(param_name: str) -> str:
        return emitter._infer_param_expr(node_spec, node_path_var, param_name)

    def read(name: str) -> str:
        return emitter._read_env_var(env, name)

    inputs = node_spec.get("_args")
    if not isinstance(inputs, list) or len(inputs) != 2:
        raise ValueError("add expects two inputs")
    a = read(str(inputs[0]))
    b = read(str(inputs[1]))
    out_name = str(node_spec.get("_bind"))
    out_var = assign_out_var(out_name)
    lines.append(
        f"{indent}if getattr(self, '_hf_align_add_fp32_accum', False) and torch.is_tensor({a}) and torch.is_tensor({b}) and {a}.is_floating_point() and {b}.is_floating_point() and {a}.dtype == {b}.dtype and {a}.dtype in {{torch.float16, torch.bfloat16}}:"
    )
    lines.append(f"{indent}    {out_var} = ({a}.float() + {b}.float()).to(dtype={a}.dtype)")
    lines.append(f"{indent}else:")
    lines.append(f"{indent}    {out_var} = {a} + {b}")
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
    "lowering_validate_signature",
    "lowering_infer_metadata",
    "interpret",
    "compile",
    "uses_node_path",
    "LOWERING_TYPE_SIGNATURE",
]
