from __future__ import annotations

from typing import Any

OP_NAME = "activation"
LOWERING_ARITY = (1, 1)
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
    raise NotImplementedError(f"TinyGrad interpret for '{OP_NAME}' not yet implemented")


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
    if kind in {"gelu", "gelu_new", "gelu_pytorch_tanh"}:
        lines.append(f"{indent}{out_var} = {src}.gelu()")
    elif kind == "relu":
        lines.append(f"{indent}{out_var} = {src}.relu()")
    elif kind == "silu":
        lines.append(f"{indent}{out_var} = {src}.silu()")
    elif kind == "swiglu":
        lines.append(f"{indent}{out_var} = {src}.silu() * {src}")
    elif kind == "sigmoid":
        lines.append(f"{indent}{out_var} = {src}.sigmoid()")
    elif kind == "tanh":
        lines.append(f"{indent}{out_var} = {src}.tanh()")
    else:
        raise ValueError(f"Unsupported activation kind: {kind}")
    return lines


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
]
