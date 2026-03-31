from __future__ import annotations

from typing import Any

OP_NAME = "softmax"
LOWERING_ARITY = (1, 1)
LOWERING_ALLOWED_KWARGS: set[str] = {"dtype", "dim"}
LOWERING_REQUIRED_KWARGS: set[str] = set()
LOWERING_KWARG_KINDS: dict[str, Any] = {"dim": "int", "dtype": "str"}
_SUPPORTED_DTYPES: set[str] = {"float32", "float16", "bfloat16"}


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return False


def lowering_validate_signature(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> None:
    del args, ctx
    if not isinstance(out, str):
        raise ValueError("softmax requires a single scalar output binding")
    dtype_name = kwargs.get("dtype")
    if dtype_name is not None:
        if not isinstance(dtype_name, str):
            raise ValueError("softmax dtype must be a string when provided")
        if dtype_name not in _SUPPORTED_DTYPES:
            supported = ", ".join(sorted(_SUPPORTED_DTYPES))
            raise ValueError(f"Unsupported softmax dtype: {dtype_name} (supported: {supported})")


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
    out_var = assign_out_var(out_name)
    dim = emitter._expr_code(node_spec.get("dim", -1), env)
    dtype_name = node_spec.get("dtype")
    if dtype_name is None:
        lines.append(f"{indent}{out_var} = {src}.softmax(axis=int({dim}))")
    else:
        if not isinstance(dtype_name, str):
            raise ValueError("softmax dtype must be a string when provided")
        dtype_map: dict[str, str] = {
            "float32": "dtypes.float32",
            "float16": "dtypes.float16",
            "bfloat16": "dtypes.bfloat16",
        }
        if dtype_name not in _SUPPORTED_DTYPES:
            raise ValueError(f"Unsupported softmax dtype: {dtype_name}")
        lines.append(
            f"{indent}{out_var} = {src}.cast({dtype_map[dtype_name]}).softmax(axis=int({dim}))"
        )
    return lines


__all__ = [
    "LOWERING_ARITY",
    "LOWERING_ALLOWED_KWARGS",
    "LOWERING_REQUIRED_KWARGS",
    "LOWERING_KWARG_KINDS",
    "OP_NAME",
    "lowering_validate_signature",
    "interpret",
    "compile",
    "uses_node_path",
]
