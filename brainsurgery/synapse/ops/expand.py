from __future__ import annotations

from typing import Any

OP_NAME = "expand"
LOWERING_ARITY = (1, 2)
LOWERING_ALLOWED_KWARGS: set[str] = set()
LOWERING_REQUIRED_KWARGS: set[str] = set()
LOWERING_KWARG_KINDS: dict[str, Any] = {}


def _raw_args(node_spec: dict[str, Any]) -> list[Any]:
    raw = node_spec.get("_args")
    if isinstance(raw, list):
        return list(raw)
    if raw is None:
        return []
    return [raw]


def _arg_or_default(args: list[Any], index: int, default: Any) -> Any:
    if index >= len(args):
        return default
    value = args[index]
    if isinstance(value, str) and value.strip().lower() == "null":
        return default
    return value


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return False


def lowering_validate_signature(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> None:
    del ctx
    if not isinstance(out, str):
        raise ValueError("expand requires a single scalar output binding")
    if kwargs:
        unknown = ", ".join(sorted(str(key) for key in kwargs))
        raise ValueError(f"expand unsupported kwargs: {unknown}")
    if len(args) < 1 or len(args) > 2:
        raise ValueError("expand requires positional args: x [shape]")


def lowering_infer_metadata(
    *,
    args: list[str],
    out: str | list[str],
    kwargs: dict[str, Any],
    ctx: Any,
) -> bool:
    if not isinstance(out, str):
        return False
    shape = _arg_or_default(args, 1, None)
    if not isinstance(shape, list) or not shape:
        return False
    ctx.tensor_shape[out] = tuple(shape)
    ctx.tensor_last_dim[out] = shape[-1]
    return True


def _resolve_shape(
    model: Any,
    raw_shape: Any,
    env: dict[str, Any],
    symbols: dict[str, int],
) -> tuple[int, ...]:
    if isinstance(raw_shape, list):
        raw_items = raw_shape
    else:
        evaluated = model._eval_expr(raw_shape, env, symbols)
        if not isinstance(evaluated, list | tuple) or not evaluated:
            raise ValueError("expand.shape must be a non-empty list")
        raw_items = list(evaluated)
    resolved: list[int] = []
    for item in raw_items:
        value = model._eval_expr(item, env, symbols)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("expand.shape entries must resolve to ints")
        resolved.append(int(value))
    return tuple(resolved)


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
    args = _raw_args(node_spec)
    if len(args) < 1 or len(args) > 2:
        raise ValueError("expand requires positional args: x [shape]")
    src = model._read_tensor_input(args[0], env)
    raw_shape = _arg_or_default(args, 1, None)
    if raw_shape is None:
        raise ValueError("expand.shape is required")
    shape = _resolve_shape(model, raw_shape, env, symbols)
    out = model._require_name(node_spec.get("_bind"), field="expand._bind")
    env[out] = src.expand(*shape)


def compile(
    emitter: Any,
    node_spec: dict[str, Any],
    env: dict[str, str],
    *,
    node_path_var: str,
    scope_var: str,
    indent: str,
) -> list[str]:
    del node_path_var, scope_var
    lines: list[str] = []
    args = _raw_args(node_spec)
    if len(args) < 1 or len(args) > 2:
        raise ValueError("expand requires positional args: x [shape]")
    src = emitter._read_env_var(env, str(args[0]))
    out_var = emitter._assign_out_var(env, str(node_spec.get("_bind")))
    raw_shape = _arg_or_default(args, 1, None)
    if raw_shape is None:
        raise ValueError("expand.shape is required")
    if isinstance(raw_shape, list) and raw_shape:
        shape_expr = (
            f"({', '.join(f'int({emitter._expr_code(item, env)})' for item in raw_shape)},)"
        )
    else:
        shape_expr = emitter._expr_code(raw_shape, env)
    shape_var = emitter._fresh("shape")
    lines.append(f"{indent}{shape_var} = tuple(int(v) for v in {shape_expr})")
    lines.append(f"{indent}if len({shape_var}) == 0:")
    lines.append(f"{indent}    raise ValueError('expand.shape must be a non-empty list')")
    lines.append(f"{indent}{out_var} = {src}.expand(*{shape_var})")
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
