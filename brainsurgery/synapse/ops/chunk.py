from __future__ import annotations

from typing import Any

OP_NAME = "chunk"
LOWERING_ARITY = (1, 3)
LOWERING_ALLOWED_KWARGS: set[str] = {"dim", "parts"}
LOWERING_REQUIRED_KWARGS: set[str] = set()
LOWERING_KWARG_KINDS: dict[str, Any] = {
    "dim": "int",
    "parts": "int",
}


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return False


def _is_null_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() == "null":
        return True
    return False


def _arg_or_default(args: list[Any], index: int, default: Any) -> Any:
    if index >= len(args):
        return default
    value = args[index]
    if _is_null_like(value):
        return default
    return value


def _is_int_token(value: str) -> bool:
    token = value.strip()
    return bool(token) and (token.isdigit() or (token[0] in {"+", "-"} and token[1:].isdigit()))


def _to_int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, str) and _is_int_token(value):
        return int(value.strip())
    return None


def _is_name_token(value: Any) -> bool:
    return isinstance(value, str) and value.strip().isidentifier()


def lowering_normalize_kwargs(
    *,
    args: list[str],
    out: str | list[str],
    kwargs: dict[str, Any],
    ctx: Any,
) -> None:
    del ctx
    if not args:
        raise ValueError("chunk requires at least one positional arg: x")
    src = args[0]
    dim_value = _arg_or_default(args, 1, -1)
    parts_value = _arg_or_default(args, 2, None)
    if len(args) > 3:
        raise ValueError(f"chunk expects at most 3 positional args, got {len(args)}")
    if "dim" in kwargs:
        if len(args) >= 2 and not _is_null_like(args[1]):
            raise ValueError("chunk received multiple values for dim")
        dim_value = kwargs["dim"]
    if "parts" in kwargs:
        if len(args) >= 3 and not _is_null_like(args[2]):
            raise ValueError("chunk received multiple values for parts")
        parts_value = kwargs["parts"]
    if _is_null_like(parts_value):
        parts_value = len(out) if isinstance(out, list) and len(out) > 0 else 1
    parts_raw = _to_int(parts_value)
    if parts_raw is None:
        if not _is_name_token(parts_value):
            raise ValueError("chunk parts must be a positive integer")
    else:
        if parts_raw <= 0:
            raise ValueError("chunk parts must be a positive integer")
        if isinstance(out, list) and len(out) != parts_raw:
            raise ValueError(
                f"chunk parts={parts_raw} requires {parts_raw} outputs, got {len(out)}"
            )
        parts_value = parts_raw
    dim_int = _to_int(dim_value)
    if dim_int is not None:
        dim_value = dim_int
    args[:] = [src, dim_value, parts_value]
    kwargs.clear()


def lowering_known_output_arity(*, kwargs: dict[str, Any]) -> int | None:
    parts = kwargs.get("parts")
    if isinstance(parts, int):
        return parts
    if isinstance(parts, str) and _is_int_token(parts):
        return int(parts.strip())
    return None


def lowering_infer_metadata(
    *,
    args: list[str],
    out: str | list[str],
    kwargs: dict[str, Any],
    ctx: Any,
) -> bool:
    del kwargs
    if not isinstance(out, list):
        return False
    dim = _arg_or_default(args, 1, -1)
    parts = _arg_or_default(args, 2, None)
    if not isinstance(dim, int) or isinstance(dim, bool):
        return False
    if not isinstance(parts, int) or isinstance(parts, bool) or parts <= 0:
        return False
    if args:
        source_name = str(args[0]).strip()
        source_shape = ctx.tensor_shape.get(source_name)
        if isinstance(source_shape, tuple):
            rank = len(source_shape)
            target_axis = dim if dim >= 0 else rank + dim
            if 0 <= target_axis < rank:
                axis_dim = source_shape[target_axis]
                if isinstance(axis_dim, int) and axis_dim % parts == 0:
                    piece = axis_dim // parts
                    for name in out:
                        new_shape = list(source_shape)
                        new_shape[target_axis] = piece
                        ctx.tensor_shape[name] = tuple(new_shape)
                        ctx.tensor_last_dim[name] = new_shape[-1]
                    return True
    return False


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
    if isinstance(raw_args, list):
        args = list(raw_args)
    elif raw_args is None:
        args = []
    else:
        args = [raw_args]
    if not args:
        raise ValueError("chunk requires positional args: x [dim parts]")
    x = model._read_tensor_input(args[0], env)
    out_bind = node_spec.get("_bind")
    dim = int(model._eval_expr(_arg_or_default(args, 1, -1), env, symbols))
    default_parts = len(out_bind) if isinstance(out_bind, list) and len(out_bind) > 0 else 1
    parts = int(model._eval_expr(_arg_or_default(args, 2, default_parts), env, symbols))
    chunks = x.chunk(parts, dim=dim)
    if isinstance(out_bind, list):
        outs = out_bind
        if len(outs) == 0:
            raise ValueError("chunk requires non-empty list out")
        if len(chunks) != len(outs):
            raise ValueError(
                f"chunk output arity mismatch: produced {len(chunks)}, expected {len(outs)}"
            )
        for name, tensor in zip(outs, chunks, strict=True):
            env[str(name)] = tensor
        return
    out_name = model._require_name(out_bind, field="chunk._bind")
    env[out_name] = list(chunks)


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
    raw_args = node_spec.get("_args")
    if isinstance(raw_args, list):
        args = list(raw_args)
    elif raw_args is None:
        args = []
    else:
        args = [raw_args]
    if not args:
        raise ValueError("chunk requires positional args: x [dim parts]")
    src = emitter._read_env_var(env, str(args[0]))
    out_bind = node_spec.get("_bind")
    tmp = emitter._fresh("chunk")
    dim = emitter._expr_code(_arg_or_default(args, 1, -1), env)
    default_parts = len(out_bind) if isinstance(out_bind, list) and len(out_bind) > 0 else 1
    parts = emitter._expr_code(_arg_or_default(args, 2, default_parts), env)
    lines.append(f"{indent}{tmp} = torch.chunk({src}, int({parts}), dim=int({dim}))")
    if isinstance(out_bind, list):
        outs = out_bind
        if len(outs) == 0:
            raise ValueError("chunk requires non-empty list out")
        for idx, out_name in enumerate(outs):
            out_var = emitter._assign_out_var(env, str(out_name))
            lines.append(f"{indent}{out_var} = {tmp}[{idx}]")
        return lines
    out_var = emitter._assign_out_var(env, str(out_bind))
    lines.append(f"{indent}{out_var} = list({tmp})")
    return lines


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any",),
    "kwargs": dict(LOWERING_KWARG_KINDS),
    "returns": "dynamic",
}

__all__ = [
    "OP_NAME",
    "LOWERING_ARITY",
    "LOWERING_ALLOWED_KWARGS",
    "LOWERING_REQUIRED_KWARGS",
    "LOWERING_KWARG_KINDS",
    "lowering_normalize_kwargs",
    "lowering_known_output_arity",
    "lowering_infer_metadata",
    "interpret",
    "compile",
    "uses_node_path",
    "LOWERING_TYPE_SIGNATURE",
]
