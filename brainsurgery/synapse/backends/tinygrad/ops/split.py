from __future__ import annotations

import re
from typing import Any

OP_NAME = "split"
LOWERING_ARITY = (1, 1)
LOWERING_ALLOWED_KWARGS: set[str] = {"dim", "parts", "sizes", "interleave"}
LOWERING_REQUIRED_KWARGS: set[str] = set()
LOWERING_KWARG_KINDS: dict[str, Any] = {
    "dim": "int",
    "parts": "int",
    "sizes": "list_dim",
    "interleave": "bool",
}


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return False


def _maybe_int_list(value: Any) -> list[int] | None:
    if isinstance(value, list):
        try:
            return [int(v) for v in value]
        except Exception:
            return None
    if isinstance(value, str) and value.strip().startswith("[") and value.strip().endswith("]"):
        inner = value.strip()[1:-1].strip()
        if not inner:
            return []
        try:
            return [int(part.strip()) for part in inner.split(",")]
        except Exception:
            return None
    return None


def _infer_split_sizes_from_last_dim(last_dim: Any, parts: int) -> list[Any] | None:
    if parts <= 0:
        return None
    if isinstance(last_dim, int):
        if last_dim % parts != 0:
            return None
        return [last_dim // parts for _ in range(parts)]
    if not isinstance(last_dim, str):
        return None
    token = last_dim.strip().replace(" ", "")
    if not token:
        return None
    if re.fullmatch(r"-?[0-9]+", token):
        value = int(token)
        if value % parts != 0:
            return None
        return [value // parts for _ in range(parts)]
    match = re.fullmatch(r"([0-9]+)\*(.+)", token)
    if match is None:
        match = re.fullmatch(r"(.+)\*([0-9]+)", token)
        if match is None:
            return None
        term = match.group(1)
        factor = int(match.group(2))
    else:
        factor = int(match.group(1))
        term = match.group(2)
    if factor % parts != 0:
        return None
    each = factor // parts
    piece: Any = term if each == 1 else f"{each}*{term}"
    return [piece for _ in range(parts)]


def lowering_normalize_kwargs(
    *,
    args: list[str],
    out: str | list[str],
    kwargs: dict[str, Any],
    ctx: Any,
) -> None:
    dim = kwargs.get("dim", -1)
    if dim != -1:
        raise ValueError("split currently supports only dim=-1 (last axis)")
    kwargs.pop("dim", None)
    if not isinstance(out, list):
        raise ValueError("split requires tuple/list binding outputs")
    has_parts = "parts" in kwargs
    has_sizes = "sizes" in kwargs
    if has_parts and has_sizes:
        raise ValueError("split accepts either parts or sizes, not both")
    interleave = bool(kwargs.get("interleave", False))
    if interleave and has_sizes:
        raise ValueError("split interleave=true does not support sizes")
    if interleave and has_parts and int(kwargs["parts"]) <= 0:
        raise ValueError("split parts must be a positive integer")
    if has_sizes and isinstance(kwargs["sizes"], str):
        parsed_sizes = _maybe_int_list(kwargs["sizes"])
        if parsed_sizes is not None:
            kwargs["sizes"] = parsed_sizes
    if not has_parts and not has_sizes and args:
        first_arg = args[0].strip()
        if first_arg.isidentifier():
            inferred = ctx.tensor_last_dim.get(first_arg)
            split_sizes = _infer_split_sizes_from_last_dim(inferred, len(out))
            if split_sizes is not None:
                kwargs["sizes"] = split_sizes
                has_sizes = True
    if has_parts:
        parts_raw = kwargs["parts"]
        if not isinstance(parts_raw, int) or isinstance(parts_raw, bool) or parts_raw <= 0:
            raise ValueError("split parts must be a positive integer")
        if len(out) != parts_raw:
            raise ValueError(
                f"split parts={parts_raw} requires {parts_raw} outputs, got {len(out)}"
            )
    if has_sizes:
        sizes_raw = kwargs["sizes"]
        if not isinstance(sizes_raw, list) or len(sizes_raw) == 0:
            raise ValueError("split sizes must be a non-empty list")
        if not all(
            (isinstance(v, int) and not isinstance(v, bool)) or isinstance(v, str)
            for v in sizes_raw
        ):
            raise ValueError("split sizes must contain only ints or symbolic dims")
        if len(out) != len(sizes_raw):
            raise ValueError(
                f"split sizes length {len(sizes_raw)} requires {len(sizes_raw)} outputs, got {len(out)}"
            )


def lowering_known_output_arity(*, kwargs: dict[str, Any]) -> int | None:
    sizes = kwargs.get("sizes")
    if isinstance(sizes, list):
        return len(sizes)
    if isinstance(sizes, str):
        text = sizes.strip()
        if text.startswith("[") and text.endswith("]"):
            inner = text[1:-1].strip()
            if not inner:
                return 0
            return len([part for part in inner.split(",") if part.strip()])
    parts = kwargs.get("parts")
    if isinstance(parts, int):
        return parts
    if isinstance(parts, str) and re.fullmatch(r"-?[0-9]+", parts.strip()):
        return int(parts.strip())
    return None


def lowering_infer_metadata(
    *,
    args: list[str],
    out: str | list[str],
    kwargs: dict[str, Any],
    ctx: Any,
) -> bool:
    del args
    if not isinstance(out, list):
        return False
    sizes = _maybe_int_list(kwargs.get("sizes"))
    if sizes is not None and len(sizes) == len(out):
        for name, dim in zip(out, sizes, strict=True):
            ctx.tensor_last_dim[name] = dim
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
    del node_path_var, scope_var
    lines: list[str] = []
    src = emitter._read_env_var(env, str(node_spec.get("_args")))
    outs = node_spec.get("_bind")
    if not isinstance(outs, list) or len(outs) == 0:
        raise ValueError("split requires non-empty list out")
    tmp = emitter._fresh("split")
    sizes = node_spec.get("sizes")
    interleave = bool(node_spec.get("interleave", False))
    if interleave:
        parts = emitter._expr_code(node_spec.get("parts", len(outs)), env)
        lines.append(f"{indent}{tmp} = int({parts})")
        lines.append(f"{indent}if {tmp} <= 0:")
        lines.append(
            f"{indent}    raise ValueError('split parts must be > 0 when interleave=true')"
        )
        lines.append(f"{indent}if {tmp} != {len(outs)}:")
        lines.append(
            f"{indent}    raise ValueError('split interleave=true requires parts to match output count ({len(outs)})')"
        )
        lines.append(f"{indent}{tmp} = [{src}[..., idx::{tmp}] for idx in range({tmp})]")
    elif sizes is not None:
        if not isinstance(sizes, list) or len(sizes) != len(outs):
            raise ValueError("split sizes must be a list with same length as out")
        sizes_code = ", ".join(emitter._expr_code(size, env) for size in sizes)
        lines.append(f"{indent}{tmp} = {src}.split([{sizes_code}], dim=-1)")
    else:
        parts = emitter._expr_code(node_spec.get("parts", len(outs)), env)
        lines.append(f"{indent}{tmp} = {src}.chunk(int({parts}), dim=-1)")
    for idx, out_name in enumerate(outs):
        out_var = emitter._assign_out_var(env, str(out_name))
        lines.append(f"{indent}{out_var} = {tmp}[{idx}]")
    return lines


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
]
