from __future__ import annotations

from typing import Any

OP_NAME = "split"
LOWERING_ARITY = (1, 1)
LOWERING_ALLOWED_KWARGS: set[str] = {"dim", "parts", "sizes"}
LOWERING_REQUIRED_KWARGS: set[str] = set()
LOWERING_KWARG_KINDS: dict[str, Any] = {
    "dim": "int",
    "parts": "int",
    "sizes": "list_dim",
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


def _is_int_token(value: str) -> bool:
    token = value.strip()
    return bool(token) and (token.isdigit() or (token[0] in {"+", "-"} and token[1:].isdigit()))


def _name_expr(name: str) -> dict[str, Any]:
    return {"_expr": "name", "id": name}


def _binary_expr(op: str, left: Any, right: Any) -> dict[str, Any]:
    return {"_expr": "binary", "op": op, "left": left, "right": right}


def _split_scaled_dim(value: Any) -> tuple[int, Any] | None:
    if isinstance(value, int):
        return value, 1
    if isinstance(value, str):
        token = value.strip().replace(" ", "")
        if not token:
            return None
        if _is_int_token(token):
            return int(token), 1
        parts = token.split("*")
        if len(parts) == 1:
            return 1, token
        if len(parts) == 2:
            a, b = parts
            if _is_int_token(a):
                return int(a), b if b else None
            if _is_int_token(b):
                return int(b), a if a else None
        return None
    if isinstance(value, dict):
        kind = value.get("_expr")
        if kind == "name":
            ident = value.get("id")
            if isinstance(ident, str) and ident:
                return 1, _name_expr(ident)
            return None
        if kind == "binary" and value.get("op") == "*":
            left = value.get("left")
            right = value.get("right")
            if isinstance(left, int) and not isinstance(left, bool):
                return left, right
            if isinstance(right, int) and not isinstance(right, bool):
                return right, left
    return None


def _scale_term(term: Any, factor: int) -> Any:
    if factor == 1:
        return term
    if isinstance(term, int):
        return factor * term
    if isinstance(term, str):
        return f"{factor}*{term}"
    return _binary_expr("*", factor, term)


def _infer_split_sizes_from_last_dim(last_dim: Any, parts: int) -> list[Any] | None:
    if parts <= 0:
        return None
    if isinstance(last_dim, int):
        if last_dim % parts != 0:
            return None
        return [last_dim // parts for _ in range(parts)]
    scaled = _split_scaled_dim(last_dim)
    if scaled is None:
        return None
    factor, term = scaled
    if not isinstance(factor, int) or isinstance(factor, bool):
        return None
    if term is None:
        return None
    if factor % parts != 0:
        return None
    each = factor // parts
    piece: Any = _scale_term(term, each)
    return [piece for _ in range(parts)]


def lowering_normalize_kwargs(
    *,
    args: list[str],
    out: str | list[str],
    kwargs: dict[str, Any],
    ctx: Any,
) -> None:
    if not isinstance(out, list):
        raise ValueError("split requires tuple/list binding outputs")
    has_parts = "parts" in kwargs
    has_sizes = "sizes" in kwargs
    if has_parts and has_sizes:
        raise ValueError("split accepts either parts or sizes, not both")
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
            (isinstance(v, int) and not isinstance(v, bool))
            or isinstance(v, str)
            or (isinstance(v, dict) and v.get("_expr") in {"name", "binary", "if", "tuple"})
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
    del args
    if not isinstance(out, list):
        return False
    dim = kwargs.get("dim", -1)
    if not isinstance(dim, int) or isinstance(dim, bool):
        return False
    sizes = _maybe_int_list(kwargs.get("sizes"))
    if sizes is not None and len(sizes) == len(out):
        axis = dim
        if args:
            source_name = str(args[0]).strip()
            source_shape = ctx.tensor_shape.get(source_name)
            if isinstance(source_shape, tuple):
                rank = len(source_shape)
                target_axis = axis if axis >= 0 else rank + axis
                if 0 <= target_axis < rank:
                    for name, split_dim in zip(out, sizes, strict=True):
                        new_shape = list(source_shape)
                        new_shape[target_axis] = split_dim
                        ctx.tensor_shape[name] = tuple(new_shape)
                        ctx.tensor_last_dim[name] = new_shape[-1]
                    return True
        if axis in (-1,):
            for name, split_dim in zip(out, sizes, strict=True):
                ctx.tensor_last_dim[name] = split_dim
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
    x = model._read_tensor_input(node_spec.get("_args"), env)
    outs = node_spec.get("_bind")
    if not isinstance(outs, list) or len(outs) == 0:
        raise ValueError("split requires non-empty list out")
    dim = int(model._eval_expr(node_spec.get("dim", -1), env, symbols))
    sizes = node_spec.get("sizes")
    if sizes is not None:
        if not isinstance(sizes, list) or len(sizes) != len(outs):
            raise ValueError("split sizes must be a list with same length as out")
        split_sizes = [int(model._eval_expr(size, env, symbols)) for size in sizes]
        chunks = x.split(split_sizes, dim=dim)
    else:
        parts = int(model._eval_expr(node_spec.get("parts", len(outs)), env, symbols))
        chunks = x.chunk(parts, dim=dim)
    for name, tensor in zip(outs, chunks, strict=True):
        env[str(name)] = tensor


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
    dim = emitter._expr_code(node_spec.get("dim", -1), env)
    sizes = node_spec.get("sizes")
    if sizes is not None:
        if not isinstance(sizes, list) or len(sizes) != len(outs):
            raise ValueError("split sizes must be a list with same length as out")
        sizes_code = ", ".join(emitter._expr_code(size, env) for size in sizes)
        lines.append(f"{indent}{tmp} = torch.split({src}, [{sizes_code}], dim=int({dim}))")
    else:
        parts = emitter._expr_code(node_spec.get("parts", len(outs)), env)
        lines.append(f"{indent}{tmp} = torch.chunk({src}, int({parts}), dim=int({dim}))")
    for idx, out_name in enumerate(outs):
        out_var = emitter._assign_out_var(env, str(out_name))
        lines.append(f"{indent}{out_var} = {tmp}[{idx}]")
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
