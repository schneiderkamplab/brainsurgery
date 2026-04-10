from __future__ import annotations

from typing import Any

OP_NAME = "split"
LOWERING_ARITY = (1, 3)
LOWERING_ALLOWED_KWARGS: set[str] = {"dim", "sizes"}
LOWERING_REQUIRED_KWARGS: set[str] = set()
LOWERING_KWARG_KINDS: dict[str, Any] = {
    "dim": "int",
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


def _to_int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, str) and _is_int_token(value):
        return int(value.strip())
    return None


def _is_int_token(value: str) -> bool:
    token = value.strip()
    return bool(token) and (token.isdigit() or (token[0] in {"+", "-"} and token[1:].isdigit()))


def _is_name_token(value: Any) -> bool:
    return isinstance(value, str) and value.strip().isidentifier()


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
    if not args:
        raise ValueError("split requires at least one positional arg: x")
    src = args[0]
    dim_value = _arg_or_default(args, 1, -1)
    sizes_value = _arg_or_default(args, 2, None)
    if len(args) > 3:
        raise ValueError(f"split expects at most 3 positional args, got {len(args)}")
    if "dim" in kwargs:
        if len(args) >= 2 and not _is_null_like(args[1]):
            raise ValueError("split received multiple values for dim")
        dim_value = kwargs["dim"]
    if "sizes" in kwargs:
        if len(args) >= 3 and not _is_null_like(args[2]):
            raise ValueError("split received multiple values for sizes")
        sizes_value = kwargs["sizes"]
    has_sizes = not _is_null_like(sizes_value)
    if has_sizes and isinstance(sizes_value, str):
        parsed_sizes = _maybe_int_list(sizes_value)
        if parsed_sizes is not None:
            sizes_value = parsed_sizes
    if not has_sizes and isinstance(out, list) and args:
        first_arg = args[0].strip()
        if first_arg.isidentifier():
            inferred = ctx.tensor_last_dim.get(first_arg)
            split_sizes = _infer_split_sizes_from_last_dim(inferred, len(out))
            if split_sizes is not None:
                sizes_value = split_sizes
                has_sizes = True
    if not has_sizes:
        raise ValueError(
            "split requires sizes (explicit or inferable from symbolic last-dim and bind arity)"
        )
    if has_sizes:
        sizes_raw = sizes_value
        if isinstance(sizes_raw, list):
            if len(sizes_raw) == 0:
                raise ValueError("split sizes must be a non-empty list")
            if not all(
                (isinstance(v, int) and not isinstance(v, bool))
                or isinstance(v, str)
                or (
                    isinstance(v, dict)
                    and v.get("_expr") in {"name", "binary", "if", "tuple", "call", "string"}
                )
                for v in sizes_raw
            ):
                raise ValueError("split sizes must contain only ints or symbolic dims")
            if isinstance(out, list) and len(out) != len(sizes_raw):
                raise ValueError(
                    f"split sizes length {len(sizes_raw)} requires {len(sizes_raw)} outputs, got {len(out)}"
                )
            sizes_value = sizes_raw
        elif _is_name_token(sizes_raw):
            sizes_value = sizes_raw
        else:
            raise ValueError("split sizes must be a non-empty list")
    dim_int = _to_int(dim_value)
    if dim_int is not None:
        dim_value = dim_int
    args[:] = [src, dim_value, sizes_value]
    kwargs.clear()


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
    return None


def lowering_infer_metadata(
    *,
    args: list[str],
    out: str | list[str],
    kwargs: dict[str, Any],
    ctx: Any,
) -> bool:
    if not isinstance(out, list):
        return False
    dim = _arg_or_default(args, 1, -1)
    if not isinstance(dim, int) or isinstance(dim, bool):
        return False
    sizes = _maybe_int_list(_arg_or_default(args, 2, None))
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
    raw_args = node_spec.get("_args")
    if isinstance(raw_args, list):
        args = list(raw_args)
    elif raw_args is None:
        args = []
    else:
        args = [raw_args]
    if not args:
        raise ValueError("split requires positional args: x [dim sizes]")
    x = model._read_tensor_input(args[0], env)
    out_bind = node_spec.get("_bind")
    dim = int(model._eval_expr(_arg_or_default(args, 1, -1), env, symbols))
    sizes = _arg_or_default(args, 2, None)
    expected_len = len(out_bind) if isinstance(out_bind, list) and len(out_bind) > 0 else None
    sizes_eval = model._eval_expr(sizes, env, symbols)
    if not isinstance(sizes_eval, list | tuple):
        raise ValueError("split sizes must be a list")
    if expected_len is not None and len(sizes_eval) != expected_len:
        raise ValueError("split sizes must be a list with same length as out")
    split_sizes = [int(model._eval_expr(size, env, symbols)) for size in sizes_eval]
    chunks = x.split(split_sizes, dim=dim)
    if isinstance(out_bind, list):
        outs = out_bind
        if len(outs) == 0:
            raise ValueError("split requires non-empty list out")
        if len(chunks) != len(outs):
            raise ValueError(
                f"split output arity mismatch: produced {len(chunks)}, expected {len(outs)}"
            )
        for name, tensor in zip(outs, chunks, strict=True):
            env[str(name)] = tensor
        return
    out_name = model._require_name(out_bind, field="split._bind")
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
        raise ValueError("split requires positional args: x [dim sizes]")
    src = emitter._read_env_var(env, str(args[0]))
    out_bind = node_spec.get("_bind")
    tmp = emitter._fresh("split")
    dim = emitter._expr_code(_arg_or_default(args, 1, -1), env)
    sizes = _arg_or_default(args, 2, None)
    sizes_code = emitter._expr_code(sizes, env)
    lines.append(f"{indent}{tmp} = torch.split({src}, {sizes_code}, dim=int({dim}))")
    if isinstance(out_bind, list):
        outs = out_bind
        if len(outs) == 0:
            raise ValueError("split requires non-empty list out")
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
