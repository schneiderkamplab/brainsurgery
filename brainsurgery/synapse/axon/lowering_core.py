from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from ..ops import (
    get_op_lowering_infer_metadata,
    get_op_lowering_known_output_arity,
    get_op_lowering_normalizer,
    get_op_lowering_signature,
    get_op_lowering_validator,
)
from ..type_inference import annotate_spec_with_block_io_types, infer_block_io_types_from_modules
from .expression_codec import axon_expr_to_runtime_value as _expr_to_runtime_value
from .typecheck import typecheck_axon_program
from .types import (
    AxonBind,
    AxonExpr,
    AxonExprBinary,
    AxonExprBind,
    AxonExprBool,
    AxonExprCall,
    AxonExprDo,
    AxonExprFloat,
    AxonExprIf,
    AxonExprInt,
    AxonExprLambda,
    AxonExprList,
    AxonExprName,
    AxonExprNull,
    AxonExprParen,
    AxonExprPipe,
    AxonExprString,
    AxonExprTernary,
    AxonExprTuple,
    AxonKwargValue,
    AxonModule,
    AxonRepeat,
    AxonReturn,
    AxonScopeBind,
    AxonStatement,
)


def _is_identifier(token: str) -> bool:
    if not token:
        return False
    if not (token[0].isalpha() or token[0] == "_"):
        return False
    return all(ch.isalnum() or ch == "_" for ch in token[1:])


def _is_name_token(token: str) -> bool:
    return _is_identifier(token.strip())


def _expr_name(expr: AxonExpr) -> str | None:
    if isinstance(expr, AxonExprName):
        return expr.name
    return None


def _is_int_literal(token: str) -> bool:
    if not token:
        return False
    if token[0] == "-":
        return token[1:].isdigit()
    return token.isdigit()


def _sanitize_token(token: str, *, default: str) -> str:
    out_chars: list[str] = []
    for ch in token:
        if ch.isalnum() or ch == "_":
            out_chars.append(ch)
        else:
            out_chars.append("_")
    normalized = "".join(out_chars).strip("_")
    return normalized or default


def _parse_scalar_token(token: str) -> Any:
    value = token.strip()
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() == "null":
        return None
    if value and ((value[0] == value[-1] == '"') or (value[0] == value[-1] == "'")):
        return value[1:-1]
    if value and (value.isdigit() or (value[0] in {"+", "-"} and value[1:].isdigit())):
        return int(value)
    try:
        return float(value)
    except ValueError:
        return value


_IMPLICIT_ACTIVATION_ALIASES: dict[str, tuple[str, int]] = {
    "gelu": ("_activations_gelu", 0),
    "gelu_new": ("_activations_gelu_new", 0),
    "gelu_pytorch_tanh": ("_activations_gelu_pytorch_tanh", 0),
    "gegelu": ("_activations_gegelu", 0),
    "relu": ("_activations_relu", 0),
    "sigmoid": ("_activations_sigmoid", 0),
    "silu": ("_activations_silu", 0),
    "swiglu": ("_activations_swiglu", 0),
}
_PRIMITIVE_NAME_ALIASES: dict[str, str] = {
    "_repeat": "repeat",
    "_list_init": "list_init",
    "_list_index": "list_index",
    "_list_append": "list_append",
    "_moe_select": "moe_select",
    "_moe_grouped_ffn": "moe_grouped_ffn",
}
_CACHE_PRIMITIVE_ALIASES: dict[str, str] = {
    "update": "cache_update",
    "seq_len": "cache_seq_len",
}


def _to_synapse_op(
    callee: str,
    args: list[str],
    kwargs: dict[str, Any],
    out: str | list[str],
) -> dict[str, Any]:
    canonical = _canonical_op_name(callee)
    if "@" in callee:
        at_op: dict[str, Any] = {"_op": canonical, "_bind": out}
        if args:
            at_op["_args"] = args[0] if len(args) == 1 else args
        for key, value in kwargs.items():
            at_op[key] = value
        return at_op

    default_op: dict[str, Any] = {"_op": canonical, "_bind": out}
    if args:
        default_op["_args"] = args[0] if len(args) == 1 else args
    for key, value in kwargs.items():
        default_op[key] = value
    return default_op


def _join_dot(left: str, right: str) -> str:
    if not left:
        return right
    if not right:
        return left
    return f"{left}.{right}"


def _normalize_root_token(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("scope root values must resolve to strings")
    token = value.strip()
    if token.startswith(".") or token.endswith(".") or ".." in token:
        raise ValueError(f"invalid scope root path {value!r}")
    return token


def _scope_root_values_from_kwarg(raw: AxonKwargValue, ctx: "_LowerCtx") -> tuple[str, ...] | None:
    def _resolve_atom(value: Any) -> Any:
        if isinstance(value, AxonExprString):
            return value.value
        if isinstance(value, AxonExprName):
            if value.name in ctx.symbol_values:
                return ctx.symbol_values[value.name]
            if value.name in ctx.symbol_names:
                raise ValueError(
                    f"scope root symbol {value.name!r} is not a concrete constant value"
                )
            return None
        if isinstance(value, AxonExprList):
            items = [_resolve_atom(item) for item in value.items]
            return None if any(item is None for item in items) else items
        if isinstance(value, AxonExprTuple):
            tuple_items = tuple(_resolve_atom(item) for item in value.items)
            return None if any(item is None for item in tuple_items) else tuple_items
        if isinstance(value, AxonExprParen):
            return _resolve_atom(value.inner)
        if isinstance(value, AxonExpr):
            return None
        return value

    resolved = _resolve_atom(raw)
    if resolved is None:
        return None
    values: list[str] = []
    if isinstance(resolved, str):
        values.append(_normalize_root_token(resolved))
    elif isinstance(resolved, list | tuple):
        for item in resolved:
            values.append(_normalize_root_token(item))
    else:
        raise ValueError("scope root must resolve to string or list/tuple of strings")
    dedup: list[str] = []
    seen: set[str] = set()
    for item in values:
        if item in seen:
            continue
        seen.add(item)
        dedup.append(item)
    return tuple(dedup if dedup else [""])


def _scope_root_expr_from_kwarg(raw: AxonKwargValue) -> Any:
    if isinstance(raw, AxonExpr):
        expr_value = _expr_to_runtime_value(raw)
        if isinstance(expr_value, (str, dict)):
            return expr_value
        raise ValueError("dynamic scope root expression must resolve to string")
    if isinstance(raw, str):
        return raw
    raise ValueError("scope root must be string or expression resolving to string")


def _current_param_roots(ctx: "_LowerCtx") -> tuple[str, ...]:
    if not ctx.param_root_stack:
        return ("",)
    combined: list[str] = [""]
    for frame in ctx.param_root_stack:
        next_values: list[str] = []
        for base in combined:
            for suffix in frame:
                next_values.append(_join_dot(base, suffix))
        combined = next_values if next_values else combined
    dedup: list[str] = []
    seen: set[str] = set()
    for item in combined:
        if item in seen:
            continue
        seen.add(item)
        dedup.append(item)
    return tuple(dedup if dedup else [""])


def _canonical_op_name(callee: str) -> str:
    base = callee.split("@", 1)[0] if "@" in callee else callee
    if base.startswith("_cache_"):
        cache_suffix = base[len("_cache_") :]
        alias = _CACHE_PRIMITIVE_ALIASES.get(cache_suffix)
        if alias is not None:
            return alias
        raise ValueError(f"unsupported cache primitive alias: {base!r}")
    alias = _PRIMITIVE_NAME_ALIASES.get(base)
    if alias is not None:
        return alias
    if base.startswith("_") and len(base) > 1 and base[1].isalpha():
        return _canonical_op_name(base[1:])
    return base


def _normalize_dim_token(value: Any) -> Any:
    if isinstance(value, str):
        token = value.strip()
        if _is_int_literal(token):
            return int(token)
        return token
    return value


def _dims_compatible(left: Any, right: Any) -> bool:
    return _normalize_dim_token(left) == _normalize_dim_token(right)


def _is_symbolic_dim_token(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    token = value.strip()
    return not _is_int_literal(token)


def _is_kind(value: Any, kind: str) -> bool:
    is_expr_payload = isinstance(value, dict) and value.get("_expr") in {
        "name",
        "binary",
        "if",
        "tuple",
    }
    if kind == "bool":
        return isinstance(value, bool | str) or is_expr_payload
    if kind == "int":
        return (
            (isinstance(value, int) and not isinstance(value, bool))
            or isinstance(value, str)
            or is_expr_payload
        )
    if kind == "number":
        if isinstance(value, bool):
            return False
        return isinstance(value, (int, float, str)) or is_expr_payload
    if kind == "str":
        return isinstance(value, str) or is_expr_payload
    if kind == "str_or_bool_or_null":
        return isinstance(value, (str, bool)) or value is None or is_expr_payload
    if kind == "dim":
        if isinstance(value, bool):
            return False
        return isinstance(value, (int, str)) or is_expr_payload
    if kind == "list_int":
        return isinstance(value, list) and all(
            isinstance(v, int) and not isinstance(v, bool) for v in value
        )
    if kind == "list_dim":
        return isinstance(value, list) and all(
            (isinstance(v, int) and not isinstance(v, bool))
            or isinstance(v, str)
            or (isinstance(v, dict) and v.get("_expr") in {"name", "binary", "if", "tuple"})
            for v in value
        )
    return True


def _validate_op_signature(op_name: str, args: list[str], kwargs: dict[str, Any]) -> None:
    signature = get_op_lowering_signature(op_name)
    if not isinstance(signature, dict):
        return
    arity = signature.get("arity")
    if arity is not None:
        min_args, max_args = arity
        if len(args) < min_args or len(args) > max_args:
            raise ValueError(
                f"{op_name} expects {min_args}"
                + (f"..{max_args}" if min_args != max_args else "")
                + f" positional args, got {len(args)}"
            )
    allowed = signature.get("allowed_kwargs")
    if allowed is not None:
        unknown = sorted(set(kwargs) - allowed)
        if unknown:
            allowed_text = ", ".join(sorted(str(name) for name in allowed))
            if allowed_text:
                raise ValueError(
                    f"{op_name} unsupported kwargs: {', '.join(unknown)}; allowed: {allowed_text}"
                )
            raise ValueError(f"{op_name} unsupported kwargs: {', '.join(unknown)}")
    required = signature.get("required_kwargs")
    if required:
        missing = sorted(required - set(kwargs))
        if missing:
            raise ValueError(f"{op_name} missing required kwargs: {', '.join(missing)}")
    kinds = signature.get("kwarg_kinds", {})
    for key, value in kwargs.items():
        expected = kinds.get(key)
        if expected is None:
            continue
        if not _is_kind(value, expected):
            raise ValueError(
                f"{op_name} kwarg {key!r} expects {expected}, got {type(value).__name__}"
            )


@dataclass
class _LowerCtx:
    counter: int = 0
    block_signatures: dict[str, tuple[list[str], list[str]]] | None = None
    block_path_params: dict[str, tuple[str, ...]] | None = None
    block_param_last_dims: dict[str, dict[str, Any]] | None = None
    block_output_last_dims: dict[str, dict[str, Any]] | None = None
    block_param_shapes: dict[str, dict[str, tuple[Any, ...]]] | None = None
    block_output_shapes: dict[str, dict[str, tuple[Any, ...]]] | None = None
    tensor_last_dim: dict[str, Any] = field(default_factory=dict)
    tensor_heads: dict[str, Any] = field(default_factory=dict)
    tensor_shape: dict[str, tuple[Any, ...]] = field(default_factory=dict)
    scope_stack: list[str] = field(default_factory=list)
    loop_scope_cover_prefixes: list[str] = field(default_factory=list)
    param_root_stack: list[tuple[str, ...]] = field(default_factory=list)
    dynamic_param_root_stack: list[Any] = field(default_factory=list)
    path_param_names: set[str] = field(default_factory=set)
    imported_namespaces: set[str] = field(default_factory=set)
    imported_member_namespaces: dict[str, set[str]] = field(default_factory=dict)
    prelude_aliases: dict[str, tuple[str, int]] = field(default_factory=dict)
    primitive_aliases: dict[str, tuple[str, int]] = field(default_factory=dict)
    current_module: str | None = None
    param_names: set[str] = field(default_factory=set)
    symbol_names: set[str] = field(default_factory=set)
    symbol_values: dict[str, Any] = field(default_factory=dict)

    def fresh(self, base: str = "t") -> str:
        self.counter += 1
        return f"{base}_{self.counter}"


def _with_guard(nodes: list[dict[str, Any]], guard: str | None) -> list[dict[str, Any]]:
    if guard is not None:
        raise ValueError("internal lowering error: guard-based lowering is no longer supported")
    return nodes


def _param_root_payload(ctx: _LowerCtx) -> Any | None:
    dynamic_root_expr = _current_dynamic_param_root(ctx)
    if dynamic_root_expr is not None:
        return dynamic_root_expr
    root_candidates = _current_param_roots(ctx)
    if root_candidates != ("",):
        return root_candidates[0] if len(root_candidates) == 1 else list(root_candidates)
    return None


def _with_scope(
    nodes: list[dict[str, Any]], scope: str | None, *, param_root: Any | None = None
) -> list[dict[str, Any]]:
    if not isinstance(scope, str) or not scope:
        scope = None

    def _annotate_node_spec(node_spec: dict[str, Any]) -> dict[str, Any]:
        spec = dict(node_spec)
        op = spec.get("_op")
        if (
            isinstance(op, str)
            and scope is not None
            and "_scope" not in spec
            and "_abs_path" not in spec
        ):
            spec["_scope"] = scope
        if (
            isinstance(op, str)
            and param_root is not None
            and "_param_root" not in spec
            and "_abs_path" not in spec
        ):
            spec["_param_root"] = copy.deepcopy(param_root)
        nested = spec.get("graph")
        if isinstance(nested, list):
            spec["graph"] = _annotate_items(nested)
        body = spec.get("_body")
        if isinstance(body, list):
            spec["_body"] = _annotate_items(body)
        then_branch = spec.get("_then")
        if isinstance(then_branch, list):
            spec["_then"] = _annotate_items(then_branch)
        else_branch = spec.get("_else")
        if isinstance(else_branch, list):
            spec["_else"] = _annotate_items(else_branch)
        return spec

    def _annotate_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out_items: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict) or len(item) != 1:
                out_items.append(item)
                continue
            name, node_spec = next(iter(item.items()))
            if not isinstance(node_spec, dict):
                out_items.append(item)
                continue
            out_items.append({name: _annotate_node_spec(node_spec)})
        return out_items

    return _annotate_items(nodes)


def _current_dynamic_param_root(ctx: _LowerCtx) -> Any | None:
    if not ctx.dynamic_param_root_stack:
        return None
    for item in reversed(ctx.dynamic_param_root_stack):
        if item is not None:
            return item
    return None


def _normalize_call_scope_for_runtime(ctx: _LowerCtx, call_scope: str) -> str:
    if not call_scope:
        return ""
    cover_prefixes = [p for p in ctx.loop_scope_cover_prefixes if isinstance(p, str) and p]
    if not cover_prefixes:
        return call_scope
    longest = max(cover_prefixes, key=len)
    if call_scope == longest:
        return ""
    if call_scope.startswith(f"{longest}."):
        return call_scope[len(longest) + 1 :]
    return call_scope


def _op_name_from_callee(callee: str) -> str:
    if "@" in callee:
        return callee.split("@", 1)[0]
    return callee


def _record_last_dim_for_call(
    *, callee: str, args: list[str], kwargs: dict[str, Any], out: str | list[str], ctx: _LowerCtx
) -> None:
    _validate_namespaced_block_call(callee, ctx)
    resolved_block = _resolve_block_call(callee, ctx)
    if resolved_block is not None and ctx.block_signatures is not None:
        block_name, _ = resolved_block
        input_names, output_names = ctx.block_signatures[block_name]
        provided: dict[str, str] = {}
        for idx, value in enumerate(args):
            if idx < len(input_names):
                provided[input_names[idx]] = value.strip()
        for key, value in kwargs.items():
            if key in input_names:
                provided[key] = str(value).strip()

        symbol_bindings: dict[str, Any] = {}
        for param_name, raw in provided.items():
            if _is_name_token(raw):
                if raw in ctx.tensor_last_dim:
                    symbol_bindings[param_name] = ctx.tensor_last_dim[raw]
                else:
                    symbol_bindings[param_name] = raw
                continue
            parsed = _parse_scalar_token(raw)
            symbol_bindings[param_name] = parsed

        param_last_dims = (
            ctx.block_param_last_dims.get(block_name, {})
            if isinstance(ctx.block_param_last_dims, dict)
            else {}
        )
        for param_name, sym in param_last_dims.items():
            if param_name in symbol_bindings and isinstance(sym, str):
                symbol_bindings[sym] = symbol_bindings[param_name]

        output_last_dims = (
            ctx.block_output_last_dims.get(block_name, {})
            if isinstance(ctx.block_output_last_dims, dict)
            else {}
        )
        output_shapes = (
            ctx.block_output_shapes.get(block_name, {})
            if isinstance(ctx.block_output_shapes, dict)
            else {}
        )
        out_targets = [out] if isinstance(out, str) else list(out)
        for output_name, target in zip(output_names, out_targets, strict=False):
            dim_token = output_last_dims.get(output_name)
            if isinstance(dim_token, str):
                resolved_dim = symbol_bindings.get(dim_token, dim_token)
                ctx.tensor_last_dim[target] = resolved_dim
            elif dim_token is not None:
                ctx.tensor_last_dim[target] = dim_token
            shape_tokens = output_shapes.get(output_name)
            if shape_tokens is not None:
                resolved_shape = tuple(symbol_bindings.get(tok, tok) for tok in shape_tokens)
                ctx.tensor_shape[target] = resolved_shape

    op_name = _canonical_op_name(callee)
    infer_metadata = get_op_lowering_infer_metadata(op_name)
    if callable(infer_metadata) and bool(
        infer_metadata(args=args, out=out, kwargs=kwargs, ctx=ctx)
    ):
        return


def _normalize_call_kwargs(
    *,
    op_name: str,
    args: list[str],
    out: str | list[str],
    kwargs: dict[str, Any],
    ctx: _LowerCtx,
) -> dict[str, Any]:
    normalized = dict(kwargs)
    op_normalizer = get_op_lowering_normalizer(op_name)
    if callable(op_normalizer):
        op_normalizer(args=args, out=out, kwargs=normalized, ctx=ctx)
    return normalized


def _validate_normalized_kwargs(op_name: str, kwargs: dict[str, Any], args: list[str]) -> None:
    _validate_op_signature(op_name, args, kwargs)


def _resolve_block_call(callee: str, ctx: _LowerCtx) -> tuple[str, dict[str, str]] | None:
    if not ctx.block_signatures:
        return None
    if callee in ctx.block_signatures:
        return callee, {}
    if "@" not in callee and "." not in callee and "::" not in callee:
        imported_namespaces = ctx.imported_member_namespaces.get(callee, set())
        if imported_namespaces:
            if len(imported_namespaces) > 1:
                choices = ", ".join(sorted(imported_namespaces))
                raise ValueError(
                    f"ambiguous imported member {callee!r}; found in namespaces: {choices}"
                )
            namespace = next(iter(imported_namespaces))
            namespaced_callee = f"{namespace}.{callee}"
            if namespaced_callee in ctx.block_signatures:
                return namespaced_callee, {}
            raise ValueError(
                f"imported member {callee!r} from {namespace!r} not found as module "
                f"{namespaced_callee!r}"
            )
    if "@" not in callee:
        return None
    parts = callee.split("@")
    base = parts[0]
    concrete_paths = parts[1:]
    if base not in ctx.block_signatures:
        return None
    path_params = (
        ctx.block_path_params.get(base, ()) if isinstance(ctx.block_path_params, dict) else ()
    )
    if not path_params:
        return None
    if len(concrete_paths) != len(path_params):
        raise ValueError(
            f"block call {callee!r} expects {len(path_params)} @path arguments, got {len(concrete_paths)}"
        )
    return base, {
        path_param: concrete
        for path_param, concrete in zip(path_params, concrete_paths, strict=True)
    }


def _validate_namespaced_block_call(callee: str, ctx: _LowerCtx) -> None:
    if "." not in callee or "@" in callee or "::" in callee:
        return
    if not ctx.block_signatures or callee not in ctx.block_signatures:
        return
    namespace = callee.split(".", 1)[0].strip()
    if not namespace:
        return
    if namespace in ctx.imported_namespaces:
        return
    if isinstance(ctx.current_module, str) and ctx.current_module.startswith(f"{namespace}."):
        return
    raise ValueError(f"namespaced call {callee!r} requires `import {namespace}` in the Axon source")


def _rewrite_prelude_alias_callee(callee: str, kwargs: dict[str, Any], ctx: _LowerCtx) -> str:
    if "::" in callee:
        return callee
    parts = callee.split("@")
    base = parts[0]
    path_parts = parts[1:]

    member_name: str | None = None
    if "." not in base:
        member_name = base
    elif base.startswith("Prelude."):
        member_name = base.split(".", 1)[1]
    if not member_name:
        return callee

    if isinstance(ctx.block_signatures, dict) and member_name in ctx.block_signatures:
        return callee
    imported_for_member = ctx.imported_member_namespaces.get(member_name, set())
    if imported_for_member and imported_for_member != {"Prelude"}:
        return callee
    alias = ctx.prelude_aliases.get(member_name)
    if alias is None:
        alias = _IMPLICIT_ACTIVATION_ALIASES.get(member_name)
    if alias is None:
        return callee
    target_base, expected_path_count = alias
    if expected_path_count != len(path_parts):
        if expected_path_count == 0 and not path_parts:
            return target_base
        if expected_path_count > 0 and not path_parts:
            return target_base
        raise ValueError(
            f"Prelude alias {member_name!r} expects {expected_path_count} @path arguments, got {len(path_parts)}"
        )
    if path_parts:
        return "@".join([target_base, *path_parts])
    return target_base


def _rewrite_primitive_alias_callee(callee: str, kwargs: dict[str, Any], ctx: _LowerCtx) -> str:
    if not ctx.primitive_aliases:
        return callee
    if "::" in callee:
        return callee
    parts = callee.split("@")
    base = parts[0]
    path_parts = parts[1:]

    full_name: str | None = None
    if "." in base:
        full_name = base
    else:
        imported_for_member = ctx.imported_member_namespaces.get(base, set())
        if len(imported_for_member) == 1:
            namespace = next(iter(imported_for_member))
            full_name = f"{namespace}.{base}"
    if not full_name:
        return callee

    alias = ctx.primitive_aliases.get(full_name)
    if alias is None:
        return callee
    target_base, expected_path_count = alias
    if expected_path_count != len(path_parts):
        if expected_path_count == 0 and not path_parts:
            return target_base
        if expected_path_count > 0 and not path_parts:
            return target_base
        raise ValueError(
            f"primitive alias {full_name!r} expects {expected_path_count} @path arguments, got {len(path_parts)}"
        )
    if path_parts:
        return "@".join([target_base, *path_parts])
    return target_base


def _known_output_arity(callee: str, kwargs: dict[str, Any], ctx: _LowerCtx) -> int | None:
    resolved = _resolve_block_call(callee, ctx)
    if resolved is not None and ctx.block_signatures:
        block_name, _ = resolved
        _, output_names = ctx.block_signatures[block_name]
        return len(output_names)

    normalized = _canonical_op_name(callee)
    op_arity = get_op_lowering_known_output_arity(normalized)
    if callable(op_arity):
        resolved = op_arity(kwargs=kwargs)
        if isinstance(resolved, int):
            return resolved
    return None


def _pipeline_temp_out(stage: AxonExpr, ctx: _LowerCtx) -> str | list[str]:
    if isinstance(stage, AxonExprName):
        arity = _known_output_arity(stage.name, {}, ctx)
    elif isinstance(stage, AxonExprCall):
        kwargs = _render_kwargs_for_call(stage.kwargs)
        arity = _known_output_arity(stage.callee, kwargs, ctx)
    else:
        return ctx.fresh("pipe")
    if arity is None or arity <= 1:
        return ctx.fresh("pipe")
    return [ctx.fresh("pipe") for _ in range(arity)]


def _expand_call_outputs_for_ternary(
    call_expr: AxonExprCall | AxonExprName, out: str | list[str], ctx: _LowerCtx
) -> str | list[str]:
    if not isinstance(out, list):
        return out
    if isinstance(call_expr, AxonExprName):
        arity = _known_output_arity(call_expr.name, {}, ctx)
    else:
        kwargs = _render_kwargs_for_call(call_expr.kwargs)
        arity = _known_output_arity(call_expr.callee, kwargs, ctx)
    if not isinstance(arity, int) or arity <= len(out):
        return out
    return [*out, *[ctx.fresh("discard") for _ in range(arity - len(out))]]


def _render_kwargs_for_call(kwargs: dict[str, AxonKwargValue]) -> dict[str, Any]:
    rendered: dict[str, Any] = {}
    for key, value in kwargs.items():
        if isinstance(value, AxonExprInt):
            rendered[key] = value.value
        elif isinstance(value, AxonExprFloat):
            rendered[key] = value.lexeme if value.lexeme else value.value
        elif isinstance(value, AxonExprBool):
            rendered[key] = value.value
        elif isinstance(value, AxonExprNull):
            rendered[key] = None
        elif isinstance(value, AxonExprString):
            rendered[key] = value.value
        elif isinstance(value, AxonExprList):
            rendered[key] = _expr_to_runtime_value(value)
        elif isinstance(value, AxonExprName):
            rendered[key] = value.name
        elif isinstance(value, AxonExpr):
            try:
                rendered[key] = _expr_to_runtime_value(value)
            except ValueError:
                rendered[key] = None
        else:
            rendered[key] = value
    return rendered


def _substitute_expr(expr: AxonExpr, var_name: str, replacement: AxonExpr) -> AxonExpr:
    if isinstance(expr, AxonExprName):
        if expr.name == var_name:
            return replacement
        return expr
    if isinstance(
        expr,
        AxonExprInt | AxonExprFloat | AxonExprBool | AxonExprNull | AxonExprString,
    ):
        return expr
    if isinstance(expr, AxonExprTuple):
        return AxonExprTuple(
            items=tuple(_substitute_expr(item, var_name, replacement) for item in expr.items)
        )
    if isinstance(expr, AxonExprList):
        return AxonExprList(
            items=tuple(_substitute_expr(item, var_name, replacement) for item in expr.items)
        )
    if isinstance(expr, AxonExprCall):
        new_kwargs: dict[str, AxonKwargValue] = {}
        for key, value in expr.kwargs.items():
            if isinstance(value, AxonExpr):
                new_kwargs[key] = _substitute_expr(value, var_name, replacement)
            else:
                new_kwargs[key] = value
        return AxonExprCall(
            callee=expr.callee,
            args=tuple(_substitute_expr(arg, var_name, replacement) for arg in expr.args),
            kwargs=new_kwargs,
        )
    if isinstance(expr, AxonExprPipe):
        return AxonExprPipe(
            value=_substitute_expr(expr.value, var_name, replacement),
            stages=tuple(_substitute_expr(stage, var_name, replacement) for stage in expr.stages),
        )
    if isinstance(expr, AxonExprBind):
        if expr.var == var_name:
            return AxonExprBind(
                value=_substitute_expr(expr.value, var_name, replacement),
                var=expr.var,
                body=expr.body,
            )
        return AxonExprBind(
            value=_substitute_expr(expr.value, var_name, replacement),
            var=expr.var,
            body=_substitute_expr(expr.body, var_name, replacement),
        )
    if isinstance(expr, AxonExprIf):
        return AxonExprIf(
            cond=_substitute_expr(expr.cond, var_name, replacement),
            true_expr=_substitute_expr(expr.true_expr, var_name, replacement),
            false_expr=_substitute_expr(expr.false_expr, var_name, replacement),
        )
    if isinstance(expr, AxonExprTernary):
        return AxonExprTernary(
            cond=_substitute_expr(expr.cond, var_name, replacement),
            true_expr=_substitute_expr(expr.true_expr, var_name, replacement),
            false_expr=_substitute_expr(expr.false_expr, var_name, replacement),
        )
    if isinstance(expr, AxonExprBinary):
        return AxonExprBinary(
            op=expr.op,
            left=_substitute_expr(expr.left, var_name, replacement),
            right=_substitute_expr(expr.right, var_name, replacement),
        )
    if isinstance(expr, AxonExprLambda):
        if expr.var == var_name:
            return expr
        return AxonExprLambda(var=expr.var, body=_substitute_expr(expr.body, var_name, replacement))
    if isinstance(expr, AxonExprParen):
        return AxonExprParen(inner=_substitute_expr(expr.inner, var_name, replacement))
    if isinstance(expr, AxonExprDo):
        body_out: list[AxonStatement] = []
        for stmt in expr.body:
            if isinstance(stmt, AxonBind):
                body_out.append(
                    AxonBind(
                        targets=stmt.targets,
                        expr=_substitute_expr(stmt.expr, var_name, replacement),
                    )
                )
                continue
            if isinstance(stmt, AxonReturn):
                body_out.append(
                    AxonReturn(
                        values=tuple(
                            _substitute_expr(value, var_name, replacement) for value in stmt.values
                        )
                    )
                )
                continue
            if isinstance(stmt, AxonRepeat):
                body_out.append(
                    AxonRepeat(
                        name=stmt.name,
                        var=stmt.var,
                        to_expr=_substitute_expr(stmt.to_expr, var_name, replacement),
                        from_expr=_substitute_expr(stmt.from_expr, var_name, replacement),
                        step_expr=_substitute_expr(stmt.step_expr, var_name, replacement),
                        body=stmt.body,
                    )
                )
                continue
            if isinstance(stmt, AxonScopeBind):
                body_out.append(stmt)
                continue
        return AxonExprDo(body=tuple(body_out))
    return expr


def _expr_is_tensorish(expr: AxonExpr, ctx: _LowerCtx) -> bool:
    if isinstance(
        expr, AxonExprCall | AxonExprPipe | AxonExprBind | AxonExprIf | AxonExprTernary | AxonExprDo
    ):
        return True
    if isinstance(expr, AxonExprName):
        token = expr.name
        if token in ctx.symbol_names:
            return False
        return (
            token in ctx.param_names
            or token in ctx.tensor_shape
            or token in ctx.tensor_last_dim
            or token in ctx.tensor_heads
        )
    if isinstance(expr, AxonExprBinary):
        return _expr_is_tensorish(expr.left, ctx) or _expr_is_tensorish(expr.right, ctx)
    if isinstance(expr, AxonExprParen):
        return _expr_is_tensorish(expr.inner, ctx)
    if isinstance(expr, AxonExprTuple):
        return any(_expr_is_tensorish(item, ctx) for item in expr.items)
    if isinstance(expr, AxonExprList):
        return any(_expr_is_tensorish(item, ctx) for item in expr.items)
    return False


def _list_item_literal_value(expr: AxonExpr) -> Any:
    if isinstance(expr, AxonExprInt):
        return expr.value
    if isinstance(expr, AxonExprFloat):
        return expr.lexeme if expr.lexeme else expr.value
    if isinstance(expr, AxonExprBool):
        return expr.value
    if isinstance(expr, AxonExprNull):
        return None
    if isinstance(expr, AxonExprString):
        return expr.value
    if isinstance(expr, AxonExprName):
        return expr.name
    if isinstance(expr, AxonExprParen):
        return _list_item_literal_value(expr.inner)
    if isinstance(expr, AxonExprList):
        return [_list_item_literal_value(item) for item in expr.items]
    raise ValueError(f"list literal item must be scalar/name/list, got {type(expr).__name__}")


def _kwarg_needs_temp_binding(expr: AxonExpr, ctx: _LowerCtx) -> bool:
    if _expr_is_tensorish(expr, ctx):
        return True
    if isinstance(expr, AxonExprCall | AxonExprPipe | AxonExprBind | AxonExprDo | AxonExprLambda):
        return True
    if isinstance(expr, AxonExprParen):
        return _kwarg_needs_temp_binding(expr.inner, ctx)
    if isinstance(expr, AxonExprIf | AxonExprTernary):
        return (
            _kwarg_needs_temp_binding(expr.cond, ctx)
            or _kwarg_needs_temp_binding(expr.true_expr, ctx)
            or _kwarg_needs_temp_binding(expr.false_expr, ctx)
        )
    if isinstance(expr, AxonExprBinary):
        return _kwarg_needs_temp_binding(expr.left, ctx) or _kwarg_needs_temp_binding(
            expr.right, ctx
        )
    if isinstance(expr, AxonExprTuple):
        return any(_kwarg_needs_temp_binding(item, ctx) for item in expr.items)
    if isinstance(expr, AxonExprList):
        return any(_kwarg_needs_temp_binding(item, ctx) for item in expr.items)
    return False


def _lower_simple_call(
    callee: str,
    args: tuple[AxonExpr, ...],
    kwargs_expr: dict[str, AxonKwargValue],
    out: str | list[str],
    ctx: _LowerCtx,
    *,
    guard: str | None = None,
) -> list[dict[str, Any]]:
    callee = callee.strip()
    _validate_namespaced_block_call(callee, ctx)
    callee = _rewrite_prelude_alias_callee(callee, kwargs_expr, ctx)
    callee = _rewrite_primitive_alias_callee(callee, kwargs_expr, ctx)
    pre_graph: list[dict[str, Any]] = []
    kwargs = _render_kwargs_for_call(kwargs_expr)
    effective_when = guard
    effective_scope = _normalize_call_scope_for_runtime(
        ctx, ".".join(part for part in ctx.scope_stack if part)
    )

    resolved_args: list[str] = []
    for arg in args:
        if isinstance(arg, AxonExprName):
            resolved_args.append(arg.name)
            continue
        if isinstance(arg, AxonExprInt):
            resolved_args.append(str(arg.value))
            continue
        if isinstance(arg, AxonExprFloat):
            resolved_args.append(arg.lexeme if arg.lexeme else str(arg.value))
            continue
        if isinstance(arg, AxonExprBool):
            resolved_args.append("true" if arg.value else "false")
            continue
        if isinstance(arg, AxonExprNull):
            resolved_args.append("null")
            continue
        if isinstance(arg, AxonExprString):
            resolved_args.append(arg.value)
            continue
        tmp = ctx.fresh("arg")
        pre_graph.extend(_lower_expr(arg, tmp, ctx, guard=guard))
        resolved_args.append(tmp)
    args_text = resolved_args

    resolved_kwargs: dict[str, Any] = {}
    for key, value_expr in kwargs_expr.items():
        if isinstance(value_expr, AxonExprName):
            resolved_kwargs[key] = value_expr.name
            continue
        if isinstance(value_expr, AxonExprInt):
            resolved_kwargs[key] = value_expr.value
            continue
        if isinstance(value_expr, AxonExprFloat):
            resolved_kwargs[key] = value_expr.lexeme if value_expr.lexeme else value_expr.value
            continue
        if isinstance(value_expr, AxonExprBool):
            resolved_kwargs[key] = value_expr.value
            continue
        if isinstance(value_expr, AxonExprNull):
            resolved_kwargs[key] = None
            continue
        if isinstance(value_expr, AxonExprString):
            resolved_kwargs[key] = value_expr.value
            continue
        if isinstance(value_expr, AxonExprList):
            if _kwarg_needs_temp_binding(value_expr, ctx):
                key_token = _sanitize_token(key, default="kwarg")
                tmp = ctx.fresh(f"kwarg_{key_token}")
                pre_graph.extend(_lower_expr(value_expr, tmp, ctx, guard=guard))
                resolved_kwargs[key] = tmp
            else:
                resolved_kwargs[key] = _expr_to_runtime_value(value_expr)
            continue
        if isinstance(value_expr, AxonExpr):
            if _kwarg_needs_temp_binding(value_expr, ctx):
                key_token = _sanitize_token(key, default="kwarg")
                tmp = ctx.fresh(f"kwarg_{key_token}")
                pre_graph.extend(_lower_expr(value_expr, tmp, ctx, guard=guard))
                resolved_kwargs[key] = tmp
            else:
                resolved_kwargs[key] = _expr_to_runtime_value(value_expr)
            continue
        resolved_kwargs[key] = value_expr
    kwargs = resolved_kwargs

    is_absolute_path = "@@" in callee
    if is_absolute_path:
        callee = callee.replace("@@", "@", 1)
    op_name = _canonical_op_name(callee)
    kwargs = _normalize_call_kwargs(
        op_name=op_name,
        args=args_text,
        out=out,
        kwargs=kwargs,
        ctx=ctx,
    )
    _validate_normalized_kwargs(op_name, kwargs, args_text)
    op_validate = get_op_lowering_validator(op_name)
    if callable(op_validate):
        op_validate(args=args_text, out=out, kwargs=kwargs, ctx=ctx)
    _validate_namespaced_block_call(callee, ctx)
    resolved_block = _resolve_block_call(callee, ctx)
    if resolved_block is None and "." in callee and "@" not in callee and "::" not in callee:
        namespace = callee.split(".", 1)[0]
        raise ValueError(
            f"unknown namespaced module call {callee!r}; add `import {namespace}` and parse from file"
        )
    if resolved_block is not None and ctx.block_signatures:
        block_name, path_bindings = resolved_block
        input_names, output_names = ctx.block_signatures[block_name]
        provided: dict[str, str] = {}
        for idx, value in enumerate(args_text):
            if idx >= len(input_names):
                raise ValueError(f"too many positional args for block call {callee!r}")
            provided[input_names[idx]] = value
        for key, value in kwargs.items():
            if key not in input_names:
                raise ValueError(f"unknown block input {key!r} for call {callee!r}")
            provided[key] = str(value)
        for key, concrete_path in path_bindings.items():
            if key not in input_names:
                raise ValueError(f"unknown block path parameter {key!r} for call {callee!r}")
            concrete_value = f"@@{concrete_path}" if is_absolute_path else concrete_path
            provided[key] = repr(concrete_value)
        if isinstance(ctx.block_param_shapes, dict):
            param_shapes = ctx.block_param_shapes.get(block_name, {})
        else:
            param_shapes = {}
        symbol_bindings: dict[str, Any] = {}
        for param_name, raw in provided.items():
            token = str(raw).strip()
            if _is_name_token(token) and token in ctx.tensor_last_dim:
                symbol_bindings[param_name] = ctx.tensor_last_dim[token]
            elif not _is_name_token(token):
                symbol_bindings[param_name] = _parse_scalar_token(token)
        for param_name, param_shape in param_shapes.items():
            if param_name not in provided:
                continue
            token = str(provided[param_name]).strip()
            if not _is_name_token(token):
                continue
            arg_shape = ctx.tensor_shape.get(token)
            if arg_shape is None:
                continue
            if len(arg_shape) != len(param_shape):
                raise ValueError(
                    f"shape mismatch in call {callee!r} for param {param_name!r}: "
                    f"expected rank {len(param_shape)} from signature {param_shape}, got rank {len(arg_shape)} from {arg_shape}"
                )
            for sym, actual in zip(param_shape, arg_shape, strict=True):
                if _is_symbolic_dim_token(sym) and sym not in symbol_bindings:
                    symbol_bindings[sym] = actual
            expected_shape = tuple(symbol_bindings.get(sym, sym) for sym in param_shape)
            if len(expected_shape) != len(arg_shape) or any(
                not _dims_compatible(exp, got)
                for exp, got in zip(expected_shape, arg_shape, strict=True)
            ):
                raise ValueError(
                    f"shape mismatch in call {callee!r} for param {param_name!r}: "
                    f"expected {expected_shape} from signature, got {arg_shape} from argument {token!r}"
                )
        out_values = [out] if isinstance(out, str) else list(out)
        if len(out_values) != len(output_names):
            raise ValueError(
                f"block call {callee!r} expects {len(output_names)} outputs, got {len(out_values)}"
            )
        positional_args: list[str] = []
        extra_kwargs: dict[str, str] = {}
        for input_name in input_names:
            if input_name not in provided:
                continue
            value = provided[input_name]
            if input_name in kwargs or input_name in path_bindings:
                extra_kwargs[input_name] = value
            elif len(positional_args) < len(args_text):
                positional_args.append(value)
            else:
                extra_kwargs[input_name] = value
        node_name = f"n_{ctx.fresh('call')}"
        node_spec: dict[str, Any] = {"_op": "call", "_target": block_name}
        if positional_args:
            node_spec["_args"] = (
                positional_args[0] if len(positional_args) == 1 else positional_args
            )
        call_scope = _normalize_call_scope_for_runtime(
            ctx, ".".join(part for part in ctx.scope_stack if part)
        )
        if path_bindings:
            # Relative @path block calls still need the runtime lexical scope so bound
            # relative parameter paths resolve under the current scope. Absolute @@path
            # calls already carry their full path and must not be scope-prefixed again.
            node_spec["_scope"] = None if is_absolute_path else call_scope
        elif call_scope:
            node_spec["_scope"] = call_scope
        else:
            # Preserve explicit "no extra call scope" intent so _with_scope does not
            # re-inject lexical scope for this node.
            node_spec["_scope"] = None
        root_candidates = _current_param_roots(ctx)
        dynamic_root_expr = _current_dynamic_param_root(ctx)
        if dynamic_root_expr is not None:
            node_spec["_param_root"] = dynamic_root_expr
        elif root_candidates != ("",):
            node_spec["_param_root"] = (
                root_candidates[0] if len(root_candidates) == 1 else list(root_candidates)
            )
        node_spec["_bind"] = out_values[0] if len(out_values) == 1 else out_values
        for key, value in extra_kwargs.items():
            node_spec[key] = value
        nodes = _with_guard([{node_name: node_spec}], effective_when)
        _record_last_dim_for_call(
            callee=block_name,
            args=args_text,
            kwargs=kwargs,
            out=out,
            ctx=ctx,
        )
        return [*pre_graph, *_with_scope(nodes, effective_scope, param_root=_param_root_payload(ctx))]

    node_spec = _to_synapse_op(callee, args_text, kwargs, out)
    if "@" in callee:
        op_name, param_path = callee.split("@", 1)
        if param_path in ctx.path_param_names:
            node_name = f"n_{ctx.fresh('op')}"
            templated_node = _to_synapse_op(op_name, args_text, kwargs, out)
            templated_node["param_base"] = param_path
            nodes = _with_guard([{node_name: templated_node}], effective_when)
            _record_last_dim_for_call(
                callee=op_name, args=args_text, kwargs=kwargs, out=out, ctx=ctx
            )
            return [*pre_graph, *_with_scope(nodes, effective_scope, param_root=_param_root_payload(ctx))]
        concrete_node = _to_synapse_op(op_name, args_text, kwargs, out)
        try:
            bound_params = _path_bound_param_names(concrete_node)
        except ValueError:
            bound_params = []
        if bound_params:
            if not param_path.strip():
                raise ValueError(f"invalid @ path in Axon call: {callee!r}")
            node_name = f"n_{ctx.fresh('op')}"
            dynamic_root_expr = _current_dynamic_param_root(ctx)
            params: dict[str, str | list[str]] = {}
            for param_name in bound_params:
                explicit_name = concrete_node.get(param_name)
                if isinstance(explicit_name, str) and explicit_name.strip():
                    explicit_token = explicit_name.strip()
                    suffix = (
                        explicit_token
                        if "." in explicit_token
                        else f"{param_path}.{explicit_token}"
                    )
                else:
                    suffix = f"{param_path}.{param_name}"
                if is_absolute_path:
                    concrete_node["_abs_path"] = param_path
                    params[param_name] = explicit_token if isinstance(explicit_name, str) and explicit_name.strip() else param_name
                else:
                    params[param_name] = suffix
            concrete_node["_params"] = params
            if dynamic_root_expr is not None:
                concrete_node["_param_root"] = dynamic_root_expr
            else:
                root_candidates = _current_param_roots(ctx)
                if root_candidates != ("",):
                    concrete_node["_param_root"] = (
                        root_candidates[0] if len(root_candidates) == 1 else list(root_candidates)
                    )
            nodes = _with_guard([{node_name: concrete_node}], effective_when)
            _record_last_dim_for_call(
                callee=callee, args=args_text, kwargs=kwargs, out=out, ctx=ctx
            )
            return [*pre_graph, *_with_scope(nodes, effective_scope, param_root=_param_root_payload(ctx))]
        segments = [part.strip() for part in param_path.split(".") if part.strip()]
        if not segments:
            raise ValueError(f"invalid @ path in Axon call: {callee!r}")
        item: dict[str, Any] = {segments[-1]: node_spec}
        for segment in reversed(segments[:-1]):
            item = {segment: {"graph": [item]}}
        nodes = _with_guard([item], effective_when)
        _record_last_dim_for_call(callee=callee, args=args_text, kwargs=kwargs, out=out, ctx=ctx)
        return [*pre_graph, *_with_scope(nodes, effective_scope, param_root=_param_root_payload(ctx))]
    node_name = f"n_{ctx.fresh('op')}"
    nodes = _with_guard([{node_name: node_spec}], effective_when)
    _record_last_dim_for_call(callee=callee, args=args_text, kwargs=kwargs, out=out, ctx=ctx)
    return [*pre_graph, *_with_scope(nodes, effective_scope, param_root=_param_root_payload(ctx))]


def _lower_alias_or_const(
    expr: AxonExpr, out: str | list[str], ctx: _LowerCtx, *, guard: str | None = None
) -> list[dict[str, Any]]:
    if isinstance(out, list):
        raise ValueError("alias/const lowering expects scalar out")
    node_name = f"n_{ctx.fresh('op')}"
    node: dict[str, Any]
    if isinstance(expr, AxonExprName):
        if expr.name in ctx.symbol_names:
            node = {"_op": "_ir_expr", "value": expr.name, "_bind": out}
        else:
            node = {"_op": "_ir_alias", "_args": expr.name, "_bind": out}
            if expr.name in ctx.tensor_last_dim:
                ctx.tensor_last_dim[out] = ctx.tensor_last_dim[expr.name]
    elif isinstance(expr, AxonExprInt):
        node = {"_op": "_ir_expr", "value": expr.value, "_bind": out}
    elif isinstance(expr, AxonExprFloat):
        node = {
            "_op": "_ir_expr",
            "value": expr.lexeme if expr.lexeme else expr.value,
            "_bind": out,
        }
    elif isinstance(expr, AxonExprBool):
        node = {"_op": "_ir_expr", "value": expr.value, "_bind": out}
    elif isinstance(expr, AxonExprNull):
        node = {"_op": "_ir_expr", "value": None, "_bind": out}
    elif isinstance(expr, AxonExprString):
        node = {"_op": "_ir_expr", "value": _expr_to_runtime_value(expr), "_bind": out}
    elif isinstance(expr, AxonExprList):
        node = {
            "_op": "_ir_expr",
            "value": _expr_to_runtime_value(expr),
            "_bind": out,
        }
    else:
        node = {"_op": "_ir_expr", "value": _expr_to_runtime_value(expr), "_bind": out}
    effective_scope = _normalize_call_scope_for_runtime(
        ctx, ".".join(part for part in ctx.scope_stack if part)
    )
    return _with_scope(
        _with_guard([{node_name: node}], guard),
        effective_scope,
        param_root=_param_root_payload(ctx),
    )


def _bind_names(out: str | list[str]) -> list[str]:
    return [out] if isinstance(out, str) else list(out)


def _bind_field(names: list[str]) -> str | list[str]:
    return names[0] if len(names) == 1 else names


def _lower_select_branch(
    *,
    branch_expr: AxonExpr,
    branch_binds: list[str],
    ctx: _LowerCtx,
) -> list[dict[str, Any]]:
    branch_graph: list[dict[str, Any]] = []
    if len(branch_binds) > 1 and isinstance(branch_expr, AxonExprCall | AxonExprName):
        expanded = _expand_call_outputs_for_ternary(branch_expr, list(branch_binds), ctx)
        branch_graph.extend(_lower_expr(branch_expr, expanded, ctx))
        return branch_graph
    branch_out: str | list[str] = _bind_field(branch_binds)
    branch_graph.extend(_lower_expr(branch_expr, branch_out, ctx))
    return branch_graph


def _lower_select_cond(
    cond_expr: AxonExpr,
    *,
    ctx: _LowerCtx,
) -> tuple[list[dict[str, Any]], Any]:
    if isinstance(cond_expr, AxonExprName):
        return [], cond_expr.name
    if isinstance(cond_expr, AxonExprBool):
        return [], cond_expr.value
    if isinstance(cond_expr, AxonExprInt):
        return [], cond_expr.value
    if isinstance(cond_expr, AxonExprFloat):
        return [], cond_expr.lexeme if cond_expr.lexeme else cond_expr.value
    if isinstance(cond_expr, AxonExprNull):
        return [], None
    if isinstance(cond_expr, AxonExprParen):
        return _lower_select_cond(cond_expr.inner, ctx=ctx)
    cond_ref = ctx.fresh("cond")
    return _lower_expr(cond_expr, cond_ref, ctx), cond_ref


def _lower_expr(
    expr: AxonExpr,
    out: str | list[str],
    ctx: _LowerCtx,
    *,
    guard: str | None = None,
) -> list[dict[str, Any]]:
    if isinstance(expr, AxonExprBind):
        bind_graph: list[dict[str, Any]] = []
        bind_ref: str
        if isinstance(expr.value, AxonExprName):
            bind_ref = expr.value.name
        else:
            bind_ref = ctx.fresh("bind")
            bind_graph.extend(_lower_expr(expr.value, bind_ref, ctx, guard=guard))
        body = _substitute_expr(expr.body, expr.var, AxonExprName(name=bind_ref))
        bind_graph.extend(_lower_expr(body, out, ctx, guard=guard))
        return bind_graph

    if isinstance(expr, AxonExprDo):
        do_graph: list[dict[str, Any]] = []
        do_outputs: dict[str, str] = {}
        _lower_statements(
            statements=expr.body,
            graph=do_graph,
            outputs=do_outputs,
            returns=(),
            ctx=ctx,
            guard=guard,
        )
        if isinstance(out, list):
            for idx, target in enumerate(out):
                output_name = f"out_{idx}"
                source = do_outputs.get(output_name)
                if source is None:
                    raise ValueError(f"do expression must return value {idx} via `return`")
                if target != source:
                    do_graph.extend(
                        _lower_expr(AxonExprName(name=source), target, ctx, guard=guard)
                    )
            return do_graph
        source0 = do_outputs.get("out_0")
        if source0 is None:
            raise ValueError("do expression must return value 0 via `return`")
        if out != source0:
            do_graph.extend(_lower_expr(AxonExprName(name=source0), out, ctx, guard=guard))
        return do_graph

    if isinstance(expr, AxonExprIf | AxonExprTernary):
        out_names = _bind_names(out)
        cond_graph, cond_value = _lower_select_cond(expr.cond, ctx=ctx)

        then_binds = [
            ctx.fresh(f"then_{_sanitize_token(name, default='out')}") for name in out_names
        ]
        else_binds = [
            ctx.fresh(f"else_{_sanitize_token(name, default='out')}") for name in out_names
        ]
        then_graph = _lower_select_branch(
            branch_expr=expr.true_expr, branch_binds=then_binds, ctx=ctx
        )
        else_graph = _lower_select_branch(
            branch_expr=expr.false_expr, branch_binds=else_binds, ctx=ctx
        )

        node_name = f"n_{ctx.fresh('select')}"
        node_spec: dict[str, Any] = {
            "_op": "select",
            "_bind": _bind_field(out_names),
            "cond": cond_value,
            "_then": then_graph,
            "_else": else_graph,
            "_then_bind": _bind_field(then_binds),
            "_else_bind": _bind_field(else_binds),
        }
        effective_scope = _normalize_call_scope_for_runtime(
            ctx, ".".join(part for part in ctx.scope_stack if part)
        )
        if effective_scope:
            node_spec["_scope"] = effective_scope
        return [
            *cond_graph,
            *(
                _with_scope(
                    [{node_name: node_spec}],
                    effective_scope,
                    param_root=_param_root_payload(ctx),
                )
            ),
        ]

    if isinstance(expr, AxonExprPipe):
        pipe_graph: list[dict[str, Any]] = []
        if len(expr.stages) == 0:
            return _lower_expr(expr.value, out, ctx, guard=guard)
        if isinstance(expr.value, AxonExprName):
            pipe_ref: str | list[str] = expr.value.name
        else:
            first_out: str | list[str] = _pipeline_temp_out(expr.value, ctx)
            pipe_graph.extend(_lower_expr(expr.value, first_out, ctx, guard=guard))
            pipe_ref = first_out

        for idx, stage in enumerate(expr.stages, start=1):
            if isinstance(stage, AxonExprCall):
                stage_callee = stage.callee
                stage_args = list(stage.args)
                stage_kwargs = dict(stage.kwargs)
            elif isinstance(stage, AxonExprName):
                stage_callee = stage.name
                stage_args = []
                stage_kwargs = {}
            else:
                raise ValueError("pipeline stage must be a call or name")
            next_out: str | list[str] = (
                out if idx == len(expr.stages) else _pipeline_temp_out(stage, ctx)
            )
            piped_refs = [pipe_ref] if isinstance(pipe_ref, str) else list(pipe_ref)
            piped_args = [AxonExprName(name=ref) for ref in piped_refs]
            if stage_args:
                same_prefix = len(stage_args) >= len(piped_refs)
                if same_prefix:
                    for idx_ref, ref in enumerate(piped_refs):
                        head = stage_args[idx_ref]
                        if not isinstance(head, AxonExprName) or head.name != ref:
                            same_prefix = False
                            break
                if same_prefix:
                    stage_args = stage_args[len(piped_refs) :]
            call_args = tuple([*piped_args, *stage_args])
            pipe_graph.extend(
                _lower_simple_call(
                    stage_callee, call_args, stage_kwargs, next_out, ctx, guard=guard
                )
            )
            pipe_ref = next_out
        return pipe_graph

    if isinstance(expr, AxonExprCall):
        return _lower_simple_call(expr.callee, expr.args, expr.kwargs, out, ctx, guard=guard)

    if isinstance(expr, AxonExprBinary) and expr.op in {"+", "*"}:
        if not (_expr_is_tensorish(expr.left, ctx) or _expr_is_tensorish(expr.right, ctx)):
            return _lower_alias_or_const(expr, out, ctx, guard=guard)
        op_name = "add" if expr.op == "+" else "mul"
        binary_graph: list[dict[str, Any]] = []
        if isinstance(expr.left, AxonExprName) and expr.left.name not in ctx.symbol_names:
            left_ref = expr.left.name
        else:
            left_ref = ctx.fresh("bin")
            binary_graph.extend(_lower_expr(expr.left, left_ref, ctx, guard=guard))
        if isinstance(expr.right, AxonExprName) and expr.right.name not in ctx.symbol_names:
            right_ref = expr.right.name
        else:
            right_ref = ctx.fresh("bin")
            binary_graph.extend(_lower_expr(expr.right, right_ref, ctx, guard=guard))
        binary_graph.extend(
            _lower_simple_call(
                op_name,
                (AxonExprName(name=left_ref), AxonExprName(name=right_ref)),
                {},
                out,
                ctx,
                guard=guard,
            )
        )
        return binary_graph

    if isinstance(expr, AxonExprTuple):
        if not isinstance(out, list) or len(out) != len(expr.items):
            raise ValueError("tuple expression arity must match binding targets")
        tuple_graph: list[dict[str, Any]] = []
        for name, item in zip(out, expr.items, strict=True):
            tuple_graph.extend(_lower_expr(item, name, ctx, guard=guard))
        return tuple_graph

    if isinstance(expr, AxonExprParen):
        return _lower_expr(expr.inner, out, ctx, guard=guard)

    if isinstance(expr, AxonExprLambda):
        raise ValueError("lambda expression cannot be lowered directly")
    return _lower_alias_or_const(expr, out, ctx, guard=guard)


def _module_return_names(module: AxonModule) -> tuple[str, ...]:
    if module.returns:
        return module.returns
    for stmt in reversed(module.statements):
        if isinstance(stmt, AxonReturn):
            inferred: list[str] = []
            for idx, value in enumerate(stmt.values):
                maybe_name = _expr_name(value)
                inferred.append(maybe_name if maybe_name is not None else f"out_{idx}")
            if inferred:
                return tuple(inferred)
    return ()


def _module_return_last_dims(module: AxonModule, returns: tuple[str, ...]) -> dict[str, Any]:
    if not returns or module.return_shape is None or len(module.return_shape) == 0:
        return {}
    if len(returns) != 1:
        return {}
    return {returns[0]: module.return_shape[-1]}


def _module_param_last_dims(module: AxonModule) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for param in module.params:
        if param.shape is None or len(param.shape) == 0:
            continue
        out[param.name] = param.shape[-1]
    return out


def _module_param_shapes(module: AxonModule) -> dict[str, tuple[Any, ...]]:
    out: dict[str, tuple[Any, ...]] = {}
    for param in module.params:
        if param.shape is None:
            continue
        out[param.name] = tuple(param.shape)
    return out


def _module_return_shapes(
    module: AxonModule, returns: tuple[str, ...]
) -> dict[str, tuple[Any, ...]]:
    if not returns or module.return_shape is None:
        return {}
    if len(returns) != 1:
        return {}
    return {returns[0]: tuple(module.return_shape)}


def _module_return_heads(module: AxonModule, returns: tuple[str, ...]) -> dict[str, Any]:
    if not returns or module.return_shape is None or len(module.return_shape) < 2:
        return {}
    if len(returns) != 1:
        return {}
    return {returns[0]: module.return_shape[1]}


def _module_inputs(module: AxonModule) -> dict[str, dict[str, bool]]:
    inputs = {param.name: {"optional": param.optional} for param in module.params}
    for path_param in module.path_params:
        inputs[path_param] = {"optional": False}
    if not module.path_params and module.path_param is not None:
        inputs[module.path_param] = {"optional": False}
    return inputs


def _module_initial_dims(module: AxonModule, returns: tuple[str, ...]) -> dict[str, Any]:
    initial_dims = {
        param.name: param.shape[-1]
        for param in module.params
        if param.shape is not None and len(param.shape) > 0
    }
    initial_dims.update(_module_return_last_dims(module, returns))
    return initial_dims


def _module_initial_shapes(
    module: AxonModule, returns: tuple[str, ...]
) -> dict[str, tuple[Any, ...]]:
    initial_shapes = {
        param.name: tuple(param.shape) for param in module.params if param.shape is not None
    }
    initial_shapes.update(_module_return_shapes(module, returns))
    return initial_shapes


def _module_initial_heads(module: AxonModule, returns: tuple[str, ...]) -> dict[str, Any]:
    initial_heads = {
        param.name: param.shape[1]
        for param in module.params
        if param.shape is not None and len(param.shape) >= 2
    }
    initial_heads.update(_module_return_heads(module, returns))
    return initial_heads


def _module_path_param_names(module: AxonModule) -> set[str]:
    names = {p for p in module.path_params if isinstance(p, str)}
    if not names and isinstance(module.path_param, str):
        names.add(module.path_param)
    return names


def _ensure_outputs_from_returns(outputs: dict[str, str], returns: tuple[str, ...]) -> None:
    if outputs:
        return
    for name in returns:
        outputs[name] = name


def _extract_primitive_aliases(modules: tuple[AxonModule, ...]) -> dict[str, tuple[str, int]]:
    allowed_namespaces = {
        "Prelude",
        "Activations",
        "Cache",
        "List",
        "MoE",
        "Config",
        "Params",
        "Position",
    }
    direct_aliases: dict[str, tuple[str, int]] = {}
    for module in modules:
        if not isinstance(module.name, str) or "." not in module.name:
            continue
        namespace, _ = module.name.split(".", 1)
        if namespace not in allowed_namespaces:
            continue
        if len(module.statements) != 1:
            continue
        stmt = module.statements[0]
        if not isinstance(stmt, AxonReturn) or len(stmt.values) != 1:
            continue
        value = stmt.values[0]
        if not isinstance(value, AxonExprCall):
            continue
        target_base = value.callee.split("@", 1)[0]
        direct_aliases[module.name] = (target_base, len(module.path_params))

    aliases: dict[str, tuple[str, int]] = {}
    for name, (target_base, expected_path_count) in direct_aliases.items():
        seen: set[str] = set()
        resolved = target_base
        while not resolved.startswith("_"):
            if resolved in seen:
                break
            seen.add(resolved)
            next_alias = direct_aliases.get(resolved)
            if next_alias is None:
                break
            next_base, next_path_count = next_alias
            if next_path_count != expected_path_count:
                break
            resolved = next_base
        if resolved.startswith("_"):
            aliases[name] = (resolved, expected_path_count)
    return aliases


def _extract_prelude_aliases(modules: tuple[AxonModule, ...]) -> dict[str, tuple[str, int]]:
    aliases: dict[str, tuple[str, int]] = {}
    primitive_aliases = _extract_primitive_aliases(modules)
    for full_name, alias in primitive_aliases.items():
        if not full_name.startswith("Prelude."):
            continue
        member_name = full_name.split(".", 1)[1]
        if not member_name:
            continue
        aliases[member_name] = alias
    return aliases


def _collect_imported_symbol_values(
    module: AxonModule,
    modules_by_name: dict[str, AxonModule] | None,
) -> dict[str, Any]:
    if not module.imported_members or not modules_by_name:
        return {}
    collected: dict[str, Any] = {}
    for namespace, members in module.imported_members.items():
        imported_module = modules_by_name.get(namespace)
        if imported_module is None or not isinstance(imported_module.symbols, dict):
            continue
        for member in members:
            if member in collected:
                continue
            if member in imported_module.symbols:
                collected[member] = imported_module.symbols[member]
    return collected


def _new_lower_ctx(
    *,
    module: AxonModule,
    returns: tuple[str, ...],
    signatures: dict[str, tuple[list[str], list[str]]] | None,
    block_path_params: dict[str, tuple[str, ...]] | None,
    block_param_last_dims: dict[str, dict[str, Any]] | None,
    block_output_last_dims: dict[str, dict[str, Any]] | None,
    block_param_shapes: dict[str, dict[str, tuple[Any, ...]]] | None = None,
    block_output_shapes: dict[str, dict[str, tuple[Any, ...]]] | None = None,
    implicit_prelude_members: set[str] | None = None,
    prelude_aliases: dict[str, tuple[str, int]] | None = None,
    primitive_aliases: dict[str, tuple[str, int]] | None = None,
    imported_symbol_values: dict[str, Any] | None = None,
) -> _LowerCtx:
    imported_member_namespaces: dict[str, set[str]] = {}
    if module.imported_members:
        for namespace, members in module.imported_members.items():
            for member in members:
                bucket = imported_member_namespaces.setdefault(member, set())
                bucket.add(namespace)
    if implicit_prelude_members:
        for member in implicit_prelude_members:
            if member in imported_member_namespaces:
                continue
            bucket = imported_member_namespaces.setdefault(member, set())
            bucket.add("Prelude")
    symbol_values = dict(imported_symbol_values or {})
    if isinstance(module.symbols, dict):
        symbol_values.update(module.symbols)
    return _LowerCtx(
        block_signatures=signatures,
        block_path_params=block_path_params,
        block_param_last_dims=block_param_last_dims,
        block_output_last_dims=block_output_last_dims,
        block_param_shapes=block_param_shapes,
        block_output_shapes=block_output_shapes,
        tensor_last_dim=_module_initial_dims(module, returns),
        tensor_heads=_module_initial_heads(module, returns),
        tensor_shape=_module_initial_shapes(module, returns),
        path_param_names=_module_path_param_names(module),
        imported_namespaces=set(module.imports) | {"Prelude"},
        imported_member_namespaces=imported_member_namespaces,
        prelude_aliases=dict(prelude_aliases or {}),
        primitive_aliases=dict(primitive_aliases or {}),
        current_module=module.name,
        param_names={param.name for param in module.params},
        symbol_names=set(symbol_values.keys()),
        symbol_values=symbol_values,
    )


def lower_axon_module_to_synapse_block(module: AxonModule) -> dict[str, Any]:
    typecheck_axon_program((module,), main_module=module.name)
    inputs = _module_inputs(module)
    graph: list[dict[str, Any]] = []
    outputs: dict[str, str] = {}
    returns = _module_return_names(module)
    ctx = _new_lower_ctx(
        module=module,
        returns=returns,
        signatures={},
        block_path_params={
            module.name: module.path_params
            if module.path_params
            else tuple([module.path_param] if module.path_param is not None else [])
        },
        block_param_last_dims={module.name: _module_param_last_dims(module)},
        block_output_last_dims={module.name: _module_return_last_dims(module, returns)},
        block_param_shapes={module.name: _module_param_shapes(module)},
        block_output_shapes={module.name: _module_return_shapes(module, returns)},
        implicit_prelude_members=set(),
        prelude_aliases={},
        imported_symbol_values={},
    )

    _lower_statements(
        statements=module.statements,
        graph=graph,
        outputs=outputs,
        returns=returns,
        ctx=ctx,
    )

    _ensure_outputs_from_returns(outputs, returns)

    return {"inputs": inputs, "graph": graph, "outputs": outputs}


def lower_axon_module_to_synapse_spec(module: AxonModule) -> dict[str, Any]:
    block = lower_axon_module_to_synapse_block(module)
    model: dict[str, Any] = {
        "inputs": block["inputs"],
        "graph": block["graph"],
        "outputs": block["outputs"],
    }
    if module.symbols:
        model["symbols"] = dict(module.symbols)
    if module.pragmas:
        model["meta"] = dict(module.pragmas)
    spec = {
        "synapse": 1,
        "model": model,
    }
    inferred_block_io_types = infer_block_io_types_from_modules(
        spec=spec,
        modules=(module,),
        selected_main=module.name,
    )
    annotate_spec_with_block_io_types(spec, block_io_types=inferred_block_io_types)
    return spec


def _lower_statements(
    *,
    statements: tuple[AxonStatement, ...],
    graph: list[dict[str, Any]],
    outputs: dict[str, str],
    returns: tuple[str, ...],
    ctx: _LowerCtx,
    guard: str | None = None,
) -> None:
    for stmt in statements:
        if isinstance(stmt, AxonRepeat):
            body_graph: list[dict[str, Any]] = []
            base_loop_scope = stmt.name if isinstance(stmt.name, str) and stmt.name else ""
            loop_cover_prefix = ".".join(part for part in ctx.scope_stack if part)
            full_loop_scope = _join_dot(loop_cover_prefix, base_loop_scope)
            ctx.loop_scope_cover_prefixes.append(full_loop_scope)
            if base_loop_scope:
                ctx.scope_stack.append(base_loop_scope)
            try:
                _lower_statements(
                    statements=stmt.body,
                    graph=body_graph,
                    outputs={},
                    returns=(),
                    ctx=ctx,
                    guard=guard,
                )
            finally:
                if base_loop_scope:
                    ctx.scope_stack.pop()
                ctx.loop_scope_cover_prefixes.pop()
            node_name = f"n_{ctx.fresh('for')}"
            base_loop_scope = stmt.name if isinstance(stmt.name, str) and stmt.name else node_name
            loop_scope = base_loop_scope
            if ctx.scope_stack:
                scope_prefix = ".".join(part for part in ctx.scope_stack if part)
                if scope_prefix:
                    loop_scope = f"{scope_prefix}.{base_loop_scope}"
            repeat_item: dict[str, Any] = {
                node_name: {
                    "_op": "for",
                    "_scope": loop_scope,
                    "_var": stmt.var,
                    "_to": _expr_to_runtime_value(stmt.to_expr),
                    "_body": body_graph,
                }
            }
            from_expr = _expr_to_runtime_value(stmt.from_expr)
            step_expr = _expr_to_runtime_value(stmt.step_expr)
            if from_expr != 0:
                repeat_item[node_name]["_from"] = from_expr
            if step_expr != 1:
                repeat_item[node_name]["_step"] = step_expr
            graph.append(repeat_item)
            continue

        if isinstance(stmt, AxonScopeBind):
            root_values: tuple[str, ...] = ("",)
            dynamic_root_expr: Any | None = None
            if "root" in stmt.kwargs:
                resolved_root_values = _scope_root_values_from_kwarg(stmt.kwargs["root"], ctx)
                if resolved_root_values is None:
                    dynamic_root_expr = _scope_root_expr_from_kwarg(stmt.kwargs["root"])
                else:
                    root_values = resolved_root_values
            ctx.scope_stack.append(stmt.prefix)
            ctx.param_root_stack.append(root_values)
            ctx.dynamic_param_root_stack.append(dynamic_root_expr)
            scoped_outputs: dict[str, str] = {}
            try:
                _lower_statements(
                    statements=stmt.body,
                    graph=graph,
                    outputs=scoped_outputs,
                    returns=(),
                    ctx=ctx,
                    guard=guard,
                )
            finally:
                ctx.dynamic_param_root_stack.pop()
                ctx.param_root_stack.pop()
                ctx.scope_stack.pop()
            for idx, target in enumerate(stmt.targets):
                output_name = f"out_{idx}"
                if output_name not in scoped_outputs:
                    raise ValueError(
                        f"scope bind for {stmt.prefix!r} must return value {idx} via `return`"
                    )
                source_name = scoped_outputs[output_name]
                if target == source_name:
                    continue
                graph.extend(_lower_expr(AxonExprName(name=source_name), target, ctx, guard=guard))
            continue

        if isinstance(stmt, AxonBind):
            out: str | list[str] = stmt.targets[0] if len(stmt.targets) == 1 else list(stmt.targets)
            graph.extend(_lower_expr(stmt.expr, out, ctx, guard=guard))
            continue

        if isinstance(stmt, AxonReturn):
            for idx, value in enumerate(stmt.values):
                output_name = returns[idx] if idx < len(returns) else f"out_{idx}"
                maybe_name = _expr_name(value)
                if maybe_name is not None:
                    outputs[output_name] = maybe_name
                    continue
                graph.extend(_lower_expr(value, output_name, ctx, guard=guard))
                outputs[output_name] = output_name
            continue


def _as_concrete_path(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    token = value.strip()
    if not token:
        return None
    if len(token) >= 2 and token[0] == token[-1] and token[0] in {"'", '"'}:
        inner = token[1:-1].strip()
        return inner or None
    return token


def _normalize_bound_params_on_node(node_spec: dict[str, Any], *, base_path: str) -> None:
    explicit_params = node_spec.get("_params")
    params = dict(explicit_params) if isinstance(explicit_params, dict) else {}
    is_absolute = base_path.startswith("@@")
    normalized_base = base_path[2:] if is_absolute else base_path
    if is_absolute:
        node_spec["_abs_path"] = normalized_base
        node_spec.pop("_scope", None)
    for param_name in _path_bound_param_names(node_spec):
        explicit_value = params.get(param_name)
        if isinstance(explicit_value, str) and explicit_value.strip():
            token = explicit_value.strip()
            if is_absolute and token.startswith("@@"):
                token = token[2:]
            if is_absolute:
                params[param_name] = token or param_name
            else:
                params[param_name] = token if "." in token else f"{normalized_base}.{token}"
            continue
        params[param_name] = param_name if is_absolute else f"{normalized_base}.{param_name}"
    node_spec["_params"] = params
    node_spec.pop("param_base", None)


def _sanitize_path_suffix(value: str) -> str:
    return _sanitize_token(value, default="path")


def _path_bound_param_names(node_spec: dict[str, Any]) -> list[str]:
    op = node_spec.get("_op")
    if op == "embedding":
        return ["weight"]
    if op == "linear":
        names = ["weight"]
        if bool(node_spec.get("bias", False)):
            names.append("bias")
        return names
    if op == "rmsnorm":
        return ["weight"]
    if op == "layernorm":
        names = ["weight"]
        raw_bias = node_spec.get("bias", True)
        has_bias = not (raw_bias is None or raw_bias is False)
        if has_bias:
            names.append("bias")
        return names
    if op == "activations_xielu":
        return ["alpha_p", "alpha_n", "beta", "eps"]
    if op == "t5_relative_position_bias":
        return ["weight"]
    raise ValueError(f"unsupported param_base resolution for op {op!r}")


def _resolve_paths_at_lowering_time(
    model: dict[str, Any], block_path_params: dict[str, tuple[str, ...]]
) -> None:
    blocks_raw = model.get("blocks")
    if not isinstance(blocks_raw, dict) or not blocks_raw:
        return

    base_blocks: dict[str, dict[str, Any]] = {
        name: copy.deepcopy(spec) for name, spec in blocks_raw.items() if isinstance(spec, dict)
    }
    all_blocks: dict[str, dict[str, Any]] = dict(base_blocks)
    path_params_by_block: dict[str, tuple[str, ...]] = {
        name: tuple(p for p in block_path_params.get(name, ()) if isinstance(p, str) and p)
        for name in base_blocks
    }
    specialization_cache: dict[tuple[str, tuple[tuple[str, str], ...]], str] = {}

    def _to_list(value: Any) -> list[Any]:
        if isinstance(value, list):
            return list(value)
        if value is None:
            return []
        return [value]

    def _substitute_names(value: Any, mapping: dict[str, Any]) -> Any:
        if isinstance(value, str):
            return mapping.get(value, value)
        if isinstance(value, list):
            return [_substitute_names(item, mapping) for item in value]
        if isinstance(value, dict):
            return {k: _substitute_names(v, mapping) for k, v in value.items()}
        return value

    def _map_call_inputs_to_values(
        *, call_spec: dict[str, Any], block_spec: dict[str, Any]
    ) -> dict[str, Any] | None:
        inputs = block_spec.get("inputs")
        if not isinstance(inputs, dict):
            return None
        ordered_inputs = list(inputs.keys())
        positional = _to_list(call_spec.get("_args"))
        kwargs = {
            key: value
            for key, value in call_spec.items()
            if isinstance(key, str) and not key.startswith("_")
        }
        env: dict[str, Any] = {}
        for idx, name in enumerate(ordered_inputs):
            if idx < len(positional):
                env[name] = positional[idx]
                continue
            if name in kwargs:
                env[name] = kwargs[name]
                continue
            return None
        return env

    def _bind_map_for_inline(*, template_bind: Any, call_bind: Any) -> dict[str, Any] | None:
        template = _to_list(template_bind)
        target = _to_list(call_bind)
        if len(template) != len(target):
            return None
        return {
            str(src): dst
            for src, dst in zip(template, target, strict=False)
            if isinstance(src, str)
        }

    def _inline_simple_call_if_possible(
        *, call_spec: dict[str, Any], target: str, concrete_paths: dict[str, str]
    ) -> dict[str, Any] | None:
        block_spec = base_blocks.get(target)
        if not isinstance(block_spec, dict):
            return None
        graph = block_spec.get("graph")
        if not isinstance(graph, list) or len(graph) != 1:
            return None
        node_item = graph[0]
        if not isinstance(node_item, dict) or len(node_item) != 1:
            return None
        _, template_spec = next(iter(node_item.items()))
        if not isinstance(template_spec, dict):
            return None
        template_op = template_spec.get("_op")
        if not isinstance(template_op, str) or template_op in {"call", "for"}:
            return None

        input_values = _map_call_inputs_to_values(call_spec=call_spec, block_spec=block_spec)
        if input_values is None:
            return None
        input_values.update(concrete_paths)

        bind_map = _bind_map_for_inline(
            template_bind=template_spec.get("_bind"),
            call_bind=call_spec.get("_bind"),
        )
        if bind_map is None:
            return None
        input_values.update(bind_map)

        inlined = copy.deepcopy(template_spec)
        inlined = _substitute_names(inlined, input_values)
        caller_scope = call_spec.get("_scope")
        if isinstance(caller_scope, str) and caller_scope and "_scope" not in inlined:
            inlined["_scope"] = caller_scope
        caller_param_root = call_spec.get("_param_root")
        if (
            caller_param_root is not None
            and "_param_root" not in inlined
            and "_abs_path" not in inlined
        ):
            inlined["_param_root"] = copy.deepcopy(caller_param_root)

        param_base = inlined.get("param_base")
        if isinstance(param_base, str):
            _normalize_bound_params_on_node(inlined, base_path=param_base)

        return inlined

    def ensure_specialized_block(block_name: str, path_bindings: dict[str, str]) -> str:
        cache_key = (block_name, tuple(sorted(path_bindings.items())))
        cached = specialization_cache.get(cache_key)
        if cached is not None:
            return cached
        base_spec = base_blocks.get(block_name)
        if base_spec is None:
            return block_name
        specialized = copy.deepcopy(base_spec)
        inputs = specialized.get("inputs")
        if isinstance(inputs, dict):
            for path_name in path_bindings:
                inputs.pop(path_name, None)
        parts = [block_name]
        for key in sorted(path_bindings):
            parts.append(f"{key}_{_sanitize_path_suffix(path_bindings[key])}")
        base_name = "__".join(parts)
        candidate = base_name
        idx = 2
        while candidate in all_blocks:
            candidate = f"{base_name}_{idx}"
            idx += 1
        rewrite_graph(
            specialized.get("graph"),
            inherited_path_bindings=path_bindings,
        )
        all_blocks[candidate] = specialized
        path_params_by_block[candidate] = ()
        specialization_cache[cache_key] = candidate
        return candidate

    def rewrite_graph(graph: Any, *, inherited_path_bindings: dict[str, str]) -> None:
        if not isinstance(graph, list):
            return
        for item in graph:
            if not isinstance(item, dict) or len(item) != 1:
                continue
            _, node_spec = next(iter(item.items()))
            if not isinstance(node_spec, dict):
                continue

            param_base = node_spec.get("param_base")
            if isinstance(param_base, str) and param_base in inherited_path_bindings:
                base_path = inherited_path_bindings[param_base]
                _normalize_bound_params_on_node(node_spec, base_path=base_path)

            if node_spec.get("_op") == "call":
                target = node_spec.get("_target")
                if isinstance(target, str):
                    required_path_params = path_params_by_block.get(target, ())
                    if required_path_params:
                        concrete: dict[str, str] = {}
                        for path_name in required_path_params:
                            concrete_value = _as_concrete_path(node_spec.get(path_name))
                            if concrete_value is None:
                                concrete = {}
                                break
                            concrete[path_name] = concrete_value
                        if concrete:
                            inlined = _inline_simple_call_if_possible(
                                call_spec=node_spec,
                                target=target,
                                concrete_paths=concrete,
                            )
                            if inlined is not None:
                                node_spec.clear()
                                node_spec.update(inlined)
                                target = None
                                required_path_params = ()
                                continue
                            specialized_name = ensure_specialized_block(target, concrete)
                            node_spec["_target"] = specialized_name
                            for path_name in required_path_params:
                                node_spec.pop(path_name, None)

            nested = node_spec.get("graph")
            if isinstance(nested, list):
                rewrite_graph(nested, inherited_path_bindings=inherited_path_bindings)
            body = node_spec.get("_body")
            if isinstance(body, list):
                rewrite_graph(body, inherited_path_bindings=inherited_path_bindings)
            then_branch = node_spec.get("_then")
            if isinstance(then_branch, list):
                rewrite_graph(then_branch, inherited_path_bindings=inherited_path_bindings)
            else_branch = node_spec.get("_else")
            if isinstance(else_branch, list):
                rewrite_graph(else_branch, inherited_path_bindings=inherited_path_bindings)

    rewrite_graph(model.get("graph"), inherited_path_bindings={})
    for block_spec in list(all_blocks.values()):
        rewrite_graph(block_spec.get("graph"), inherited_path_bindings={})

    reachable: set[str] = set()

    def collect_called_blocks(graph: Any) -> list[str]:
        called: list[str] = []
        if not isinstance(graph, list):
            return called
        for item in graph:
            if not isinstance(item, dict) or len(item) != 1:
                continue
            _, node_spec = next(iter(item.items()))
            if not isinstance(node_spec, dict):
                continue
            if node_spec.get("_op") == "call" and isinstance(node_spec.get("_target"), str):
                called.append(node_spec["_target"])
            nested = node_spec.get("graph")
            if isinstance(nested, list):
                called.extend(collect_called_blocks(nested))
            body = node_spec.get("_body")
            if isinstance(body, list):
                called.extend(collect_called_blocks(body))
            then_branch = node_spec.get("_then")
            if isinstance(then_branch, list):
                called.extend(collect_called_blocks(then_branch))
            else_branch = node_spec.get("_else")
            if isinstance(else_branch, list):
                called.extend(collect_called_blocks(else_branch))
        return called

    worklist = collect_called_blocks(model.get("graph"))
    while worklist:
        target = worklist.pop()
        if target in reachable:
            continue
        if target not in all_blocks:
            continue
        block_spec = all_blocks[target]
        reachable.add(target)
        worklist.extend(collect_called_blocks(block_spec.get("graph")))

    def _assert_no_unresolved_param_base(graph: Any, *, block_name: str) -> None:
        if not isinstance(graph, list):
            return
        for item in graph:
            if not isinstance(item, dict) or len(item) != 1:
                continue
            _, node_spec = next(iter(item.items()))
            if not isinstance(node_spec, dict):
                continue
            if "param_base" in node_spec:
                raise ValueError(f"unresolved param_base in reachable block {block_name!r}")
            for key in ("graph", "_body", "_then", "_else"):
                nested = node_spec.get(key)
                if isinstance(nested, list):
                    _assert_no_unresolved_param_base(nested, block_name=block_name)

    for block_name in sorted(reachable):
        block_spec = all_blocks[block_name]
        _assert_no_unresolved_param_base(block_spec.get("graph"), block_name=block_name)

    if reachable:
        model["blocks"] = {name: all_blocks[name] for name in all_blocks if name in reachable}
    else:
        model.pop("blocks", None)


def _canonical_type_source_block_name(block_name: str) -> str:
    if "__" in block_name:
        return block_name.split("__", 1)[0]
    return block_name


def _finalize_block_io_types_for_model(
    *,
    model: dict[str, Any],
    inferred: dict[str, dict[str, dict[str, str]]],
) -> dict[str, dict[str, dict[str, str]]]:
    final_types: dict[str, dict[str, dict[str, str]]] = {}
    main_types = inferred.get("main")
    if isinstance(main_types, dict):
        final_types["main"] = main_types

    blocks = model.get("blocks")
    if not isinstance(blocks, dict):
        return final_types

    for block_name in blocks:
        if not isinstance(block_name, str):
            continue
        direct = inferred.get(block_name)
        if isinstance(direct, dict):
            final_types[block_name] = direct
            continue
        source_name = _canonical_type_source_block_name(block_name)
        source = inferred.get(source_name)
        if isinstance(source, dict):
            final_types[block_name] = source
    return final_types


def lower_axon_program_to_synapse_spec(
    modules: tuple[AxonModule, ...], *, main_module: str | None = None
) -> dict[str, Any]:
    typecheck_axon_program(modules, main_module=main_module)
    if not modules:
        raise ValueError("Axon program must contain at least one module")

    by_name = {module.name: module for module in modules}
    if len(by_name) != len(modules):
        raise ValueError("Axon program contains duplicate module names")

    main_name = modules[-1].name if main_module is None else main_module
    if main_name not in by_name:
        raise ValueError(f"Unknown main module: {main_name!r}")

    signatures: dict[str, tuple[list[str], list[str]]] = {}
    block_path_params: dict[str, tuple[str, ...]] = {}
    block_param_last_dims: dict[str, dict[str, Any]] = {}
    block_output_last_dims: dict[str, dict[str, Any]] = {}
    block_param_shapes: dict[str, dict[str, tuple[Any, ...]]] = {}
    block_output_shapes: dict[str, dict[str, tuple[Any, ...]]] = {}
    for module in modules:
        input_names = [param.name for param in module.params]
        path_params = module.path_params
        if not path_params and module.path_param is not None:
            path_params = (module.path_param,)
        input_names.extend(path_params)
        output_names = list(_module_return_names(module))
        signatures[module.name] = (input_names, output_names)
        block_path_params[module.name] = path_params
        block_param_last_dims[module.name] = _module_param_last_dims(module)
        block_output_last_dims[module.name] = _module_return_last_dims(module, tuple(output_names))
        block_param_shapes[module.name] = _module_param_shapes(module)
        block_output_shapes[module.name] = _module_return_shapes(module, tuple(output_names))
    primitive_aliases = _extract_primitive_aliases(modules)
    prelude_aliases = _extract_prelude_aliases(modules)
    implicit_prelude_members = {
        name.split(".", 1)[1]
        for name in signatures
        if isinstance(name, str) and name.startswith("Prelude.") and "." in name
    }

    main = by_name[main_name]
    main_returns = _module_return_names(main)
    main_inputs = _module_inputs(main)
    main_graph: list[dict[str, Any]] = []
    main_outputs: dict[str, str] = {}
    _lower_statements(
        statements=main.statements,
        graph=main_graph,
        outputs=main_outputs,
        returns=main_returns,
        ctx=_new_lower_ctx(
            module=main,
            returns=main_returns,
            signatures=signatures,
            block_path_params=block_path_params,
            block_param_last_dims=block_param_last_dims,
            block_output_last_dims=block_output_last_dims,
            block_param_shapes=block_param_shapes,
            block_output_shapes=block_output_shapes,
            implicit_prelude_members=implicit_prelude_members,
            prelude_aliases=prelude_aliases,
            primitive_aliases=primitive_aliases,
            imported_symbol_values=_collect_imported_symbol_values(main, by_name),
        ),
    )
    _ensure_outputs_from_returns(main_outputs, main_returns)
    model: dict[str, Any] = {"inputs": main_inputs, "graph": main_graph, "outputs": main_outputs}
    if main.symbols:
        model["symbols"] = dict(main.symbols)
    if main.pragmas:
        model["meta"] = dict(main.pragmas)
    spec: dict[str, Any] = {"synapse": 1, "model": model}

    blocks: dict[str, Any] = {}
    for module in modules:
        if module.name == main_name:
            continue
        if module.name in primitive_aliases:
            continue
        block_inputs = _module_inputs(module)
        block_returns = _module_return_names(module)
        block_graph: list[dict[str, Any]] = []
        block_outputs: dict[str, str] = {}
        _lower_statements(
            statements=module.statements,
            graph=block_graph,
            outputs=block_outputs,
            returns=block_returns,
            ctx=_new_lower_ctx(
                module=module,
                returns=block_returns,
                signatures=signatures,
                block_path_params=block_path_params,
                block_param_last_dims=block_param_last_dims,
                block_output_last_dims=block_output_last_dims,
                block_param_shapes=block_param_shapes,
                block_output_shapes=block_output_shapes,
                implicit_prelude_members=implicit_prelude_members,
                prelude_aliases=prelude_aliases,
                primitive_aliases=primitive_aliases,
                imported_symbol_values=_collect_imported_symbol_values(module, by_name),
            ),
        )
        _ensure_outputs_from_returns(block_outputs, block_returns)
        blocks[module.name] = {
            "inputs": block_inputs,
            "graph": block_graph,
            "outputs": block_outputs,
        }
    if blocks:
        spec["model"]["blocks"] = blocks

    _resolve_paths_at_lowering_time(spec["model"], block_path_params)
    inferred_block_io_types = infer_block_io_types_from_modules(
        spec=spec,
        modules=modules,
        selected_main=main_name,
    )
    final_block_io_types = _finalize_block_io_types_for_model(
        model=spec["model"],
        inferred=inferred_block_io_types,
    )
    annotate_spec_with_block_io_types(spec, block_io_types=final_block_io_types)

    return spec


__all__ = [
    "lower_axon_module_to_synapse_block",
    "lower_axon_module_to_synapse_spec",
    "lower_axon_program_to_synapse_spec",
]
