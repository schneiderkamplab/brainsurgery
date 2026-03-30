from __future__ import annotations

import ast
import copy
import re
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
from .call_parser import (
    looks_like_call as _looks_like_call,
)
from .call_parser import (
    parse_call as _parse_call,
)
from .call_parser import (
    parse_scalar as _parse_scalar,
)
from .call_parser import (
    render_call as _render_call,
)
from .call_parser import (
    split_top_level as _split_top_level,
)
from .call_parser import (
    strip_wrapping_parens as _strip_wrapping_parens,
)
from .expressions import (
    is_name_token as _is_name_token,
)
from .expressions import (
    split_binary as _split_binary,
)
from .expressions import (
    split_if_then_else as _split_if_then_else,
)
from .expressions import (
    split_ternary as _split_ternary,
)
from .expressions import (
    substitute_var as _substitute_var,
)
from .expressions import (
    tuple_items as _tuple_items,
)
from .types import (
    AxonBind,
    AxonModule,
    AxonRepeat,
    AxonReturn,
    AxonScope,
    AxonScopeBind,
    AxonStatement,
)

_LAMBDA_RE = re.compile(r"^\\([A-Za-z_][A-Za-z0-9_]*)\s*->\s*(.+)$")

_IMPLICIT_ACTIVATION_ALIASES: dict[str, tuple[str, int]] = {
    "gelu": ("_activations_gelu", 0),
    "gelu_new": ("_activations_gelu_new", 0),
    "gelu_pytorch_tanh": ("_activations_gelu_pytorch_tanh", 0),
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
        if re.fullmatch(r"-?[0-9]+", token):
            return int(token)
        return token
    return value


def _dims_compatible(left: Any, right: Any) -> bool:
    return _normalize_dim_token(left) == _normalize_dim_token(right)


def _is_symbolic_dim_token(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    token = value.strip()
    return re.fullmatch(r"-?[0-9]+", token) is None


def _is_kind(value: Any, kind: str) -> bool:
    if kind == "bool":
        return isinstance(value, bool)
    if kind == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        if isinstance(value, bool):
            return False
        return isinstance(value, (int, float, str))
    if kind == "str":
        return isinstance(value, str)
    if kind == "dim":
        if isinstance(value, bool):
            return False
        return isinstance(value, (int, str))
    if kind == "list_int":
        return isinstance(value, list) and all(
            isinstance(v, int) and not isinstance(v, bool) for v in value
        )
    if kind == "list_dim":
        return isinstance(value, list) and all(
            (isinstance(v, int) and not isinstance(v, bool)) or isinstance(v, str) for v in value
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


def _merge_when(outer: str | None, inner: Any) -> str | None:
    if inner is None:
        return outer
    if isinstance(inner, bool):
        inner_text = "true" if inner else "false"
    else:
        inner_text = str(inner).strip()
    if not inner_text or inner_text == "true":
        return outer
    if inner_text == "false":
        return "false"
    if outer is None or not outer.strip() or outer.strip() == "true":
        return inner_text
    if outer.strip() == "false":
        return "false"
    return f"({outer}) and ({inner_text})"


@dataclass
class _LowerCtx:
    counter: int = 0
    block_signatures: dict[str, tuple[list[str], list[str]]] | None = None
    block_path_params: dict[str, tuple[str, ...]] | None = None
    block_param_last_dims: dict[str, dict[str, Any]] | None = None
    block_output_last_dims: dict[str, dict[str, Any]] | None = None
    block_param_shapes: dict[str, dict[str, tuple[str, ...]]] | None = None
    block_output_shapes: dict[str, dict[str, tuple[str, ...]]] | None = None
    tensor_last_dim: dict[str, Any] = field(default_factory=dict)
    tensor_heads: dict[str, Any] = field(default_factory=dict)
    tensor_shape: dict[str, tuple[Any, ...]] = field(default_factory=dict)
    scope_stack: list[str] = field(default_factory=list)
    path_param_names: set[str] = field(default_factory=set)
    imported_namespaces: set[str] = field(default_factory=set)
    imported_member_namespaces: dict[str, set[str]] = field(default_factory=dict)
    prelude_aliases: dict[str, tuple[str, int]] = field(default_factory=dict)
    primitive_aliases: dict[str, tuple[str, int]] = field(default_factory=dict)
    current_module: str | None = None
    param_names: set[str] = field(default_factory=set)

    def fresh(self, base: str = "t") -> str:
        self.counter += 1
        return f"{base}_{self.counter}"


def _with_when(nodes: list[dict[str, Any]], when: str | None) -> list[dict[str, Any]]:
    if when is None:
        return nodes
    out: list[dict[str, Any]] = []
    for item in nodes:
        name, node_spec = next(iter(item.items()))
        spec = dict(node_spec)
        spec["when"] = when
        out.append({name: spec})
    return out


def _with_scope(nodes: list[dict[str, Any]], scope: str | None) -> list[dict[str, Any]]:
    if not isinstance(scope, str) or not scope:
        return nodes

    def _annotate_node_spec(node_spec: dict[str, Any]) -> dict[str, Any]:
        spec = dict(node_spec)
        op = spec.get("_op")
        if isinstance(op, str) and "_scope" not in spec:
            spec["_scope"] = scope
        nested = spec.get("graph")
        if isinstance(nested, list):
            spec["graph"] = _annotate_items(nested)
        body = spec.get("_body")
        if isinstance(body, list):
            spec["_body"] = _annotate_items(body)
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
            parsed = _parse_scalar(raw)
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


def _reject_obsolete_namespace_call(callee: str) -> None:
    if _op_name_from_callee(callee) == "reshape_heads_triplet":
        raise ValueError(
            "obsolete compatibility call 'reshape_heads_triplet'; "
            "call reshape_heads on q/k/v individually"
        )
    if callee.startswith("_act_"):
        raise ValueError(f"obsolete activation primitive {callee!r}; use _activations_<kind>")
    if callee.startswith("activations_"):
        raise ValueError(f"obsolete activation call {callee!r}; use _activations_<kind>")
    if "::" not in callee:
        return
    if callee.startswith("act::"):
        raise ValueError(
            f"obsolete call syntax {callee!r}; use _activations_<kind> primitive calls"
        )
    if callee.startswith("cache::"):
        raise ValueError(f"obsolete call syntax {callee!r}; use _cache_update/_cache_seq_len")
    raise ValueError(f"obsolete namespaced call syntax {callee!r}; '::' is not supported in calls")


def _lower_simple_call(
    expr: str, out: str | list[str], ctx: _LowerCtx, *, when: str | None = None
) -> list[dict[str, Any]]:
    callee, args, kwargs = _parse_call(expr)
    callee = _rewrite_prelude_alias_callee(callee, kwargs, ctx)
    callee = _rewrite_primitive_alias_callee(callee, kwargs, ctx)
    _reject_obsolete_namespace_call(callee)
    pre_graph: list[dict[str, Any]] = []
    effective_when = _merge_when(when, kwargs.pop("when", None))
    effective_scope = ".".join(part for part in ctx.scope_stack if part)

    resolved_args: list[str] = []
    raw_op_name = _op_name_from_callee(callee)

    def _should_materialize_inline_value(token_text: str) -> bool:
        stripped = token_text.strip()
        if _is_name_token(stripped):
            return False
        if not stripped:
            return False
        if (stripped.startswith("[") and stripped.endswith("]")) or (
            stripped.startswith("{") and stripped.endswith("}")
        ):
            # Keep structured literals (e.g., split sizes) inline for op normalizers.
            return False
        if re.fullmatch(r"-?[0-9]+(\.[0-9]+([eE][+-]?[0-9]+)?)?", stripped):
            return False
        if stripped.lower() in {"true", "false", "null"}:
            return False
        try:
            parsed = ast.parse(stripped, mode="eval")
        except SyntaxError:
            # Fallback: materialize complex non-name tokens.
            return True
        node = parsed.body
        if isinstance(node, (ast.Name, ast.Constant, ast.List, ast.Tuple, ast.Dict, ast.Set)):
            return False
        return True

    for idx, arg in enumerate(args):
        token = arg.strip()
        inner = token
        if token.startswith("(") and token.endswith(")"):
            inner = token[1:-1].strip()
        if _should_materialize_inline_value(inner):
            if raw_op_name in {"repeat", "_repeat"} and idx >= 1:
                resolved_args.append(inner)
                continue
            tmp = ctx.fresh("arg")
            pre_graph.extend(_lower_expr(inner, tmp, ctx, when=when))
            resolved_args.append(tmp)
        else:
            resolved_args.append(token)
    args = resolved_args

    resolved_kwargs: dict[str, Any] = {}
    for key, value in kwargs.items():
        if isinstance(value, str):
            stripped_value = _strip_wrapping_parens(value)
            if _looks_like_call(stripped_value):
                key_token = re.sub(r"[^A-Za-z0-9_]+", "_", key).strip("_") or "kwarg"
                tmp = ctx.fresh(f"kwarg_{key_token}")
                pre_graph.extend(_lower_expr(stripped_value, tmp, ctx, when=when))
                resolved_kwargs[key] = tmp
                continue
        resolved_kwargs[key] = value
    kwargs = resolved_kwargs

    is_absolute_path = "@@" in callee
    if is_absolute_path:
        callee = callee.replace("@@", "@", 1)
    if "@" in callee and ctx.scope_stack and not is_absolute_path:
        scope_prefix = ".".join(part for part in ctx.scope_stack if part)
        if scope_prefix:
            parts = callee.split("@")
            base = parts[0]
            if ctx.block_signatures and base in ctx.block_signatures and len(parts) > 1:
                scoped_paths = [
                    f"{scope_prefix}.{param_path}" if param_path.strip() else scope_prefix
                    for param_path in parts[1:]
                ]
                callee = "@".join([base, *scoped_paths])
            else:
                op_name_with_at, param_path = callee.split("@", 1)
                scoped_path = f"{scope_prefix}.{param_path}" if param_path.strip() else scope_prefix
                callee = f"{op_name_with_at}@{scoped_path}"
    op_name = _canonical_op_name(callee)
    kwargs = _normalize_call_kwargs(op_name=op_name, args=args, out=out, kwargs=kwargs, ctx=ctx)
    _validate_normalized_kwargs(op_name, kwargs, args)
    op_validate = get_op_lowering_validator(op_name)
    if callable(op_validate):
        op_validate(args=args, out=out, kwargs=kwargs, ctx=ctx)
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
        for idx, value in enumerate(args):
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
            provided[key] = repr(concrete_path)
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
                symbol_bindings[param_name] = _parse_scalar(token)
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
            elif len(positional_args) < len(args):
                positional_args.append(value)
            else:
                extra_kwargs[input_name] = value
        node_name = f"n_{ctx.fresh('call')}"
        node_spec: dict[str, Any] = {"_op": "call", "_target": block_name}
        if positional_args:
            node_spec["_args"] = (
                positional_args[0] if len(positional_args) == 1 else positional_args
            )
        call_scope = ".".join(part for part in ctx.scope_stack if part)
        if call_scope and not path_bindings:
            node_spec["_scope"] = call_scope
        node_spec["_bind"] = out_values[0] if len(out_values) == 1 else out_values
        for key, value in extra_kwargs.items():
            node_spec[key] = value
        nodes = _with_when([{node_name: node_spec}], effective_when)
        _record_last_dim_for_call(callee=block_name, args=args, kwargs=kwargs, out=out, ctx=ctx)
        return [*pre_graph, *_with_scope(nodes, effective_scope)]

    node_spec = _to_synapse_op(callee, args, kwargs, out)
    if "@" in callee:
        op_name, param_path = callee.split("@", 1)
        if param_path in ctx.path_param_names:
            node_name = f"n_{ctx.fresh('op')}"
            templated_node = _to_synapse_op(op_name, args, kwargs, out)
            templated_node["param_base"] = param_path
            nodes = _with_when([{node_name: templated_node}], effective_when)
            _record_last_dim_for_call(callee=op_name, args=args, kwargs=kwargs, out=out, ctx=ctx)
            return [*pre_graph, *_with_scope(nodes, effective_scope)]
        concrete_node = _to_synapse_op(op_name, args, kwargs, out)
        try:
            bound_params = _path_bound_param_names(concrete_node)
        except ValueError:
            bound_params = []
        if bound_params:
            if not param_path.strip():
                raise ValueError(f"invalid @ path in Axon call: {expr!r}")
            node_name = f"n_{ctx.fresh('op')}"
            params = {param_name: f"{param_path}.{param_name}" for param_name in bound_params}
            concrete_node["_params"] = params
            nodes = _with_when([{node_name: concrete_node}], effective_when)
            _record_last_dim_for_call(callee=callee, args=args, kwargs=kwargs, out=out, ctx=ctx)
            return [*pre_graph, *_with_scope(nodes, effective_scope)]
        segments = [part.strip() for part in param_path.split(".") if part.strip()]
        if not segments:
            raise ValueError(f"invalid @ path in Axon call: {expr!r}")
        item: dict[str, Any] = {segments[-1]: node_spec}
        for segment in reversed(segments[:-1]):
            item = {segment: {"graph": [item]}}
        nodes = _with_when([item], effective_when)
        _record_last_dim_for_call(callee=callee, args=args, kwargs=kwargs, out=out, ctx=ctx)
        return [*pre_graph, *_with_scope(nodes, effective_scope)]
    node_name = f"n_{ctx.fresh('op')}"
    nodes = _with_when([{node_name: node_spec}], effective_when)
    _record_last_dim_for_call(callee=callee, args=args, kwargs=kwargs, out=out, ctx=ctx)
    return [*pre_graph, *_with_scope(nodes, effective_scope)]


def _lower_alias_or_const(
    expr: str, out: str | list[str], ctx: _LowerCtx, *, when: str | None = None
) -> list[dict[str, Any]]:
    if isinstance(out, list):
        raise ValueError("alias/const lowering expects scalar out")
    token = expr.strip()
    node_name = f"n_{ctx.fresh('op')}"
    scalar = _parse_scalar(token)
    if (
        isinstance(scalar, str)
        and scalar == token
        and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token)
    ):
        node = {"_op": "_ir_alias", "_args": token, "_bind": out}
        if token in ctx.tensor_last_dim:
            ctx.tensor_last_dim[out] = ctx.tensor_last_dim[token]
    else:
        node = {"_op": "_ir_expr", "value": scalar, "_bind": out}
    effective_scope = ".".join(part for part in ctx.scope_stack if part)
    if effective_scope:
        node["_scope"] = effective_scope
    return _with_when([{node_name: node}], when)


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


def _known_output_arity(callee: str, ctx: _LowerCtx) -> int | None:
    resolved = _resolve_block_call(callee, ctx)
    if resolved is not None and ctx.block_signatures:
        block_name, _ = resolved
        _, output_names = ctx.block_signatures[block_name]
        return len(output_names)

    normalized = _canonical_op_name(callee)
    op_arity = get_op_lowering_known_output_arity(normalized)
    if callable(op_arity):
        resolved = op_arity(kwargs={})
        if isinstance(resolved, int):
            return resolved
    return None


def _pipeline_temp_out(stage: str, ctx: _LowerCtx) -> str | list[str]:
    if not _looks_like_call(stage):
        return ctx.fresh("pipe")
    callee, _, kwargs = _parse_call(stage)
    normalized = _canonical_op_name(callee)
    op_arity = get_op_lowering_known_output_arity(normalized)
    if callable(op_arity):
        arity = op_arity(kwargs=kwargs)
    else:
        arity = _known_output_arity(callee, ctx)
    if arity is None or arity <= 1:
        return ctx.fresh("pipe")
    return [ctx.fresh("pipe") for _ in range(arity)]


def _call_output_arity(expr: str, ctx: _LowerCtx) -> int | None:
    if not _looks_like_call(expr):
        return None
    callee, _, kwargs = _parse_call(expr)
    resolved = _resolve_block_call(callee, ctx)
    if resolved is not None and ctx.block_signatures:
        block_name, _ = resolved
        _, output_names = ctx.block_signatures[block_name]
        return len(output_names)

    normalized = _canonical_op_name(callee)
    op_arity = get_op_lowering_known_output_arity(normalized)
    if callable(op_arity):
        arity = op_arity(kwargs=kwargs)
        if isinstance(arity, int):
            return arity
    return None


def _expand_call_outputs_for_ternary(
    call_expr: str, out: str | list[str], ctx: _LowerCtx
) -> str | list[str]:
    if not isinstance(out, list):
        return out
    arity = _call_output_arity(call_expr, ctx)
    if not isinstance(arity, int) or arity <= len(out):
        return out
    return [*out, *[ctx.fresh("discard") for _ in range(arity - len(out))]]


def _lower_expr(
    expr: str,
    out: str | list[str],
    ctx: _LowerCtx,
    *,
    when: str | None = None,
) -> list[dict[str, Any]]:
    expr = expr.strip()

    bind_parts = _split_top_level(expr, ">>=")
    if len(bind_parts) > 1:
        bind_graph: list[dict[str, Any]] = []
        first_bind = bind_parts[0].strip()
        if _is_name_token(first_bind):
            bind_ref = first_bind
        else:
            bind_ref = ctx.fresh("bind")
            bind_graph.extend(_lower_expr(first_bind, bind_ref, ctx, when=when))
        for idx, part in enumerate(bind_parts[1:], start=1):
            match = _LAMBDA_RE.match(part.strip())
            if match is None:
                raise ValueError(f"expected lambda after >>=, got: {part!r}")
            var_name = match.group(1)
            body = _substitute_var(match.group(2), var_name, bind_ref)
            bind_next_out: str | list[str] = (
                out if idx == len(bind_parts) - 1 else ctx.fresh("bind")
            )
            bind_graph.extend(_lower_expr(body, bind_next_out, ctx, when=when))
            if isinstance(bind_next_out, str):
                bind_ref = bind_next_out
        return bind_graph

    conditional = _split_if_then_else(expr)
    if conditional is None:
        conditional = _split_ternary(expr)
    if conditional is not None:
        cond, true_expr, false_expr = conditional
        if isinstance(out, list):
            true_items = _tuple_items(true_expr)
            false_items = _tuple_items(false_expr)
            ternary_graph: list[dict[str, Any]] = []
            if len(true_items) == 1 and _looks_like_call(true_items[0]):
                ternary_out = _expand_call_outputs_for_ternary(true_items[0], out, ctx)
                ternary_graph.extend(_lower_expr(true_items[0], ternary_out, ctx, when=cond))
            elif len(true_items) == len(out):
                for name, item in zip(out, true_items, strict=True):
                    ternary_graph.extend(_lower_expr(item, name, ctx, when=cond))
            else:
                raise ValueError("ternary true-branch arity must match binding targets")

            if len(false_items) == 1 and _looks_like_call(false_items[0]):
                ternary_out = _expand_call_outputs_for_ternary(false_items[0], out, ctx)
                ternary_graph.extend(
                    _lower_expr(false_items[0], ternary_out, ctx, when=f"not ({cond})")
                )
            elif len(false_items) == len(out):
                for name, item in zip(out, false_items, strict=True):
                    ternary_graph.extend(_lower_expr(item, name, ctx, when=f"not ({cond})"))
            else:
                raise ValueError("ternary tuple arity must match binding targets")
            return ternary_graph

        cond_graph: list[dict[str, Any]] = []
        cond_graph.extend(_lower_expr(true_expr, out, ctx, when=cond))
        cond_graph.extend(_lower_expr(false_expr, out, ctx, when=f"not ({cond})"))
        return cond_graph

    if "|>" in expr:
        stages = _split_top_level(expr, "|>")
        if not stages:
            raise ValueError("empty pipeline")
        pipe_graph: list[dict[str, Any]] = []

        first = stages[0]
        if _looks_like_call(first):
            first_out: str | list[str] = out if len(stages) == 1 else _pipeline_temp_out(first, ctx)
            pipe_graph.extend(_lower_simple_call(first, first_out, ctx, when=when))
            pipe_ref = first_out
        else:
            pipe_ref = first.strip()

        for idx, stage in enumerate(stages[1:], start=1):
            stage = stage.strip()
            if _looks_like_call(stage):
                callee, args, kwargs = _parse_call(stage)
            else:
                callee = stage
                args, kwargs = [], {}
            next_out: str | list[str] = (
                out if idx == len(stages) - 1 else _pipeline_temp_out(stage, ctx)
            )
            piped_args = [pipe_ref] if isinstance(pipe_ref, str) else list(pipe_ref)
            stage_args = list(args)
            if stage_args:
                if isinstance(pipe_ref, str):
                    if stage_args[0].strip() == pipe_ref:
                        stage_args = stage_args[1:]
                else:
                    n = len(piped_args)
                    if len(stage_args) >= n and all(
                        stage_args[i].strip() == piped_args[i] for i in range(n)
                    ):
                        stage_args = stage_args[n:]
            call_expr = _render_call(callee, [*piped_args, *stage_args], kwargs)
            pipe_graph.extend(_lower_simple_call(call_expr, next_out, ctx, when=when))
            pipe_ref = next_out
        return pipe_graph

    if _looks_like_call(expr):
        return _lower_simple_call(expr, out, ctx, when=when)

    def _expr_is_tensorish(candidate: str) -> bool:
        token = candidate.strip()
        if _looks_like_call(token):
            return True
        if _is_name_token(token):
            return (
                token in ctx.param_names
                or token in ctx.tensor_shape
                or token in ctx.tensor_last_dim
                or token in ctx.tensor_heads
            )
        return False

    plus = _split_binary(expr, "+")
    if plus is not None:
        left_expr, right_expr = plus
        if not (_expr_is_tensorish(left_expr) or _expr_is_tensorish(right_expr)):
            return _lower_alias_or_const(expr, out, ctx, when=when)
        add_graph: list[dict[str, Any]] = []
        left_ref = left_expr.strip() if _is_name_token(left_expr) else ctx.fresh("bin")
        right_ref = right_expr.strip() if _is_name_token(right_expr) else ctx.fresh("bin")
        if not _is_name_token(left_expr):
            add_graph.extend(_lower_expr(left_expr, left_ref, ctx, when=when))
        if not _is_name_token(right_expr):
            add_graph.extend(_lower_expr(right_expr, right_ref, ctx, when=when))
        add_graph.extend(_lower_simple_call(f"add({left_ref}, {right_ref})", out, ctx, when=when))
        return add_graph

    mul = _split_binary(expr, "*")
    if mul is not None:
        left_expr, right_expr = mul
        if not (_expr_is_tensorish(left_expr) or _expr_is_tensorish(right_expr)):
            return _lower_alias_or_const(expr, out, ctx, when=when)
        left_ref = left_expr.strip() if _is_name_token(left_expr) else ctx.fresh("bin")
        right_ref = right_expr.strip() if _is_name_token(right_expr) else ctx.fresh("bin")
        mul_graph: list[dict[str, Any]] = []
        if not _is_name_token(left_expr):
            mul_graph.extend(_lower_expr(left_expr, left_ref, ctx, when=when))
        if not _is_name_token(right_expr):
            mul_graph.extend(_lower_expr(right_expr, right_ref, ctx, when=when))
        mul_graph.extend(_lower_simple_call(f"mul({left_ref}, {right_ref})", out, ctx, when=when))
        return mul_graph

    return _lower_alias_or_const(expr, out, ctx, when=when)


def _module_return_names(module: AxonModule) -> tuple[str, ...]:
    if module.returns:
        return module.returns
    for stmt in reversed(module.statements):
        if isinstance(stmt, AxonReturn):
            inferred: list[str] = []
            for idx, value in enumerate(stmt.values):
                inferred.append(value if _is_name_token(value) else f"out_{idx}")
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


def _module_param_shapes(module: AxonModule) -> dict[str, tuple[str, ...]]:
    out: dict[str, tuple[str, ...]] = {}
    for param in module.params:
        if param.shape is None:
            continue
        out[param.name] = tuple(param.shape)
    return out


def _module_return_shapes(
    module: AxonModule, returns: tuple[str, ...]
) -> dict[str, tuple[str, ...]]:
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
    direct_aliases: dict[str, tuple[str, int]] = {}
    for module in modules:
        if not isinstance(module.name, str) or "." not in module.name:
            continue
        if len(module.statements) != 1:
            continue
        stmt = module.statements[0]
        if not isinstance(stmt, AxonReturn) or len(stmt.values) != 1:
            continue
        expr = stmt.values[0].strip()
        try:
            callee, _, _ = _parse_call(expr)
        except ValueError:
            continue
        target_base = callee.split("@", 1)[0]
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


def _new_lower_ctx(
    *,
    module: AxonModule,
    returns: tuple[str, ...],
    signatures: dict[str, tuple[list[str], list[str]]] | None,
    block_path_params: dict[str, tuple[str, ...]] | None,
    block_param_last_dims: dict[str, dict[str, Any]] | None,
    block_output_last_dims: dict[str, dict[str, Any]] | None,
    block_param_shapes: dict[str, dict[str, tuple[str, ...]]] | None = None,
    block_output_shapes: dict[str, dict[str, tuple[str, ...]]] | None = None,
    implicit_prelude_members: set[str] | None = None,
    prelude_aliases: dict[str, tuple[str, int]] | None = None,
    primitive_aliases: dict[str, tuple[str, int]] | None = None,
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
    )


def lower_axon_module_to_synapse_block(module: AxonModule) -> dict[str, Any]:
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
) -> None:
    for stmt in statements:
        if isinstance(stmt, AxonRepeat):
            body_graph: list[dict[str, Any]] = []
            _lower_statements(
                statements=stmt.body,
                graph=body_graph,
                outputs={},
                returns=(),
                ctx=ctx,
            )
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
                    "_to": stmt.to_expr,
                    "_body": body_graph,
                }
            }
            if stmt.from_expr != "0":
                repeat_item[node_name]["_from"] = stmt.from_expr
            if stmt.step_expr != "1":
                repeat_item[node_name]["_step"] = stmt.step_expr
            graph.append(repeat_item)
            continue

        if isinstance(stmt, AxonScope):
            ctx.scope_stack.append(stmt.prefix)
            try:
                _lower_statements(
                    statements=stmt.body,
                    graph=graph,
                    outputs=outputs,
                    returns=returns,
                    ctx=ctx,
                )
            finally:
                ctx.scope_stack.pop()
            continue

        if isinstance(stmt, AxonScopeBind):
            ctx.scope_stack.append(stmt.prefix)
            scoped_outputs: dict[str, str] = {}
            try:
                _lower_statements(
                    statements=stmt.body,
                    graph=graph,
                    outputs=scoped_outputs,
                    returns=(),
                    ctx=ctx,
                )
            finally:
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
                graph.extend(_lower_expr(source_name, target, ctx))
            continue

        if isinstance(stmt, AxonBind):
            out: str | list[str] = stmt.targets[0] if len(stmt.targets) == 1 else list(stmt.targets)
            graph.extend(_lower_expr(stmt.expr, out, ctx))
            continue

        if isinstance(stmt, AxonReturn):
            for idx, value in enumerate(stmt.values):
                output_name = returns[idx] if idx < len(returns) else f"out_{idx}"
                if _is_name_token(value):
                    outputs[output_name] = value
                    continue
                graph.extend(_lower_expr(value, output_name, ctx))
                outputs[output_name] = output_name
            continue


def _as_concrete_path(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    token = value.strip()
    if not token:
        return None
    try:
        parsed = ast.literal_eval(token)
    except Exception:
        parsed = token
    if isinstance(parsed, str) and parsed.strip():
        return parsed.strip()
    return None


def _sanitize_path_suffix(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")
    return sanitized or "path"


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
        return ["weight", "bias"]
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

        param_base = inlined.get("param_base")
        if isinstance(param_base, str):
            explicit_params = inlined.get("_params")
            params = dict(explicit_params) if isinstance(explicit_params, dict) else {}
            for param_name in _path_bound_param_names(inlined):
                params.setdefault(param_name, f"{param_base}.{param_name}")
            inlined["_params"] = params
            inlined.pop("param_base", None)

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
                explicit_params = node_spec.get("_params")
                params: dict[str, str]
                if isinstance(explicit_params, dict):
                    params = dict(explicit_params)
                else:
                    params = {}
                for param_name in _path_bound_param_names(node_spec):
                    params.setdefault(param_name, f"{base_path}.{param_name}")
                node_spec["_params"] = params
                node_spec.pop("param_base", None)

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

    for block_name in sorted(reachable):
        block_spec = all_blocks[block_name]
        for item in block_spec.get("graph", []):
            if not isinstance(item, dict) or len(item) != 1:
                continue
            _, node_spec = next(iter(item.items()))
            if isinstance(node_spec, dict) and "param_base" in node_spec:
                raise ValueError(f"unresolved param_base in reachable block {block_name!r}")

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
    block_param_shapes: dict[str, dict[str, tuple[str, ...]]] = {}
    block_output_shapes: dict[str, dict[str, tuple[str, ...]]] = {}
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
