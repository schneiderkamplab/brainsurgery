from __future__ import annotations

import inspect
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .axon import lower_axon_program_to_synapse_spec, parse_axon_program_from_path
from .ops import get_op_lowering_signature, get_op_module
from .type_inference import (
    extract_block_io_types_from_spec,
    infer_block_io_types_from_modules,
    infer_input_slot_types,
    infer_output_types_for_node,
)

_WORD_TOKEN_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_PARAM_INFER_RE = re.compile(r"_infer_param_(?:expr|path)\([^)]*?[\"']([A-Za-z0-9_]+)[\"']\)")
_INFER_PARAM_CALL_RE = re.compile(r"infer_param\([\"']([A-Za-z0-9_]+)[\"']\)")
_PORT_TOKEN_RE = re.compile(r"[^0-9A-Za-z_]")
_EXPECTS_ARGS_LIST_RE = re.compile(r"expects(?:\s+_args\s+as|\s+args)\s+\[([^\]]+)\]")
_EXPECTS_POSITIONAL_RE = re.compile(r"expects exactly \d+ positional args:\s*([^\n\"']+)")
type _Endpoint = tuple[str, str | None]

_OP_POSITIONAL_ARG_OVERRIDES: dict[str, tuple[str, ...]] = {
    "_ir_alias": ("x",),
    "_ir_expr": (),
    "activation": ("x",),
    "add": ("x", "y"),
    "attention": ("q", "k", "v"),
    "cache_seq_len": ("entry",),
    "cache_update": ("past", "k", "v"),
    "causal_conv1d": ("x", "state"),
    "causal_mask": ("query", "key"),
    "clamp": ("x",),
    "concat": ("x", "y"),
    "embedding": ("x",),
    "layernorm": ("x",),
    "linear": ("x",),
    "linear_position_bias": ("attention_mask",),
    "list_append": ("xs", "x"),
    "list_index": ("collection", "index"),
    "list_init": (),
    "mamba_scan": ("u", "delta", "b", "c"),
    "merge_heads": ("x",),
    "moe_grouped_ffn": ("x", "scores", "indices"),
    "moe_scatter_add": ("acc", "idx", "upd", "w"),
    "moe_select": ("x", "scores", "indices"),
    "mul": ("x", "y"),
    "position_ids": ("input_ids", "attn_mask"),
    "repeat": ("x",),
    "reshape_heads": ("x",),
    "rmsnorm": ("x",),
    "rope_pair": ("q", "k"),
    "softmax": ("x",),
    "split": ("x",),
    "split_qkv_heads": ("x",),
    "topk": ("x",),
    "zeros_like": ("x",),
}


def _dot_escape(text: str) -> str:
    # Keep Graphviz escapes like "\n" intact so labels can use multiline rendering.
    return text.replace('"', '\\"').replace("\n", "\\n")


def _html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def _dot_id(*parts: str) -> str:
    merged = "::".join(parts)
    sanitized = re.sub(r"[^0-9A-Za-z_:.]", "_", merged)
    return f"n_{sanitized}"


def _format_block_label(text: str) -> str:
    if "::" not in text:
        return text
    head, tail = text.split("::", 1)
    parts = [part.strip() for part in tail.split("->")]
    if len(parts) <= 1:
        return text
    max_width = 56
    lines: list[str] = []
    current = ""
    for part in parts:
        token = part.strip()
        if not token:
            continue
        if not current:
            current = token
            continue
        candidate = f"{current} -> {token}"
        if len(candidate) <= max_width:
            current = candidate
            continue
        lines.append(current)
        current = token
    if current:
        lines.append(current)
    wrapped_tail = " ->\n".join(lines) if lines else " -> ".join(parts)
    return f"{head.strip()} ::\n{wrapped_tail}"


def _slot_label_html(
    name: str,
    type_hint: str | None,
    expr_hint: str | None = None,
) -> str:
    base = f'<FONT POINT-SIZE="7">{_html_escape(name)}</FONT>'
    extras: list[str] = []
    if isinstance(type_hint, str) and type_hint:
        extras.append(f'<FONT POINT-SIZE="6" COLOR="gray50">{_html_escape(type_hint)}</FONT>')
    if isinstance(expr_hint, str) and expr_hint:
        extras.append(f'<FONT POINT-SIZE="6" COLOR="gray35">{_html_escape(expr_hint)}</FONT>')
    if not extras:
        return base
    return f'{base}<BR ALIGN="CENTER"/>' + '<BR ALIGN="CENTER"/>'.join(extras)


def _arg_slot_bg(kind: str) -> str:
    if kind == "pos":
        return "lemonchiffon"
    if kind == "mixed":
        return "thistle1"
    return "azure"


def _cluster_id(block_name: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z_]", "_", block_name)
    return f"cluster_{sanitized}"


def _loop_cluster_id(*parts: str) -> str:
    merged = "__".join(parts)
    sanitized = re.sub(r"[^0-9A-Za-z_]", "_", merged)
    return f"cluster_loop_{sanitized}"


def _scope_cluster_id(*parts: str) -> str:
    merged = "__".join(parts)
    sanitized = re.sub(r"[^0-9A-Za-z_]", "_", merged)
    return f"cluster_scope_{sanitized}"


def _port_id(name: str) -> str:
    sanitized = _PORT_TOKEN_RE.sub("_", name)
    if not sanitized:
        sanitized = "p"
    if sanitized[0].isdigit():
        sanitized = f"p_{sanitized}"
    return f"p_{sanitized}"


def _dot_endpoint(endpoint: _Endpoint) -> str:
    node_id, port_id = endpoint
    if isinstance(port_id, str) and port_id:
        return f'"{node_id}":"{port_id}"'
    return f'"{node_id}"'


def _dot_endpoint_with_compass(endpoint: _Endpoint, compass: str) -> str:
    base = _dot_endpoint(endpoint)
    return f"{base}:{compass}"


def _edge_sort_key(item: tuple[tuple[_Endpoint, _Endpoint], set[str]]) -> tuple[str, str, str, str]:
    (src, dst), _labels = item
    src_node, src_port = src
    dst_node, dst_port = dst
    return (
        src_node,
        src_port or "",
        dst_node,
        dst_port or "",
    )


def _append_edge(
    edge_labels: dict[tuple[_Endpoint, _Endpoint], set[str]],
    *,
    src: _Endpoint,
    dst: _Endpoint,
    label: str,
) -> None:
    key = (src, dst)
    bucket = edge_labels.setdefault(key, set())
    bucket.add(label)


_OP_PARAM_FIELDS_CACHE: dict[str, tuple[str, ...]] = {}


def _op_param_fields(op_name: str) -> tuple[str, ...]:
    cached = _OP_PARAM_FIELDS_CACHE.get(op_name)
    if cached is not None:
        return cached
    op_module = get_op_module(op_name)
    if op_module is None:
        _OP_PARAM_FIELDS_CACHE[op_name] = ()
        return ()
    fields: set[str] = set()
    for fn_name in ("run", "compile"):
        fn = getattr(op_module, fn_name, None)
        if fn is None:
            continue
        try:
            source = inspect.getsource(fn)
        except (OSError, TypeError):
            continue
        fields.update(_PARAM_INFER_RE.findall(source))
        fields.update(_INFER_PARAM_CALL_RE.findall(source))
    result = tuple(sorted(fields))
    _OP_PARAM_FIELDS_CACHE[op_name] = result
    return result


def _node_param_paths(
    node_spec: Mapping[str, Any],
    *,
    op_name: str,
    node_runtime_path: str,
    scope_prefix: str | None = None,
) -> list[str]:
    paths: set[str] = set()
    explicit_params = node_spec.get("_params")
    explicit_param_map: dict[str, str] = {}
    if isinstance(explicit_params, Mapping):
        for key, value in explicit_params.items():
            if not isinstance(value, str) or not value:
                continue
            paths.add(value)
            if isinstance(key, str) and key:
                explicit_param_map[key] = value

    fields = _op_param_fields(op_name)
    param_base = node_spec.get("param_base")
    param_base_name = param_base if isinstance(param_base, str) and param_base else None
    for field in fields:
        explicit_field_value = node_spec.get(field)
        if isinstance(explicit_field_value, str) and explicit_field_value:
            paths.add(explicit_field_value)
            continue
        mapped_field_value = explicit_param_map.get(field)
        if isinstance(mapped_field_value, str) and mapped_field_value:
            paths.add(mapped_field_value)
            continue
        if param_base_name is not None:
            paths.add(f"{param_base_name}.{field}")
            continue
        paths.add(f"{node_runtime_path}.{field}")
    for key, value in node_spec.items():
        if not isinstance(key, str) or not key.endswith("_param"):
            continue
        if not isinstance(value, str) or not value:
            continue
        if "." in value:
            paths.add(value)
            continue
        if param_base_name is not None:
            paths.add(f"{param_base_name}.{value}")
            continue
        paths.add(f"{node_runtime_path}.{value}")
    prefixed: set[str] = set()
    for path in paths:
        if not isinstance(scope_prefix, str) or not scope_prefix:
            prefixed.add(path)
            continue
        if path.startswith(scope_prefix + "."):
            prefixed.add(path)
            continue
        prefixed.add(f"{scope_prefix}.{path}")
    return sorted(prefixed)


def _collect_var_refs(value: Any, *, known_vars: set[str], out: set[str]) -> None:
    if isinstance(value, str):
        if value in known_vars:
            out.add(value)
            return
        for token in _WORD_TOKEN_RE.findall(value):
            if token in known_vars:
                out.add(token)
        return
    if isinstance(value, Mapping):
        for child in value.values():
            _collect_var_refs(child, known_vars=known_vars, out=out)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            _collect_var_refs(child, known_vars=known_vars, out=out)


def _looks_identifier(value: str) -> bool:
    return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", value))


def _parse_name_list(text: str) -> list[str]:
    raw = text.replace(" and ", ",")
    names = [part.strip().strip("'\"") for part in raw.split(",")]
    return [name for name in names if _looks_identifier(name)]


_OP_POSITIONAL_NAMES_CACHE: dict[str, tuple[str, ...]] = {}


def _positional_slot_name(
    *,
    idx: int,
    value: Any,
    positional_names: tuple[str, ...],
) -> str:
    if idx < len(positional_names):
        return positional_names[idx]
    if isinstance(value, str) and _looks_identifier(value):
        return value
    return f"arg_{idx}"


def _op_positional_arg_names(op_name: str) -> tuple[str, ...]:
    overridden = _OP_POSITIONAL_ARG_OVERRIDES.get(op_name)
    if overridden is not None:
        return overridden
    cached = _OP_POSITIONAL_NAMES_CACHE.get(op_name)
    if cached is not None:
        return cached
    module = get_op_module(op_name)
    if module is None:
        _OP_POSITIONAL_NAMES_CACHE[op_name] = ()
        return ()
    candidates: list[list[str]] = []
    for fn_name in ("interpret", "compile"):
        fn = getattr(module, fn_name, None)
        if fn is None:
            continue
        try:
            source = inspect.getsource(fn)
        except (OSError, TypeError):
            continue
        for match in _EXPECTS_ARGS_LIST_RE.finditer(source):
            parsed = _parse_name_list(match.group(1))
            if parsed:
                candidates.append(parsed)
        for match in _EXPECTS_POSITIONAL_RE.finditer(source):
            parsed = _parse_name_list(match.group(1))
            if parsed:
                candidates.append(parsed)
    if not candidates:
        signature = get_op_lowering_signature(op_name)
        arity = signature.get("arity") if isinstance(signature, dict) else None
        is_single_arity = (
            isinstance(arity, tuple) and len(arity) == 2 and arity[0] == 1 and arity[1] == 1
        )
        if is_single_arity:
            hinted: set[str] = set()
            for fn_name in ("interpret", "compile"):
                fn = getattr(module, fn_name, None)
                if fn is None:
                    continue
                try:
                    source = inspect.getsource(fn)
                except (OSError, TypeError):
                    continue
                for field in re.findall(
                    rf"{re.escape(op_name)}\.([A-Za-z_][A-Za-z0-9_]*)",
                    source,
                ):
                    if field in {"_args", "_bind", "bind", "args", "out"}:
                        continue
                    hinted.add(field)
            if "in" in hinted:
                _OP_POSITIONAL_NAMES_CACHE[op_name] = ("in",)
                return ("in",)
            if hinted:
                chosen = sorted(hinted)[0]
                _OP_POSITIONAL_NAMES_CACHE[op_name] = (chosen,)
                return (chosen,)
        _OP_POSITIONAL_NAMES_CACHE[op_name] = ()
        return ()
    best = max(candidates, key=len)
    result = tuple(best)
    _OP_POSITIONAL_NAMES_CACHE[op_name] = result
    return result


def _node_input_slots(
    node_spec: Mapping[str, Any],
    *,
    op_name: str,
    known_vars: set[str],
) -> list[tuple[str, set[str]]]:
    slots: dict[str, set[str]] = {}

    raw_args = node_spec.get("_args")
    positional_values: list[Any]
    if raw_args is None:
        positional_values = []
    elif isinstance(raw_args, list):
        positional_values = list(raw_args)
    else:
        positional_values = [raw_args]
    positional_names = _op_positional_arg_names(op_name)
    for idx, value in enumerate(positional_values):
        slot_name = _positional_slot_name(idx=idx, value=value, positional_names=positional_names)
        refs: set[str] = set()
        _collect_var_refs(value, known_vars=known_vars, out=refs)
        slots.setdefault(slot_name, set()).update(refs)

    for key, value in node_spec.items():
        if not isinstance(key, str):
            continue
        if key.startswith("_") or key in {"graph", "when"}:
            continue
        kw_refs: set[str] = set()
        _collect_var_refs(value, known_vars=known_vars, out=kw_refs)
        slots.setdefault(key, set()).update(kw_refs)

    ordered: list[tuple[str, set[str]]] = []
    for idx, value in enumerate(positional_values):
        slot_name = _positional_slot_name(idx=idx, value=value, positional_names=positional_names)
        if slot_name in slots and all(name != slot_name for name, _ in ordered):
            ordered.append((slot_name, slots[slot_name]))
    for key in slots:
        if all(name != key for name, _ in ordered):
            ordered.append((key, slots[key]))
    return ordered


def _node_input_slot_kinds(node_spec: Mapping[str, Any], *, op_name: str) -> dict[str, str]:
    kinds: dict[str, str] = {}
    raw_args = node_spec.get("_args")
    positional_values: list[Any]
    if raw_args is None:
        positional_values = []
    elif isinstance(raw_args, list):
        positional_values = list(raw_args)
    else:
        positional_values = [raw_args]
    positional_names = _op_positional_arg_names(op_name)
    for idx, value in enumerate(positional_values):
        slot_name = _positional_slot_name(idx=idx, value=value, positional_names=positional_names)
        kinds[slot_name] = "pos"
    for key in node_spec:
        if not isinstance(key, str):
            continue
        if key.startswith("_") or key in {"graph", "when"}:
            continue
        previous = kinds.get(key)
        if previous == "pos":
            kinds[key] = "mixed"
        elif previous is None:
            kinds[key] = "kw"
    return kinds


def _value_expr_hint(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if _looks_identifier(text):
            return None
        return text
    if isinstance(value, (int, float, bool)) or value is None:
        return repr(value)
    if isinstance(value, list):
        return repr(value)
    return None


def _node_input_slot_expr_hints(
    node_spec: Mapping[str, Any],
    *,
    op_name: str,
) -> dict[str, str]:
    hints: dict[str, str] = {}
    raw_args = node_spec.get("_args")
    positional_values: list[Any]
    if raw_args is None:
        positional_values = []
    elif isinstance(raw_args, list):
        positional_values = list(raw_args)
    else:
        positional_values = [raw_args]
    positional_names = _op_positional_arg_names(op_name)
    for idx, value in enumerate(positional_values):
        slot_name = _positional_slot_name(idx=idx, value=value, positional_names=positional_names)
        hint = _value_expr_hint(value)
        if isinstance(hint, str) and hint:
            hints[slot_name] = hint
    for key, value in node_spec.items():
        if not isinstance(key, str):
            continue
        if key.startswith("_") or key in {"graph", "when"}:
            continue
        hint = _value_expr_hint(value)
        if isinstance(hint, str) and hint:
            hints[key] = hint
    return hints


def _node_input_vars(node_spec: Mapping[str, Any], *, known_vars: set[str]) -> list[str]:
    refs: set[str] = set()
    raw_args = node_spec.get("_args")
    if raw_args is not None:
        _collect_var_refs(raw_args, known_vars=known_vars, out=refs)
    for key, value in node_spec.items():
        if key.startswith("_") or key in {"graph", "when"}:
            continue
        _collect_var_refs(value, known_vars=known_vars, out=refs)
    return sorted(refs)


def _node_output_vars(node_spec: Mapping[str, Any]) -> list[str]:
    bound = node_spec.get("_bind")
    if isinstance(bound, str):
        return [bound]
    if isinstance(bound, list):
        return [str(item) for item in bound]
    return []


def _collect_graph_bound_vars(graph: list[Any]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for item in graph:
        if not isinstance(item, Mapping) or len(item) != 1:
            continue
        _name, node_spec = next(iter(item.items()))
        if not isinstance(node_spec, Mapping):
            continue
        for out_name in _node_output_vars(node_spec):
            if out_name in seen:
                continue
            ordered.append(out_name)
            seen.add(out_name)
        nested_graph = node_spec.get("graph")
        if isinstance(nested_graph, list):
            for out_name in _collect_graph_bound_vars(nested_graph):
                if out_name in seen:
                    continue
                ordered.append(out_name)
                seen.add(out_name)
        body_graph = node_spec.get("_body")
        if isinstance(body_graph, list):
            for out_name in _collect_graph_bound_vars(body_graph):
                if out_name in seen:
                    continue
                ordered.append(out_name)
                seen.add(out_name)
    return ordered


def _collect_graph_var_refs(graph: list[Any], *, known_vars: set[str]) -> list[str]:
    refs: set[str] = set()
    for item in graph:
        if not isinstance(item, Mapping) or len(item) != 1:
            continue
        _name, node_spec = next(iter(item.items()))
        if not isinstance(node_spec, Mapping):
            continue
        _collect_var_refs(node_spec.get("_args"), known_vars=known_vars, out=refs)
        for key, value in node_spec.items():
            if not isinstance(key, str):
                continue
            if key.startswith("_") or key in {"graph", "when"}:
                continue
            _collect_var_refs(value, known_vars=known_vars, out=refs)
        nested_graph = node_spec.get("graph")
        if isinstance(nested_graph, list):
            refs.update(_collect_graph_var_refs(nested_graph, known_vars=known_vars))
        body_graph = node_spec.get("_body")
        if isinstance(body_graph, list):
            refs.update(_collect_graph_var_refs(body_graph, known_vars=known_vars))
    return sorted(refs)


def _collect_call_scope_hints(
    graph: list[Any],
    *,
    by_target: dict[str, set[str]],
    by_edge: list[tuple[str, str, str]],
    current_block: str,
    loop_scopes: tuple[tuple[str, str], ...] = (),
) -> None:
    def _resolve_scope_with_loop_placeholders(
        base_scope: str, scopes: tuple[tuple[str, str], ...]
    ) -> str:
        relevant: list[tuple[list[str], str]] = []
        base_parts = [part for part in base_scope.split(".") if part]
        for loop_scope, loop_var in scopes:
            loop_parts = [part for part in loop_scope.split(".") if part]
            if loop_parts == base_parts or (
                len(loop_parts) > len(base_parts) and loop_parts[: len(base_parts)] == base_parts
            ):
                relevant.append((loop_parts, loop_var))
        if not relevant:
            return base_scope
        relevant.sort(key=lambda item: len(item[0]))
        resolved_parts = list(base_parts)
        plain_parts = list(base_parts)
        for loop_parts, loop_var in relevant:
            suffix = loop_parts[len(base_parts) :]
            expected_prefix = loop_parts[: len(plain_parts)]
            if plain_parts != expected_prefix:
                # Ambiguous divergent loop-paths for the same call scope.
                return base_scope
            for seg in suffix[len(plain_parts) - len(base_parts) :]:
                resolved_parts.append(seg)
                plain_parts.append(seg)
            resolved_parts.append(f"{{{loop_var}}}")
        return ".".join(resolved_parts)

    for item in graph:
        if not isinstance(item, Mapping) or len(item) != 1:
            continue
        _name, node_spec = next(iter(item.items()))
        if not isinstance(node_spec, Mapping):
            continue
        next_loop_scopes = loop_scopes
        if node_spec.get("_op") == "for":
            raw_scope = node_spec.get("_scope")
            raw_var = node_spec.get("_var")
            if isinstance(raw_scope, str) and raw_scope and isinstance(raw_var, str) and raw_var:
                next_loop_scopes = (*loop_scopes, (raw_scope, raw_var))
        if node_spec.get("_op") == "call":
            target = node_spec.get("_target")
            scope = node_spec.get("_scope")
            if isinstance(target, str):
                scope_base: str | None = None
                if isinstance(scope, str) and scope:
                    scope_base = scope
                elif next_loop_scopes:
                    scope_base = next_loop_scopes[-1][0]
                if isinstance(scope_base, str) and scope_base:
                    resolved_scope = _resolve_scope_with_loop_placeholders(
                        scope_base, next_loop_scopes
                    )
                    by_target.setdefault(target, set()).add(resolved_scope)
                    by_edge.append((current_block, target, resolved_scope))
                    continue
                continue
        nested_graph = node_spec.get("graph")
        if isinstance(nested_graph, list):
            _collect_call_scope_hints(
                nested_graph,
                by_target=by_target,
                by_edge=by_edge,
                current_block=current_block,
                loop_scopes=next_loop_scopes,
            )
        body_graph = node_spec.get("_body")
        if isinstance(body_graph, list):
            _collect_call_scope_hints(
                body_graph,
                by_target=by_target,
                by_edge=by_edge,
                current_block=current_block,
                loop_scopes=next_loop_scopes,
            )


def _normalize_expr(expr: str) -> str:
    return " ".join(expr.strip().split())


def _split_when_polarity(expr: str) -> tuple[bool, str]:
    normalized = _normalize_expr(expr)
    match = re.match(r"^not\s*\((.+)\)$", normalized)
    if match is not None:
        return False, _normalize_expr(match.group(1))
    match = re.match(r"^not\s+(.+)$", normalized)
    if match is not None:
        return False, _normalize_expr(match.group(1))
    return True, normalized


def _ordered_rows(rows: list[str], *, top_down: bool) -> list[str]:
    return rows if top_down else list(reversed(rows))


def _transpose_table(rows: list[list[str]]) -> list[str]:
    if not rows:
        return []
    width = max(len(row) for row in rows)
    normalized = [row + ['<TD BGCOLOR="white"></TD>'] * (width - len(row)) for row in rows]
    out: list[str] = []
    for col in range(width):
        out.append(f"      <TR>{''.join(row[col] for row in normalized)}</TR>\n")
    return out


def _render_gateway_row_table(
    *,
    label: str,
    label_bg: str,
    slot_cells: list[str],
    empty_bg: str,
    border: int,
    cellborder: int,
    cellspacing: int,
    cellpadding: int,
    color: str | None,
    transpose_layout: bool,
) -> str:
    if not slot_cells:
        slot_cells = [f'<TD BGCOLOR="{empty_bg}"></TD>']
    color_attr = f' COLOR="{color}"' if isinstance(color, str) else ""
    if not transpose_layout:
        return (
            f'    <TABLE BORDER="{border}" CELLBORDER="{cellborder}" '
            f'CELLSPACING="{cellspacing}" CELLPADDING="{cellpadding}"{color_attr}>\n'
            "      <TR>"
            f'<TD BGCOLOR="{label_bg}" ALIGN="CENTER"><B>{_html_escape(label)}</B></TD>'
            f"{''.join(slot_cells)}</TR>\n"
            "    </TABLE>"
        )
    rows = [
        [
            f'<TD BGCOLOR="{label_bg}" ALIGN="CENTER"><B>{_html_escape(label)}</B></TD>',
            *[f'<TD BGCOLOR="{empty_bg}"></TD>' for _ in slot_cells],
        ],
        [*slot_cells],
    ]
    transposed = _transpose_table(rows)
    return (
        f'    <TABLE BORDER="{border}" CELLBORDER="{cellborder}" '
        f'CELLSPACING="{cellspacing}" CELLPADDING="{cellpadding}"{color_attr}>\n'
        f"{''.join(transposed)}"
        "    </TABLE>"
    )


def _walk_graph(
    graph: list[Any],
    *,
    block_name: str,
    scope: str,
    lines: list[str],
    var_sources: dict[str, _Endpoint],
    edge_labels: dict[tuple[_Endpoint, _Endpoint], set[str]],
    flow_edges: set[tuple[str, str]],
    loop_control_edges: set[tuple[str, str, str]],
    call_return_edges: set[tuple[_Endpoint, _Endpoint, str]],
    loop_subgraphs: list[tuple[str, str, tuple[str, ...], str | None]],
    scope_subgraphs: list[tuple[str, str, tuple[str, ...], str | None]],
    scope_nodes_emitted: set[str],
    scope_in_emitted_ports: set[tuple[str, str]],
    scope_out_emitted_ports: set[tuple[str, str]],
    blocks_by_name: Mapping[str, Mapping[str, Any]],
    runtime_scope: str,
    required_downstream_vars: set[str],
    var_types: dict[str, str],
    block_io_types: Mapping[str, Mapping[str, Mapping[str, str]]] | None,
    block_scope_prefixes: Mapping[str, str] | None,
    current_block_name: str,
    top_down: bool,
    transpose_layout: bool,
    scope_context: dict[str, Any] | None = None,
    input_gateway_node_ids: set[str] | None = None,
    output_gateway_node_ids: set[str] | None = None,
) -> list[str]:
    created_nodes: list[str] = []
    root_scope_walk = scope_context is None
    if root_scope_walk:
        scope_order: list[str] = []
        scope_node_ids: dict[str, list[str]] = {}
        scope_refs: dict[str, set[str]] = {}
        scope_produced: dict[str, set[str]] = {}
        scope_consumed: dict[str, set[str]] = {}
        scope_produced_forward: dict[str, set[str]] = {}
        node_scope_by_id: dict[str, str] = {}
        all_known_vars = set(var_sources.keys()).union(_collect_graph_bound_vars(graph))
        var_producer_scope_all: dict[str, str] = {}
        var_used_scopes: dict[str, set[str]] = {}

        def _collect_scope_prepass(items: list[Any], *, item_scope: str) -> None:
            for index, item in enumerate(items):
                if not isinstance(item, Mapping) or len(item) != 1:
                    continue
                node_name, node_spec = next(iter(item.items()))
                if not isinstance(node_name, str) or not isinstance(node_spec, Mapping):
                    continue
                op = node_spec.get("_op")
                op_name = op if isinstance(op, str) else "group"
                all_input_slots = _node_input_slots(
                    node_spec, op_name=op_name, known_vars=all_known_vars
                )
                all_refs = {ref for _slot_name, slot_refs in all_input_slots for ref in slot_refs}
                raw_scope = node_spec.get("_scope")
                usage_scope = (
                    raw_scope if isinstance(raw_scope, str) and raw_scope else "__outside__"
                )
                for ref in all_refs:
                    var_used_scopes.setdefault(ref, set()).add(usage_scope)
                if isinstance(raw_scope, str) and raw_scope and op_name != "for":
                    node_id = _dot_id("op", block_name, item_scope, f"{index:04d}_{node_name}")
                    node_scope_by_id[node_id] = raw_scope
                    if raw_scope not in scope_node_ids:
                        scope_order.append(raw_scope)
                        scope_node_ids[raw_scope] = []
                        scope_refs[raw_scope] = set()
                        scope_produced[raw_scope] = set()
                    scope_node_ids[raw_scope].append(node_id)
                    scope_refs[raw_scope].update(all_refs)
                    for out_var in _node_output_vars(node_spec):
                        scope_produced[raw_scope].add(out_var)
                        var_producer_scope_all[out_var] = raw_scope

                nested_graph = node_spec.get("graph")
                if isinstance(nested_graph, list):
                    _collect_scope_prepass(nested_graph, item_scope=f"{item_scope}.{node_name}")
                body_graph = node_spec.get("_body")
                if isinstance(body_graph, list):
                    _collect_scope_prepass(body_graph, item_scope=f"{item_scope}.{node_name}._body")

        _collect_scope_prepass(graph, item_scope=scope)

        def _loop_vars_under_scope(items: list[Any], *, scope_name: str) -> set[str]:
            out: set[str] = set()
            for item in items:
                if not isinstance(item, Mapping) or len(item) != 1:
                    continue
                _node_name, node_spec = next(iter(item.items()))
                if not isinstance(node_spec, Mapping):
                    continue
                if node_spec.get("_op") == "for":
                    raw_scope = node_spec.get("_scope")
                    raw_var = node_spec.get("_var")
                    if (
                        isinstance(raw_scope, str)
                        and raw_scope
                        and isinstance(raw_var, str)
                        and raw_var
                        and (raw_scope == scope_name or raw_scope.startswith(scope_name + "."))
                    ):
                        out.add(raw_var)
                nested_graph = node_spec.get("graph")
                if isinstance(nested_graph, list):
                    out.update(_loop_vars_under_scope(nested_graph, scope_name=scope_name))
                body_graph = node_spec.get("_body")
                if isinstance(body_graph, list):
                    out.update(_loop_vars_under_scope(body_graph, scope_name=scope_name))
            return out

        for scope_name in scope_order:
            refs = scope_refs.get(scope_name, set())
            produced = scope_produced.get(scope_name, set())
            scope_consumed[scope_name] = {
                var_name for var_name in refs if var_producer_scope_all.get(var_name) != scope_name
            }
            local_loop_vars = _loop_vars_under_scope(graph, scope_name=scope_name)
            if local_loop_vars:
                scope_consumed[scope_name] = {
                    var_name
                    for var_name in scope_consumed[scope_name]
                    if var_name not in local_loop_vars
                }
            scope_produced_forward[scope_name] = {
                var_name
                for var_name in produced
                if (
                    any(
                        use_scope != scope_name
                        for use_scope in var_used_scopes.get(var_name, set())
                    )
                    or var_name in required_downstream_vars
                )
            }

        scope_in_node: dict[str, str] = {}
        scope_out_node: dict[str, str] = {}
        scope_in_ports: dict[str, dict[str, str]] = {}
        scope_out_ports: dict[str, dict[str, str]] = {}
        scope_in_linked: set[tuple[str, str]] = set()
        runtime_var_scope: dict[str, str] = {}
        scope_out_latest_source: dict[tuple[str, str], _Endpoint] = {}

        for scope_name in scope_order:
            cluster_id = _scope_cluster_id(block_name, scope_name)
            in_vars = sorted(scope_consumed.get(scope_name, set()))
            out_vars = sorted(scope_produced_forward.get(scope_name, set()))
            in_id = _dot_id("scope_in", block_name, scope_name)
            out_id = _dot_id("scope_out", block_name, scope_name)
            scope_in_node[scope_name] = in_id
            scope_out_node[scope_name] = out_id

            in_port_map: dict[str, str] = {}
            in_cells: list[str] = []
            for var_name in in_vars:
                port = _port_id(f"in_{var_name}")
                in_port_map[var_name] = port
                in_cells.append(
                    f'<TD PORT="{port}" BGCOLOR="azure" ALIGN="CENTER"><FONT POINT-SIZE="7">{_html_escape(var_name)}</FONT></TD>'
                )
            if not in_cells:
                in_cells.append('<TD BGCOLOR="azure"></TD>')
            scope_in_ports[scope_name] = in_port_map
            in_table = _render_gateway_row_table(
                label="SCOPE",
                label_bg="azure",
                slot_cells=in_cells,
                empty_bg="azure",
                border=2,
                cellborder=1,
                cellspacing=0,
                cellpadding=6,
                color="steelblue4",
                transpose_layout=transpose_layout,
            )
            if in_id not in scope_nodes_emitted:
                lines.append(f'  "{in_id}" [shape=plain, label=<{in_table}>];')
                created_nodes.append(in_id)
                scope_nodes_emitted.add(in_id)
                if input_gateway_node_ids is not None:
                    input_gateway_node_ids.add(in_id)
                for port in in_port_map.values():
                    scope_in_emitted_ports.add((in_id, port))

            out_port_map: dict[str, str] = {}
            out_cells: list[str] = []
            for var_name in out_vars:
                port = _port_id(f"out_{var_name}")
                out_port_map[var_name] = port
                out_cells.append(
                    f'<TD PORT="{port}" BGCOLOR="honeydew" ALIGN="CENTER"><FONT POINT-SIZE="7">{_html_escape(var_name)}</FONT></TD>'
                )
            if not out_cells:
                out_cells.append('<TD BGCOLOR="honeydew"></TD>')
            scope_out_ports[scope_name] = out_port_map
            out_table = _render_gateway_row_table(
                label="SCOPE",
                label_bg="honeydew",
                slot_cells=out_cells,
                empty_bg="honeydew",
                border=2,
                cellborder=1,
                cellspacing=0,
                cellpadding=6,
                color="steelblue4",
                transpose_layout=transpose_layout,
            )
            if out_id not in scope_nodes_emitted:
                lines.append(f'  "{out_id}" [shape=plain, label=<{out_table}>];')
                created_nodes.append(out_id)
                scope_nodes_emitted.add(out_id)
                if output_gateway_node_ids is not None:
                    output_gateway_node_ids.add(out_id)
                for port in out_port_map.values():
                    scope_out_emitted_ports.add((out_id, port))

            parent_scope_name: str | None = None
            parts = [part for part in scope_name.split(".") if part]
            for cut in range(len(parts) - 1, 0, -1):
                candidate = ".".join(parts[:cut])
                if candidate in scope_node_ids:
                    parent_scope_name = candidate
                    break
            parent_cluster = (
                _scope_cluster_id(block_name, parent_scope_name)
                if isinstance(parent_scope_name, str)
                else None
            )
            scope_subgraphs.append(
                (
                    cluster_id,
                    f"scope {scope_name}",
                    tuple([in_id, *scope_node_ids.get(scope_name, []), out_id]),
                    parent_cluster,
                )
            )

        scope_context = {
            "scope_order": scope_order,
            "scope_node_ids": scope_node_ids,
            "scope_consumed": scope_consumed,
            "scope_produced_forward": scope_produced_forward,
            "node_scope_by_id": node_scope_by_id,
            "scope_in_node": scope_in_node,
            "scope_out_node": scope_out_node,
            "scope_in_ports": scope_in_ports,
            "scope_out_ports": scope_out_ports,
            "scope_in_linked": scope_in_linked,
            "runtime_var_scope": runtime_var_scope,
            "scope_out_latest_source": scope_out_latest_source,
        }
    else:
        assert isinstance(scope_context, dict)
        scope_order = scope_context["scope_order"]
        scope_node_ids = scope_context["scope_node_ids"]
        scope_consumed = scope_context["scope_consumed"]
        scope_produced_forward = scope_context["scope_produced_forward"]
        node_scope_by_id = scope_context["node_scope_by_id"]
        scope_in_node = scope_context["scope_in_node"]
        scope_out_node = scope_context["scope_out_node"]
        scope_in_ports = scope_context["scope_in_ports"]
        scope_out_ports = scope_context["scope_out_ports"]
        scope_in_linked = scope_context["scope_in_linked"]
        runtime_var_scope = scope_context["runtime_var_scope"]
        scope_out_latest_source = scope_context["scope_out_latest_source"]

    prev_node_id: str | None = None
    pending_ternary: dict[str, tuple[_Endpoint, bool, str, str | None]] = {}
    for index, item in enumerate(graph):
        if not isinstance(item, Mapping) or len(item) != 1:
            continue
        node_name, node_spec = next(iter(item.items()))
        if not isinstance(node_name, str) or not isinstance(node_spec, Mapping):
            continue
        node_id = _dot_id("op", block_name, scope, f"{index:04d}_{node_name}")
        op = node_spec.get("_op")
        op_name = op if isinstance(op, str) else "group"
        node_runtime_path = f"{runtime_scope}.{node_name}" if runtime_scope else node_name
        known_vars = set(var_sources.keys())
        input_slots = _node_input_slots(node_spec, op_name=op_name, known_vars=known_vars)
        input_slot_kinds = _node_input_slot_kinds(node_spec, op_name=op_name)
        input_slot_expr_hints = _node_input_slot_expr_hints(node_spec, op_name=op_name)
        input_slot_names = [slot for slot, _ in input_slots]
        output_vars = _node_output_vars(node_spec)
        input_slot_display: dict[str, str] = {name: name for name in input_slot_names}
        output_slot_display: dict[str, str] = {name: name for name in output_vars}
        body_graph = node_spec.get("_body")
        loop_var_name: str | None = None
        loop_body_bound_vars: list[str] = []
        if op_name == "for":
            loop_var = node_spec.get("_var")
            loop_from = node_spec.get("_from", 0)
            loop_to = node_spec.get("_to")
            loop_step = node_spec.get("_step", 1)
            var_text = loop_var if isinstance(loop_var, str) and loop_var else "i"
            if isinstance(loop_var, str) and loop_var:
                loop_var_name = loop_var
            if isinstance(body_graph, list):
                body_ref_vars = _collect_graph_var_refs(body_graph, known_vars=known_vars)
            else:
                body_ref_vars = []
            if isinstance(body_graph, list):
                loop_body_bound_vars = _collect_graph_bound_vars(body_graph)
                for carried_name in loop_body_bound_vars:
                    if carried_name == loop_var_name:
                        continue
                    if carried_name not in output_vars:
                        output_vars.append(carried_name)
            future_known_vars = set(known_vars).union(output_vars).union(required_downstream_vars)
            tail_refs = _collect_graph_var_refs(graph[index + 1 :], known_vars=future_known_vars)
            live_after_loop = set(tail_refs).union(required_downstream_vars)
            output_vars = [name for name in output_vars if name in live_after_loop]
            slot_map: dict[str, set[str]] = {name: set(refs) for name, refs in input_slots}
            loop_inputs: list[tuple[str, Any]] = [
                ("from", loop_from),
                ("to", loop_to),
                ("step", loop_step),
            ]
            if loop_var_name is not None:
                slot_map.setdefault(loop_var_name, set())
            for slot_name, raw_value in loop_inputs:
                loop_input_refs: set[str] = set()
                _collect_var_refs(raw_value, known_vars=known_vars, out=loop_input_refs)
                if loop_input_refs:
                    slot_map.setdefault(slot_name, set()).update(loop_input_refs)
            for body_var in body_ref_vars:
                if body_var == loop_var_name:
                    continue
                slot_map.setdefault(body_var, set()).add(body_var)
            preferred_order: list[str] = []
            if loop_var_name is not None:
                preferred_order.append(loop_var_name)
            preferred_order.extend(["from", "to", "step"])
            preferred_order.extend(input_slot_names)
            preferred_order.extend(body_ref_vars)
            ordered_slots: list[tuple[str, set[str]]] = []
            seen_slots: set[str] = set()
            for slot_name in preferred_order:
                if slot_name in slot_map and slot_name not in seen_slots:
                    ordered_slots.append((slot_name, slot_map[slot_name]))
                    seen_slots.add(slot_name)
            for slot_name, refs in slot_map.items():
                if slot_name in seen_slots:
                    continue
                ordered_slots.append((slot_name, refs))
            input_slots = ordered_slots
            input_slot_names = [slot for slot, _ in input_slots]
        param_paths = _node_param_paths(
            node_spec,
            op_name=op_name,
            node_runtime_path=node_runtime_path,
            scope_prefix=(
                block_scope_prefixes.get(current_block_name)
                if isinstance(block_scope_prefixes, Mapping)
                else None
            ),
        )
        arg_ports: dict[str, str] = {}
        out_ports: dict[str, str] = {}
        input_slot_types = infer_input_slot_types(input_slots=input_slots, var_types=var_types)
        output_slot_types: dict[str, str] = {}
        output_slot_types.update(
            infer_output_types_for_node(
                op_name=op_name,
                node_spec=node_spec,
                input_slots=input_slots,
                output_vars=output_vars,
                var_types=var_types,
            )
        )
        if op_name == "_ir_alias" and output_vars:
            if input_slots:
                first_refs = sorted(input_slots[0][1])
                if len(first_refs) == 1:
                    alias_type = var_types.get(first_refs[0])
                    if isinstance(alias_type, str):
                        for out_name in output_vars:
                            output_slot_types[out_name] = alias_type
        if op_name == "call":
            target_name = node_spec.get("_target")
            if isinstance(target_name, str) and isinstance(block_io_types, Mapping):
                target_io = block_io_types.get(target_name, {})
                target_out_map = (
                    target_io.get("outputs", {}) if isinstance(target_io, Mapping) else {}
                )
                if isinstance(target_out_map, Mapping):
                    for out_name, (_target_out_name, target_type) in zip(
                        output_vars,
                        target_out_map.items(),
                        strict=False,
                    ):
                        if isinstance(target_type, str) and target_type:
                            output_slot_types[out_name] = target_type
            target_block = blocks_by_name.get(target_name) if isinstance(target_name, str) else None
            if isinstance(target_block, Mapping):
                target_inputs = target_block.get("inputs")
                if isinstance(target_inputs, Mapping):
                    formal_inputs = [name for name in target_inputs if isinstance(name, str)]
                    raw_args = node_spec.get("_args")
                    if raw_args is None:
                        positional_arity = 0
                    elif isinstance(raw_args, list):
                        positional_arity = len(raw_args)
                    else:
                        positional_arity = 1
                    for idx, slot_name in enumerate(input_slot_names[:positional_arity]):
                        if idx < len(formal_inputs):
                            input_slot_display[slot_name] = formal_inputs[idx]
                target_outputs = target_block.get("outputs")
                if isinstance(target_outputs, Mapping):
                    formal_outputs = [name for name in target_outputs if isinstance(name, str)]
                    for idx, out_name in enumerate(output_vars):
                        if idx < len(formal_outputs):
                            display_name = formal_outputs[idx]
                            if re.fullmatch(r"out_\d+", display_name):
                                candidate = re.sub(r"^(kwarg_|arg_)", "", out_name)
                                candidate = re.sub(r"_\d+$", "", candidate)
                                if candidate and not re.fullmatch(r"(tmp|discard)_\d+", candidate):
                                    display_name = candidate
                            output_slot_display[out_name] = display_name
        slot_cols = max(1, len(input_slot_names), len(output_vars))
        node_in_cells: list[str] = []
        for slot_name in input_slot_names:
            port = _port_id(f"arg_{slot_name}")
            arg_ports[slot_name] = port
            slot_kind = input_slot_kinds.get(slot_name, "kw")
            slot_bg = _arg_slot_bg(slot_kind)
            node_in_cells.append(
                f'<TD PORT="{port}" BGCOLOR="{slot_bg}" ALIGN="CENTER">'
                f"{_slot_label_html(input_slot_display.get(slot_name, slot_name), input_slot_types.get(slot_name), input_slot_expr_hints.get(slot_name))}</TD>"
            )
        while len(node_in_cells) < slot_cols:
            node_in_cells.append('<TD BGCOLOR="azure"></TD>')

        node_out_cells: list[str] = []
        for out_name in output_vars:
            port = _port_id(f"out_{out_name}")
            out_ports[out_name] = port
            node_out_cells.append(
                f'<TD PORT="{port}" BGCOLOR="honeydew" ALIGN="CENTER">'
                f"{_slot_label_html(output_slot_display.get(out_name, out_name), output_slot_types.get(out_name))}</TD>"
            )
        while len(node_out_cells) < slot_cols:
            node_out_cells.append('<TD BGCOLOR="honeydew"></TD>')

        loop_output_id: str | None = None
        loop_cluster_label: str | None = None
        if op_name == "for":
            if loop_step == 1:
                loop_cluster_label = f"for {var_text} <- [{loop_from}..{loop_to})"
            else:
                loop_cluster_label = f"for {var_text} <- [{loop_from}..{loop_to}) step {loop_step}"
            for_in_cells = [
                (
                    f'<TD PORT="{arg_ports[slot_name]}" '
                    f'BGCOLOR="{_arg_slot_bg(input_slot_kinds.get(slot_name, "kw"))}" '
                    f'ALIGN="CENTER">'
                    f"{_slot_label_html(slot_name, input_slot_types.get(slot_name), input_slot_expr_hints.get(slot_name))}</TD>"
                )
                for slot_name in input_slot_names
                if slot_name in arg_ports
            ]
            if not for_in_cells:
                for_in_cells = ['<TD BGCOLOR="azure"></TD>']
            header_table = _render_gateway_row_table(
                label="FOR",
                label_bg="azure",
                slot_cells=for_in_cells,
                empty_bg="azure",
                border=2,
                cellborder=1,
                cellspacing=0,
                cellpadding=6,
                color="deepskyblue4",
                transpose_layout=transpose_layout,
            )
            lines.append(f'  "{node_id}" [shape=plain, label=<{header_table}>];')
            created_nodes.append(node_id)
            if input_gateway_node_ids is not None:
                input_gateway_node_ids.add(node_id)
            loop_output_id = _dot_id("for_outputs", block_name, scope, f"{index:04d}_{node_name}")
            for_out_cells = [
                (
                    f'<TD PORT="{out_ports[out_name]}" BGCOLOR="honeydew" ALIGN="CENTER">'
                    f"{_slot_label_html(out_name, output_slot_types.get(out_name))}</TD>"
                )
                for out_name in output_vars
                if out_name in out_ports
            ]
            if not for_out_cells:
                for_out_cells = ['<TD BGCOLOR="honeydew"></TD>']
            output_table = _render_gateway_row_table(
                label="FOR",
                label_bg="honeydew",
                slot_cells=for_out_cells,
                empty_bg="honeydew",
                border=2,
                cellborder=1,
                cellspacing=0,
                cellpadding=6,
                color="deepskyblue4",
                transpose_layout=transpose_layout,
            )
            lines.append(f'  "{loop_output_id}" [shape=plain, label=<{output_table}>];')
            created_nodes.append(loop_output_id)
            if output_gateway_node_ids is not None:
                output_gateway_node_ids.add(loop_output_id)
            flow_edges.add((node_id, loop_output_id))
        else:
            const_detail = ""
            if op_name == "_ir_expr" and "value" in node_spec:
                raw_value = node_spec.get("value")
                if isinstance(raw_value, str):
                    detail_text = raw_value
                else:
                    detail_text = repr(raw_value)
                const_detail = f'<BR ALIGN="LEFT"/><FONT POINT-SIZE="8" COLOR="gray35">{_html_escape(detail_text)}</FONT>'
            op_content = (
                f'<FONT POINT-SIZE="10">{_html_escape(op_name)}</FONT>'
                f'<BR ALIGN="LEFT"/><FONT POINT-SIZE="8" COLOR="gray35">{_html_escape(node_name)}</FONT>'
                f"{const_detail}"
            )
            row_label_bg = "gray90"
            op_cell_bg = "white"
            par_row = ""
            if param_paths:
                par_text = _html_escape(", ".join(param_paths))
                par_row = (
                    "      <TR>"
                    f'<TD BGCOLOR="{row_label_bg}" ALIGN="CENTER"><FONT POINT-SIZE="7"><B>PAR</B></FONT></TD>'
                    f'<TD BGCOLOR="{op_cell_bg}" ALIGN="LEFT" COLSPAN="{slot_cols}">'
                    f'<FONT POINT-SIZE="7" COLOR="gray35">{par_text}</FONT></TD>'
                    "</TR>\n"
                )
            if transpose_layout:
                slot_row_count = max(1, slot_cols)
                rows: list[str] = []
                header_cells = [
                    f'<TD BGCOLOR="{row_label_bg}" ALIGN="CENTER"><FONT POINT-SIZE="7"><B>IN</B></FONT></TD>',
                    f'<TD BGCOLOR="{row_label_bg}" ALIGN="CENTER"><FONT POINT-SIZE="7"><B>OP</B></FONT></TD>',
                ]
                if param_paths:
                    header_cells.append(
                        f'<TD BGCOLOR="{row_label_bg}" ALIGN="CENTER"><FONT POINT-SIZE="7"><B>PAR</B></FONT></TD>'
                    )
                header_cells.append(
                    f'<TD BGCOLOR="{row_label_bg}" ALIGN="CENTER"><FONT POINT-SIZE="7"><B>OUT</B></FONT></TD>'
                )
                rows.append(f"      <TR>{''.join(header_cells)}</TR>\n")
                par_cell = (
                    f'<TD BGCOLOR="{op_cell_bg}" ALIGN="LEFT" ROWSPAN="{slot_row_count}">'
                    f'<FONT POINT-SIZE="7" COLOR="gray35">{_html_escape(", ".join(param_paths))}</FONT></TD>'
                    if param_paths
                    else ""
                )
                for i in range(slot_row_count):
                    in_cell = (
                        node_in_cells[i] if i < len(node_in_cells) else '<TD BGCOLOR="azure"></TD>'
                    )
                    out_cell = (
                        node_out_cells[i]
                        if i < len(node_out_cells)
                        else '<TD BGCOLOR="honeydew"></TD>'
                    )
                    cells = [in_cell]
                    if i == 0:
                        cells.append(
                            f'<TD BGCOLOR="{op_cell_bg}" ALIGN="LEFT" ROWSPAN="{slot_row_count}">{op_content}</TD>'
                        )
                        if par_cell:
                            cells.append(par_cell)
                    cells.append(out_cell)
                    rows.append(f"      <TR>{''.join(cells)}</TR>\n")
                op_table = (
                    '    <TABLE BORDER="1" CELLBORDER="1" CELLSPACING="0" CELLPADDING="3">\n'
                    f"{''.join(rows)}"
                    "    </TABLE>"
                )
            else:
                op_rows = [
                    (
                        "      <TR>"
                        f'<TD BGCOLOR="{row_label_bg}" ALIGN="CENTER"><FONT POINT-SIZE="7"><B>IN</B></FONT></TD>'
                        f"{''.join(node_in_cells)}</TR>\n"
                    ),
                    (
                        "      <TR>"
                        f'<TD BGCOLOR="{row_label_bg}" ALIGN="CENTER"><FONT POINT-SIZE="7"><B>OP</B></FONT></TD>'
                        f'<TD BGCOLOR="{op_cell_bg}" ALIGN="LEFT" COLSPAN="{slot_cols}">{op_content}</TD>'
                        "</TR>\n"
                    ),
                ]
                if par_row:
                    op_rows.append(par_row)
                op_rows.append(
                    "      <TR>"
                    f'<TD BGCOLOR="{row_label_bg}" ALIGN="CENTER"><FONT POINT-SIZE="7"><B>OUT</B></FONT></TD>'
                    f"{''.join(node_out_cells)}</TR>\n"
                )
                op_table = (
                    '    <TABLE BORDER="1" CELLBORDER="1" CELLSPACING="0" CELLPADDING="3">\n'
                    f"{''.join(_ordered_rows(op_rows, top_down=top_down))}"
                    "    </TABLE>"
                )
            lines.append(f'  "{node_id}" [shape=plain, label=<{op_table}>];')
            created_nodes.append(node_id)
        if prev_node_id is not None:
            flow_edges.add((prev_node_id, node_id))
        prev_node_id = loop_output_id if isinstance(loop_output_id, str) else node_id

        node_scope = node_scope_by_id.get(node_id)
        routing_scope: str | None = node_scope
        if routing_scope is None:
            raw_node_scope = node_spec.get("_scope")
            if isinstance(raw_node_scope, str) and raw_node_scope:
                scope_parts = [part for part in raw_node_scope.split(".") if part]
                for cut in range(len(scope_parts) - 1, 0, -1):
                    candidate = ".".join(scope_parts[:cut])
                    if candidate in scope_node_ids:
                        routing_scope = candidate
                        break
        for slot_name, refs in input_slots:
            dst_port = arg_ports.get(slot_name)
            if dst_port is None:
                continue
            for var_name in sorted(refs):
                src_endpoint = var_sources.get(var_name)
                if src_endpoint is None:
                    continue
                if isinstance(routing_scope, str):
                    consumed = scope_consumed.get(routing_scope, set())
                    in_node_id = scope_in_node.get(routing_scope)
                    in_port = scope_in_ports.get(routing_scope, {}).get(var_name)
                    if (
                        not (
                            op_name == "for"
                            and isinstance(loop_var_name, str)
                            and var_name == loop_var_name
                        )
                        and var_name in consumed
                        and runtime_var_scope.get(var_name) != routing_scope
                        and isinstance(in_node_id, str)
                        and isinstance(in_port, str)
                        and (in_node_id, in_port) in scope_in_emitted_ports
                    ):
                        link_key = (routing_scope, var_name)
                        if link_key not in scope_in_linked:
                            _append_edge(
                                edge_labels,
                                src=src_endpoint,
                                dst=(in_node_id, in_port),
                                label=var_name,
                            )
                            scope_in_linked.add(link_key)
                        src_endpoint = (in_node_id, in_port)
                producer_scope = runtime_var_scope.get(var_name)
                if (
                    isinstance(producer_scope, str)
                    and producer_scope != routing_scope
                    and var_name in scope_produced_forward.get(producer_scope, set())
                ):
                    out_node_id = scope_out_node.get(producer_scope)
                    out_port = scope_out_ports.get(producer_scope, {}).get(var_name)
                    if isinstance(out_node_id, str) and isinstance(out_port, str):
                        src_endpoint = (out_node_id, out_port)
                _append_edge(edge_labels, src=src_endpoint, dst=(node_id, dst_port), label=var_name)
        loop_input_prior_sources: dict[str, _Endpoint | None] = {}
        loop_input_prior_scopes: dict[str, str | None] = {}
        if op_name == "for":
            for slot_name, refs in input_slots:
                dst_port = arg_ports.get(slot_name)
                if dst_port is None:
                    continue
                for var_name in sorted(refs):
                    if var_name in loop_input_prior_sources:
                        continue
                    loop_input_prior_sources[var_name] = var_sources.get(var_name)
                    loop_input_prior_scopes[var_name] = runtime_var_scope.get(var_name)
                    var_sources[var_name] = (node_id, dst_port)
                    if isinstance(routing_scope, str):
                        runtime_var_scope[var_name] = routing_scope
        prior_loop_var_endpoint: _Endpoint | None = None
        had_loop_var_prior = False
        if op_name == "for" and isinstance(loop_var_name, str):
            had_loop_var_prior = loop_var_name in var_sources
            if had_loop_var_prior:
                prior_loop_var_endpoint = var_sources[loop_var_name]
            loop_var_port = arg_ports.get(loop_var_name)
            var_sources[loop_var_name] = (node_id, loop_var_port)

        if op_name == "call":
            block_target = node_spec.get("_target")
            if isinstance(block_target, str):
                target_id = _dot_id("block", block_target)
                _append_edge(edge_labels, src=(node_id, None), dst=(target_id, None), label="call")
                target_block = blocks_by_name.get(block_target)
                if isinstance(target_block, Mapping):
                    target_outputs = target_block.get("outputs")
                    if isinstance(target_outputs, Mapping):
                        callee_output_names = [
                            out_name for out_name in target_outputs if isinstance(out_name, str)
                        ]
                        callee_outputs_id = _dot_id("outputs", block_target)
                        for caller_out_name, callee_out_name in zip(
                            output_vars, callee_output_names, strict=False
                        ):
                            caller_port = out_ports.get(caller_out_name)
                            if not isinstance(caller_port, str):
                                continue
                            call_return_edges.add(
                                (
                                    (callee_outputs_id, _port_id(callee_out_name)),
                                    (node_id, caller_port),
                                    callee_out_name,
                                )
                            )

        if op_name != "for":
            output_source_node_id = loop_output_id if isinstance(loop_output_id, str) else node_id
            for out_var in output_vars:
                var_sources[out_var] = (output_source_node_id, out_ports.get(out_var))
                node_scope = node_scope_by_id.get(node_id)
                if isinstance(node_scope, str):
                    runtime_var_scope[out_var] = node_scope
                    if out_var in scope_produced_forward.get(node_scope, set()):
                        out_endpoint = var_sources.get(out_var)
                        if isinstance(out_endpoint, tuple):
                            scope_out_latest_source[(node_scope, out_var)] = out_endpoint
                else:
                    runtime_var_scope.pop(out_var, None)
                out_type = output_slot_types.get(out_var)
                if isinstance(out_type, str):
                    var_types[out_var] = out_type
                when_expr = node_spec.get("when")
                if not isinstance(when_expr, str) or not when_expr.strip():
                    pending_ternary.pop(out_var, None)
                    continue
                is_positive, cond_expr = _split_when_polarity(when_expr)
                current_out_endpoint: _Endpoint = (output_source_node_id, out_ports.get(out_var))
                current_out_type = output_slot_types.get(out_var) or var_types.get(out_var)
                previous = pending_ternary.get(out_var)
                if previous is None:
                    pending_ternary[out_var] = (
                        current_out_endpoint,
                        is_positive,
                        cond_expr,
                        current_out_type,
                    )
                    continue
                prev_out_endpoint, prev_is_positive, prev_cond_expr, prev_out_type = previous
                if prev_cond_expr != cond_expr or prev_is_positive == is_positive:
                    pending_ternary[out_var] = (
                        current_out_endpoint,
                        is_positive,
                        cond_expr,
                        current_out_type,
                    )
                    continue

                select_id = _dot_id("ternary", block_name, scope, f"{index:04d}_{out_var}")
                cond_port = _port_id("arg_cond")
                then_port = _port_id("arg_then")
                else_port = _port_id("arg_else")
                out_port = _port_id(f"out_{out_var}")
                if transpose_layout:
                    select_rows_cells = [
                        '<TR><TD BGCOLOR="lightpink" ALIGN="CENTER"><FONT POINT-SIZE="7"><B>IN</B></FONT></TD>'
                        '<TD BGCOLOR="lightpink" ALIGN="CENTER"><FONT POINT-SIZE="7"><B>OP</B></FONT></TD>'
                        '<TD BGCOLOR="lightpink" ALIGN="CENTER"><FONT POINT-SIZE="7"><B>OUT</B></FONT></TD></TR>\n',
                        "<TR>"
                        f'<TD PORT="{cond_port}" BGCOLOR="lightyellow" ALIGN="CENTER"><FONT POINT-SIZE="7"><B>cond</B></FONT></TD>'
                        '<TD BGCOLOR="seashell2" ALIGN="LEFT" ROWSPAN="3"><FONT POINT-SIZE="10"><B>?:</B></FONT>'
                        f'<BR ALIGN="LEFT"/><FONT POINT-SIZE="8" COLOR="gray35">{_html_escape(out_var)}</FONT></TD>'
                        f'<TD PORT="{out_port}" BGCOLOR="honeydew" ALIGN="CENTER" ROWSPAN="3"><FONT POINT-SIZE="7">{_html_escape(out_var)}</FONT></TD>'
                        "</TR>\n",
                        "<TR>"
                        f'<TD PORT="{then_port}" BGCOLOR="honeydew" ALIGN="CENTER"><FONT POINT-SIZE="7"><B>then</B></FONT></TD>'
                        "</TR>\n",
                        "<TR>"
                        f'<TD PORT="{else_port}" BGCOLOR="mistyrose" ALIGN="CENTER"><FONT POINT-SIZE="7"><B>else</B></FONT></TD>'
                        "</TR>\n",
                    ]
                    select_table = (
                        '    <TABLE BORDER="2" CELLBORDER="1" CELLSPACING="0" CELLPADDING="3" COLOR="deeppink4" BGCOLOR="lavenderblush">\n'
                        f"{''.join(select_rows_cells)}"
                        "    </TABLE>"
                    )
                else:
                    select_rows = [
                        (
                            "      <TR>"
                            '<TD BGCOLOR="lightpink" ALIGN="CENTER"><FONT POINT-SIZE="7"><B>IN</B></FONT></TD>'
                            f'<TD PORT="{cond_port}" BGCOLOR="lightyellow" ALIGN="CENTER"><FONT POINT-SIZE="7"><B>cond</B></FONT></TD>'
                            f'<TD PORT="{then_port}" BGCOLOR="honeydew" ALIGN="CENTER"><FONT POINT-SIZE="7"><B>then</B></FONT></TD>'
                            f'<TD PORT="{else_port}" BGCOLOR="mistyrose" ALIGN="CENTER"><FONT POINT-SIZE="7"><B>else</B></FONT></TD>'
                            "</TR>\n"
                        ),
                        (
                            "      <TR>"
                            '<TD BGCOLOR="lightpink" ALIGN="CENTER"><FONT POINT-SIZE="7"><B>OP</B></FONT></TD>'
                            '<TD BGCOLOR="seashell2" ALIGN="LEFT" COLSPAN="3"><FONT POINT-SIZE="10"><B>?:</B></FONT>'
                            f'<BR ALIGN="LEFT"/><FONT POINT-SIZE="8" COLOR="gray35">{_html_escape(out_var)}</FONT></TD>'
                            "</TR>\n"
                        ),
                        (
                            "      <TR>"
                            '<TD BGCOLOR="lightpink" ALIGN="CENTER"><FONT POINT-SIZE="7"><B>OUT</B></FONT></TD>'
                            f'<TD PORT="{out_port}" BGCOLOR="honeydew" ALIGN="CENTER" COLSPAN="3">'
                            f'<FONT POINT-SIZE="7">{_html_escape(out_var)}</FONT></TD>'
                            "</TR>\n"
                        ),
                    ]
                    select_table = (
                        '    <TABLE BORDER="2" CELLBORDER="1" CELLSPACING="0" CELLPADDING="3" COLOR="deeppink4" BGCOLOR="lavenderblush">\n'
                        f"{''.join(_ordered_rows(select_rows, top_down=top_down))}"
                        "    </TABLE>"
                    )
                lines.append(f'  "{select_id}" [shape=plain, label=<{select_table}>];')
                created_nodes.append(select_id)
                if isinstance(node_scope, str):
                    node_scope_by_id[select_id] = node_scope
                    if select_id not in scope_node_ids.setdefault(node_scope, []):
                        scope_node_ids[node_scope].append(select_id)
                    scope_cluster = _scope_cluster_id(block_name, node_scope)
                    for i, (cluster_name, cluster_label, cluster_nodes, parent_scope) in enumerate(
                        scope_subgraphs
                    ):
                        if cluster_name != scope_cluster:
                            continue
                        if select_id in cluster_nodes:
                            break
                        scope_subgraphs[i] = (
                            cluster_name,
                            cluster_label,
                            tuple([*cluster_nodes, select_id]),
                            parent_scope,
                        )
                        break
                if prev_is_positive:
                    true_src = prev_out_endpoint
                    false_src = current_out_endpoint
                else:
                    true_src = current_out_endpoint
                    false_src = prev_out_endpoint
                _append_edge(
                    edge_labels,
                    src=true_src,
                    dst=(select_id, then_port),
                    label=f"{out_var}_then",
                )
                _append_edge(
                    edge_labels,
                    src=false_src,
                    dst=(select_id, else_port),
                    label=f"{out_var}_else",
                )

                condition_refs: set[str] = set()
                _collect_var_refs(cond_expr, known_vars=set(var_sources.keys()), out=condition_refs)
                for cond_var in sorted(condition_refs):
                    cond_src = var_sources.get(cond_var)
                    if cond_src is None:
                        continue
                    _append_edge(
                        edge_labels, src=cond_src, dst=(select_id, cond_port), label=cond_var
                    )

                var_sources[out_var] = (select_id, out_port)
                if isinstance(prev_out_type, str) and prev_out_type == current_out_type:
                    var_types[out_var] = prev_out_type
                pending_ternary.pop(out_var, None)

        nested_graph = node_spec.get("graph")
        if isinstance(nested_graph, list):
            created_nodes.extend(
                _walk_graph(
                    nested_graph,
                    block_name=block_name,
                    scope=f"{scope}.{node_name}",
                    lines=lines,
                    var_sources=var_sources,
                    edge_labels=edge_labels,
                    flow_edges=flow_edges,
                    loop_control_edges=loop_control_edges,
                    call_return_edges=call_return_edges,
                    loop_subgraphs=loop_subgraphs,
                    scope_subgraphs=scope_subgraphs,
                    scope_nodes_emitted=scope_nodes_emitted,
                    scope_in_emitted_ports=scope_in_emitted_ports,
                    scope_out_emitted_ports=scope_out_emitted_ports,
                    blocks_by_name=blocks_by_name,
                    runtime_scope=node_runtime_path,
                    required_downstream_vars=required_downstream_vars,
                    var_types=var_types,
                    block_io_types=block_io_types,
                    block_scope_prefixes=block_scope_prefixes,
                    current_block_name=current_block_name,
                    top_down=top_down,
                    transpose_layout=transpose_layout,
                    scope_context=scope_context,
                    input_gateway_node_ids=input_gateway_node_ids,
                    output_gateway_node_ids=output_gateway_node_ids,
                )
            )

        if isinstance(body_graph, list):
            body_scope = f"{scope}.{node_name}._body"
            body_node_ids: list[str] = []
            for body_index, body_item in enumerate(body_graph):
                if not isinstance(body_item, Mapping) or len(body_item) != 1:
                    continue
                body_name, body_spec = next(iter(body_item.items()))
                if not isinstance(body_name, str) or not isinstance(body_spec, Mapping):
                    continue
                body_node_ids.append(
                    _dot_id("op", block_name, body_scope, f"{body_index:04d}_{body_name}")
                )
            if body_node_ids:
                loop_control_edges.add((node_id, body_node_ids[0], "loop"))
                loop_control_edges.add((body_node_ids[-1], node_id, "next"))
            body_created = _walk_graph(
                body_graph,
                block_name=block_name,
                scope=body_scope,
                lines=lines,
                var_sources=var_sources,
                edge_labels=edge_labels,
                flow_edges=flow_edges,
                loop_control_edges=loop_control_edges,
                call_return_edges=call_return_edges,
                loop_subgraphs=loop_subgraphs,
                scope_subgraphs=scope_subgraphs,
                scope_nodes_emitted=scope_nodes_emitted,
                scope_in_emitted_ports=scope_in_emitted_ports,
                scope_out_emitted_ports=scope_out_emitted_ports,
                blocks_by_name=blocks_by_name,
                runtime_scope=node_runtime_path,
                required_downstream_vars=set(output_vars).union(set(loop_body_bound_vars)),
                var_types=var_types,
                block_io_types=block_io_types,
                block_scope_prefixes=block_scope_prefixes,
                current_block_name=current_block_name,
                top_down=top_down,
                transpose_layout=transpose_layout,
                scope_context=scope_context,
                input_gateway_node_ids=input_gateway_node_ids,
                output_gateway_node_ids=output_gateway_node_ids,
            )
            created_nodes.extend(body_created)
            if op_name == "for":
                if isinstance(loop_output_id, str):
                    loop_parent_scope: str | None = None
                    raw_loop_scope = node_spec.get("_scope")
                    if isinstance(raw_loop_scope, str) and raw_loop_scope:
                        parts = [part for part in raw_loop_scope.split(".") if part]
                        for cut in range(len(parts) - 1, 0, -1):
                            candidate = ".".join(parts[:cut])
                            if candidate in scope_node_ids:
                                loop_parent_scope = candidate
                                break
                    for out_var in output_vars:
                        out_port = out_ports.get(out_var)
                        if not isinstance(out_port, str):
                            continue
                        loop_dst_endpoint: _Endpoint = (loop_output_id, out_port)
                        src_endpoint = var_sources.get(out_var)
                        if src_endpoint is None:
                            continue
                        if src_endpoint != loop_dst_endpoint:
                            _append_edge(
                                edge_labels,
                                src=src_endpoint,
                                dst=loop_dst_endpoint,
                                label=out_var,
                            )
                        var_sources[out_var] = loop_dst_endpoint
                        if isinstance(
                            loop_parent_scope, str
                        ) and out_var in scope_produced_forward.get(loop_parent_scope, set()):
                            out_node_id = scope_out_node.get(loop_parent_scope)
                            out_scope_port = scope_out_ports.get(loop_parent_scope, {}).get(out_var)
                            if isinstance(out_node_id, str) and isinstance(out_scope_port, str):
                                if (out_node_id, out_scope_port) not in scope_out_emitted_ports:
                                    continue
                                scope_out_latest_source[(loop_parent_scope, out_var)] = (
                                    loop_dst_endpoint
                                )
                                runtime_var_scope[out_var] = loop_parent_scope
                        type_hint = var_types.get(out_var)
                        if isinstance(type_hint, str):
                            var_types[out_var] = type_hint
                loop_cluster = _loop_cluster_id(block_name, scope, f"{index:04d}_{node_name}")
                loop_label = (
                    loop_cluster_label
                    if isinstance(loop_cluster_label, str)
                    else f"for {node_name}"
                )
                loop_node_list = [node_id, *body_created]
                if isinstance(loop_output_id, str):
                    loop_node_list.append(loop_output_id)
                if loop_node_list:
                    loop_parent_cluster: str | None = None
                    raw_loop_scope = node_spec.get("_scope")
                    if isinstance(raw_loop_scope, str) and raw_loop_scope:
                        parts = [part for part in raw_loop_scope.split(".") if part]
                        for cut in range(len(parts) - 1, 0, -1):
                            candidate = ".".join(parts[:cut])
                            if candidate in scope_node_ids:
                                loop_parent_cluster = _scope_cluster_id(block_name, candidate)
                                break
                    loop_subgraphs.append(
                        (loop_cluster, loop_label, tuple(loop_node_list), loop_parent_cluster)
                    )
        if op_name == "for" and isinstance(loop_var_name, str):
            if had_loop_var_prior and prior_loop_var_endpoint is not None:
                var_sources[loop_var_name] = prior_loop_var_endpoint
            else:
                var_sources.pop(loop_var_name, None)
        if op_name == "for":
            for var_name, prior_source in loop_input_prior_sources.items():
                if var_name in output_vars:
                    continue
                if prior_source is None:
                    var_sources.pop(var_name, None)
                else:
                    var_sources[var_name] = prior_source
                prior_scope = loop_input_prior_scopes.get(var_name)
                if isinstance(prior_scope, str):
                    runtime_var_scope[var_name] = prior_scope
                else:
                    runtime_var_scope.pop(var_name, None)
        for out_name, out_type in output_slot_types.items():
            var_types[out_name] = out_type

    if root_scope_walk:
        for (scope_name, var_name), src_endpoint in sorted(scope_out_latest_source.items()):
            out_node_id = scope_out_node.get(scope_name)
            out_port = scope_out_ports.get(scope_name, {}).get(var_name)
            if not (isinstance(out_node_id, str) and isinstance(out_port, str)):
                continue
            if (out_node_id, out_port) not in scope_out_emitted_ports:
                continue
            # Only route through scope-out when this scope still owns the latest
            # runtime value; otherwise a later outer-scope rebind should win.
            if runtime_var_scope.get(var_name) != scope_name:
                continue
            scope_dst_endpoint: _Endpoint = (out_node_id, out_port)
            _append_edge(edge_labels, src=src_endpoint, dst=scope_dst_endpoint, label=var_name)
            var_sources[var_name] = scope_dst_endpoint
            runtime_var_scope[var_name] = scope_name

    return created_nodes


def _render_block(
    *,
    block_name: str,
    block_spec: Mapping[str, Any],
    lines: list[str],
    edge_labels: dict[tuple[_Endpoint, _Endpoint], set[str]],
    block_clusters: dict[str, str],
    flow_edges: set[tuple[str, str]],
    loop_control_edges: set[tuple[str, str, str]],
    call_return_edges: set[tuple[_Endpoint, _Endpoint, str]],
    blocks_by_name: Mapping[str, Mapping[str, Any]],
    block_label_by_block: Mapping[str, str] | None = None,
    block_io_types: Mapping[str, Mapping[str, Mapping[str, str]]] | None = None,
    block_scope_prefixes: Mapping[str, str] | None = None,
    top_down: bool = True,
    transpose_layout: bool = False,
    input_gateway_node_ids: set[str] | None = None,
    output_gateway_node_ids: set[str] | None = None,
) -> None:
    def _emit_cluster(
        *,
        cluster_name: str,
        cluster_label: str,
        node_ids: tuple[str, ...],
        style: str,
        color: str,
        fillcolor: str | None = None,
        nested: list[tuple[str, str, tuple[str, ...], str, str, str | None]] | None = None,
        indent: str = "  ",
    ) -> None:
        lines.append(f"{indent}subgraph {cluster_name} {{")
        lines.append(f'{indent}  style="{style}";')
        lines.append(f'{indent}  color="{color}";')
        if isinstance(fillcolor, str):
            lines.append(f'{indent}  fillcolor="{fillcolor}";')
        lines.append(f'{indent}  label="{_dot_escape(cluster_label)}";')
        for node_id in node_ids:
            lines.append(f'{indent}  "{node_id}";')
        if nested:
            for (
                child_name,
                child_label,
                child_nodes,
                child_style,
                child_color,
                child_fill,
            ) in nested:
                _emit_cluster(
                    cluster_name=child_name,
                    cluster_label=child_label,
                    node_ids=child_nodes,
                    style=child_style,
                    color=child_color,
                    fillcolor=child_fill,
                    nested=None,
                    indent=indent + "  ",
                )
        lines.append(f"{indent}}}")

    block_id = _dot_id("block", block_name)
    cluster_id = _cluster_id(block_name)
    block_clusters[block_id] = cluster_id
    lines.append(f"  subgraph {cluster_id} {{")
    lines.append('    style="rounded,filled";')
    lines.append('    color="gray70";')
    lines.append('    fillcolor="cornsilk";')
    block_label = block_name
    if block_label_by_block is not None:
        mapped = block_label_by_block.get(block_name)
        if isinstance(mapped, str) and mapped:
            block_label = mapped
    cluster_label = f"block {block_label}"
    lines.append(f'    label="{_dot_escape(cluster_label)}";')
    var_sources: dict[str, _Endpoint] = {}
    input_names: list[str] = []
    input_types: dict[str, str] = {}
    inputs = block_spec.get("inputs")
    if isinstance(inputs, Mapping):
        input_names = [name for name in inputs if isinstance(name, str)]
    if isinstance(block_io_types, Mapping):
        typed = block_io_types.get(block_name, {})
        typed_inputs = typed.get("inputs", {}) if isinstance(typed, Mapping) else {}
        if isinstance(typed_inputs, Mapping):
            input_types = {
                k: v for k, v in typed_inputs.items() if isinstance(k, str) and isinstance(v, str)
            }
    var_types: dict[str, str] = {}

    if input_names:
        title = _html_escape("BLOCK")
        port_cells = []
        for input_name in input_names:
            port = _port_id(input_name)
            port_cells.append(
                f'<TD PORT="{port}" BGCOLOR="aliceblue" ALIGN="CENTER">{_slot_label_html(input_name, input_types.get(input_name))}</TD>'
            )
            var_sources[input_name] = (block_id, port)
            if input_name in input_types:
                var_types[input_name] = input_types[input_name]
        table = _render_gateway_row_table(
            label=title,
            label_bg="azure",
            slot_cells=port_cells,
            empty_bg="azure",
            border=1,
            cellborder=1,
            cellspacing=0,
            cellpadding=6,
            color=None,
            transpose_layout=transpose_layout,
        )
        lines.append(f'    "{block_id}" [shape=plain, label=<{table}>];')
        if input_gateway_node_ids is not None:
            input_gateway_node_ids.add(block_id)
    else:
        lines.append(
            f'    "{block_id}" [shape=box3d, style="filled,bold", fillcolor="lightgoldenrod1", '
            f'label="{_dot_escape("block " + block_name)}"];'
        )

    graph = block_spec.get("graph")
    if isinstance(graph, list):
        loop_subgraphs: list[tuple[str, str, tuple[str, ...], str | None]] = []
        scope_subgraphs: list[tuple[str, str, tuple[str, ...], str | None]] = []
        scope_nodes_emitted: set[str] = set()
        scope_in_emitted_ports: set[tuple[str, str]] = set()
        scope_out_emitted_ports: set[tuple[str, str]] = set()
        outputs_required: set[str] = set()
        outputs_spec = block_spec.get("outputs")
        if isinstance(outputs_spec, Mapping):
            for source_name in outputs_spec.values():
                if isinstance(source_name, str):
                    outputs_required.add(source_name)
        _walk_graph(
            graph,
            block_name=block_name,
            scope="graph",
            lines=lines,
            var_sources=var_sources,
            edge_labels=edge_labels,
            flow_edges=flow_edges,
            loop_control_edges=loop_control_edges,
            call_return_edges=call_return_edges,
            loop_subgraphs=loop_subgraphs,
            scope_subgraphs=scope_subgraphs,
            scope_nodes_emitted=scope_nodes_emitted,
            scope_in_emitted_ports=scope_in_emitted_ports,
            scope_out_emitted_ports=scope_out_emitted_ports,
            blocks_by_name=blocks_by_name,
            runtime_scope="",
            required_downstream_vars=outputs_required,
            var_types=var_types,
            block_io_types=block_io_types,
            block_scope_prefixes=block_scope_prefixes,
            current_block_name=block_name,
            top_down=top_down,
            transpose_layout=transpose_layout,
            input_gateway_node_ids=input_gateway_node_ids,
            output_gateway_node_ids=output_gateway_node_ids,
        )
        merged_scope_subgraphs: dict[str, tuple[str, set[str], str | None]] = {}
        for scope_name, scope_label, scope_nodes, parent_scope in scope_subgraphs:
            existing = merged_scope_subgraphs.get(scope_name)
            if existing is None:
                merged_scope_subgraphs[scope_name] = (scope_label, set(scope_nodes), parent_scope)
                continue
            existing_label, existing_nodes, existing_parent = existing
            existing_nodes.update(scope_nodes)
            chosen_parent = existing_parent if isinstance(existing_parent, str) else parent_scope
            merged_scope_subgraphs[scope_name] = (existing_label, existing_nodes, chosen_parent)
        scope_subgraphs = [
            (name, label, tuple(sorted(nodes)), parent)
            for name, (label, nodes, parent) in merged_scope_subgraphs.items()
        ]
        scope_children_by_parent: dict[str, list[tuple[str, str, tuple[str, ...]]]] = {}
        loops_by_scope: dict[str, list[tuple[str, str, tuple[str, ...]]]] = {}
        top_scope_clusters: list[tuple[str, str, tuple[str, ...]]] = []
        for scope_name, scope_label, scope_nodes, parent_scope in scope_subgraphs:
            if isinstance(parent_scope, str):
                scope_children_by_parent.setdefault(parent_scope, []).append(
                    (scope_name, scope_label, scope_nodes)
                )
            else:
                top_scope_clusters.append((scope_name, scope_label, scope_nodes))
        top_level_loops: list[tuple[str, str, tuple[str, ...]]] = []
        for cluster_name, cluster_label, node_ids, parent_scope in loop_subgraphs:
            if isinstance(parent_scope, str):
                loops_by_scope.setdefault(parent_scope, []).append(
                    (cluster_name, cluster_label, node_ids)
                )
            else:
                top_level_loops.append((cluster_name, cluster_label, node_ids))
        for cluster_name, cluster_label, node_ids in top_level_loops:
            _emit_cluster(
                cluster_name=cluster_name,
                cluster_label=cluster_label,
                node_ids=node_ids,
                style="rounded,dashed",
                color="deepskyblue4",
                fillcolor="azure",
            )

        scope_meta_by_name = {
            scope_name: (scope_label, scope_nodes)
            for scope_name, scope_label, scope_nodes, _parent in scope_subgraphs
        }
        emitted_scopes: set[str] = set()

        def _emit_scope_tree(scope_name: str) -> None:
            if scope_name in emitted_scopes:
                return
            scope_meta = scope_meta_by_name.get(scope_name)
            if scope_meta is None:
                return
            scope_label, scope_nodes = scope_meta
            child_clusters = scope_children_by_parent.get(scope_name, [])
            nested_items: list[tuple[str, str, tuple[str, ...], str, str, str | None]] = []
            for child_name, child_label, child_nodes in child_clusters:
                nested_items.append(
                    (
                        child_name,
                        child_label,
                        child_nodes,
                        "rounded,dashed",
                        "lightsteelblue4",
                        None,
                    )
                )
            for loop_name, loop_label, loop_nodes in loops_by_scope.get(scope_name, []):
                nested_items.append(
                    (loop_name, loop_label, loop_nodes, "rounded,dashed", "deepskyblue4", "azure")
                )
            _emit_cluster(
                cluster_name=scope_name,
                cluster_label=scope_label,
                node_ids=scope_nodes,
                style="rounded,dashed",
                color="lightsteelblue4",
                nested=nested_items,
            )
            emitted_scopes.add(scope_name)
            for child_name, _child_label, _child_nodes in child_clusters:
                emitted_scopes.add(child_name)

        loop_cluster_names = {
            cluster_name for cluster_name, _cluster_label, _node_ids, _parent in loop_subgraphs
        }
        for scope_name, _scope_label, _scope_nodes in top_scope_clusters:
            _emit_scope_tree(scope_name)
        for parent_scope, children in scope_children_by_parent.items():
            if parent_scope in loop_cluster_names:
                continue
            if parent_scope in scope_meta_by_name and parent_scope not in emitted_scopes:
                _emit_scope_tree(parent_scope)
            if parent_scope not in scope_meta_by_name:
                for child_name, _child_label, _child_nodes in children:
                    _emit_scope_tree(child_name)

    outputs = block_spec.get("outputs")
    if isinstance(outputs, Mapping):
        output_names = [name for name in outputs if isinstance(name, str)]
        output_types: dict[str, str] = {}
        if isinstance(block_io_types, Mapping):
            typed = block_io_types.get(block_name, {})
            typed_outputs = typed.get("outputs", {}) if isinstance(typed, Mapping) else {}
            if isinstance(typed_outputs, Mapping):
                output_types = {
                    k: v
                    for k, v in typed_outputs.items()
                    if isinstance(k, str) and isinstance(v, str)
                }
        output_group_id = _dot_id("outputs", block_name)
        if output_names:
            output_cells = []
            for output_name in output_names:
                output_cells.append(
                    f'<TD PORT="{_port_id(output_name)}" BGCOLOR="honeydew" ALIGN="CENTER">{_slot_label_html(output_name, output_types.get(output_name) or var_types.get(output_name))}</TD>'
                )
            output_block_title = _html_escape("BLOCK")
            output_table = _render_gateway_row_table(
                label=output_block_title,
                label_bg="honeydew",
                slot_cells=output_cells,
                empty_bg="honeydew",
                border=1,
                cellborder=1,
                cellspacing=0,
                cellpadding=6,
                color=None,
                transpose_layout=transpose_layout,
            )
            lines.append(f'    "{output_group_id}" [shape=plain, label=<{output_table}>];')
            if output_gateway_node_ids is not None:
                output_gateway_node_ids.add(output_group_id)
        for output_name, source_name in outputs.items():
            if not isinstance(output_name, str):
                continue
            output_endpoint: _Endpoint = (
                output_group_id,
                _port_id(output_name),
            )
            if isinstance(source_name, str):
                src_endpoint = var_sources.get(source_name)
                if src_endpoint is not None:
                    _append_edge(
                        edge_labels,
                        src=src_endpoint,
                        dst=output_endpoint,
                        label=source_name,
                    )
    lines.append("  }")


def render_synapse_spec_to_dot(
    spec: Mapping[str, Any],
    *,
    block_label_by_block: Mapping[str, str] | None = None,
    block_io_types: Mapping[str, Mapping[str, Mapping[str, str]]] | None = None,
    show_control_flow: bool = True,
    direction: str = "top-down",
) -> str:
    model = spec.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("spec.model must be a mapping")
    graph = model.get("graph")
    if not isinstance(graph, list):
        raise ValueError("model.graph must be a list")
    resolved_block_io_types = (
        block_io_types if block_io_types is not None else extract_block_io_types_from_spec(spec)
    )

    direction_value = direction.strip().lower()
    if direction_value == "top-down":
        rankdir = "TB"
        top_down = True
        transpose_layout = False
        src_compass = "s"
        dst_compass = "n"
    elif direction_value == "bottom-up":
        rankdir = "BT"
        top_down = False
        transpose_layout = False
        src_compass = "n"
        dst_compass = "s"
    elif direction_value == "left-right":
        rankdir = "LR"
        top_down = True
        transpose_layout = True
        src_compass = "e"
        dst_compass = "w"
    elif direction_value == "right-left":
        rankdir = "RL"
        top_down = True
        transpose_layout = True
        src_compass = "w"
        dst_compass = "e"
    else:
        raise ValueError("direction must be one of: top-down, bottom-up, left-right, right-left")

    lines: list[str] = [
        "digraph synapse {",
        f"  rankdir={rankdir};",
        "  compound=true;",
        "  newrank=true;",
        '  graph [fontname="Helvetica"];',
        '  node [fontname="Helvetica"];',
        '  edge [fontname="Helvetica"];',
    ]
    edge_labels: dict[tuple[_Endpoint, _Endpoint], set[str]] = {}
    block_clusters: dict[str, str] = {}
    flow_edges: set[tuple[str, str]] = set()
    loop_control_edges: set[tuple[str, str, str]] = set()
    call_return_edges: set[tuple[_Endpoint, _Endpoint, str]] = set()
    input_gateway_node_ids: set[str] = set()
    output_gateway_node_ids: set[str] = set()
    blocks = model.get("blocks")
    blocks_by_name: dict[str, Mapping[str, Any]] = (
        {
            name: value
            for name, value in blocks.items()
            if isinstance(name, str) and isinstance(value, Mapping)
        }
        if isinstance(blocks, Mapping)
        else {}
    )
    call_scopes_by_target: dict[str, set[str]] = {}
    call_scope_edges: list[tuple[str, str, str]] = []
    _collect_call_scope_hints(
        graph,
        by_target=call_scopes_by_target,
        by_edge=call_scope_edges,
        current_block="main",
    )
    for block_name, block_spec in blocks_by_name.items():
        block_graph = block_spec.get("graph")
        if isinstance(block_graph, list):
            _collect_call_scope_hints(
                block_graph,
                by_target=call_scopes_by_target,
                by_edge=call_scope_edges,
                current_block=block_name,
            )
    scope_candidates_by_block: dict[str, set[str]] = {"main": {""}}
    max_iters = max(1, len(call_scope_edges) * max(1, len(blocks_by_name) + 1))
    for _ in range(max_iters):
        changed = False
        for caller_block, target_block, local_scope in call_scope_edges:
            caller_candidates = scope_candidates_by_block.get(caller_block)
            if not caller_candidates:
                continue
            target_candidates = scope_candidates_by_block.setdefault(target_block, set())
            before = len(target_candidates)
            for caller_prefix in caller_candidates:
                if caller_prefix and local_scope:
                    target_candidates.add(f"{caller_prefix}.{local_scope}")
                else:
                    target_candidates.add(local_scope or caller_prefix)
            if len(target_candidates) != before:
                changed = True
        if not changed:
            break

    block_scope_prefixes: dict[str, str] = {}
    for block_name, candidates in scope_candidates_by_block.items():
        if block_name == "main":
            continue
        clean_candidates = {candidate for candidate in candidates if isinstance(candidate, str)}
        if len(clean_candidates) == 1:
            block_scope_prefixes[block_name] = next(iter(clean_candidates))
            continue
        direct_scopes = call_scopes_by_target.get(block_name, set())
        if len(direct_scopes) == 1:
            direct = next(iter(direct_scopes))
            if direct in clean_candidates or not clean_candidates:
                block_scope_prefixes[block_name] = direct

    main_block: dict[str, Any] = {"graph": graph}
    model_inputs = model.get("inputs")
    if isinstance(model_inputs, Mapping):
        main_block["inputs"] = dict(model_inputs)
    model_outputs = model.get("outputs")
    if isinstance(model_outputs, Mapping):
        main_block["outputs"] = dict(model_outputs)
    _render_block(
        block_name="main",
        block_spec=main_block,
        lines=lines,
        edge_labels=edge_labels,
        block_clusters=block_clusters,
        flow_edges=flow_edges,
        loop_control_edges=loop_control_edges,
        call_return_edges=call_return_edges,
        blocks_by_name=blocks_by_name,
        block_label_by_block=block_label_by_block,
        block_io_types=resolved_block_io_types,
        block_scope_prefixes=block_scope_prefixes,
        top_down=top_down,
        transpose_layout=transpose_layout,
        input_gateway_node_ids=input_gateway_node_ids,
        output_gateway_node_ids=output_gateway_node_ids,
    )

    if isinstance(blocks, Mapping):
        for block_name, block_spec in blocks.items():
            if not isinstance(block_name, str) or not isinstance(block_spec, Mapping):
                continue
            _render_block(
                block_name=block_name,
                block_spec=block_spec,
                lines=lines,
                edge_labels=edge_labels,
                block_clusters=block_clusters,
                flow_edges=flow_edges,
                loop_control_edges=loop_control_edges,
                call_return_edges=call_return_edges,
                blocks_by_name=blocks_by_name,
                block_label_by_block=block_label_by_block,
                block_io_types=resolved_block_io_types,
                block_scope_prefixes=block_scope_prefixes,
                top_down=top_down,
                transpose_layout=transpose_layout,
                input_gateway_node_ids=input_gateway_node_ids,
                output_gateway_node_ids=output_gateway_node_ids,
            )

    gateway_nodes = input_gateway_node_ids.union(output_gateway_node_ids)

    def _render_edge_src(endpoint: _Endpoint) -> str:
        node_id, _port = endpoint
        if node_id in gateway_nodes:
            return _dot_endpoint_with_compass(endpoint, src_compass)
        return _dot_endpoint(endpoint)

    def _render_edge_dst(endpoint: _Endpoint) -> str:
        node_id, _port = endpoint
        if node_id in gateway_nodes:
            return _dot_endpoint_with_compass(endpoint, dst_compass)
        return _dot_endpoint(endpoint)

    for src, dst in sorted(flow_edges):
        src_render = (
            _dot_endpoint_with_compass((src, None), src_compass)
            if src in gateway_nodes
            else f'"{src}"'
        )
        dst_render = (
            _dot_endpoint_with_compass((dst, None), dst_compass)
            if dst in gateway_nodes
            else f'"{dst}"'
        )
        if show_control_flow:
            lines.append(
                f'  {src_render} -> {dst_render} [style=dashed, color="gray65", arrowhead=vee];'
            )
        else:
            lines.append(f"  {src_render} -> {dst_render} [style=invis];")
    for src, dst, label in sorted(loop_control_edges):
        src_render = (
            _dot_endpoint_with_compass((src, None), src_compass)
            if src in gateway_nodes
            else f'"{src}"'
        )
        dst_render = (
            _dot_endpoint_with_compass((dst, None), dst_compass)
            if dst in gateway_nodes
            else f'"{dst}"'
        )
        lines.append(
            f'  {src_render} -> {dst_render} [style=dashed, color="deepskyblue4", '
            f'fontcolor="deepskyblue4", arrowhead=normal, label="{_dot_escape(label)}"];'
        )
    for call_src, call_dst, _label in sorted(
        call_return_edges,
        key=lambda item: (_dot_endpoint(item[0]), _dot_endpoint(item[1]), item[2]),
    ):
        src_render = _render_edge_src(call_src)
        dst_render = _render_edge_dst(call_dst)
        lines.append(
            f'  {src_render} -> {dst_render} [style="dotted", color="seagreen4", arrowhead=normal];'
        )

    for (data_src, data_dst), labels in sorted(edge_labels.items(), key=_edge_sort_key):
        label = ",".join(sorted(labels))
        src_node, _src_port = data_src
        dst_node, _ = data_dst
        _dst_port = data_dst[1]
        cluster_id = block_clusters.get(dst_node)
        src_render = _render_edge_src(data_src)
        dst_render = _render_edge_dst(data_dst)
        if labels == {"call"} and isinstance(cluster_id, str):
            lines.append(
                f"  {src_render} -> {dst_render} "
                f'[label="{_dot_escape(label)}", lhead={cluster_id}, style="dashed", '
                f'color="darkorange3", fontcolor="darkorange4", penwidth=2, arrowhead=vee];'
            )
            continue
        lines.append(f"  {src_render} -> {dst_render};")

    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def run_axon_visualize(
    *,
    axon_file: Path,
    output_path: Path,
    main_module: str | None = None,
    show_control_flow: bool = True,
    direction: str = "top-down",
) -> Path:
    modules = parse_axon_program_from_path(axon_file)
    spec = lower_axon_program_to_synapse_spec(modules, main_module=main_module)
    selected_main = (
        str(main_module) if isinstance(main_module, str) and main_module else modules[-1].name
    )

    block_label_by_block: dict[str, str] = {"main": f"{selected_main}.main"}
    module_names = {m.name for m in modules if isinstance(getattr(m, "name", None), str)}
    for module_name in module_names:
        block_label_by_block[module_name] = module_name
    blocks = spec.get("model", {}).get("blocks", {})
    if isinstance(blocks, Mapping):
        for block_name in blocks:
            if not isinstance(block_name, str):
                continue
            if "." in block_name:
                block_label_by_block.setdefault(block_name, block_name)
                continue
            block_label_by_block.setdefault(block_name, f"{selected_main}.{block_name}")

    block_io_types = extract_block_io_types_from_spec(spec)
    inferred_block_io_types = infer_block_io_types_from_modules(
        spec=spec,
        modules=modules,
        selected_main=selected_main,
    )
    for block_name, io_spec in inferred_block_io_types.items():
        current = block_io_types.setdefault(block_name, {"inputs": {}, "outputs": {}})
        current_inputs = current.setdefault("inputs", {})
        current_outputs = current.setdefault("outputs", {})
        inferred_inputs = io_spec.get("inputs", {}) if isinstance(io_spec, Mapping) else {}
        inferred_outputs = io_spec.get("outputs", {}) if isinstance(io_spec, Mapping) else {}
        if isinstance(inferred_inputs, Mapping):
            for name, type_expr in inferred_inputs.items():
                if (
                    isinstance(name, str)
                    and isinstance(type_expr, str)
                    and name not in current_inputs
                ):
                    current_inputs[name] = type_expr
        if isinstance(inferred_outputs, Mapping):
            for name, type_expr in inferred_outputs.items():
                if (
                    isinstance(name, str)
                    and isinstance(type_expr, str)
                    and name not in current_outputs
                ):
                    current_outputs[name] = type_expr

    dot_text = render_synapse_spec_to_dot(
        spec,
        block_label_by_block=block_label_by_block,
        block_io_types=block_io_types,
        show_control_flow=show_control_flow,
        direction=direction,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(dot_text, encoding="utf-8")
    return output_path


__all__ = ["render_synapse_spec_to_dot", "run_axon_visualize"]
