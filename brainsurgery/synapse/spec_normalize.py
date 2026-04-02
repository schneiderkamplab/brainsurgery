from __future__ import annotations

from copy import deepcopy
from typing import Any

from .axon import parse_expression_to_runtime_value
from .ops import get_op_lowering_signature

_EXPR_KINDS = {"int", "number", "bool", "dim"}


def _parse_expr_value(raw: Any, *, ctx: str) -> Any:
    if not isinstance(raw, str):
        return raw
    if raw == "":
        return {"_expr": "string", "value": ""}
    try:
        return parse_expression_to_runtime_value(raw)
    except ValueError as exc:
        raise ValueError(f"{ctx}: invalid expression value {raw!r}") from exc


def _normalize_kind_value(value: Any, kind: str, *, ctx: str) -> Any:
    if kind in _EXPR_KINDS:
        return _parse_expr_value(value, ctx=ctx)
    if kind == "list_dim":
        if isinstance(value, list):
            return [_parse_expr_value(item, ctx=f"{ctx}[{idx}]") for idx, item in enumerate(value)]
        return _parse_expr_value(value, ctx=ctx)
    return value


def _normalize_node(node_spec: dict[str, Any], *, ctx: str) -> None:
    op_name = node_spec.get("_op")
    if isinstance(op_name, str):
        if op_name in {"_ir_const", "_ir_expr"} and "value" in node_spec:
            node_spec["value"] = _parse_expr_value(node_spec["value"], ctx=f"{ctx}.value")
        if op_name == "for":
            for field_name in ("_from", "_to", "_step"):
                if field_name in node_spec:
                    node_spec[field_name] = _parse_expr_value(
                        node_spec[field_name], ctx=f"{ctx}.{field_name}"
                    )
        signature = get_op_lowering_signature(op_name)
        if isinstance(signature, dict):
            kwarg_kinds = signature.get("kwarg_kinds")
            if isinstance(kwarg_kinds, dict):
                for key, kind in kwarg_kinds.items():
                    if key in node_spec and isinstance(kind, str):
                        node_spec[key] = _normalize_kind_value(
                            node_spec[key],
                            kind,
                            ctx=f"{ctx}.{key}",
                        )

    nested_graph = node_spec.get("graph")
    if isinstance(nested_graph, list):
        _normalize_graph(nested_graph, graph_ctx=f"{ctx}.graph")
    body_graph = node_spec.get("_body")
    if isinstance(body_graph, list):
        _normalize_graph(body_graph, graph_ctx=f"{ctx}._body")
    then_graph = node_spec.get("_then")
    if isinstance(then_graph, list):
        _normalize_graph(then_graph, graph_ctx=f"{ctx}._then")
    else_graph = node_spec.get("_else")
    if isinstance(else_graph, list):
        _normalize_graph(else_graph, graph_ctx=f"{ctx}._else")


def _normalize_graph(graph: list[Any], *, graph_ctx: str) -> None:
    for index, item in enumerate(graph):
        if not isinstance(item, dict) or len(item) != 1:
            continue
        _, node_spec = next(iter(item.items()))
        if not isinstance(node_spec, dict):
            continue
        _normalize_node(node_spec, ctx=f"{graph_ctx}[{index}]")


def normalize_synapse_spec_expressions(spec: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(spec)
    model = normalized.get("model")
    if not isinstance(model, dict):
        return normalized
    graph = model.get("graph")
    if isinstance(graph, list):
        _normalize_graph(graph, graph_ctx="model.graph")
    blocks = model.get("blocks")
    if isinstance(blocks, dict):
        for block_name, block_spec in blocks.items():
            if not isinstance(block_spec, dict):
                continue
            block_graph = block_spec.get("graph")
            if isinstance(block_graph, list):
                _normalize_graph(block_graph, graph_ctx=f"model.blocks[{block_name!r}].graph")
    return normalized


__all__ = ["normalize_synapse_spec_expressions"]
