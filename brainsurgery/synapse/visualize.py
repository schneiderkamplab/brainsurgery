from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .axon import lower_axon_program_to_synapse_spec, parse_axon_program_from_path

_WORD_TOKEN_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")


def _dot_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _dot_id(*parts: str) -> str:
    merged = "::".join(parts)
    sanitized = re.sub(r"[^0-9A-Za-z_:.]", "_", merged)
    return f"n_{sanitized}"


def _cluster_id(block_name: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z_]", "_", block_name)
    return f"cluster_{sanitized}"


def _append_edge(
    edge_labels: dict[tuple[str, str], set[str]],
    *,
    src: str,
    dst: str,
    label: str,
) -> None:
    key = (src, dst)
    bucket = edge_labels.setdefault(key, set())
    bucket.add(label)


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


def _node_input_vars(node_spec: Mapping[str, Any], *, known_vars: set[str]) -> list[str]:
    refs: set[str] = set()
    raw_args = node_spec.get("_args")
    if raw_args is not None:
        _collect_var_refs(raw_args, known_vars=known_vars, out=refs)
    for key, value in node_spec.items():
        if key.startswith("_") or key in {"when", "graph"}:
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


def _walk_graph(
    graph: list[Any],
    *,
    block_name: str,
    scope: str,
    lines: list[str],
    var_sources: dict[str, str],
    edge_labels: dict[tuple[str, str], set[str]],
) -> None:
    for index, item in enumerate(graph):
        if not isinstance(item, Mapping) or len(item) != 1:
            continue
        node_name, node_spec = next(iter(item.items()))
        if not isinstance(node_name, str) or not isinstance(node_spec, Mapping):
            continue
        node_id = _dot_id("op", block_name, scope, f"{index:04d}_{node_name}")
        op = node_spec.get("_op")
        op_name = op if isinstance(op, str) else "group"
        node_label = f"{node_name}\\n{op_name}"
        lines.append(
            f'  "{node_id}" [shape=ellipse, style="filled", fillcolor="white", '
            f'label="{_dot_escape(node_label)}"];'
        )

        known_vars = set(var_sources.keys())
        for var_name in _node_input_vars(node_spec, known_vars=known_vars):
            src_id = var_sources.get(var_name)
            if src_id is None:
                continue
            _append_edge(edge_labels, src=src_id, dst=node_id, label=var_name)

        if op_name == "call":
            block_target = node_spec.get("_target")
            if isinstance(block_target, str):
                target_id = _dot_id("block", block_target)
                _append_edge(edge_labels, src=node_id, dst=target_id, label="calls")

        for out_var in _node_output_vars(node_spec):
            var_sources[out_var] = node_id

        nested_graph = node_spec.get("graph")
        if isinstance(nested_graph, list):
            _walk_graph(
                nested_graph,
                block_name=block_name,
                scope=f"{scope}.{node_name}",
                lines=lines,
                var_sources=var_sources,
                edge_labels=edge_labels,
            )

        body_graph = node_spec.get("_body")
        if isinstance(body_graph, list):
            _walk_graph(
                body_graph,
                block_name=block_name,
                scope=f"{scope}.{node_name}._body",
                lines=lines,
                var_sources=var_sources,
                edge_labels=edge_labels,
            )


def _render_block(
    *,
    block_name: str,
    block_spec: Mapping[str, Any],
    lines: list[str],
    edge_labels: dict[tuple[str, str], set[str]],
    block_clusters: dict[str, str],
) -> None:
    block_id = _dot_id("block", block_name)
    cluster_id = _cluster_id(block_name)
    block_clusters[block_id] = cluster_id
    lines.append(f"  subgraph {cluster_id} {{")
    lines.append('    style="rounded,filled";')
    lines.append('    color="gray70";')
    lines.append('    fillcolor="cornsilk";')
    lines.append(f'    label="{_dot_escape("block " + block_name)}";')
    lines.append(
        f'    "{block_id}" [shape=box3d, style="filled,bold", fillcolor="lightgoldenrod1", '
        f'label="{_dot_escape("block " + block_name)}"];'
    )

    var_sources: dict[str, str] = {}
    inputs = block_spec.get("inputs")
    if isinstance(inputs, Mapping):
        for input_name in inputs:
            if not isinstance(input_name, str):
                continue
            input_id = _dot_id("input", block_name, input_name)
            lines.append(
                f'    "{input_id}" [shape=oval, style="filled", fillcolor="aliceblue", '
                f'label="{_dot_escape("in " + input_name)}"];'
            )
            _append_edge(edge_labels, src=block_id, dst=input_id, label=input_name)
            var_sources[input_name] = input_id

    graph = block_spec.get("graph")
    if isinstance(graph, list):
        _walk_graph(
            graph,
            block_name=block_name,
            scope="graph",
            lines=lines,
            var_sources=var_sources,
            edge_labels=edge_labels,
        )

    outputs = block_spec.get("outputs")
    if isinstance(outputs, Mapping):
        for output_name, source_name in outputs.items():
            if not isinstance(output_name, str):
                continue
            output_id = _dot_id("output", block_name, output_name)
            lines.append(
                f'    "{output_id}" [shape=oval, style="filled", fillcolor="honeydew", '
                f'label="{_dot_escape("out " + output_name)}"];'
            )
            _append_edge(edge_labels, src=block_id, dst=output_id, label=output_name)
            if isinstance(source_name, str):
                src_id = var_sources.get(source_name)
                if src_id is not None:
                    _append_edge(edge_labels, src=src_id, dst=output_id, label=source_name)
    lines.append("  }")


def render_synapse_spec_to_dot(spec: Mapping[str, Any]) -> str:
    model = spec.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("spec.model must be a mapping")
    graph = model.get("graph")
    if not isinstance(graph, list):
        raise ValueError("model.graph must be a list")

    lines: list[str] = [
        "digraph synapse {",
        "  rankdir=TB;",
        "  compound=true;",
        "  newrank=true;",
        '  graph [fontname="Helvetica"];',
        '  node [fontname="Helvetica"];',
        '  edge [fontname="Helvetica"];',
    ]
    edge_labels: dict[tuple[str, str], set[str]] = {}
    block_clusters: dict[str, str] = {}

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
    )

    blocks = model.get("blocks")
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
            )

    for (src, dst), labels in sorted(edge_labels.items()):
        label = ",".join(sorted(labels))
        cluster_id = block_clusters.get(dst)
        if labels == {"calls"} and isinstance(cluster_id, str):
            lines.append(
                f'  "{src}" -> "{dst}" [label="{_dot_escape(label)}", lhead={cluster_id}];'
            )
            continue
        lines.append(f'  "{src}" -> "{dst}" [label="{_dot_escape(label)}"];')

    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def run_axon_visualize(
    *,
    axon_file: Path,
    output_path: Path,
    main_module: str | None = None,
) -> Path:
    modules = parse_axon_program_from_path(axon_file)
    spec = lower_axon_program_to_synapse_spec(modules, main_module=main_module)
    dot_text = render_synapse_spec_to_dot(spec)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(dot_text, encoding="utf-8")
    return output_path


__all__ = ["render_synapse_spec_to_dot", "run_axon_visualize"]
