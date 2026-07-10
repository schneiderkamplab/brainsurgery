from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from ..ast import TypeAny, TypeExpr, render_type
from .core import (
    GraphExpr,
    GraphLiteral,
    GraphModule,
    GraphOperand,
    GraphPath,
    GraphProgram,
    GraphValue,
    GraphValueRef,
    validate_graph_program,
)


@dataclass(frozen=True)
class _Endpoint:
    node: str
    port: str | None = None


@dataclass(frozen=True)
class _RefUse:
    name: str
    port: str
    label: str


@dataclass(frozen=True)
class _CallUse:
    target: str
    label: str


def _html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _dot_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _dot_id(*parts: str) -> str:
    raw = "__".join(str(part) for part in parts if str(part))
    sanitized = re.sub(r"[^0-9A-Za-z_]", "_", raw)
    if not sanitized:
        sanitized = "node"
    if sanitized[0].isdigit():
        sanitized = f"n_{sanitized}"
    return f"n_{sanitized}"


def _port_id(name: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z_]", "_", name)
    if not sanitized:
        sanitized = "p"
    if sanitized[0].isdigit():
        sanitized = f"p_{sanitized}"
    return sanitized


def _endpoint(endpoint: _Endpoint) -> str:
    if endpoint.port:
        return f'"{endpoint.node}":"{endpoint.port}"'
    return f'"{endpoint.node}"'


def _type_label(type_expr: TypeExpr | None) -> str:
    return render_type(type_expr or TypeAny())


def _path_label(path: GraphPath) -> str:
    prefix = "@@" if path.absolute else "@"
    body = ".".join(path.parts)
    if not body:
        return prefix
    if any(char in body for char in "{}'\" "):
        return f"{prefix}'{body}'"
    return f"{prefix}{body}"


def _literal_label(literal: GraphLiteral) -> str:
    value = literal.value
    if value is True:
        text = "true"
    elif value is False:
        text = "false"
    elif value is None:
        text = "null"
    else:
        text = repr(value)
    return f"{text} :: {_type_label(literal.type_expr)}"


def _operand_label(operand: GraphOperand, *, depth: int = 0) -> str:
    if isinstance(operand, GraphValueRef):
        return f"{operand.name} :: {_type_label(operand.type_expr)}"
    if isinstance(operand, GraphLiteral):
        return _literal_label(operand)
    if isinstance(operand, GraphPath):
        return f"{_path_label(operand)} :: Path"
    if isinstance(operand, GraphExpr):
        return _expr_label(operand, depth=depth)
    raise TypeError(f"unsupported graph operand: {type(operand).__name__}")


def _expr_label(expr: GraphExpr, *, depth: int = 0) -> str:
    if depth >= 2:
        return f"{expr.op.name}(...) :: {_type_label(expr.type_expr)}"
    args = [_operand_label(item, depth=depth + 1) for item in expr.inputs]
    kwargs = [
        f"{name}={_operand_label(value, depth=depth + 1)}"
        for name, value in sorted(expr.attrs.items())
    ]
    joined = ", ".join([*args, *kwargs])
    return f"{expr.op.name}({joined}) :: {_type_label(expr.type_expr)}"


def _collect_ref_uses(
    operand: GraphOperand,
    *,
    port: str,
    label: str,
    out: list[_RefUse],
) -> None:
    if isinstance(operand, GraphValueRef):
        out.append(_RefUse(name=operand.name, port=port, label=label))
        return
    if isinstance(operand, GraphExpr):
        for index, item in enumerate(operand.inputs):
            _collect_ref_uses(
                item,
                port=port,
                label=f"{label}.{index}",
                out=out,
            )
        for attr_name, value in sorted(operand.attrs.items()):
            _collect_ref_uses(
                value,
                port=port,
                label=f"{label}.{attr_name}",
                out=out,
            )


def _collect_call_uses(
    operand: GraphOperand,
    *,
    module_names: set[str],
    label: str,
    out: list[_CallUse],
) -> None:
    if not isinstance(operand, GraphExpr):
        return
    if operand.op.name in module_names:
        out.append(_CallUse(target=operand.op.name, label=label))
    for index, item in enumerate(operand.inputs):
        _collect_call_uses(
            item,
            module_names=module_names,
            label=f"{label}.{index}",
            out=out,
        )
    for attr_name, value in sorted(operand.attrs.items()):
        _collect_call_uses(
            value,
            module_names=module_names,
            label=f"{label}.{attr_name}",
            out=out,
        )


def _slot_cell(
    *,
    port: str,
    title: str,
    detail: str | None,
    bgcolor: str,
    expr: str | None = None,
) -> str:
    lines = [f'<FONT POINT-SIZE="8"><B>{_html_escape(title)}</B></FONT>']
    if detail:
        lines.append(f'<FONT POINT-SIZE="7" COLOR="gray35">{_html_escape(detail)}</FONT>')
    if expr:
        lines.append(f'<FONT POINT-SIZE="7" COLOR="gray45">{_html_escape(expr)}</FONT>')
    body = '<BR ALIGN="CENTER"/>'.join(lines)
    return f'<TD PORT="{port}" BGCOLOR="{bgcolor}" ALIGN="CENTER">{body}</TD>'


def _table_node(
    *,
    node_id: str,
    rows: Iterable[str],
    color: str = "gray65",
) -> str:
    table = (
        f'<TABLE BORDER="1" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4" COLOR="{color}">'
        + "".join(rows)
        + "</TABLE>"
    )
    return f'    "{node_id}" [shape=plain, label=<{table}>];'


def _gateway_node(
    *,
    node_id: str,
    label: str,
    values: tuple[GraphValue, ...] | tuple[GraphOperand, ...],
    output_names: tuple[str, ...] = (),
    input_side: bool,
) -> str:
    bgcolor = "aliceblue" if input_side else "honeydew"
    cells: list[str] = []
    for index, value in enumerate(values):
        if isinstance(value, GraphValue):
            name = value.name
            detail = _type_label(value.type_expr)
        else:
            name = output_names[index] if index < len(output_names) else f"out{index}"
            detail = _operand_label(value)
        cells.append(
            _slot_cell(
                port=_port_id(name),
                title=name,
                detail=detail,
                bgcolor=bgcolor,
            )
        )
    if not cells:
        cells.append(f'<TD BGCOLOR="{bgcolor}"></TD>')
    rows = [
        f'<TR><TD BGCOLOR="{bgcolor}" ALIGN="CENTER"><FONT POINT-SIZE="9"><B>{_html_escape(label)}</B></FONT></TD>{"" .join(cells)}</TR>'
    ]
    return _table_node(node_id=node_id, rows=rows, color="gray50")


def _node_table(module_name: str, index: int, node: Any) -> str:
    node_id = _node_id(module_name, index)
    input_cells: list[str] = []
    for input_index, operand in enumerate(node.inputs):
        input_cells.append(
            _slot_cell(
                port=_input_port(input_index),
                title=f"arg{input_index}",
                detail=_operand_label(operand),
                bgcolor="lemonchiffon",
            )
        )
    for attr_name, operand in sorted(node.attrs.items()):
        input_cells.append(
            _slot_cell(
                port=_attr_port(attr_name),
                title=attr_name,
                detail=_operand_label(operand),
                bgcolor="thistle1",
            )
        )
    if not input_cells:
        input_cells.append('<TD BGCOLOR="lemonchiffon"></TD>')

    output_cells = [
        _slot_cell(
            port=_output_port(value.name),
            title=value.name,
            detail=_type_label(value.type_expr),
            bgcolor="honeydew",
        )
        for value in node.outputs
    ]
    if not output_cells:
        output_cells.append('<TD BGCOLOR="honeydew"></TD>')

    span = max(len(input_cells), len(output_cells), 1)
    while len(input_cells) < span:
        input_cells.append('<TD BGCOLOR="lemonchiffon"></TD>')
    while len(output_cells) < span:
        output_cells.append('<TD BGCOLOR="honeydew"></TD>')

    op_label = (
        f'<FONT POINT-SIZE="10"><B>{_html_escape(node.op.name)}</B></FONT>'
        f'<BR ALIGN="LEFT"/><FONT POINT-SIZE="7" COLOR="gray35">{_html_escape(node.id)}</FONT>'
        f'<BR ALIGN="LEFT"/><FONT POINT-SIZE="7" COLOR="gray45">{_html_escape(_type_label(node.type_expr))}</FONT>'
    )
    rows = [
        f'<TR><TD BGCOLOR="gray90"><FONT POINT-SIZE="7"><B>IN</B></FONT></TD>{"".join(input_cells)}</TR>',
        f'<TR><TD BGCOLOR="gray90"><FONT POINT-SIZE="7"><B>OP</B></FONT></TD><TD COLSPAN="{span}" BGCOLOR="white" ALIGN="LEFT">{op_label}</TD></TR>',
        f'<TR><TD BGCOLOR="gray90"><FONT POINT-SIZE="7"><B>OUT</B></FONT></TD>{"".join(output_cells)}</TR>',
    ]
    return _table_node(node_id=node_id, rows=rows)


def _node_id(module_name: str, index: int) -> str:
    return _dot_id("op", module_name, f"{index:04d}")


def _input_node_id(module_name: str) -> str:
    return _dot_id("inputs", module_name)


def _output_node_id(module_name: str) -> str:
    return _dot_id("outputs", module_name)


def _module_cluster_id(module_name: str) -> str:
    return "cluster_" + re.sub(r"[^0-9A-Za-z_]", "_", module_name or "module")


def _input_port(index: int) -> str:
    return _port_id(f"in_{index}")


def _attr_port(name: str) -> str:
    return _port_id(f"attr_{name}")


def _output_port(name: str) -> str:
    return _port_id(f"out_{name}")


def _module_input_sources(module: GraphModule) -> dict[str, _Endpoint]:
    node_id = _input_node_id(module.name)
    return {value.name: _Endpoint(node_id, _port_id(value.name)) for value in module.inputs}


def _append_edge(
    edges: dict[tuple[_Endpoint, _Endpoint], set[str]],
    *,
    src: _Endpoint,
    dst: _Endpoint,
    label: str,
) -> None:
    edges.setdefault((src, dst), set()).add(label)


def _render_module(
    module: GraphModule,
    *,
    module_names: set[str],
    edges: dict[tuple[_Endpoint, _Endpoint], set[str]],
    call_edges: set[tuple[str, str, str]],
) -> list[str]:
    lines: list[str] = []
    cluster_id = _module_cluster_id(module.name)
    lines.append(f"  subgraph {cluster_id} {{")
    lines.append('    style="rounded,filled";')
    lines.append('    color="gray70";')
    fill = "lightgoldenrod1" if module.name else "cornsilk"
    lines.append(f'    fillcolor="{fill}";')
    label_parts = [f"module {module.name}"]
    if module.return_type_expr is not None:
        label_parts.append(f"returns {_type_label(module.return_type_expr)}")
    if module.constraints:
        label_parts.append(f"constraints {len(module.constraints)}")
    module_label = "\n".join(label_parts)
    lines.append(f'    label="{_dot_escape(module_label)}";')

    input_id = _input_node_id(module.name)
    output_id = _output_node_id(module.name)
    lines.append(
        _gateway_node(
            node_id=input_id,
            label="INPUT",
            values=module.inputs,
            input_side=True,
        )
    )

    sources = _module_input_sources(module)
    for index, node in enumerate(module.nodes):
        node_id = _node_id(module.name, index)
        lines.append(_node_table(module.name, index, node))
        if node.op.name in module_names:
            call_edges.add((node_id, node.op.name, node.op.name))
        for input_index, operand in enumerate(node.inputs):
            call_uses: list[_CallUse] = []
            _collect_call_uses(
                operand,
                module_names=module_names,
                label=f"arg{input_index}",
                out=call_uses,
            )
            for call_use in call_uses:
                call_edges.add((node_id, call_use.target, call_use.label))
            uses: list[_RefUse] = []
            _collect_ref_uses(
                operand,
                port=_input_port(input_index),
                label=f"arg{input_index}",
                out=uses,
            )
            for use in uses:
                src = sources.get(use.name)
                if src is not None:
                    _append_edge(edges, src=src, dst=_Endpoint(node_id, use.port), label=use.label)
        for attr_name, operand in sorted(node.attrs.items()):
            call_uses = []
            _collect_call_uses(
                operand,
                module_names=module_names,
                label=attr_name,
                out=call_uses,
            )
            for call_use in call_uses:
                call_edges.add((node_id, call_use.target, call_use.label))
            uses = []
            _collect_ref_uses(
                operand,
                port=_attr_port(attr_name),
                label=attr_name,
                out=uses,
            )
            for use in uses:
                src = sources.get(use.name)
                if src is not None:
                    _append_edge(edges, src=src, dst=_Endpoint(node_id, use.port), label=use.label)
        for value in node.outputs:
            sources[value.name] = _Endpoint(node_id, _output_port(value.name))

    lines.append(
        _gateway_node(
            node_id=output_id,
            label="OUTPUT",
            values=module.outputs,
            output_names=module.output_names,
            input_side=False,
        )
    )
    for index, output in enumerate(module.outputs):
        uses: list[_RefUse] = []
        output_name = module.output_names[index] if index < len(module.output_names) else f"out{index}"
        _collect_ref_uses(
            output,
            port=_port_id(output_name),
            label=output_name,
            out=uses,
        )
        for use in uses:
            src = sources.get(use.name)
            if src is not None:
                _append_edge(edges, src=src, dst=_Endpoint(output_id, use.port), label=use.label)

    lines.append("  }")
    return lines


def render_graph_program_to_dot(
    graph: GraphProgram,
    *,
    direction: str = "top-down",
    show_data_labels: bool = True,
    show_control_flow: bool = True,
) -> str:
    """Render typed Graph IR as Graphviz DOT without serialized compatibility formats."""

    validate_graph_program(graph)
    direction_value = direction.strip().lower()
    rankdirs = {
        "top-down": "TB",
        "bottom-up": "BT",
        "left-right": "LR",
        "right-left": "RL",
    }
    if direction_value not in rankdirs:
        raise ValueError("direction must be one of: top-down, bottom-up, left-right, right-left")

    lines = [
        "digraph GraphIR {",
        f"  rankdir={rankdirs[direction_value]};",
        "  compound=true;",
        "  newrank=true;",
        '  graph [fontname="Helvetica"];',
        '  node [fontname="Helvetica"];',
        '  edge [fontname="Helvetica"];',
        f'  label="Graph IR main: {_dot_escape(graph.main_module)}";',
        "  labelloc=t;",
    ]
    edges: dict[tuple[_Endpoint, _Endpoint], set[str]] = {}
    call_edges: set[tuple[str, str, str]] = set()
    module_names = {module.name for module in graph.modules}
    for module in graph.modules:
        lines.extend(
            _render_module(
                module,
                module_names=module_names,
                edges=edges,
                call_edges=call_edges,
            )
        )

    if show_control_flow:
        for module in graph.modules:
            previous: str | None = _input_node_id(module.name)
            for index, _node in enumerate(module.nodes):
                current = _node_id(module.name, index)
                if previous is not None:
                    lines.append(
                        f'  "{previous}" -> "{current}" [style=dashed, color="gray70", arrowhead=vee];'
                    )
                previous = current
            if previous is not None:
                lines.append(
                    f'  "{previous}" -> "{_output_node_id(module.name)}" [style=dashed, color="gray70", arrowhead=vee];'
                )

    for src_id, target, label in sorted(call_edges):
        cluster_id = _module_cluster_id(target)
        target_input_id = _input_node_id(target)
        lines.append(
            f'  "{src_id}" -> "{target_input_id}" [label="{_dot_escape(label)}", '
            'style="dotted", color="darkorange3", fontcolor="darkorange4", '
            "lhead="
            f"{cluster_id}, arrowhead=vee];"
        )

    for (src, dst), labels in sorted(
        edges.items(),
        key=lambda item: (
            item[0][0].node,
            item[0][0].port or "",
            item[0][1].node,
            item[0][1].port or "",
        ),
    ):
        attrs = ['color="black"']
        if show_data_labels and labels:
            attrs.append(f'label="{_dot_escape(",".join(sorted(labels)))}"')
        lines.append(f"  {_endpoint(src)} -> {_endpoint(dst)} [{', '.join(attrs)}];")

    lines.append("}")
    lines.append("")
    return "\n".join(lines)


__all__ = ["render_graph_program_to_dot"]
