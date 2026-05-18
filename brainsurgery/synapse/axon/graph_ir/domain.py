from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

from .core import (
    GraphExpr,
    GraphLiteral,
    GraphModule,
    GraphNode,
    GraphOperand,
    GraphOp,
    GraphPath,
    GraphProgram,
    GraphValueRef,
)


class GraphDomainKind(Enum):
    UNKNOWN = "unknown"
    NULL = "null"
    NOT_NULL = "not_null"
    LITERAL = "literal"
    PATH = "path"
    GLOBAL_VALUE = "global_value"


DomainLiteral: TypeAlias = bool | int | float | str


@dataclass(frozen=True)
class GraphDomainFact:
    kind: GraphDomainKind
    value: DomainLiteral | GraphPath | str | None = None


@dataclass(frozen=True)
class GraphDomainAnalysis:
    module_input_facts: dict[str, dict[str, GraphDomainFact]]
    module_local_facts: dict[str, dict[str, GraphDomainFact]]


UNKNOWN_FACT = GraphDomainFact(GraphDomainKind.UNKNOWN)
NULL_FACT = GraphDomainFact(GraphDomainKind.NULL)
NOT_NULL_FACT = GraphDomainFact(GraphDomainKind.NOT_NULL)


def literal_domain_fact(value: DomainLiteral) -> GraphDomainFact:
    return GraphDomainFact(GraphDomainKind.LITERAL, value)


def path_domain_fact(path: GraphPath) -> GraphDomainFact:
    return GraphDomainFact(GraphDomainKind.PATH, path)


def global_value_domain_fact(name: str) -> GraphDomainFact:
    return GraphDomainFact(GraphDomainKind.GLOBAL_VALUE, name)


def infer_main_module_domain_facts(graph: GraphProgram) -> GraphDomainAnalysis:
    modules_by_name = {module.name: module for module in graph.modules}
    global_value_names = {
        module.name
        for module in graph.modules
        if module.is_global_binding and not module.inputs
    }
    reachable = _reachable_modules(graph, modules_by_name)
    callsites = _reachable_calls_by_callee(graph, reachable, modules_by_name)
    module_input_facts: dict[str, dict[str, GraphDomainFact]] = {}
    for module_name in reachable:
        module = modules_by_name[module_name]
        if module.name == graph.main_module:
            module_input_facts[module.name] = {
                value.name: UNKNOWN_FACT
                for value in module.inputs
            }
            continue
        calls = callsites.get(module.name, ())
        facts: dict[str, GraphDomainFact] = {}
        for index, formal in enumerate(module.inputs):
            actual_facts: list[GraphDomainFact] = []
            for call in calls:
                if index >= len(call.inputs):
                    actual_facts = []
                    break
                actual_facts.append(
                    _operand_domain_fact(
                        call.inputs[index],
                        local_facts={},
                        global_value_names=global_value_names,
                    )
                )
            facts[formal.name] = _meet_domain_facts(actual_facts)
        module_input_facts[module.name] = facts
    module_local_facts = {
        module_name: _infer_module_local_facts(
            modules_by_name[module_name],
            input_facts=module_input_facts.get(module_name, {}),
            global_value_names=global_value_names,
        )
        for module_name in reachable
    }
    return GraphDomainAnalysis(
        module_input_facts=module_input_facts,
        module_local_facts=module_local_facts,
    )


def _reachable_modules(
    graph: GraphProgram,
    modules_by_name: dict[str, GraphModule],
) -> set[str]:
    seen: set[str] = set()
    stack = [graph.main_module]
    while stack:
        name = stack.pop()
        if name in seen or name not in modules_by_name:
            continue
        seen.add(name)
        module = modules_by_name[name]
        calls: set[str] = set()
        for node in module.nodes:
            _collect_operand_calls_from_node(node, modules_by_name, calls)
        for output in module.outputs:
            _collect_operand_calls(output, modules_by_name, calls)
        stack.extend(sorted(calls - seen))
    return seen


def _reachable_calls_by_callee(
    graph: GraphProgram,
    reachable: set[str],
    modules_by_name: dict[str, GraphModule],
) -> dict[str, tuple[GraphNode | GraphExpr, ...]]:
    calls: dict[str, list[GraphNode | GraphExpr]] = {}
    for module_name in reachable:
        module = modules_by_name[module_name]
        for node in module.nodes:
            if node.op.name in modules_by_name:
                calls.setdefault(node.op.name, []).append(node)
            for operand in (*node.inputs, *node.attrs.values()):
                _collect_operand_call_nodes(operand, modules_by_name, calls)
        for output in module.outputs:
            _collect_operand_call_nodes(output, modules_by_name, calls)
    return {name: tuple(items) for name, items in calls.items()}


def _collect_operand_calls_from_node(
    node: GraphNode,
    modules_by_name: dict[str, GraphModule],
    calls: set[str],
) -> None:
    if node.op.name in modules_by_name:
        calls.add(node.op.name)
    for operand in (*node.inputs, *node.attrs.values()):
        _collect_operand_calls(operand, modules_by_name, calls)


def _collect_operand_calls(
    operand: GraphOperand,
    modules_by_name: dict[str, GraphModule],
    calls: set[str],
) -> None:
    if not isinstance(operand, GraphExpr):
        return
    if operand.op.name in modules_by_name:
        calls.add(operand.op.name)
    for item in (*operand.inputs, *operand.attrs.values()):
        _collect_operand_calls(item, modules_by_name, calls)


def _collect_operand_call_nodes(
    operand: GraphOperand,
    modules_by_name: dict[str, GraphModule],
    calls: dict[str, list[GraphNode | GraphExpr]],
) -> None:
    if not isinstance(operand, GraphExpr):
        return
    if operand.op.name in modules_by_name:
        calls.setdefault(operand.op.name, []).append(operand)
    for item in (*operand.inputs, *operand.attrs.values()):
        _collect_operand_call_nodes(item, modules_by_name, calls)


def _infer_module_local_facts(
    module: GraphModule,
    *,
    input_facts: dict[str, GraphDomainFact],
    global_value_names: set[str],
) -> dict[str, GraphDomainFact]:
    facts = dict(input_facts)
    for node in module.nodes:
        output_fact = _node_output_domain_fact(
            node,
            local_facts=facts,
            global_value_names=global_value_names,
        )
        if output_fact is None:
            for output in node.outputs:
                facts[output.name] = UNKNOWN_FACT
            continue
        for output in node.outputs:
            facts[output.name] = output_fact
    return facts


def _node_output_domain_fact(
    node: GraphNode,
    *,
    local_facts: dict[str, GraphDomainFact],
    global_value_names: set[str],
) -> GraphDomainFact | None:
    if len(node.outputs) != 1:
        return None
    expr = GraphExpr(
        op=node.op,
        inputs=node.inputs,
        attrs=node.attrs,
        type_expr=node.type_expr,
        dims=node.dims,
    )
    return _expr_domain_fact(
        expr,
        local_facts=local_facts,
        global_value_names=global_value_names,
    )


def _operand_domain_fact(
    operand: GraphOperand,
    *,
    local_facts: dict[str, GraphDomainFact],
    global_value_names: set[str],
) -> GraphDomainFact:
    if isinstance(operand, GraphLiteral):
        if operand.value is None:
            return NULL_FACT
        return literal_domain_fact(operand.value)
    if isinstance(operand, GraphPath):
        return path_domain_fact(operand)
    if isinstance(operand, GraphValueRef):
        if operand.name in local_facts:
            return local_facts[operand.name]
        if operand.name in global_value_names:
            return global_value_domain_fact(operand.name)
        return UNKNOWN_FACT
    return _expr_domain_fact(
        operand,
        local_facts=local_facts,
        global_value_names=global_value_names,
    )


def _expr_domain_fact(
    expr: GraphExpr,
    *,
    local_facts: dict[str, GraphDomainFact],
    global_value_names: set[str],
) -> GraphDomainFact:
    if expr.op.name in {"core.alias", "core.ascribe"} and len(expr.inputs) == 1 and not expr.attrs:
        return _operand_domain_fact(
            expr.inputs[0],
            local_facts=local_facts,
            global_value_names=global_value_names,
        )
    if expr.op.name == "core.select" and len(expr.inputs) == 3 and not expr.attrs:
        cond_fact = _operand_domain_fact(
            expr.inputs[0],
            local_facts=local_facts,
            global_value_names=global_value_names,
        )
        if cond_fact == literal_domain_fact(True):
            return _operand_domain_fact(
                expr.inputs[1],
                local_facts=local_facts,
                global_value_names=global_value_names,
            )
        if cond_fact == literal_domain_fact(False):
            return _operand_domain_fact(
                expr.inputs[2],
                local_facts=local_facts,
                global_value_names=global_value_names,
            )
        return UNKNOWN_FACT
    if expr.op.name.startswith("core.binary.") and len(expr.inputs) == 2 and not expr.attrs:
        return _binary_domain_fact(
            expr.op,
            expr.inputs[0],
            expr.inputs[1],
            local_facts=local_facts,
            global_value_names=global_value_names,
        )
    if expr.op.name in global_value_names and not expr.inputs and not expr.attrs:
        return global_value_domain_fact(expr.op.name)
    return UNKNOWN_FACT


def _binary_domain_fact(
    op: GraphOp,
    left: GraphOperand,
    right: GraphOperand,
    *,
    local_facts: dict[str, GraphDomainFact],
    global_value_names: set[str],
) -> GraphDomainFact:
    operator = op.name.removeprefix("core.binary.")
    left_fact = _operand_domain_fact(left, local_facts=local_facts, global_value_names=global_value_names)
    right_fact = _operand_domain_fact(right, local_facts=local_facts, global_value_names=global_value_names)
    if operator in {"==", "!="}:
        equality = _domain_facts_equal(left_fact, right_fact)
        if equality is not None:
            return literal_domain_fact(equality if operator == "==" else not equality)
    return UNKNOWN_FACT


def _domain_facts_equal(left: GraphDomainFact, right: GraphDomainFact) -> bool | None:
    if left.kind == GraphDomainKind.UNKNOWN or right.kind == GraphDomainKind.UNKNOWN:
        return None
    if left.kind != right.kind:
        if {left.kind, right.kind} == {GraphDomainKind.NULL, GraphDomainKind.NOT_NULL}:
            return False
        if left.kind == GraphDomainKind.NULL and right.kind in {
            GraphDomainKind.LITERAL,
            GraphDomainKind.PATH,
            GraphDomainKind.GLOBAL_VALUE,
        }:
            return False
        if right.kind == GraphDomainKind.NULL and left.kind in {
            GraphDomainKind.LITERAL,
            GraphDomainKind.PATH,
            GraphDomainKind.GLOBAL_VALUE,
        }:
            return False
        return None
    return left.value == right.value


def _meet_domain_facts(facts: list[GraphDomainFact]) -> GraphDomainFact:
    if not facts:
        return UNKNOWN_FACT
    first = facts[0]
    if first.kind == GraphDomainKind.UNKNOWN:
        return UNKNOWN_FACT
    if all(fact == first for fact in facts[1:]):
        return first
    if all(fact.kind != GraphDomainKind.NULL for fact in facts):
        return NOT_NULL_FACT
    return UNKNOWN_FACT


__all__ = [
    "GraphDomainAnalysis",
    "GraphDomainFact",
    "GraphDomainKind",
    "NOT_NULL_FACT",
    "NULL_FACT",
    "UNKNOWN_FACT",
    "global_value_domain_fact",
    "infer_main_module_domain_facts",
    "literal_domain_fact",
    "path_domain_fact",
]
