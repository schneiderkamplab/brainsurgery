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
    GraphValue,
    GraphValueRef,
    graph_operand_type,
)
from ..ast import (
    TypeAny,
    TypeBool,
    TypeDim,
    TypeExpr,
    TypeFloat,
    TypeInt,
    TypeNull,
    TypeOptional,
    TypePath,
    TypeString,
    TypeVar,
)


class GraphDomainKind(Enum):
    UNKNOWN = "unknown"
    NULL = "null"
    NOT_NULL = "not_null"
    LITERAL = "literal"
    INTERVAL = "interval"
    PATH = "path"
    GLOBAL_VALUE = "global_value"


DomainLiteral: TypeAlias = bool | int | float | str
DomainIntervalBound: TypeAlias = int | float | None


@dataclass(frozen=True)
class GraphDomainInterval:
    lower: DomainIntervalBound
    upper: DomainIntervalBound


@dataclass(frozen=True)
class GraphDomainFact:
    kind: GraphDomainKind
    value: DomainLiteral | GraphDomainInterval | GraphPath | str | None = None


@dataclass(frozen=True)
class GraphDomainAnalysis:
    module_input_facts: dict[str, dict[str, GraphDomainFact]]
    module_local_facts: dict[str, dict[str, GraphDomainFact]]
    module_output_facts: dict[str, tuple[GraphDomainFact, ...]]


UNKNOWN_FACT = GraphDomainFact(GraphDomainKind.UNKNOWN)
NULL_FACT = GraphDomainFact(GraphDomainKind.NULL)
NOT_NULL_FACT = GraphDomainFact(GraphDomainKind.NOT_NULL)


def literal_domain_fact(value: DomainLiteral) -> GraphDomainFact:
    return GraphDomainFact(GraphDomainKind.LITERAL, value)


def interval_domain_fact(
    lower: DomainIntervalBound,
    upper: DomainIntervalBound,
) -> GraphDomainFact:
    return GraphDomainFact(GraphDomainKind.INTERVAL, GraphDomainInterval(lower=lower, upper=upper))


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
    module_input_facts: dict[str, dict[str, GraphDomainFact]] = {}
    for module_name in reachable:
        module = modules_by_name[module_name]
        module_input_facts[module.name] = {value.name: UNKNOWN_FACT for value in module.inputs}
    module_local_facts = _infer_reachable_module_local_facts(
        modules_by_name=modules_by_name,
        reachable=reachable,
        module_input_facts=module_input_facts,
        global_value_names=global_value_names,
    )
    module_output_facts = _infer_reachable_module_output_facts(
        modules_by_name=modules_by_name,
        reachable=reachable,
        module_local_facts=module_local_facts,
        global_value_names=global_value_names,
    )
    callsites = _reachable_calls_by_callee(graph, reachable, modules_by_name)
    module_local_conditions = {
        module_name: _module_local_conditions(modules_by_name[module_name])
        for module_name in reachable
    }
    converged = False
    for _ in range(64):
        next_input_facts: dict[str, dict[str, GraphDomainFact]] = {}
        for module_name in reachable:
            module = modules_by_name[module_name]
            if module.name == graph.main_module:
                next_input_facts[module.name] = {
                    value.name: UNKNOWN_FACT
                    for value in module.inputs
                }
                continue
            calls = callsites.get(module.name, ())
            facts: dict[str, GraphDomainFact] = {}
            for index, formal in enumerate(module.inputs):
                actual_facts: list[GraphDomainFact] = []
                for caller_name, call, branch_refinements in calls:
                    if index >= len(call.inputs):
                        actual_facts = []
                        break
                    caller_facts = module_local_facts.get(caller_name, {})
                    caller_conditions = module_local_conditions.get(caller_name, {})
                    for condition, branch_value in branch_refinements:
                        caller_facts = _refine_facts_for_branch(
                            condition,
                            branch_value,
                            caller_facts,
                            caller_conditions,
                        )
                    actual_facts.append(
                        _operand_domain_fact(
                            call.inputs[index],
                            local_facts=caller_facts,
                            local_conditions=caller_conditions,
                            global_value_names=global_value_names,
                        )
                    )
                facts[formal.name] = _meet_domain_facts(actual_facts)
            next_input_facts[module.name] = facts
        next_local_facts = _infer_reachable_module_local_facts(
            modules_by_name=modules_by_name,
            reachable=reachable,
            module_input_facts=next_input_facts,
            global_value_names=global_value_names,
        )
        next_output_facts = _infer_reachable_module_output_facts(
            modules_by_name=modules_by_name,
            reachable=reachable,
            module_local_facts=next_local_facts,
            global_value_names=global_value_names,
        )
        if (
            next_input_facts == module_input_facts
            and next_local_facts == module_local_facts
            and next_output_facts == module_output_facts
        ):
            module_input_facts = next_input_facts
            module_local_facts = next_local_facts
            module_output_facts = next_output_facts
            converged = True
            break
        module_input_facts = next_input_facts
        module_local_facts = next_local_facts
        module_output_facts = next_output_facts
    if not converged:
        raise RuntimeError("graph domain analysis did not converge after 64 iterations")
    analysis = GraphDomainAnalysis(
        module_input_facts=module_input_facts,
        module_local_facts=module_local_facts,
        module_output_facts=module_output_facts,
    )
    validate_graph_domain_analysis(graph, analysis, reachable=reachable)
    return analysis


def validate_graph_domain_analysis(
    graph: GraphProgram,
    analysis: GraphDomainAnalysis,
    *,
    reachable: set[str] | None = None,
) -> None:
    modules_by_name = {module.name: module for module in graph.modules}
    reachable_modules = reachable if reachable is not None else _reachable_modules(graph, modules_by_name)
    _validate_analysis_module_set("input", analysis.module_input_facts, reachable_modules)
    _validate_analysis_module_set("local", analysis.module_local_facts, reachable_modules)
    _validate_analysis_module_set("output", analysis.module_output_facts, reachable_modules)
    for module_name in sorted(reachable_modules):
        module = modules_by_name[module_name]
        input_facts = analysis.module_input_facts[module_name]
        expected_input_names = {value.name for value in module.inputs}
        if set(input_facts) != expected_input_names:
            raise ValueError(
                f"domain analysis for module {module_name!r} has input fact keys "
                f"{sorted(input_facts)!r}; expected {sorted(expected_input_names)!r}"
            )
        for value in module.inputs:
            _validate_fact(
                input_facts[value.name],
                expected_type=value.type_expr,
                allow_null=value.optional,
                context=f"{module_name}.{value.name}",
            )
        local_facts = analysis.module_local_facts[module_name]
        declared_values = _module_declared_values(module)
        missing = sorted(name for name in declared_values if name not in local_facts)
        if missing:
            raise ValueError(f"domain analysis for module {module_name!r} is missing local facts for {missing!r}")
        for name, value in declared_values.items():
            _validate_fact(
                local_facts[name],
                expected_type=value.type_expr,
                allow_null=value.optional,
                context=f"{module_name}.{name}",
            )
        output_facts = analysis.module_output_facts[module_name]
        if len(output_facts) != len(module.outputs):
            raise ValueError(
                f"domain analysis for module {module_name!r} has {len(output_facts)} output facts; "
                f"expected {len(module.outputs)}"
            )
        for index, (fact, output) in enumerate(zip(output_facts, module.outputs, strict=True)):
            _validate_fact(
                fact,
                expected_type=graph_operand_type(output),
                allow_null=False,
                context=f"{module_name}.output[{index}]",
            )


def _validate_analysis_module_set(
    label: str,
    facts_by_module: dict[str, object],
    reachable: set[str],
) -> None:
    found = set(facts_by_module)
    if found != reachable:
        raise ValueError(
            f"domain analysis {label} modules are {sorted(found)!r}; expected reachable modules "
            f"{sorted(reachable)!r}"
        )


def _module_declared_values(module: GraphModule) -> dict[str, GraphValue]:
    values = {value.name: value for value in module.inputs}
    for node in module.nodes:
        for output in node.outputs:
            values[output.name] = output
    return values


def _validate_fact(
    fact: GraphDomainFact,
    *,
    expected_type: TypeExpr | None,
    allow_null: bool,
    context: str,
) -> None:
    if fact.kind in {GraphDomainKind.UNKNOWN, GraphDomainKind.NULL, GraphDomainKind.NOT_NULL}:
        if fact.value is not None:
            raise ValueError(f"domain fact for {context} has unexpected payload {fact.value!r}")
    elif fact.kind == GraphDomainKind.LITERAL:
        if type(fact.value) not in {bool, int, float, str}:
            raise ValueError(f"domain literal fact for {context} has invalid value {fact.value!r}")
    elif fact.kind == GraphDomainKind.INTERVAL:
        if not isinstance(fact.value, GraphDomainInterval):
            raise ValueError(f"domain interval fact for {context} has invalid value {fact.value!r}")
        _validate_interval(fact.value, context)
    elif fact.kind == GraphDomainKind.PATH:
        if not isinstance(fact.value, GraphPath):
            raise ValueError(f"domain path fact for {context} has invalid value {fact.value!r}")
    elif fact.kind == GraphDomainKind.GLOBAL_VALUE:
        if type(fact.value) is not str:
            raise ValueError(f"domain global-value fact for {context} has invalid value {fact.value!r}")
    else:
        raise ValueError(f"unknown domain fact kind {fact.kind!r} for {context}")
    if expected_type is not None and not _fact_compatible_with_type(fact, expected_type, allow_null=allow_null):
        raise ValueError(f"domain fact {fact!r} for {context} is incompatible with type {expected_type!r}")


def _validate_interval(interval: GraphDomainInterval, context: str) -> None:
    for label, value in (("lower", interval.lower), ("upper", interval.upper)):
        if value is not None and type(value) not in {int, float}:
            raise ValueError(f"domain interval {label} bound for {context} is invalid: {value!r}")
    if interval.lower is not None and interval.upper is not None and interval.lower > interval.upper:
        raise ValueError(f"domain interval for {context} has lower bound greater than upper bound: {interval!r}")


def _fact_compatible_with_type(fact: GraphDomainFact, type_expr: TypeExpr, *, allow_null: bool) -> bool:
    if isinstance(type_expr, (TypeAny, TypeVar)):
        return True
    if isinstance(type_expr, TypeOptional):
        if fact.kind == GraphDomainKind.NULL:
            return True
        return _fact_compatible_with_type(fact, type_expr.inner, allow_null=allow_null)
    if fact.kind == GraphDomainKind.UNKNOWN:
        return True
    if fact.kind == GraphDomainKind.NULL:
        return allow_null or isinstance(type_expr, TypeNull)
    if fact.kind == GraphDomainKind.NOT_NULL:
        return not isinstance(type_expr, TypeNull)
    if fact.kind == GraphDomainKind.PATH:
        return isinstance(type_expr, TypePath)
    if fact.kind == GraphDomainKind.GLOBAL_VALUE:
        return True
    if fact.kind == GraphDomainKind.INTERVAL:
        return isinstance(type_expr, (TypeDim, TypeFloat, TypeInt))
    if fact.kind != GraphDomainKind.LITERAL:
        return True
    value = fact.value
    if type(value) is bool:
        return isinstance(type_expr, TypeBool)
    if type(value) is int:
        return isinstance(type_expr, (TypeDim, TypeFloat, TypeInt))
    if type(value) is float:
        return isinstance(type_expr, TypeFloat)
    if type(value) is str:
        return isinstance(type_expr, TypeString)
    return False


def _infer_reachable_module_local_facts(
    *,
    modules_by_name: dict[str, GraphModule],
    reachable: set[str],
    module_input_facts: dict[str, dict[str, GraphDomainFact]],
    global_value_names: set[str],
) -> dict[str, dict[str, GraphDomainFact]]:
    return {
        module_name: _infer_module_local_facts(
            modules_by_name[module_name],
            input_facts=module_input_facts.get(module_name, {}),
            global_value_names=global_value_names,
        )
        for module_name in reachable
    }


def _infer_reachable_module_output_facts(
    *,
    modules_by_name: dict[str, GraphModule],
    reachable: set[str],
    module_local_facts: dict[str, dict[str, GraphDomainFact]],
    global_value_names: set[str],
) -> dict[str, tuple[GraphDomainFact, ...]]:
    return {
        module_name: tuple(
            _operand_domain_fact(
                output,
                local_facts=module_local_facts.get(module_name, {}),
                global_value_names=global_value_names,
            )
            for output in modules_by_name[module_name].outputs
        )
        for module_name in reachable
    }


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
) -> dict[str, tuple[tuple[str, GraphNode | GraphExpr, tuple[tuple[GraphOperand, bool], ...]], ...]]:
    calls: dict[str, list[tuple[str, GraphNode | GraphExpr, tuple[tuple[GraphOperand, bool], ...]]]] = {}
    for module_name in reachable:
        module = modules_by_name[module_name]
        for node in module.nodes:
            if node.op.name in modules_by_name:
                calls.setdefault(node.op.name, []).append((module_name, node, ()))
            if node.op.name == "core.select" and len(node.inputs) == 3 and not node.attrs:
                _collect_operand_call_nodes(node.inputs[0], module_name, modules_by_name, calls, ())
                _collect_operand_call_nodes(
                    node.inputs[1],
                    module_name,
                    modules_by_name,
                    calls,
                    ((node.inputs[0], True),),
                )
                _collect_operand_call_nodes(
                    node.inputs[2],
                    module_name,
                    modules_by_name,
                    calls,
                    ((node.inputs[0], False),),
                )
            else:
                for operand in (*node.inputs, *node.attrs.values()):
                    _collect_operand_call_nodes(operand, module_name, modules_by_name, calls, ())
        for output in module.outputs:
            _collect_operand_call_nodes(output, module_name, modules_by_name, calls, ())
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
    caller_name: str,
    modules_by_name: dict[str, GraphModule],
    calls: dict[str, list[tuple[str, GraphNode | GraphExpr, tuple[tuple[GraphOperand, bool], ...]]]],
    branch_refinements: tuple[tuple[GraphOperand, bool], ...],
) -> None:
    if not isinstance(operand, GraphExpr):
        return
    if operand.op.name in modules_by_name:
        calls.setdefault(operand.op.name, []).append((caller_name, operand, branch_refinements))
    if operand.op.name == "core.select" and len(operand.inputs) == 3 and not operand.attrs:
        _collect_operand_call_nodes(
            operand.inputs[0],
            caller_name,
            modules_by_name,
            calls,
            branch_refinements,
        )
        _collect_operand_call_nodes(
            operand.inputs[1],
            caller_name,
            modules_by_name,
            calls,
            (*branch_refinements, (operand.inputs[0], True)),
        )
        _collect_operand_call_nodes(
            operand.inputs[2],
            caller_name,
            modules_by_name,
            calls,
            (*branch_refinements, (operand.inputs[0], False)),
        )
        return
    for item in (*operand.inputs, *operand.attrs.values()):
        _collect_operand_call_nodes(item, caller_name, modules_by_name, calls, branch_refinements)


def _module_local_conditions(module: GraphModule) -> dict[str, GraphExpr]:
    conditions: dict[str, GraphExpr] = {}
    for node in module.nodes:
        if (
            len(node.outputs) == 1
            and node.op.name.startswith("core.binary.")
            and len(node.inputs) == 2
            and not node.attrs
        ):
            conditions[node.outputs[0].name] = GraphExpr(
                op=node.op,
                inputs=node.inputs,
                attrs=node.attrs,
                type_expr=node.type_expr,
                dims=node.dims,
            )
    return conditions


def _infer_module_local_facts(
    module: GraphModule,
    *,
    input_facts: dict[str, GraphDomainFact],
    global_value_names: set[str],
) -> dict[str, GraphDomainFact]:
    facts = dict(input_facts)
    conditions: dict[str, GraphExpr] = {}
    for node in module.nodes:
        output_fact = _node_output_domain_fact(
            node,
            local_facts=facts,
            local_conditions=conditions,
            global_value_names=global_value_names,
        )
        if output_fact is None:
            for output in node.outputs:
                facts[output.name] = UNKNOWN_FACT
            continue
        for output in node.outputs:
            facts[output.name] = output_fact
        if (
            len(node.outputs) == 1
            and node.op.name.startswith("core.binary.")
            and len(node.inputs) == 2
            and not node.attrs
        ):
            conditions[node.outputs[0].name] = GraphExpr(
                op=node.op,
                inputs=node.inputs,
                attrs=node.attrs,
                type_expr=node.type_expr,
                dims=node.dims,
            )
    return facts


def _node_output_domain_fact(
    node: GraphNode,
    *,
    local_facts: dict[str, GraphDomainFact],
    local_conditions: dict[str, GraphExpr],
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
        local_conditions=local_conditions,
        global_value_names=global_value_names,
    )


def _operand_domain_fact(
    operand: GraphOperand,
    *,
    local_facts: dict[str, GraphDomainFact],
    local_conditions: dict[str, GraphExpr] | None = None,
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
        local_conditions=local_conditions or {},
        global_value_names=global_value_names,
    )


def _expr_domain_fact(
    expr: GraphExpr,
    *,
    local_facts: dict[str, GraphDomainFact],
    local_conditions: dict[str, GraphExpr],
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
            local_conditions=local_conditions,
            global_value_names=global_value_names,
        )
        if cond_fact == literal_domain_fact(True):
            return _operand_domain_fact(
                expr.inputs[1],
                local_facts=_refine_facts_for_branch(expr.inputs[0], True, local_facts, local_conditions),
                local_conditions=local_conditions,
                global_value_names=global_value_names,
            )
        if cond_fact == literal_domain_fact(False):
            return _operand_domain_fact(
                expr.inputs[2],
                local_facts=_refine_facts_for_branch(expr.inputs[0], False, local_facts, local_conditions),
                local_conditions=local_conditions,
                global_value_names=global_value_names,
            )
        true_fact = _operand_domain_fact(
            expr.inputs[1],
            local_facts=_refine_facts_for_branch(expr.inputs[0], True, local_facts, local_conditions),
            local_conditions=local_conditions,
            global_value_names=global_value_names,
        )
        false_fact = _operand_domain_fact(
            expr.inputs[2],
            local_facts=_refine_facts_for_branch(expr.inputs[0], False, local_facts, local_conditions),
            local_conditions=local_conditions,
            global_value_names=global_value_names,
        )
        joined = _meet_domain_facts([true_fact, false_fact])
        if joined.kind != GraphDomainKind.UNKNOWN:
            return joined
        return UNKNOWN_FACT
    if expr.op.name.startswith("core.binary.") and len(expr.inputs) == 2 and not expr.attrs:
        return _binary_domain_fact(
            expr.op,
            expr.inputs[0],
            expr.inputs[1],
            local_facts=local_facts,
            local_conditions=local_conditions,
            global_value_names=global_value_names,
        )
    if expr.op.name in global_value_names and not expr.inputs and not expr.attrs:
        return global_value_domain_fact(expr.op.name)
    return UNKNOWN_FACT


def _refine_facts_for_branch(
    condition: GraphOperand,
    branch_value: bool,
    local_facts: dict[str, GraphDomainFact],
    local_conditions: dict[str, GraphExpr],
) -> dict[str, GraphDomainFact]:
    if not isinstance(condition, GraphValueRef):
        if (
            isinstance(condition, GraphExpr)
            and condition.op.name.startswith("core.binary.")
            and len(condition.inputs) == 2
            and not condition.attrs
        ):
            return _refine_facts_for_binary_condition(
                condition.op.name.removeprefix("core.binary."),
                condition.inputs[0],
                condition.inputs[1],
                branch_value=branch_value,
                local_facts=local_facts,
            )
        return local_facts
    if condition.name in local_conditions:
        return _refine_facts_for_branch(
            local_conditions[condition.name],
            branch_value,
            local_facts,
            local_conditions,
        )
    fact = local_facts.get(condition.name)
    if fact is None or fact.kind != GraphDomainKind.LITERAL or type(fact.value) is not bool:
        return local_facts
    return local_facts


def _refine_facts_for_binary_condition(
    operator: str,
    left: GraphOperand,
    right: GraphOperand,
    *,
    branch_value: bool,
    local_facts: dict[str, GraphDomainFact],
) -> dict[str, GraphDomainFact]:
    if operator not in {"==", "!="}:
        return local_facts
    equality_branch = branch_value if operator == "==" else not branch_value
    left_ref = left if isinstance(left, GraphValueRef) else None
    right_ref = right if isinstance(right, GraphValueRef) else None
    left_literal = _literal_or_null_fact(right)
    right_literal = _literal_or_null_fact(left)
    refined = dict(local_facts)
    if left_ref is not None and left_literal is not None:
        refined[left_ref.name] = left_literal if equality_branch else _negated_equality_fact(left_literal)
    if right_ref is not None and right_literal is not None:
        refined[right_ref.name] = right_literal if equality_branch else _negated_equality_fact(right_literal)
    return refined


def _literal_or_null_fact(operand: GraphOperand) -> GraphDomainFact | None:
    if isinstance(operand, GraphLiteral):
        if operand.value is None:
            return NULL_FACT
        return literal_domain_fact(operand.value)
    if isinstance(operand, GraphPath):
        return path_domain_fact(operand)
    return None


def _negated_equality_fact(fact: GraphDomainFact) -> GraphDomainFact:
    if fact.kind == GraphDomainKind.NULL:
        return NOT_NULL_FACT
    return UNKNOWN_FACT


def _binary_domain_fact(
    op: GraphOp,
    left: GraphOperand,
    right: GraphOperand,
    *,
    local_facts: dict[str, GraphDomainFact],
    local_conditions: dict[str, GraphExpr],
    global_value_names: set[str],
) -> GraphDomainFact:
    operator = op.name.removeprefix("core.binary.")
    left_fact = _operand_domain_fact(
        left,
        local_facts=local_facts,
        local_conditions=local_conditions,
        global_value_names=global_value_names,
    )
    right_fact = _operand_domain_fact(
        right,
        local_facts=local_facts,
        local_conditions=local_conditions,
        global_value_names=global_value_names,
    )
    if operator in {"==", "!="}:
        equality = _domain_facts_equal(left_fact, right_fact)
        if equality is not None:
            return literal_domain_fact(equality if operator == "==" else not equality)
    if operator in {"+", "-", "*", "/"}:
        return _numeric_interval_binary_fact(operator, left_fact, right_fact)
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


def _fact_interval(fact: GraphDomainFact) -> GraphDomainInterval | None:
    if fact.kind == GraphDomainKind.INTERVAL and isinstance(fact.value, GraphDomainInterval):
        return fact.value
    if fact.kind == GraphDomainKind.LITERAL and type(fact.value) in {int, float}:
        return GraphDomainInterval(lower=fact.value, upper=fact.value)
    return None


def _interval_fact_from_bounds(
    lower: DomainIntervalBound,
    upper: DomainIntervalBound,
) -> GraphDomainFact:
    if lower is not None and upper is not None and lower == upper:
        return literal_domain_fact(lower)
    return interval_domain_fact(lower, upper)


def _numeric_interval_binary_fact(
    operator: str,
    left: GraphDomainFact,
    right: GraphDomainFact,
) -> GraphDomainFact:
    left_interval = _fact_interval(left)
    right_interval = _fact_interval(right)
    if left_interval is None or right_interval is None:
        return UNKNOWN_FACT
    if operator == "+":
        lower = (
            None
            if left_interval.lower is None or right_interval.lower is None
            else left_interval.lower + right_interval.lower
        )
        upper = (
            None
            if left_interval.upper is None or right_interval.upper is None
            else left_interval.upper + right_interval.upper
        )
        return _interval_fact_from_bounds(lower, upper)
    if operator == "-":
        lower = (
            None
            if left_interval.lower is None or right_interval.upper is None
            else left_interval.lower - right_interval.upper
        )
        upper = (
            None
            if left_interval.upper is None or right_interval.lower is None
            else left_interval.upper - right_interval.lower
        )
        return _interval_fact_from_bounds(lower, upper)
    if operator == "*":
        if (
            left_interval.lower is None
            or left_interval.upper is None
            or right_interval.lower is None
            or right_interval.upper is None
        ):
            return UNKNOWN_FACT
        candidates = (
            left_interval.lower * right_interval.lower,
            left_interval.lower * right_interval.upper,
            left_interval.upper * right_interval.lower,
            left_interval.upper * right_interval.upper,
        )
        return _interval_fact_from_bounds(min(candidates), max(candidates))
    if operator == "/":
        if (
            left_interval.lower is None
            or left_interval.upper is None
            or right_interval.lower is None
            or right_interval.upper is None
            or right_interval.lower <= 0 <= right_interval.upper
        ):
            return UNKNOWN_FACT
        candidates = (
            left_interval.lower / right_interval.lower,
            left_interval.lower / right_interval.upper,
            left_interval.upper / right_interval.lower,
            left_interval.upper / right_interval.upper,
        )
        return _interval_fact_from_bounds(min(candidates), max(candidates))
    return UNKNOWN_FACT


def _meet_domain_facts(facts: list[GraphDomainFact]) -> GraphDomainFact:
    if not facts:
        return UNKNOWN_FACT
    first = facts[0]
    if first.kind == GraphDomainKind.UNKNOWN:
        return UNKNOWN_FACT
    if all(fact == first for fact in facts[1:]):
        return first
    intervals = [_fact_interval(fact) for fact in facts]
    if all(interval is not None for interval in intervals):
        concrete = [interval for interval in intervals if interval is not None]
        lower = None if any(interval.lower is None for interval in concrete) else min(
            interval.lower for interval in concrete if interval.lower is not None
        )
        upper = None if any(interval.upper is None for interval in concrete) else max(
            interval.upper for interval in concrete if interval.upper is not None
        )
        return _interval_fact_from_bounds(lower, upper)
    if all(fact.kind != GraphDomainKind.NULL for fact in facts):
        return NOT_NULL_FACT
    return UNKNOWN_FACT


__all__ = [
    "GraphDomainAnalysis",
    "GraphDomainFact",
    "GraphDomainInterval",
    "GraphDomainKind",
    "NOT_NULL_FACT",
    "NULL_FACT",
    "UNKNOWN_FACT",
    "global_value_domain_fact",
    "infer_main_module_domain_facts",
    "interval_domain_fact",
    "literal_domain_fact",
    "path_domain_fact",
    "validate_graph_domain_analysis",
]
