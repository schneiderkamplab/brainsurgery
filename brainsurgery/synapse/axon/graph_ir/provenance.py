from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .core import (
    GraphExpr,
    GraphLiteral,
    GraphModule,
    GraphNode,
    GraphOperand,
    GraphPath,
    GraphProgram,
    GraphValueRef,
)


ProvenanceKind = Literal["unknown", "input", "literal", "path", "global", "op", "tuple"]


@dataclass(frozen=True)
class GraphProvenance:
    kind: ProvenanceKind
    name: str | None = None
    value: object | None = None
    op: str | None = None
    args: tuple["GraphProvenance", ...] = ()


@dataclass(frozen=True)
class GraphSdpaGqaFact:
    q: str
    k: str
    v: str
    additive_mask: str
    keep: str
    default_scale: bool = True


@dataclass(frozen=True)
class GraphRopeApplyFactorsFact:
    x: GraphProvenance
    sin: GraphProvenance
    cos: GraphProvenance
    interleaved: bool = False


@dataclass(frozen=True)
class GraphDerivedProvenanceFact:
    kind: Literal[
        "additive_mask_from_keep",
        "nonempty_keep_rows",
        "sdpa_gqa",
        "rope_apply_factors",
    ]
    value: GraphProvenance | GraphSdpaGqaFact | GraphRopeApplyFactorsFact


@dataclass(frozen=True)
class GraphProvenanceAnalysis:
    module_summary_provenance: dict[str, tuple[GraphProvenance, ...]]
    module_input_provenance: dict[str, dict[str, GraphProvenance]]
    module_local_provenance: dict[str, dict[str, GraphProvenance]]
    module_output_provenance: dict[str, tuple[GraphProvenance, ...]]
    module_input_facts: dict[str, dict[str, tuple[GraphDerivedProvenanceFact, ...]]]
    module_local_facts: dict[str, dict[str, tuple[GraphDerivedProvenanceFact, ...]]]
    module_output_facts: dict[str, tuple[tuple[GraphDerivedProvenanceFact, ...], ...]]


UNKNOWN_PROVENANCE = GraphProvenance("unknown")


def infer_graph_provenance(program: GraphProgram) -> GraphProvenanceAnalysis:
    modules_by_name = {module.name: module for module in program.modules}
    global_names = {
        module.name
        for module in program.modules
        if module.is_global_binding and not module.inputs
    }
    summaries: dict[str, tuple[GraphProvenance, ...]] = {}
    visiting: set[str] = set()

    def module_summary(module_name: str) -> tuple[GraphProvenance, ...]:
        if module_name in summaries:
            return summaries[module_name]
        module = modules_by_name.get(module_name)
        if module is None or module_name in visiting:
            return ()
        visiting.add(module_name)
        input_prov = {
            value.name: GraphProvenance("input", name=value.name)
            for value in module.inputs
        }
        local = _infer_module_local_provenance(
            module,
            input_provenance=input_prov,
            modules_by_name=modules_by_name,
            module_summary=module_summary,
            global_names=global_names,
        )
        outputs = tuple(
            _operand_provenance(
                output,
                local_provenance=local,
                modules_by_name=modules_by_name,
                module_summary=module_summary,
                global_names=global_names,
            )
            for output in module.outputs
        )
        visiting.remove(module_name)
        summaries[module_name] = outputs
        return outputs

    for module in program.modules:
        module_summary(module.name)

    reachable = _reachable_modules(program, modules_by_name)
    input_prov: dict[str, dict[str, GraphProvenance]] = {
        name: {
            value.name: (
                GraphProvenance("input", name=value.name)
                if name == program.main_module
                else UNKNOWN_PROVENANCE
            )
            for value in modules_by_name[name].inputs
        }
        for name in reachable
    }
    local_prov = _infer_reachable_local_provenance(
        reachable,
        modules_by_name=modules_by_name,
        module_input_provenance=input_prov,
        module_summary=module_summary,
        global_names=global_names,
    )
    output_prov = _infer_reachable_output_provenance(
        reachable,
        modules_by_name=modules_by_name,
        module_local_provenance=local_prov,
        module_summary=module_summary,
        global_names=global_names,
    )
    callsites = _reachable_call_actuals(
        reachable,
        modules_by_name=modules_by_name,
        module_local_provenance=local_prov,
        module_summary=module_summary,
        global_names=global_names,
    )
    for _ in range(64):
        next_input = {}
        for module_name in reachable:
            module = modules_by_name[module_name]
            if module_name == program.main_module:
                next_input[module_name] = {
                    value.name: GraphProvenance("input", name=value.name)
                    for value in module.inputs
                }
                continue
            by_formal: dict[str, GraphProvenance] = {}
            calls = callsites.get(module_name, ())
            for index, formal in enumerate(module.inputs):
                actuals = [actuals[index] for actuals in calls if index < len(actuals)]
                by_formal[formal.name] = _meet_provenance(actuals)
            next_input[module_name] = by_formal
        next_local = _infer_reachable_local_provenance(
            reachable,
            modules_by_name=modules_by_name,
            module_input_provenance=next_input,
            module_summary=module_summary,
            global_names=global_names,
        )
        next_output = _infer_reachable_output_provenance(
            reachable,
            modules_by_name=modules_by_name,
            module_local_provenance=next_local,
            module_summary=module_summary,
            global_names=global_names,
        )
        next_calls = _reachable_call_actuals(
            reachable,
            modules_by_name=modules_by_name,
            module_local_provenance=next_local,
            module_summary=module_summary,
            global_names=global_names,
        )
        if (
            next_input == input_prov
            and next_local == local_prov
            and next_output == output_prov
        ):
            input_prov = next_input
            local_prov = next_local
            output_prov = next_output
            break
        input_prov = next_input
        local_prov = next_local
        output_prov = next_output
        callsites = next_calls
    else:
        raise RuntimeError("graph provenance analysis did not converge after 64 iterations")

    return GraphProvenanceAnalysis(
        module_summary_provenance=dict(summaries),
        module_input_provenance=input_prov,
        module_local_provenance=local_prov,
        module_output_provenance=output_prov,
        module_input_facts={
            module: {
                name: _derived_facts(prov)
                for name, prov in values.items()
            }
            for module, values in input_prov.items()
        },
        module_local_facts={
            module: {
                name: _derived_facts(prov)
                for name, prov in values.items()
            }
            for module, values in local_prov.items()
        },
        module_output_facts={
            module: tuple(_derived_facts(prov) for prov in values)
            for module, values in output_prov.items()
        },
    )


def graph_provenance_facts(
    provenance: GraphProvenance,
) -> tuple[GraphDerivedProvenanceFact, ...]:
    return _derived_facts(provenance)


def format_graph_provenance(provenance: GraphProvenance, *, max_depth: int = 4) -> str:
    if max_depth <= 0:
        return "..."
    if provenance.kind == "unknown":
        return "unknown"
    if provenance.kind == "input":
        return provenance.name or "input"
    if provenance.kind == "global":
        return provenance.name or "global"
    if provenance.kind == "literal":
        return repr(provenance.value)
    if provenance.kind == "path":
        return str(provenance.value)
    if provenance.kind == "tuple":
        return "(" + ", ".join(
            format_graph_provenance(item, max_depth=max_depth - 1)
            for item in provenance.args
        ) + ")"
    args = ", ".join(
        format_graph_provenance(item, max_depth=max_depth - 1)
        for item in provenance.args
    )
    return f"{provenance.op}({args})"


def format_derived_provenance_fact(fact: GraphDerivedProvenanceFact) -> str:
    if fact.kind == "additive_mask_from_keep":
        assert isinstance(fact.value, GraphProvenance)
        return "additive_mask_from_keep=" + format_graph_provenance(fact.value)
    if fact.kind == "nonempty_keep_rows":
        assert isinstance(fact.value, GraphProvenance)
        return "nonempty_keep_rows=" + format_graph_provenance(fact.value)
    if fact.kind == "sdpa_gqa":
        assert isinstance(fact.value, GraphSdpaGqaFact)
        scale = "default" if fact.value.default_scale else "explicit"
        return (
            "sdpa_gqa="
            f"q:{fact.value.q},k:{fact.value.k},v:{fact.value.v},"
            f"additive_mask:{fact.value.additive_mask},keep:{fact.value.keep},scale:{scale}"
        )
    if fact.kind == "rope_apply_factors":
        assert isinstance(fact.value, GraphRopeApplyFactorsFact)
        mode = "interleaved" if fact.value.interleaved else "noninterleaved"
        return (
            "rope_apply_factors="
            f"x:{format_graph_provenance(fact.value.x)},"
            f"sin:{format_graph_provenance(fact.value.sin)},"
            f"cos:{format_graph_provenance(fact.value.cos)},mode:{mode}"
        )
    return f"{fact.kind}={format_graph_provenance(fact.value)}"


def _reachable_modules(
    program: GraphProgram,
    modules_by_name: dict[str, GraphModule],
) -> set[str]:
    seen: set[str] = set()
    stack = [program.main_module]
    while stack:
        name = stack.pop()
        if name in seen or name not in modules_by_name:
            continue
        seen.add(name)
        module = modules_by_name[name]
        for called in _module_called_names(module, modules_by_name):
            if called not in seen:
                stack.append(called)
    return seen


def _module_called_names(
    module: GraphModule,
    modules_by_name: dict[str, GraphModule],
) -> set[str]:
    out: set[str] = set()
    for node in module.nodes:
        if node.op.name in modules_by_name:
            out.add(node.op.name)
        if node.op.name == "core.repeat":
            callee = _repeat_callee(node)
            if callee is not None and callee in modules_by_name:
                out.add(callee)
        for operand in (*node.inputs, *node.attrs.values()):
            _operand_called_names(operand, modules_by_name, out)
    for output in module.outputs:
        _operand_called_names(output, modules_by_name, out)
    return out


def _operand_called_names(
    operand: GraphOperand,
    modules_by_name: dict[str, GraphModule],
    out: set[str],
) -> None:
    if not isinstance(operand, GraphExpr):
        return
    if operand.op.name in modules_by_name:
        out.add(operand.op.name)
    for item in (*operand.inputs, *operand.attrs.values()):
        _operand_called_names(item, modules_by_name, out)


def _infer_reachable_local_provenance(
    reachable: set[str],
    *,
    modules_by_name: dict[str, GraphModule],
    module_input_provenance: dict[str, dict[str, GraphProvenance]],
    module_summary,
    global_names: set[str],
) -> dict[str, dict[str, GraphProvenance]]:
    return {
        name: _infer_module_local_provenance(
            modules_by_name[name],
            input_provenance=module_input_provenance.get(name, {}),
            modules_by_name=modules_by_name,
            module_summary=module_summary,
            global_names=global_names,
        )
        for name in reachable
    }


def _infer_reachable_output_provenance(
    reachable: set[str],
    *,
    modules_by_name: dict[str, GraphModule],
    module_local_provenance: dict[str, dict[str, GraphProvenance]],
    module_summary,
    global_names: set[str],
) -> dict[str, tuple[GraphProvenance, ...]]:
    return {
        name: tuple(
            _operand_provenance(
                output,
                local_provenance=module_local_provenance.get(name, {}),
                modules_by_name=modules_by_name,
                module_summary=module_summary,
                global_names=global_names,
            )
            for output in modules_by_name[name].outputs
        )
        for name in reachable
    }


def _infer_module_local_provenance(
    module: GraphModule,
    *,
    input_provenance: dict[str, GraphProvenance],
    modules_by_name: dict[str, GraphModule],
    module_summary,
    global_names: set[str],
) -> dict[str, GraphProvenance]:
    local = dict(input_provenance)
    for node in module.nodes:
        outputs = _node_output_provenance(
            node,
            local_provenance=local,
            modules_by_name=modules_by_name,
            module_summary=module_summary,
            global_names=global_names,
        )
        if len(outputs) != len(node.outputs):
            outputs = tuple(UNKNOWN_PROVENANCE for _ in node.outputs)
        for value, prov in zip(node.outputs, outputs, strict=True):
            local[value.name] = prov
    return local


def _node_output_provenance(
    node: GraphNode,
    *,
    local_provenance: dict[str, GraphProvenance],
    modules_by_name: dict[str, GraphModule],
    module_summary,
    global_names: set[str],
) -> tuple[GraphProvenance, ...]:
    if node.op.name == "core.repeat":
        return tuple(
            GraphProvenance(
                "op",
                op="core.repeat",
                args=tuple(
                    _operand_provenance(
                        item,
                        local_provenance=local_provenance,
                        modules_by_name=modules_by_name,
                        module_summary=module_summary,
                        global_names=global_names,
                    )
                    for item in node.inputs
                ),
            )
            for _ in node.outputs
        )
    if node.op.name in modules_by_name:
        actuals = tuple(
            _operand_provenance(
                item,
                local_provenance=local_provenance,
                modules_by_name=modules_by_name,
                module_summary=module_summary,
                global_names=global_names,
            )
            for item in node.inputs
        )
        return _instantiate_summary(
            modules_by_name[node.op.name],
            module_summary(node.op.name),
            actuals,
        )
    if len(node.outputs) == 1:
        return (
            GraphProvenance(
                "op",
                op=node.op.name,
                args=tuple(
                    _operand_provenance(
                        item,
                        local_provenance=local_provenance,
                        modules_by_name=modules_by_name,
                        module_summary=module_summary,
                        global_names=global_names,
                    )
                    for item in (*node.inputs, *node.attrs.values())
                ),
            ),
        )
    return tuple(
        GraphProvenance(
            "op",
            op=f"{node.op.name}[{index}]",
            args=tuple(
                _operand_provenance(
                    item,
                    local_provenance=local_provenance,
                    modules_by_name=modules_by_name,
                    module_summary=module_summary,
                    global_names=global_names,
                )
                for item in (*node.inputs, *node.attrs.values())
            ),
        )
        for index, _ in enumerate(node.outputs)
    )


def _operand_provenance(
    operand: GraphOperand,
    *,
    local_provenance: dict[str, GraphProvenance],
    modules_by_name: dict[str, GraphModule],
    module_summary,
    global_names: set[str],
) -> GraphProvenance:
    if isinstance(operand, GraphValueRef):
        if operand.name in local_provenance:
            return local_provenance[operand.name]
        if operand.name in global_names:
            return GraphProvenance("global", name=operand.name)
        return GraphProvenance("input", name=operand.name)
    if isinstance(operand, GraphLiteral):
        return GraphProvenance("literal", value=operand.value)
    if isinstance(operand, GraphPath):
        prefix = "@@" if operand.absolute else "@"
        return GraphProvenance("path", value=prefix + ".".join(operand.parts))
    if isinstance(operand, GraphExpr):
        actuals = tuple(
            _operand_provenance(
                item,
                local_provenance=local_provenance,
                modules_by_name=modules_by_name,
                module_summary=module_summary,
                global_names=global_names,
            )
            for item in (*operand.inputs, *operand.attrs.values())
        )
        if operand.op.name in modules_by_name:
            summary = module_summary(operand.op.name)
            outputs = _instantiate_summary(modules_by_name[operand.op.name], summary, actuals)
            if len(outputs) == 1:
                return outputs[0]
        return GraphProvenance("op", op=operand.op.name, args=actuals)
    return UNKNOWN_PROVENANCE


def _instantiate_summary(
    module: GraphModule,
    summary: tuple[GraphProvenance, ...],
    actuals: tuple[GraphProvenance, ...],
) -> tuple[GraphProvenance, ...]:
    if not summary:
        return tuple(UNKNOWN_PROVENANCE for _ in module.outputs)
    subst = {
        formal.name: actual
        for formal, actual in zip(module.inputs, actuals, strict=False)
    }
    return tuple(_subst_provenance(item, subst) for item in summary)


def _subst_provenance(
    provenance: GraphProvenance,
    subst: dict[str, GraphProvenance],
) -> GraphProvenance:
    if provenance.kind == "input" and provenance.name in subst:
        return subst[provenance.name]
    if not provenance.args:
        return provenance
    return GraphProvenance(
        provenance.kind,
        name=provenance.name,
        value=provenance.value,
        op=provenance.op,
        args=tuple(_subst_provenance(arg, subst) for arg in provenance.args),
    )


def _reachable_call_actuals(
    reachable: set[str],
    *,
    modules_by_name: dict[str, GraphModule],
    module_local_provenance: dict[str, dict[str, GraphProvenance]],
    module_summary,
    global_names: set[str],
) -> dict[str, tuple[tuple[GraphProvenance, ...], ...]]:
    calls: dict[str, list[tuple[GraphProvenance, ...]]] = {}
    for module_name in reachable:
        module = modules_by_name[module_name]
        local = module_local_provenance.get(module_name, {})
        for node in module.nodes:
            if node.op.name in modules_by_name:
                calls.setdefault(node.op.name, []).append(
                    tuple(
                        _operand_provenance(
                            item,
                            local_provenance=local,
                            modules_by_name=modules_by_name,
                            module_summary=module_summary,
                            global_names=global_names,
                        )
                        for item in node.inputs
                    )
                )
            if node.op.name == "core.repeat":
                callee = _repeat_callee(node)
                actuals = _repeat_actuals(
                    node,
                    local_provenance=local,
                    modules_by_name=modules_by_name,
                    module_summary=module_summary,
                    global_names=global_names,
                )
                if callee is not None and callee in modules_by_name and actuals is not None:
                    calls.setdefault(callee, []).append(actuals)
            for operand in (*node.inputs, *node.attrs.values()):
                _collect_operand_call_actuals(
                    operand,
                    calls,
                    local_provenance=local,
                    modules_by_name=modules_by_name,
                    module_summary=module_summary,
                    global_names=global_names,
                )
        for output in module.outputs:
            _collect_operand_call_actuals(
                output,
                calls,
                local_provenance=local,
                modules_by_name=modules_by_name,
                module_summary=module_summary,
                global_names=global_names,
            )
    return {key: tuple(value) for key, value in calls.items()}


def _collect_operand_call_actuals(
    operand: GraphOperand,
    calls: dict[str, list[tuple[GraphProvenance, ...]]],
    *,
    local_provenance: dict[str, GraphProvenance],
    modules_by_name: dict[str, GraphModule],
    module_summary,
    global_names: set[str],
) -> None:
    if not isinstance(operand, GraphExpr):
        return
    if operand.op.name in modules_by_name:
        calls.setdefault(operand.op.name, []).append(
            tuple(
                _operand_provenance(
                    item,
                    local_provenance=local_provenance,
                    modules_by_name=modules_by_name,
                    module_summary=module_summary,
                    global_names=global_names,
                )
                for item in operand.inputs
            )
        )
    for item in (*operand.inputs, *operand.attrs.values()):
        _collect_operand_call_actuals(
            item,
            calls,
            local_provenance=local_provenance,
            modules_by_name=modules_by_name,
            module_summary=module_summary,
            global_names=global_names,
        )


def _repeat_callee(node: GraphNode) -> str | None:
    callee = node.attrs.get("callee")
    return callee.value if isinstance(callee, GraphLiteral) and isinstance(callee.value, str) else None


def _repeat_actuals(
    node: GraphNode,
    *,
    local_provenance: dict[str, GraphProvenance],
    modules_by_name: dict[str, GraphModule],
    module_summary,
    global_names: set[str],
) -> tuple[GraphProvenance, ...] | None:
    arg_count = node.attrs.get("arg_count")
    if not isinstance(arg_count, GraphLiteral) or type(arg_count.value) is not int:
        return None
    actuals: list[GraphProvenance] = []
    for index in range(arg_count.value):
        role = node.attrs.get(f"arg_{index}")
        if not isinstance(role, GraphLiteral) or not isinstance(role.value, str):
            return None
        text = role.value
        if text == "iter":
            actuals.append(GraphProvenance("op", op="core.repeat.iter"))
        elif text.startswith("input:"):
            input_index = int(text.removeprefix("input:"))
            if input_index >= len(node.inputs):
                return None
            actuals.append(
                _operand_provenance(
                    node.inputs[input_index],
                    local_provenance=local_provenance,
                    modules_by_name=modules_by_name,
                    module_summary=module_summary,
                    global_names=global_names,
                )
            )
        elif text.startswith("carry:"):
            carry_index = int(text.removeprefix("carry:"))
            input_index = 3 + carry_index
            if input_index >= len(node.inputs):
                return None
            actuals.append(
                GraphProvenance(
                    "op",
                    op="core.repeat.carry",
                    args=(
                        _operand_provenance(
                            node.inputs[input_index],
                            local_provenance=local_provenance,
                            modules_by_name=modules_by_name,
                            module_summary=module_summary,
                            global_names=global_names,
                        ),
                    ),
                )
            )
        else:
            return None
    return tuple(actuals)


def _meet_provenance(values: list[GraphProvenance]) -> GraphProvenance:
    if not values:
        return UNKNOWN_PROVENANCE
    first = values[0]
    if all(value == first for value in values):
        return first
    return UNKNOWN_PROVENANCE


def _derived_facts(provenance: GraphProvenance) -> tuple[GraphDerivedProvenanceFact, ...]:
    facts: list[GraphDerivedProvenanceFact] = []
    keep = _additive_mask_keep(provenance)
    if keep is not None:
        facts.append(GraphDerivedProvenanceFact("additive_mask_from_keep", keep))
    if _is_nonempty_keep_rows(provenance):
        facts.append(GraphDerivedProvenanceFact("nonempty_keep_rows", provenance))
    sdpa = _sdpa_gqa_fact(provenance)
    if sdpa is not None:
        facts.append(GraphDerivedProvenanceFact("sdpa_gqa", sdpa))
    rope = _rope_apply_factors_fact(provenance)
    if rope is not None:
        facts.append(GraphDerivedProvenanceFact("rope_apply_factors", rope))
    return tuple(facts)


def _additive_mask_keep(provenance: GraphProvenance) -> GraphProvenance | None:
    if provenance.kind != "op" or provenance.op not in {"_where", "core.where"}:
        return None
    if len(provenance.args) != 3:
        return None
    keep, yes, no = provenance.args
    if not _is_zero_tensor_provenance(yes):
        return None
    if not _is_dtype_min_tensor_for(no, yes):
        return None
    return keep


def _is_zero_tensor_provenance(provenance: GraphProvenance) -> bool:
    return provenance.kind == "op" and provenance.op in {"_zeros", "_zeros_like"}


def _is_dtype_min_tensor_for(
    candidate: GraphProvenance,
    zero: GraphProvenance,
) -> bool:
    if candidate.kind != "op" or candidate.op != "_fill" or len(candidate.args) < 2:
        return False
    base, value = candidate.args[:2]
    if not (
        base.kind == "op"
        and base.op == "_empty_like"
        and len(base.args) >= 1
        and base.args[0] == zero
    ):
        return False
    return (
        value.kind == "op"
        and value.op == "_dtype_value"
        and len(value.args) >= 2
        and value.args[0] == zero
        and value.args[1] == GraphProvenance("literal", value="min")
    )


def _is_nonempty_keep_rows(provenance: GraphProvenance) -> bool:
    """Return True for keep masks whose construction proves one true per query row.

    This intentionally accepts only the unpadded causal-mask structure. A masked
    padding branch can contain fully masked rows, so select/and-with-padding
    forms are rejected unless future analyses prove stronger facts.
    """
    if provenance.kind != "op" or provenance.op != "_expand" or len(provenance.args) < 1:
        return False
    reshaped = provenance.args[0]
    if reshaped.kind != "op" or reshaped.op != "_reshape" or len(reshaped.args) < 1:
        return False
    keep_2d = reshaped.args[0]
    if keep_2d.kind != "op" or keep_2d.op not in {"_le", "_and"}:
        return False
    if keep_2d.op == "_le":
        return _looks_like_causal_upper_bound(keep_2d)
    return any(_looks_like_causal_upper_bound(arg) for arg in keep_2d.args)


def _looks_like_causal_upper_bound(provenance: GraphProvenance) -> bool:
    if provenance.kind != "op" or provenance.op != "_le" or len(provenance.args) != 2:
        return False
    left, right = provenance.args
    return _contains_op(left, "_arange") and _contains_op(right, "_arange")


def _contains_op(provenance: GraphProvenance, op_name: str, *, depth: int = 12) -> bool:
    if depth <= 0:
        return False
    if provenance.kind == "op" and provenance.op == op_name:
        return True
    return any(_contains_op(arg, op_name, depth=depth - 1) for arg in provenance.args)


def _sdpa_gqa_fact(provenance: GraphProvenance) -> GraphSdpaGqaFact | None:
    # reshape(matmul(where(keep_g, probs, 0), unsqueeze(v, 2)), ...)
    if provenance.kind != "op" or provenance.op != "_reshape" or len(provenance.args) < 1:
        return None
    matmul_out = provenance.args[0]
    if matmul_out.kind != "op" or matmul_out.op != "_matmul" or len(matmul_out.args) != 2:
        return None
    probs_masked, vg = matmul_out.args
    v_name = _match_unsqueeze_input(vg, dim=2)
    if v_name is None:
        return None
    if probs_masked.kind != "op" or probs_masked.op != "_where" or len(probs_masked.args) != 3:
        return None
    keep_g, probs, zero = probs_masked.args
    if zero != GraphProvenance("literal", value=0):
        return None
    keep_name = _match_gqa_keep_expand(keep_g)
    if keep_name is None:
        return None
    softmax_in = _match_probs_slice_softmax(probs)
    if softmax_in is None:
        return None
    scores_scaled, additive_mask_g = _match_binary_op(softmax_in, "core.binary.+")
    if scores_scaled is None or additive_mask_g is None:
        return None
    additive_mask_name = _match_reshape_input(additive_mask_g)
    if additive_mask_name is None:
        return None
    scores, scale = _match_binary_op(scores_scaled, "core.binary.*")
    if scores is None or scale is None:
        return None
    if not _is_default_scale(scale):
        return None
    q_name, k_name = _match_qk_scores(scores)
    if q_name is None or k_name is None:
        return None
    return GraphSdpaGqaFact(
        q=q_name,
        k=k_name,
        v=v_name,
        additive_mask=additive_mask_name,
        keep=keep_name,
        default_scale=True,
    )


def _match_binary_op(
    provenance: GraphProvenance,
    op_name: str,
) -> tuple[GraphProvenance | None, GraphProvenance | None]:
    if provenance.kind == "op" and provenance.op == op_name and len(provenance.args) == 2:
        return provenance.args[0], provenance.args[1]
    return None, None


def _match_probs_slice_softmax(provenance: GraphProvenance) -> GraphProvenance | None:
    if provenance.kind == "op" and provenance.op == "_slice" and provenance.args:
        provenance = provenance.args[0]
    if provenance.kind == "op" and provenance.op == "_softmax" and provenance.args:
        return provenance.args[0]
    return None


def _match_qk_scores(provenance: GraphProvenance) -> tuple[str | None, str | None]:
    if provenance.kind != "op" or provenance.op != "_matmul" or len(provenance.args) != 2:
        return None, None
    q_name = _match_reshape_input(provenance.args[0])
    kt = provenance.args[1]
    if kt.kind != "op" or kt.op != "_transpose" or not kt.args:
        return None, None
    k_name = _match_unsqueeze_input(kt.args[0], dim=2)
    return q_name, k_name


def _match_gqa_keep_expand(provenance: GraphProvenance) -> str | None:
    if provenance.kind != "op" or provenance.op != "_expand" or not provenance.args:
        return None
    return _match_reshape_input(provenance.args[0])


def _match_reshape_input(provenance: GraphProvenance) -> str | None:
    if provenance.kind == "op" and provenance.op == "_reshape" and provenance.args:
        return _input_name(provenance.args[0])
    return None


def _match_unsqueeze_input(provenance: GraphProvenance, *, dim: int) -> str | None:
    if provenance.kind != "op" or provenance.op != "_unsqueeze" or len(provenance.args) < 2:
        return None
    if provenance.args[1] != GraphProvenance("literal", value=dim):
        return None
    return _input_name(provenance.args[0])


def _input_name(provenance: GraphProvenance) -> str | None:
    return provenance.name if provenance.kind == "input" else None


def _is_default_scale(provenance: GraphProvenance) -> bool:
    if provenance.kind != "op" or provenance.op != "core.binary./" or len(provenance.args) != 2:
        return False
    numerator, denominator = provenance.args
    if numerator != GraphProvenance("literal", value=1.0):
        return False
    return denominator.kind == "op" and denominator.op == "_sqrt" and len(denominator.args) == 1


def _rope_apply_factors_fact(provenance: GraphProvenance) -> GraphRopeApplyFactorsFact | None:
    left, right = _match_binary_op(provenance, "core.binary.+")
    if left is None or right is None:
        return None
    first = _match_rope_scaled_term(left)
    second = _match_rope_scaled_term(right)
    if first is None or second is None:
        return None
    x_a, factor_a, rotated_a = first
    x_b, factor_b, rotated_b = second
    if not rotated_a and rotated_b and x_a == x_b:
        return GraphRopeApplyFactorsFact(
            x=x_a,
            cos=factor_a,
            sin=factor_b,
            interleaved=False,
        )
    if not rotated_b and rotated_a and x_a == x_b:
        return GraphRopeApplyFactorsFact(
            x=x_a,
            cos=factor_b,
            sin=factor_a,
            interleaved=False,
        )
    return None


def _match_rope_scaled_term(
    provenance: GraphProvenance,
) -> tuple[GraphProvenance, GraphProvenance, bool] | None:
    left, right = _match_binary_op(provenance, "core.binary.*")
    if left is None or right is None:
        return None
    left_rot = _match_rope_rotate_half_noninterleaved(left)
    if left_rot is not None:
        factor = _match_expand_source(right)
        return (left_rot, factor, True) if factor is not None else None
    right_rot = _match_rope_rotate_half_noninterleaved(right)
    if right_rot is not None:
        factor = _match_expand_source(left)
        return (right_rot, factor, True) if factor is not None else None
    factor = _match_expand_source(right)
    if factor is not None:
        return left, factor, False
    factor = _match_expand_source(left)
    if factor is not None:
        return right, factor, False
    return None


def _match_expand_source(provenance: GraphProvenance) -> GraphProvenance | None:
    if provenance.kind == "op" and provenance.op == "_expand" and provenance.args:
        return provenance.args[0]
    return None


def _match_rope_rotate_half_noninterleaved(
    provenance: GraphProvenance,
) -> GraphProvenance | None:
    if provenance.kind != "op" or provenance.op != "_concat" or len(provenance.args) < 3:
        return None
    first, second, dim = provenance.args[:3]
    if dim != GraphProvenance("literal", value=-1):
        return None
    negated = _match_negated_slice(first)
    plain = _match_slice(second)
    if negated is None or plain is None:
        return None
    x_hi, hi_dim, hi_start, hi_end = negated
    x_lo, lo_dim, lo_start, lo_end = plain
    if x_hi != x_lo:
        return None
    if hi_dim != GraphProvenance("literal", value=-1) or lo_dim != GraphProvenance("literal", value=-1):
        return None
    if lo_start != GraphProvenance("literal", value=0):
        return None
    if hi_start != lo_end:
        return None
    # hi_end is the last shape symbol and is intentionally not name-matched.
    return x_hi


def _match_negated_slice(
    provenance: GraphProvenance,
) -> tuple[GraphProvenance, GraphProvenance, GraphProvenance, GraphProvenance] | None:
    left, right = _match_binary_op(provenance, "core.binary.-")
    if left != GraphProvenance("literal", value=0) or right is None:
        return None
    return _match_slice(right)


def _match_slice(
    provenance: GraphProvenance,
) -> tuple[GraphProvenance, GraphProvenance, GraphProvenance, GraphProvenance] | None:
    if provenance.kind != "op" or provenance.op != "_slice" or len(provenance.args) < 4:
        return None
    return provenance.args[0], provenance.args[1], provenance.args[2], provenance.args[3]


__all__ = [
    "GraphDerivedProvenanceFact",
    "GraphRopeApplyFactorsFact",
    "GraphSdpaGqaFact",
    "GraphProvenance",
    "GraphProvenanceAnalysis",
    "format_derived_provenance_fact",
    "format_graph_provenance",
    "graph_provenance_facts",
    "infer_graph_provenance",
]
