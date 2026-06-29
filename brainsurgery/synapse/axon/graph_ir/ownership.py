from __future__ import annotations

from dataclasses import dataclass

from .core import GraphExpr, GraphModule, GraphNode, GraphOperand, GraphProgram, GraphValueRef
from .effects import GraphEffect, UsageClass, graph_node_effect, graph_node_usage


_FRESH_ALLOC_OPS = {
    "_empty_like",
    "_fill",
    "_tensor_like",
    "_zeros",
    "_zeros_like",
}

_ALIAS_VIEW_OPS = {
    "core.alias",
    "core.ascribe",
    "_reshape",
    "_slice",
    "_transpose",
    "_unsqueeze",
}


@dataclass(frozen=True)
class GraphOwnershipAnalysis:
    inplace_assign_slice_node_ids: frozenset[str]
    value_owner: dict[str, str]


def _operand_refs(operand: GraphOperand, out: set[str]) -> None:
    if isinstance(operand, GraphValueRef):
        out.add(operand.name)
        return
    if isinstance(operand, GraphExpr):
        for item in operand.inputs:
            _operand_refs(item, out)
        for item in operand.attrs.values():
            _operand_refs(item, out)


def _node_refs(node: GraphNode) -> set[str]:
    refs: set[str] = set()
    for operand in (*node.inputs, *node.attrs.values()):
        _operand_refs(operand, refs)
    return refs


def _module_suffix_refs(module: GraphModule) -> list[set[str]]:
    suffix: list[set[str]] = [set() for _ in range(len(module.nodes) + 1)]
    output_refs: set[str] = set()
    for output in module.outputs:
        _operand_refs(output, output_refs)
    suffix[len(module.nodes)] = output_refs
    for index in range(len(module.nodes) - 1, -1, -1):
        suffix[index] = set(suffix[index + 1])
        suffix[index].update(_node_refs(module.nodes[index]))
    return suffix


def _single_ref_operand(operand: GraphOperand) -> str | None:
    return operand.name if isinstance(operand, GraphValueRef) else None


def _fresh_allocation_node(node: GraphNode) -> bool:
    if node.op.name in _FRESH_ALLOC_OPS:
        return True
    return (
        graph_node_effect(node) == GraphEffect.TOTAL_PURE
        and graph_node_usage(node) in {UsageClass.AFFINE, UsageClass.LINEAR}
    )


def _alias_view_node(node: GraphNode) -> bool:
    return node.op.name in _ALIAS_VIEW_OPS and len(node.inputs) >= 1


def infer_graph_ownership(
    graph: GraphProgram,
    *,
    assume_module_inputs_owned: bool = False,
) -> GraphOwnershipAnalysis:
    value_owner: dict[str, str] = {}
    inplace_assign_slice_node_ids: set[str] = set()

    for module in graph.modules:
        owner_by_value: dict[str, str] = {}
        aliases_by_owner: dict[str, set[str]] = {}
        suffix_refs = _module_suffix_refs(module)

        def set_owner(name: str, owner: str) -> None:
            owner_by_value[name] = owner
            aliases_by_owner.setdefault(owner, set()).add(name)
            value_owner[f"{module.name}:{name}"] = owner

        if assume_module_inputs_owned:
            for value in module.inputs:
                set_owner(value.name, value.name)

        for index, node in enumerate(module.nodes):
            if (
                node.op.name == "_assign_slice"
                and len(node.inputs) >= 5
                and len(node.outputs) == 1
                and isinstance(node.inputs[0], GraphValueRef)
            ):
                base_name = node.inputs[0].name
                owner = owner_by_value.get(base_name)
                if owner is not None:
                    aliases = aliases_by_owner.get(owner, set())
                    future_refs = suffix_refs[index + 1]
                    stale_future_refs = (aliases & future_refs) - {base_name}
                    base_reused = base_name in future_refs
                    if not stale_future_refs and not base_reused:
                        inplace_assign_slice_node_ids.add(node.id)
                        set_owner(node.outputs[0].name, owner)
                        continue

            if len(node.outputs) == 1 and _fresh_allocation_node(node):
                set_owner(node.outputs[0].name, node.outputs[0].name)
                continue

            if len(node.outputs) == 1 and _alias_view_node(node):
                input_name = _single_ref_operand(node.inputs[0])
                if input_name is not None and input_name in owner_by_value:
                    set_owner(node.outputs[0].name, owner_by_value[input_name])
                    continue

            for output in node.outputs:
                owner_by_value.pop(output.name, None)

    return GraphOwnershipAnalysis(
        inplace_assign_slice_node_ids=frozenset(inplace_assign_slice_node_ids),
        value_owner=value_owner,
    )


__all__ = [
    "GraphOwnershipAnalysis",
    "infer_graph_ownership",
]
