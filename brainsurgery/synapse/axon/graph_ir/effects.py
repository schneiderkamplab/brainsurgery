from __future__ import annotations

from enum import Enum

from .core import GraphExpr, GraphModule, GraphOperand


class GraphEffect(str, Enum):
    TOTAL_PURE = "total_pure"
    PARTIAL_PURE = "partial_pure"
    EFFECTFUL = "effectful"


_TOTAL_CORE_PREFIXES = ("core.binary.",)
_TOTAL_CORE_OPS = {
    "core.alias",
    "core.ascribe",
    "core.list",
    "core.tuple",
}


def join_graph_effect(left: GraphEffect, right: GraphEffect) -> GraphEffect:
    if GraphEffect.EFFECTFUL in {left, right}:
        return GraphEffect.EFFECTFUL
    if GraphEffect.PARTIAL_PURE in {left, right}:
        return GraphEffect.PARTIAL_PURE
    return GraphEffect.TOTAL_PURE


def graph_op_effect(op_name: str) -> GraphEffect:
    if op_name in _TOTAL_CORE_OPS or op_name.startswith(_TOTAL_CORE_PREFIXES):
        return GraphEffect.TOTAL_PURE
    if op_name.startswith("core."):
        return GraphEffect.PARTIAL_PURE
    return GraphEffect.PARTIAL_PURE


def graph_operand_effect(
    operand: GraphOperand,
    *,
    module_effects: dict[str, GraphEffect] | None = None,
) -> GraphEffect:
    if not isinstance(operand, GraphExpr):
        return GraphEffect.TOTAL_PURE
    effect = (
        module_effects.get(operand.op.name, graph_op_effect(operand.op.name))
        if module_effects
        else graph_op_effect(operand.op.name)
    )
    for item in operand.inputs:
        effect = join_graph_effect(effect, graph_operand_effect(item, module_effects=module_effects))
    for item in operand.attrs.values():
        effect = join_graph_effect(effect, graph_operand_effect(item, module_effects=module_effects))
    return effect


def graph_module_effect(
    module: GraphModule,
    *,
    module_effects: dict[str, GraphEffect] | None = None,
) -> GraphEffect:
    effect = GraphEffect.TOTAL_PURE
    for node in module.nodes:
        node_effect = (
            module_effects.get(node.op.name, graph_op_effect(node.op.name))
            if module_effects
            else graph_op_effect(node.op.name)
        )
        effect = join_graph_effect(effect, node_effect)
        for item in node.inputs:
            effect = join_graph_effect(effect, graph_operand_effect(item, module_effects=module_effects))
        for item in node.attrs.values():
            effect = join_graph_effect(effect, graph_operand_effect(item, module_effects=module_effects))
    for item in module.outputs:
        effect = join_graph_effect(effect, graph_operand_effect(item, module_effects=module_effects))
    return effect


def infer_graph_module_effects(
    modules: tuple[GraphModule, ...],
    *,
    max_iterations: int = 8,
) -> dict[str, GraphEffect]:
    effects = {module.name: GraphEffect.PARTIAL_PURE for module in modules}
    for _ in range(max_iterations):
        changed = False
        for module in modules:
            inferred = graph_module_effect(module, module_effects=effects)
            if effects[module.name] != inferred:
                effects[module.name] = inferred
                changed = True
        if not changed:
            break
    return effects


__all__ = [
    "GraphEffect",
    "graph_module_effect",
    "graph_op_effect",
    "graph_operand_effect",
    "infer_graph_module_effects",
    "join_graph_effect",
]
