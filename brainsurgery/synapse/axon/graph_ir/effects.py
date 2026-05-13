from __future__ import annotations

from ..analysis import PurityEffect as GraphEffect
from ..analysis import join_effect as join_graph_effect
from ..analysis import op_effect
from .core import GraphExpr, GraphLiteral, GraphModule, GraphNode, GraphOperand, GraphValueRef


def graph_op_effect(op_name: str) -> GraphEffect:
    return op_effect(op_name)


def _graph_operand_non_null(operand: GraphOperand) -> bool:
    if isinstance(operand, GraphLiteral):
        return operand.value is not None
    return False


def _graph_op_call_effect(
    op_name: str,
    *,
    inputs: tuple[GraphOperand, ...],
    attrs: dict[str, GraphOperand],
    module_effects: dict[str, GraphEffect] | None = None,
    modules_by_name: dict[str, GraphModule] | None = None,
    active_modules: frozenset[str] = frozenset(),
) -> GraphEffect:
    if modules_by_name is not None and op_name in modules_by_name and op_name not in active_modules:
        callee = modules_by_name[op_name]
        if len(inputs) == len(callee.inputs):
            subst = {
                formal.name: actual
                for formal, actual in zip(callee.inputs, inputs, strict=True)
            }
            return graph_module_effect(
                callee,
                module_effects=module_effects,
                modules_by_name=modules_by_name,
                active_modules=active_modules | {op_name},
                subst=subst,
            )
    if module_effects is not None and op_name in module_effects:
        return module_effects[op_name]
    normalized = op_name[1:] if op_name.startswith("_") else op_name
    if normalized in {
        "config_bool",
        "config_dim",
        "config_float",
        "config_int",
        "config_list",
        "config_str",
        "config_value",
    }:
        default = attrs.get("default")
        if default is None and len(inputs) >= 2:
            default = inputs[1]
        return GraphEffect.TOTAL_PURE if default is not None and _graph_operand_non_null(default) else GraphEffect.PARTIAL_PURE
    return op_effect(op_name, attrs=attrs)


def graph_operand_effect(
    operand: GraphOperand,
    *,
    module_effects: dict[str, GraphEffect] | None = None,
    modules_by_name: dict[str, GraphModule] | None = None,
    active_modules: frozenset[str] = frozenset(),
    subst: dict[str, GraphOperand] | None = None,
) -> GraphEffect:
    if isinstance(operand, GraphValueRef) and subst is not None and operand.name in subst:
        if subst[operand.name] == operand:
            return GraphEffect.TOTAL_PURE
        return graph_operand_effect(
            subst[operand.name],
            module_effects=module_effects,
            modules_by_name=modules_by_name,
            active_modules=active_modules,
            subst=subst,
        )
    if not isinstance(operand, GraphExpr):
        return GraphEffect.TOTAL_PURE
    inputs = tuple(_substitute_operand(item, subst or {}) for item in operand.inputs)
    attrs = {key: _substitute_operand(value, subst or {}) for key, value in operand.attrs.items()}
    effect = _graph_op_call_effect(
        operand.op.name,
        inputs=inputs,
        attrs=attrs,
        module_effects=module_effects,
        modules_by_name=modules_by_name,
        active_modules=active_modules,
    )
    for item in inputs:
        effect = join_graph_effect(
            effect,
            graph_operand_effect(
                item,
                module_effects=module_effects,
                modules_by_name=modules_by_name,
                active_modules=active_modules,
                subst=subst,
            ),
        )
    for item in attrs.values():
        effect = join_graph_effect(
            effect,
            graph_operand_effect(
                item,
                module_effects=module_effects,
                modules_by_name=modules_by_name,
                active_modules=active_modules,
                subst=subst,
            ),
        )
    return effect


def _substitute_operand(operand: GraphOperand, subst: dict[str, GraphOperand]) -> GraphOperand:
    if isinstance(operand, GraphValueRef):
        return subst.get(operand.name, operand)
    if isinstance(operand, GraphExpr):
        return GraphExpr(
            op=operand.op,
            inputs=tuple(_substitute_operand(item, subst) for item in operand.inputs),
            attrs={key: _substitute_operand(value, subst) for key, value in operand.attrs.items()},
            type_expr=operand.type_expr,
            dims=operand.dims,
        )
    return operand


def graph_module_effect(
    module: GraphModule,
    *,
    module_effects: dict[str, GraphEffect] | None = None,
    modules_by_name: dict[str, GraphModule] | None = None,
    active_modules: frozenset[str] = frozenset(),
    subst: dict[str, GraphOperand] | None = None,
) -> GraphEffect:
    effect = GraphEffect.TOTAL_PURE
    for node in module.nodes:
        effect = join_graph_effect(
            effect,
            graph_node_effect(
                node,
                module_effects=module_effects,
                modules_by_name=modules_by_name,
                active_modules=active_modules,
                subst=subst,
            ),
        )
    for item in module.outputs:
        effect = join_graph_effect(
            effect,
            graph_operand_effect(
                item,
                module_effects=module_effects,
                modules_by_name=modules_by_name,
                active_modules=active_modules,
                subst=subst,
            ),
        )
    return effect


def graph_node_effect(
    node: GraphNode,
    *,
    module_effects: dict[str, GraphEffect] | None = None,
    modules_by_name: dict[str, GraphModule] | None = None,
    active_modules: frozenset[str] = frozenset(),
    subst: dict[str, GraphOperand] | None = None,
) -> GraphEffect:
    inputs = tuple(_substitute_operand(item, subst or {}) for item in node.inputs)
    attrs = {key: _substitute_operand(value, subst or {}) for key, value in node.attrs.items()}
    effect = _graph_op_call_effect(
        node.op.name,
        inputs=inputs,
        attrs=attrs,
        module_effects=module_effects,
        modules_by_name=modules_by_name,
        active_modules=active_modules,
    )
    for item in inputs:
        effect = join_graph_effect(
            effect,
            graph_operand_effect(
                item,
                module_effects=module_effects,
                modules_by_name=modules_by_name,
                active_modules=active_modules,
                subst=subst,
            ),
        )
    for item in attrs.values():
        effect = join_graph_effect(
            effect,
            graph_operand_effect(
                item,
                module_effects=module_effects,
                modules_by_name=modules_by_name,
                active_modules=active_modules,
                subst=subst,
            ),
        )
    return effect


def infer_graph_module_effects(
    modules: tuple[GraphModule, ...],
    *,
    max_iterations: int = 16,
) -> dict[str, GraphEffect]:
    effects = {module.name: GraphEffect.PARTIAL_PURE for module in modules}
    modules_by_name = {module.name: module for module in modules}
    for _ in range(max_iterations):
        changed = False
        for module in modules:
            inferred = graph_module_effect(
                module,
                module_effects=effects,
                modules_by_name=modules_by_name,
                active_modules=frozenset({module.name}),
            )
            if effects[module.name] != inferred:
                effects[module.name] = inferred
                changed = True
        if not changed:
            break
    return effects


__all__ = [
    "GraphEffect",
    "graph_module_effect",
    "graph_node_effect",
    "graph_op_effect",
    "graph_operand_effect",
    "infer_graph_module_effects",
    "join_graph_effect",
]
