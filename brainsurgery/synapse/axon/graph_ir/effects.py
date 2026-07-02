from __future__ import annotations

from enum import Enum

from ...ops import get_op_semantics
from ..analysis import PurityEffect as GraphEffect
from ..analysis import join_effect as join_graph_effect
from ..analysis import op_effect
from .core import GraphExpr, GraphLiteral, GraphModule, GraphNode, GraphOperand, GraphValueRef


class UsageClass(str, Enum):
    UNRESTRICTED = "unrestricted"
    AFFINE = "affine"
    LINEAR = "linear"


def join_usage(left: UsageClass, right: UsageClass) -> UsageClass:
    if UsageClass.LINEAR in {left, right}:
        return UsageClass.LINEAR
    if UsageClass.AFFINE in {left, right}:
        return UsageClass.AFFINE
    return UsageClass.UNRESTRICTED


def graph_op_effect(op_name: str) -> GraphEffect:
    if op_name.startswith(("__torch_", "__tinygrad_", "__triton_", "__vllm_")):
        return GraphEffect.TOTAL_PURE
    return op_effect(op_name)


def graph_op_usage(op_name: str) -> UsageClass:
    if op_name.startswith(("__torch_", "__tinygrad_", "__triton_", "__vllm_")):
        return UsageClass.UNRESTRICTED
    normalized = op_name[1:] if op_name.startswith("_") else op_name
    semantics = get_op_semantics(normalized)
    usage = semantics.get("usage")
    if usage == "unrestricted":
        return UsageClass.UNRESTRICTED
    if usage == "affine":
        return UsageClass.AFFINE
    if usage == "linear":
        return UsageClass.LINEAR
    if op_effect(op_name) == GraphEffect.EFFECTFUL:
        return UsageClass.LINEAR
    return UsageClass.UNRESTRICTED


def _graph_operand_non_null(operand: GraphOperand) -> bool:
    if isinstance(operand, GraphLiteral):
        return operand.value is not None
    if isinstance(operand, GraphExpr) and operand.op.name in {"core.list", "core.tuple"}:
        return True
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
    if op_name.startswith(("__torch_", "__tinygrad_", "__triton_", "__vllm_")):
        return GraphEffect.TOTAL_PURE
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
    effect = _graph_op_call_effect(
        operand.op.name,
        inputs=operand.inputs,
        attrs=operand.attrs,
        module_effects=module_effects,
        modules_by_name=modules_by_name,
        active_modules=active_modules,
    )
    for item in operand.inputs:
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
    for item in operand.attrs.values():
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


def _graph_op_call_usage(
    op_name: str,
    *,
    inputs: tuple[GraphOperand, ...],
    attrs: dict[str, GraphOperand],
    module_usages: dict[str, UsageClass] | None = None,
    modules_by_name: dict[str, GraphModule] | None = None,
    active_modules: frozenset[str] = frozenset(),
) -> UsageClass:
    if modules_by_name is not None and op_name in modules_by_name and op_name not in active_modules:
        callee = modules_by_name[op_name]
        if len(inputs) == len(callee.inputs):
            subst = {
                formal.name: actual
                for formal, actual in zip(callee.inputs, inputs, strict=True)
            }
            return graph_module_usage(
                callee,
                module_usages=module_usages,
                modules_by_name=modules_by_name,
                active_modules=active_modules | {op_name},
                subst=subst,
            )
    if module_usages is not None and op_name in module_usages:
        return module_usages[op_name]
    return graph_op_usage(op_name)


def graph_operand_usage(
    operand: GraphOperand,
    *,
    module_usages: dict[str, UsageClass] | None = None,
    modules_by_name: dict[str, GraphModule] | None = None,
    active_modules: frozenset[str] = frozenset(),
    subst: dict[str, GraphOperand] | None = None,
    active_refs: frozenset[str] = frozenset(),
) -> UsageClass:
    if isinstance(operand, GraphValueRef) and subst is not None and operand.name in subst:
        if subst[operand.name] == operand or operand.name in active_refs:
            return UsageClass.UNRESTRICTED
        return graph_operand_usage(
            subst[operand.name],
            module_usages=module_usages,
            modules_by_name=modules_by_name,
            active_modules=active_modules,
            subst=subst,
            active_refs=active_refs | {operand.name},
        )
    if not isinstance(operand, GraphExpr):
        return UsageClass.UNRESTRICTED
    usage = _graph_op_call_usage(
        operand.op.name,
        inputs=operand.inputs,
        attrs=operand.attrs,
        module_usages=module_usages,
        modules_by_name=modules_by_name,
        active_modules=active_modules,
    )
    for item in operand.inputs:
        usage = join_usage(
            usage,
            graph_operand_usage(
                item,
                module_usages=module_usages,
                modules_by_name=modules_by_name,
                active_modules=active_modules,
                subst=subst,
                active_refs=active_refs,
            ),
        )
    for item in operand.attrs.values():
        usage = join_usage(
            usage,
            graph_operand_usage(
                item,
                module_usages=module_usages,
                modules_by_name=modules_by_name,
                active_modules=active_modules,
                subst=subst,
                active_refs=active_refs,
            ),
        )
    return usage


def _substitute_operand(
    operand: GraphOperand,
    subst: dict[str, GraphOperand],
    active_refs: frozenset[str] = frozenset(),
    depth: int = 0,
    cache: dict[tuple[int, frozenset[str]], GraphOperand] | None = None,
) -> GraphOperand:
    if len(active_refs) > 64 or depth > 64:
        return operand
    cache_key = (id(operand), active_refs)
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    if isinstance(operand, GraphValueRef):
        replacement = subst.get(operand.name)
        if replacement is None or operand.name in active_refs:
            return operand
        result = _substitute_operand(
            replacement,
            subst,
            active_refs | {operand.name},
            depth + 1,
            cache=cache,
        )
        if cache is not None:
            cache[cache_key] = result
        return result
    if isinstance(operand, GraphExpr):
        result = GraphExpr(
            op=operand.op,
            inputs=tuple(
                _substitute_operand(item, subst, active_refs, depth + 1, cache=cache)
                for item in operand.inputs
            ),
            attrs={
                key: _substitute_operand(value, subst, active_refs, depth + 1, cache=cache)
                for key, value in operand.attrs.items()
            },
            type_expr=operand.type_expr,
            dims=operand.dims,
        )
        if cache is not None:
            cache[cache_key] = result
        return result
    return operand


def graph_module_effect(
    module: GraphModule,
    *,
    module_effects: dict[str, GraphEffect] | None = None,
    modules_by_name: dict[str, GraphModule] | None = None,
    active_modules: frozenset[str] = frozenset(),
    subst: dict[str, GraphOperand] | None = None,
) -> GraphEffect:
    producers = {
        output.name: node
        for node in module.nodes
        for output in node.outputs
    }
    demanded_nodes: set[str] = set()
    visiting: set[tuple[str, bool]] = set()
    memo: dict[tuple[str, bool], GraphEffect] = {}

    def resolve_ref_operand(
        operand: GraphOperand,
        active_refs: frozenset[str] = frozenset(),
    ) -> GraphOperand:
        if subst is None or not isinstance(operand, GraphValueRef):
            return operand
        replacement = subst.get(operand.name)
        if replacement is None or operand.name in active_refs:
            return operand
        return resolve_ref_operand(replacement, active_refs | {operand.name})

    def value_ref_effect(ref: GraphValueRef, *, demand_content: bool) -> GraphEffect:
        key = (ref.name, demand_content)
        if key in memo:
            return memo[key]
        if key in visiting:
            return GraphEffect.PARTIAL_PURE
        producer = producers.get(ref.name)
        if producer is None:
            if (
                modules_by_name is not None
                and ref.name in modules_by_name
                and ref.name not in active_modules
            ):
                return graph_module_effect(
                    modules_by_name[ref.name],
                    module_effects=module_effects,
                    modules_by_name=modules_by_name,
                    active_modules=active_modules | {ref.name},
                    subst=subst,
                )
            if module_effects is not None and ref.name in module_effects:
                return module_effects[ref.name]
            return GraphEffect.TOTAL_PURE
        visiting.add(key)
        effect = node_effect(producer, demand_content=demand_content)
        visiting.remove(key)
        memo[key] = effect
        return effect

    def operand_effect(
        operand: GraphOperand,
        *,
        demand_content: bool,
        depth: int = 0,
    ) -> GraphEffect:
        if depth > 128:
            return GraphEffect.PARTIAL_PURE
        operand = resolve_ref_operand(operand)
        if isinstance(operand, GraphValueRef):
            return value_ref_effect(operand, demand_content=demand_content)
        if not isinstance(operand, GraphExpr):
            return GraphEffect.TOTAL_PURE
        inputs = tuple(resolve_ref_operand(item) for item in operand.inputs)
        attrs = {key: resolve_ref_operand(value) for key, value in operand.attrs.items()}
        effect = _graph_op_call_effect_for_demand(
            operand.op.name,
            demand_content=demand_content,
            inputs=inputs,
            attrs=attrs,
            module_effects=module_effects,
            modules_by_name=modules_by_name,
            active_modules=active_modules,
        )
        for item, item_demand in zip(
            inputs,
            _input_content_demands(operand.op.name, len(inputs), demand_content=demand_content),
            strict=False,
        ):
            effect = join_graph_effect(effect, operand_effect(item, demand_content=item_demand, depth=depth + 1))
        for item in attrs.values():
            effect = join_graph_effect(effect, operand_effect(item, demand_content=True, depth=depth + 1))
        return effect

    def node_effect(node: GraphNode, *, demand_content: bool) -> GraphEffect:
        demanded_nodes.add(node.id)
        inputs = tuple(resolve_ref_operand(item) for item in node.inputs)
        attrs = {key: resolve_ref_operand(value) for key, value in node.attrs.items()}
        effect = _graph_op_call_effect_for_demand(
            node.op.name,
            demand_content=demand_content,
            inputs=inputs,
            attrs=attrs,
            module_effects=module_effects,
            modules_by_name=modules_by_name,
            active_modules=active_modules,
        )
        for item, item_demand in zip(
            inputs,
            _input_content_demands(node.op.name, len(inputs), demand_content=demand_content),
            strict=False,
        ):
            effect = join_graph_effect(effect, operand_effect(item, demand_content=item_demand))
        for item in attrs.values():
            effect = join_graph_effect(effect, operand_effect(item, demand_content=True))
        return effect

    effect = GraphEffect.TOTAL_PURE
    for item in module.outputs:
        effect = join_graph_effect(
            effect,
            operand_effect(item, demand_content=True),
        )
    for node in module.nodes:
        if node.id in demanded_nodes:
            continue
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
    return effect


def _normalized_op_name(op_name: str) -> str:
    return op_name[1:] if op_name.startswith("_") else op_name


def _graph_op_call_effect_for_demand(
    op_name: str,
    *,
    demand_content: bool,
    inputs: tuple[GraphOperand, ...],
    attrs: dict[str, GraphOperand],
    module_effects: dict[str, GraphEffect] | None,
    modules_by_name: dict[str, GraphModule] | None,
    active_modules: frozenset[str],
) -> GraphEffect:
    if not demand_content and _normalized_op_name(op_name) == "empty_like":
        return GraphEffect.TOTAL_PURE
    return _graph_op_call_effect(
        op_name,
        inputs=inputs,
        attrs=attrs,
        module_effects=module_effects,
        modules_by_name=modules_by_name,
        active_modules=active_modules,
    )


def _input_content_demands(
    op_name: str,
    input_count: int,
    *,
    demand_content: bool,
) -> tuple[bool, ...]:
    if _normalized_op_name(op_name) == "fill" and input_count:
        return (False, *([True] * (input_count - 1)))
    return tuple(True for _ in range(input_count))


def graph_node_effect(
    node: GraphNode,
    *,
    module_effects: dict[str, GraphEffect] | None = None,
    modules_by_name: dict[str, GraphModule] | None = None,
    active_modules: frozenset[str] = frozenset(),
    subst: dict[str, GraphOperand] | None = None,
) -> GraphEffect:
    def resolve_ref_operand(
        operand: GraphOperand,
        active_refs: frozenset[str] = frozenset(),
    ) -> GraphOperand:
        if subst is None or not isinstance(operand, GraphValueRef):
            return operand
        replacement = subst.get(operand.name)
        if replacement is None or operand.name in active_refs:
            return operand
        return resolve_ref_operand(replacement, active_refs | {operand.name})

    inputs = tuple(resolve_ref_operand(item) for item in node.inputs)
    attrs = {key: resolve_ref_operand(value) for key, value in node.attrs.items()}
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


def graph_node_usage(
    node: GraphNode,
    *,
    module_usages: dict[str, UsageClass] | None = None,
    modules_by_name: dict[str, GraphModule] | None = None,
    active_modules: frozenset[str] = frozenset(),
    subst: dict[str, GraphOperand] | None = None,
) -> UsageClass:
    def resolve_ref_operand(
        operand: GraphOperand,
        active_refs: frozenset[str] = frozenset(),
    ) -> GraphOperand:
        if subst is None or not isinstance(operand, GraphValueRef):
            return operand
        replacement = subst.get(operand.name)
        if replacement is None or operand.name in active_refs:
            return operand
        return resolve_ref_operand(replacement, active_refs | {operand.name})

    inputs = tuple(resolve_ref_operand(item) for item in node.inputs)
    attrs = {key: resolve_ref_operand(value) for key, value in node.attrs.items()}
    usage = _graph_op_call_usage(
        node.op.name,
        inputs=inputs,
        attrs=attrs,
        module_usages=module_usages,
        modules_by_name=modules_by_name,
        active_modules=active_modules,
    )
    for item in inputs:
        usage = join_usage(
            usage,
            graph_operand_usage(
                item,
                module_usages=module_usages,
                modules_by_name=modules_by_name,
                active_modules=active_modules,
                subst=subst,
            ),
        )
    for item in attrs.values():
        usage = join_usage(
            usage,
            graph_operand_usage(
                item,
                module_usages=module_usages,
                modules_by_name=modules_by_name,
                active_modules=active_modules,
                subst=subst,
            ),
        )
    return usage


def graph_module_usage(
    module: GraphModule,
    *,
    module_usages: dict[str, UsageClass] | None = None,
    modules_by_name: dict[str, GraphModule] | None = None,
    active_modules: frozenset[str] = frozenset(),
    subst: dict[str, GraphOperand] | None = None,
) -> UsageClass:
    usage = UsageClass.UNRESTRICTED
    for item in module.outputs:
        usage = join_usage(
            usage,
            graph_operand_usage(
                item,
                module_usages=module_usages,
                modules_by_name=modules_by_name,
                active_modules=active_modules,
                subst=subst,
            ),
        )
    for node in module.nodes:
        usage = join_usage(
            usage,
            graph_node_usage(
                node,
                module_usages=module_usages,
                modules_by_name=modules_by_name,
                active_modules=active_modules,
                subst=subst,
            ),
        )
    return usage


def infer_graph_module_effects(
    modules: tuple[GraphModule, ...],
    *,
    max_iterations: int = 16,
) -> dict[str, GraphEffect]:
    effects = {module.name: GraphEffect.PARTIAL_PURE for module in modules}
    for _ in range(max_iterations):
        changed = False
        for module in modules:
            # Summary inference must not inline callees at every call site.  The
            # fixpoint over the previous iteration's summaries is conservative
            # and avoids exponential re-walking of nested/inlined call graphs.
            inferred = graph_module_effect(
                module,
                module_effects=effects,
                active_modules=frozenset({module.name}),
            )
            if effects[module.name] != inferred:
                effects[module.name] = inferred
                changed = True
        if not changed:
            break
    return effects


def infer_graph_module_usages(
    modules: tuple[GraphModule, ...],
    *,
    max_iterations: int = 16,
) -> dict[str, UsageClass]:
    usages = {module.name: UsageClass.UNRESTRICTED for module in modules}
    for _ in range(max_iterations):
        changed = False
        for module in modules:
            # As with effects, usage summaries are inferred by fixpoint over
            # summaries rather than by call-site expansion.
            inferred = graph_module_usage(
                module,
                module_usages=usages,
                active_modules=frozenset({module.name}),
            )
            if usages[module.name] != inferred:
                usages[module.name] = inferred
                changed = True
        if not changed:
            break
    return usages


__all__ = [
    "GraphEffect",
    "UsageClass",
    "graph_module_effect",
    "graph_module_usage",
    "graph_node_effect",
    "graph_node_usage",
    "graph_op_effect",
    "graph_op_usage",
    "graph_operand_effect",
    "graph_operand_usage",
    "infer_graph_module_effects",
    "infer_graph_module_usages",
    "join_graph_effect",
    "join_usage",
]
