from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, replace

from ..ast import (
    Constraint,
    ConstraintAtom,
    ConstraintOperand,
    DimExprBinary,
    DimToken,
    TypeAny,
    TypeDim,
    TypeExpr,
    TypeInt,
    TypeList,
    TypeNamed,
    TypeOptional,
    TypePath,
    TypeTensor,
    TypeTuple,
    dim_token_names,
)
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
    graph_path_template_names,
    graph_type_compatible,
    validate_graph_program,
)
from .effects import GraphEffect, graph_op_effect, graph_operand_effect, infer_graph_module_effects


@dataclass(frozen=True)
class GraphOptimizeConfig:
    prune_to_main: bool = True
    atomic_alias_cleanup: bool = True
    dead_temp_elimination: bool = True
    constant_folding: bool = True
    constant_dim_substitution: bool = False
    specialize_definitions: str = "single-callsite"
    inline_safe: bool = True
    max_iterations: int = 8


_SPECIALIZE_MODES = {"off", "single-callsite", "monomorphize"}


def _is_atomic_operand(operand: GraphOperand) -> bool:
    return isinstance(operand, GraphValueRef | GraphLiteral | GraphPath)


def _path_has_template(path: GraphPath) -> bool:
    return any("{" in part or "}" in part for part in path.parts)


def _is_safe_specialization_operand(operand: GraphOperand) -> bool:
    if isinstance(operand, GraphPath):
        return not _path_has_template(operand)
    if isinstance(operand, GraphLiteral):
        return True
    return False


def _is_total_pure_op(op_name: str, module_effects: Mapping[str, GraphEffect] | None = None) -> bool:
    if module_effects is not None and op_name in module_effects:
        return module_effects[op_name] == GraphEffect.TOTAL_PURE
    return graph_op_effect(op_name) == GraphEffect.TOTAL_PURE


def _literal_like(value: object, type_like: GraphOperand | GraphNode | GraphExpr) -> GraphLiteral:
    type_expr = getattr(type_like, "type_expr")
    return GraphLiteral(value=value, type_expr=type_expr)


def _substitute_dim_token(dim: DimToken, subst: Mapping[str, DimToken]) -> DimToken:
    if isinstance(dim, str):
        return subst.get(dim, dim)
    if isinstance(dim, DimExprBinary):
        left = _substitute_dim_token(dim.left, subst)
        right = _substitute_dim_token(dim.right, subst)
        if type(left) is int and type(right) is int:
            if dim.op == "+":
                return left + right
            if dim.op == "-":
                return left - right
            if dim.op == "*":
                return left * right
            if dim.op == "/" and right != 0 and left % right == 0:
                return left // right
        return DimExprBinary(
            op=dim.op,
            left=left,
            right=right,
        )
    return dim


def _substitute_type_expr(tp: TypeExpr, subst: Mapping[str, DimToken]) -> TypeExpr:
    if isinstance(tp, TypeNamed):
        return TypeNamed(
            name=tp.name,
            args=tuple(_substitute_dim_token(dim, subst) for dim in tp.args),
        )
    if isinstance(tp, TypeOptional):
        return TypeOptional(_substitute_type_expr(tp.inner, subst))
    if isinstance(tp, TypeTensor):
        return TypeTensor(
            base=tp.base,
            dims=tuple(_substitute_dim_token(dim, subst) for dim in tp.dims),
        )
    if isinstance(tp, TypeList):
        return TypeList(_substitute_type_expr(tp.item, subst))
    if isinstance(tp, TypeTuple):
        return TypeTuple(tuple(_substitute_type_expr(item, subst) for item in tp.items))
    return tp


def _substitute_constraint_atom(
    atom: ConstraintAtom,
    subst: Mapping[str, DimToken],
) -> ConstraintAtom:
    if isinstance(atom, str):
        replacement = subst.get(atom)
        return replacement if isinstance(replacement, int | str | DimExprBinary) else atom
    if isinstance(atom, DimExprBinary):
        return _substitute_dim_token(atom, subst)
    return atom


def _substitute_constraint_operand(
    operand: ConstraintOperand,
    subst: Mapping[str, DimToken],
) -> ConstraintOperand:
    if isinstance(operand, tuple):
        return tuple(_substitute_constraint_atom(item, subst) for item in operand)
    return _substitute_constraint_atom(operand, subst)


def _substitute_constraint(constraint: Constraint, subst: Mapping[str, DimToken]) -> Constraint:
    return Constraint(
        relation=constraint.relation,
        left=_substitute_constraint_operand(constraint.left, subst),
        right=(
            None
            if constraint.right is None
            else _substitute_constraint_operand(constraint.right, subst)
        ),
        guards=tuple(_substitute_constraint(guard, subst) for guard in constraint.guards),
    )


def _substitute_graph_operand_dims(
    operand: GraphOperand,
    subst: Mapping[str, DimToken],
) -> GraphOperand:
    if isinstance(operand, GraphValueRef):
        return replace(
            operand,
            type_expr=_substitute_type_expr(operand.type_expr, subst),
            dims=(
                None
                if operand.dims is None
                else tuple(_substitute_dim_token(dim, subst) for dim in operand.dims)
            ),
        )
    if isinstance(operand, GraphLiteral):
        return replace(operand, type_expr=_substitute_type_expr(operand.type_expr, subst))
    if isinstance(operand, GraphExpr):
        return replace(
            operand,
            inputs=tuple(_substitute_graph_operand_dims(item, subst) for item in operand.inputs),
            attrs={
                key: _substitute_graph_operand_dims(value, subst)
                for key, value in operand.attrs.items()
            },
            type_expr=_substitute_type_expr(operand.type_expr, subst),
            dims=(
                None
                if operand.dims is None
                else tuple(_substitute_dim_token(dim, subst) for dim in operand.dims)
            ),
        )
    return operand


def _substitute_graph_value_dims(value: GraphValue, subst: Mapping[str, DimToken]) -> GraphValue:
    return replace(
        value,
        type_expr=_substitute_type_expr(value.type_expr, subst),
        dims=(
            None
            if value.dims is None
            else tuple(_substitute_dim_token(dim, subst) for dim in value.dims)
        ),
    )


def _substitute_graph_node_dims(node: GraphNode, subst: Mapping[str, DimToken]) -> GraphNode:
    return replace(
        node,
        inputs=tuple(_substitute_graph_operand_dims(item, subst) for item in node.inputs),
        attrs={
            key: _substitute_graph_operand_dims(value, subst)
            for key, value in node.attrs.items()
        },
        outputs=tuple(_substitute_graph_value_dims(output, subst) for output in node.outputs),
        type_expr=_substitute_type_expr(node.type_expr, subst),
        dims=(
            None
            if node.dims is None
            else tuple(_substitute_dim_token(dim, subst) for dim in node.dims)
        ),
    )


def _substitute_graph_module_dims(
    module: GraphModule,
    subst: Mapping[str, DimToken],
) -> GraphModule:
    return replace(
        module,
        inputs=tuple(_substitute_graph_value_dims(value, subst) for value in module.inputs),
        outputs=tuple(_substitute_graph_operand_dims(output, subst) for output in module.outputs),
        nodes=tuple(_substitute_graph_node_dims(node, subst) for node in module.nodes),
        return_type_expr=(
            None
            if module.return_type_expr is None
            else _substitute_type_expr(module.return_type_expr, subst)
        ),
        constraints=tuple(_substitute_constraint(item, subst) for item in module.constraints),
    )


def _result_types(type_expr: TypeExpr, output_count: int) -> tuple[TypeExpr, ...]:
    if output_count == 1:
        return (type_expr,)
    if isinstance(type_expr, TypeTuple) and len(type_expr.items) == output_count:
        return type_expr.items
    if isinstance(type_expr, TypeList):
        return tuple(type_expr.item for _ in range(output_count))
    return tuple(TypeAny() for _ in range(output_count))


def _module_output_types(module: GraphModule) -> tuple[TypeExpr, ...]:
    if module.return_type_expr is not None:
        return _result_types(module.return_type_expr, len(module.outputs))
    return tuple(graph_operand_type(output) for output in module.outputs)


def _module_output_types_for_arity(module: GraphModule, output_count: int) -> tuple[TypeExpr, ...]:
    if module.return_type_expr is not None:
        return _result_types(module.return_type_expr, output_count)
    if output_count == len(module.outputs):
        return tuple(graph_operand_type(output) for output in module.outputs)
    if len(module.outputs) == 1:
        return _result_types(graph_operand_type(module.outputs[0]), output_count)
    return tuple(TypeAny() for _ in range(output_count))


def _bind_dim_sequence_map(
    formal_dims: tuple[DimToken, ...],
    actual_dims: tuple[DimToken, ...],
    dim_map: dict[str, DimToken],
) -> None:
    variadic_indexes = [
        index
        for index, dim in enumerate(formal_dims)
        if isinstance(dim, str) and dim.startswith("..")
    ]
    if len(variadic_indexes) > 1:
        return
    pairs: list[tuple[DimToken, DimToken]] = []
    if not variadic_indexes:
        pairs = list(zip(formal_dims, actual_dims, strict=False))
    else:
        variadic_index = variadic_indexes[0]
        prefix = formal_dims[:variadic_index]
        suffix = formal_dims[variadic_index + 1 :]
        if len(actual_dims) < len(prefix) + len(suffix):
            return
        pairs.extend(zip(prefix, actual_dims[: len(prefix)], strict=False))
        if suffix:
            pairs.extend(zip(suffix, actual_dims[-len(suffix) :], strict=False))
    for formal_dim, actual_dim in pairs:
        if isinstance(formal_dim, str) and not formal_dim.startswith(".."):
            dim_map.setdefault(formal_dim, actual_dim)


def _bind_type_dim_map(formal: TypeExpr, actual: TypeExpr, dim_map: dict[str, DimToken]) -> None:
    if isinstance(formal, TypeTensor) and isinstance(actual, TypeTensor):
        _bind_dim_sequence_map(formal.dims, actual.dims, dim_map)
        return
    if isinstance(formal, TypeNamed) and isinstance(actual, TypeNamed):
        _bind_dim_sequence_map(formal.args, actual.args, dim_map)
        return
    if isinstance(formal, TypeOptional) and isinstance(actual, TypeOptional):
        _bind_type_dim_map(formal.inner, actual.inner, dim_map)
        return
    if isinstance(formal, TypeList) and isinstance(actual, TypeList):
        _bind_type_dim_map(formal.item, actual.item, dim_map)
        return
    if isinstance(formal, TypeTuple) and isinstance(actual, TypeTuple):
        for formal_item, actual_item in zip(formal.items, actual.items, strict=False):
            _bind_type_dim_map(formal_item, actual_item, dim_map)


def _operand_dim_token(
    operand: GraphOperand,
    dim_values: Mapping[str, DimToken] | None = None,
) -> DimToken | None:
    if isinstance(operand, GraphLiteral) and type(operand.value) is int:
        return operand.value
    if isinstance(operand, GraphValueRef):
        if dim_values is not None and operand.name in dim_values:
            return dim_values[operand.name]
        return operand.name
    return None


def _bind_value_dim_map(
    formal: GraphValue,
    actual: GraphOperand,
    dim_map: dict[str, DimToken],
    *,
    dim_values: Mapping[str, DimToken] | None = None,
) -> None:
    formal_type = formal.type_expr
    if isinstance(formal_type, TypeOptional):
        formal_type = formal_type.inner
    if not isinstance(formal_type, TypeDim):
        return
    actual_dim = _operand_dim_token(actual, dim_values)
    if actual_dim is not None:
        dim_map.setdefault(formal.name, actual_dim)


def _call_actuals(
    node: GraphNode | GraphExpr,
    callee: GraphModule,
) -> tuple[GraphOperand, ...]:
    actuals: list[GraphOperand | None] = [None] * len(callee.inputs)
    for index, operand in enumerate(node.inputs):
        if index < len(actuals):
            actuals[index] = operand
    formal_by_name = {formal.name: index for index, formal in enumerate(callee.inputs)}
    for key, operand in node.attrs.items():
        index = formal_by_name.get(key)
        if index is not None:
            actuals[index] = operand
    return tuple(actual for actual in actuals if actual is not None)


def _refresh_graph_operand_types(
    operand: GraphOperand,
    *,
    env: Mapping[str, GraphValue],
    globals_env: Mapping[str, GraphValue],
    modules_by_name: Mapping[str, GraphModule],
    dim_values: Mapping[str, DimToken] | None = None,
) -> GraphOperand:
    if isinstance(operand, GraphValueRef):
        value = env.get(operand.name) or globals_env.get(operand.name)
        if value is None:
            return operand
        return replace(operand, type_expr=value.type_expr, dims=value.dims)
    if not isinstance(operand, GraphExpr):
        return operand
    inputs = tuple(
        _refresh_graph_operand_types(
            item,
            env=env,
            globals_env=globals_env,
            modules_by_name=modules_by_name,
            dim_values=dim_values,
        )
        for item in operand.inputs
    )
    attrs = {
        key: _refresh_graph_operand_types(
            value,
            env=env,
            globals_env=globals_env,
            modules_by_name=modules_by_name,
            dim_values=dim_values,
        )
        for key, value in operand.attrs.items()
    }
    callee = modules_by_name.get(operand.op.name)
    if callee is None:
        return replace(operand, inputs=inputs, attrs=attrs)
    dim_map: dict[str, DimToken] = {}
    call = replace(operand, inputs=inputs, attrs=attrs)
    for formal, actual in zip(callee.inputs, _call_actuals(call, callee), strict=False):
        _bind_type_dim_map(formal.type_expr, graph_operand_type(actual), dim_map)
        _bind_value_dim_map(formal, actual, dim_map, dim_values=dim_values)
    result_types = tuple(
        _substitute_type_expr(tp, dim_map)
        for tp in _module_output_types_for_arity(callee, 1)
    )
    if len(result_types) != 1:
        return replace(call, type_expr=TypeTuple(result_types))
    result_type = result_types[0]
    dims = result_type.dims if isinstance(result_type, TypeTensor) else None
    return replace(call, type_expr=result_type, dims=dims)


def _refresh_graph_module_types(
    module: GraphModule,
    *,
    globals_env: Mapping[str, GraphValue],
    modules_by_name: Mapping[str, GraphModule],
) -> GraphModule:
    env = {value.name: value for value in module.inputs}
    dim_values: dict[str, DimToken] = {}
    nodes: list[GraphNode] = []
    for node in module.nodes:
        inputs = tuple(
            _refresh_graph_operand_types(
                item,
                env=env,
                globals_env=globals_env,
                modules_by_name=modules_by_name,
                dim_values=dim_values,
            )
            for item in node.inputs
        )
        attrs = {
            key: _refresh_graph_operand_types(
                value,
                env=env,
                globals_env=globals_env,
                modules_by_name=modules_by_name,
                dim_values=dim_values,
            )
            for key, value in node.attrs.items()
        }
        type_expr = node.type_expr
        output_types = _result_types(type_expr, len(node.outputs))
        callee = modules_by_name.get(node.op.name)
        if callee is not None:
            dim_map: dict[str, DimToken] = {}
            call = replace(node, inputs=inputs, attrs=attrs)
            for formal, actual in zip(callee.inputs, _call_actuals(call, callee), strict=False):
                _bind_type_dim_map(formal.type_expr, graph_operand_type(actual), dim_map)
                _bind_value_dim_map(formal, actual, dim_map, dim_values=dim_values)
            output_types = tuple(
                _substitute_type_expr(tp, dim_map)
                for tp in _module_output_types_for_arity(callee, len(node.outputs))
            )
            type_expr = output_types[0] if len(output_types) == 1 else TypeTuple(output_types)
        elif node.op.name == "core.tuple":
            output_types = tuple(graph_operand_type(item) for item in inputs)
            type_expr = output_types[0] if len(output_types) == 1 else TypeTuple(output_types)
        elif node.op.name in {"core.alias", "core.ascribe"} and len(inputs) == 1:
            input_type = graph_operand_type(inputs[0])
            output_types = _result_types(input_type, len(node.outputs))
            type_expr = output_types[0] if len(output_types) == 1 else TypeTuple(output_types)
        outputs = tuple(
            replace(
                output,
                type_expr=output_types[index] if index < len(output_types) else output.type_expr,
                dims=(
                    output_types[index].dims
                    if index < len(output_types) and isinstance(output_types[index], TypeTensor)
                    else output.dims
                ),
            )
            for index, output in enumerate(node.outputs)
        )
        rewritten = replace(
            node,
            inputs=inputs,
            attrs=attrs,
            outputs=outputs,
            type_expr=type_expr,
            dims=type_expr.dims if isinstance(type_expr, TypeTensor) else node.dims,
        )
        nodes.append(rewritten)
        env.update({output.name: output for output in outputs})
        if len(outputs) == 1:
            output = outputs[0]
            if node.op.name in globals_env and isinstance(globals_env[node.op.name].type_expr, TypeDim | TypeInt):
                dim_values[output.name] = node.op.name
            elif node.op.name in {"core.alias", "core.ascribe"} and len(inputs) == 1:
                dim = _operand_dim_token(inputs[0], dim_values)
                if dim is not None:
                    dim_values[output.name] = dim
                elif (
                    isinstance(inputs[0], GraphExpr)
                    and not inputs[0].inputs
                    and not inputs[0].attrs
                    and inputs[0].op.name in globals_env
                    and isinstance(globals_env[inputs[0].op.name].type_expr, TypeDim | TypeInt)
                ):
                    dim_values[output.name] = inputs[0].op.name
    outputs = tuple(
        _refresh_graph_operand_types(
            output,
            env=env,
            globals_env=globals_env,
            modules_by_name=modules_by_name,
            dim_values=dim_values,
        )
        for output in module.outputs
    )
    return replace(module, nodes=tuple(nodes), outputs=outputs)


def _refresh_graph_program_types(graph: GraphProgram) -> GraphProgram:
    modules_by_name = {module.name: module for module in graph.modules}
    globals_env = {
        module.name: GraphValue(
            name=module.name,
            type_expr=_module_output_types(module)[0],
            dims=None,
        )
        for module in graph.modules
        if not module.inputs and len(module.outputs) == 1
    }
    return replace(
        graph,
        modules=tuple(
            _refresh_graph_module_types(
                module,
                globals_env=globals_env,
                modules_by_name=modules_by_name,
            )
            for module in graph.modules
        ),
    )


def _atomic_int_constant_dims(graph: GraphProgram) -> dict[str, DimToken]:
    modules_by_name = {module.name: module for module in graph.modules}
    evaluating: set[str] = set()
    memo: dict[str, int | bool] = {}

    def eval_operand(operand: GraphOperand, env: Mapping[str, int]) -> int | bool | None:
        if isinstance(operand, GraphLiteral):
            if type(operand.value) is int or isinstance(operand.value, bool):
                return operand.value
            return None
        if isinstance(operand, GraphValueRef):
            return env.get(operand.name)
        if not isinstance(operand, GraphExpr):
            return None
        values = tuple(eval_operand(item, env) for item in operand.inputs)
        if any(value is None for value in values):
            return None
        if operand.op.name == "core.alias" and len(values) == 1:
            return values[0]
        if operand.op.name == "core.ascribe" and len(values) == 1:
            return values[0]
        if operand.op.name == "core.select" and len(values) == 3 and isinstance(values[0], bool):
            return values[1] if values[0] else values[2]
        if operand.op.name.startswith("core.binary.") and len(values) == 2:
            left, right = values
            if type(left) is not int or type(right) is not int:
                return None
            op = operand.op.name.removeprefix("core.binary.")
            if op == "+":
                return left + right
            if op == "-":
                return left - right
            if op == "*":
                return left * right
            if op == "/" and right != 0 and left % right == 0:
                return left // right
            return None
        if operand.op.name in modules_by_name and not operand.inputs and not operand.attrs:
            return eval_module(operand.op.name)
        return None

    def eval_module(name: str) -> int | bool | None:
        if name in memo:
            return memo[name]
        if name in evaluating:
            return None
        module = modules_by_name.get(name)
        if module is None or module.inputs or len(module.outputs) != 1:
            return None
        evaluating.add(name)
        env: dict[str, int] = {}
        for node in module.nodes:
            if len(node.outputs) != 1 or node.attrs:
                evaluating.remove(name)
                return None
            if node.op.name in modules_by_name and not node.inputs:
                value = eval_module(node.op.name)
            else:
                value = eval_operand(
                    GraphExpr(
                        op=node.op,
                        inputs=node.inputs,
                        attrs=node.attrs,
                        type_expr=node.type_expr,
                        dims=node.dims,
                    ),
                    env,
                )
            if type(value) is not int and type(value) is not bool:
                evaluating.remove(name)
                return None
            env[node.outputs[0].name] = value
        value = eval_operand(module.outputs[0], env)
        evaluating.remove(name)
        if type(value) is int or type(value) is bool:
            memo[name] = value
            return value
        return None

    for module in graph.modules:
        value = eval_module(module.name)
        if value is not None:
            memo[module.name] = value
    return {name: value for name, value in memo.items() if type(value) is int}


def _module_dim_refs(module: GraphModule) -> set[str]:
    refs: set[str] = set()
    _module_metadata_refs(module, set(), refs)

    def collect_type(tp: TypeExpr | None) -> None:
        if tp is None:
            return
        if isinstance(tp, TypeTensor):
            for dim in tp.dims:
                refs.update(dim_token_names(dim))
            return
        if isinstance(tp, TypeNamed):
            for dim in tp.args:
                refs.update(dim_token_names(dim))
            return
        if isinstance(tp, TypeOptional):
            collect_type(tp.inner)
            return
        if isinstance(tp, TypeList):
            collect_type(tp.item)
            return
        if isinstance(tp, TypeTuple):
            for item in tp.items:
                collect_type(item)

    for value in module.inputs:
        collect_type(value.type_expr)
        if value.dims is not None:
            for dim in value.dims:
                refs.update(dim_token_names(dim))
    for node in module.nodes:
        collect_type(node.type_expr)
        if node.dims is not None:
            for dim in node.dims:
                refs.update(dim_token_names(dim))
        for value in node.outputs:
            collect_type(value.type_expr)
            if value.dims is not None:
                for dim in value.dims:
                    refs.update(dim_token_names(dim))
        for operand in (*node.inputs, *node.attrs.values()):
            collect_type(graph_operand_type(operand))
    for output in module.outputs:
        collect_type(graph_operand_type(output))
    collect_type(module.return_type_expr)
    return refs


def _is_dim_equality_to_constant(constraint: Constraint, name: str, value: int) -> bool:
    if constraint.relation != "=":
        return False
    return (constraint.left == name and constraint.right == value) or (
        constraint.left == value and constraint.right == name
    )


def _module_allows_constant_dim_substitution(
    module: GraphModule,
    *,
    name: str,
    value: int,
) -> bool:
    if name not in _module_dim_refs(module):
        return False
    if any(_is_dim_equality_to_constant(constraint, name, value) for constraint in module.constraints):
        return True
    return False


def _substitute_atomic_constant_dims_local(graph: GraphProgram) -> GraphProgram:
    constants = _atomic_int_constant_dims(graph)
    if not constants:
        return graph
    modules = list(graph.modules)
    changed = False
    for index, module in enumerate(tuple(modules)):
        allowed = {
            name: value
            for name, value in constants.items()
            if _module_allows_constant_dim_substitution(module, name=name, value=value)
        }
        if not allowed:
            continue
        candidate_module = _substitute_graph_module_dims(module, allowed)
        candidate_modules = list(modules)
        candidate_modules[index] = candidate_module
        candidate = replace(graph, modules=tuple(candidate_modules))
        candidate = _refresh_graph_program_types(candidate)
        validate_graph_program(candidate)
        modules = list(candidate.modules)
        graph = candidate
        changed = True
    if not changed:
        return graph
    validate_graph_program(graph)
    return graph


def _fold_graph_binary(op_name: str, left: GraphLiteral, right: GraphLiteral, template: GraphExpr | GraphNode) -> GraphLiteral | None:
    op = op_name.removeprefix("core.binary.")
    lval = left.value
    rval = right.value
    if isinstance(lval, bool) and isinstance(rval, bool):
        if op == "and":
            return _literal_like(lval and rval, template)
        if op == "or":
            return _literal_like(lval or rval, template)
        if op == "==":
            return _literal_like(lval == rval, template)
        if op == "!=":
            return _literal_like(lval != rval, template)
    if type(lval) is int and type(rval) is int:
        if op == "+":
            return _literal_like(lval + rval, template)
        if op == "-":
            return _literal_like(lval - rval, template)
        if op == "*":
            return _literal_like(lval * rval, template)
        if op == "/" and rval != 0 and lval % rval == 0:
            return _literal_like(lval // rval, template)
        if op == "==":
            return _literal_like(lval == rval, template)
        if op == "!=":
            return _literal_like(lval != rval, template)
        if op == "<":
            return _literal_like(lval < rval, template)
        if op == "<=":
            return _literal_like(lval <= rval, template)
        if op == ">":
            return _literal_like(lval > rval, template)
        if op == ">=":
            return _literal_like(lval >= rval, template)
    if isinstance(lval, float) and isinstance(rval, float):
        if op == "+":
            return _literal_like(lval + rval, template)
        if op == "-":
            return _literal_like(lval - rval, template)
        if op == "*":
            return _literal_like(lval * rval, template)
        if op == "/" and rval != 0.0:
            return _literal_like(lval / rval, template)
    if lval is None and rval is None:
        if op == "==":
            return _literal_like(True, template)
        if op == "!=":
            return _literal_like(False, template)
    return None


def _operand_refs(operand: GraphOperand, out: set[str]) -> None:
    if isinstance(operand, GraphValueRef):
        out.add(operand.name)
        return
    if isinstance(operand, GraphPath):
        out.update(graph_path_template_names(operand))
        return
    if isinstance(operand, GraphExpr):
        for item in operand.inputs:
            _operand_refs(item, out)
        for item in operand.attrs.values():
            _operand_refs(item, out)


def _operand_module_calls(operand: GraphOperand, module_names: set[str], out: set[str]) -> None:
    if isinstance(operand, GraphPath):
        out.update(name for name in graph_path_template_names(operand) if name in module_names)
        return
    if not isinstance(operand, GraphExpr):
        return
    if operand.op.name in module_names:
        out.add(operand.op.name)
    for item in operand.inputs:
        _operand_module_calls(item, module_names, out)
    for item in operand.attrs.values():
        _operand_module_calls(item, module_names, out)


def _type_module_refs(type_expr: TypeExpr | None, module_names: set[str], out: set[str]) -> None:
    if type_expr is None:
        return
    if isinstance(type_expr, TypeTensor):
        for dim in type_expr.dims:
            out.update(name for name in dim_token_names(dim) if name in module_names)
        return
    if isinstance(type_expr, TypeNamed):
        for dim in type_expr.args:
            out.update(name for name in dim_token_names(dim) if name in module_names)
        return
    if isinstance(type_expr, TypeOptional):
        _type_module_refs(type_expr.inner, module_names, out)
        return
    if isinstance(type_expr, TypeList):
        _type_module_refs(type_expr.item, module_names, out)
        return
    if isinstance(type_expr, TypeTuple):
        for item in type_expr.items:
            _type_module_refs(item, module_names, out)


def _dim_module_refs(dim: DimToken, module_names: set[str], out: set[str]) -> None:
    out.update(name for name in dim_token_names(dim) if name in module_names)


def _constraint_atom_module_refs(
    atom: ConstraintAtom,
    module_names: set[str],
    out: set[str],
) -> None:
    if isinstance(atom, str):
        if atom in module_names:
            out.add(atom)
        return
    if isinstance(atom, DimExprBinary):
        _dim_module_refs(atom, module_names, out)


def _constraint_operand_module_refs(
    operand: ConstraintOperand,
    module_names: set[str],
    out: set[str],
) -> None:
    if isinstance(operand, tuple):
        for item in operand:
            _constraint_atom_module_refs(item, module_names, out)
        return
    _constraint_atom_module_refs(operand, module_names, out)


def _constraint_module_refs(
    constraint: Constraint,
    module_names: set[str],
    out: set[str],
) -> None:
    _constraint_operand_module_refs(constraint.left, module_names, out)
    if constraint.right is not None:
        _constraint_operand_module_refs(constraint.right, module_names, out)
    for guard in constraint.guards:
        _constraint_module_refs(guard, module_names, out)


def _module_metadata_refs(module: GraphModule, module_names: set[str], out: set[str]) -> None:
    for value in module.inputs:
        _type_module_refs(value.type_expr, module_names, out)
        if value.dims is not None:
            for dim in value.dims:
                _dim_module_refs(dim, module_names, out)
    for node in module.nodes:
        _type_module_refs(node.type_expr, module_names, out)
        if node.dims is not None:
            for dim in node.dims:
                _dim_module_refs(dim, module_names, out)
        for value in node.outputs:
            _type_module_refs(value.type_expr, module_names, out)
            if value.dims is not None:
                for dim in value.dims:
                    _dim_module_refs(dim, module_names, out)
        for operand in (*node.inputs, *node.attrs.values()):
            _type_module_refs(graph_operand_type(operand), module_names, out)
    for output in module.outputs:
        _type_module_refs(graph_operand_type(output), module_names, out)
    _type_module_refs(module.return_type_expr, module_names, out)
    for constraint in module.constraints:
        _constraint_module_refs(constraint, module_names, out)


def _count_operand_module_calls(
    operand: GraphOperand,
    module_names: set[str],
    counts: Counter[str],
) -> None:
    if not isinstance(operand, GraphExpr):
        return
    if operand.op.name in module_names:
        counts[operand.op.name] += 1
    for item in operand.inputs:
        _count_operand_module_calls(item, module_names, counts)
    for item in operand.attrs.values():
        _count_operand_module_calls(item, module_names, counts)


def _replace_operand_refs(
    operand: GraphOperand,
    subst: Mapping[str, GraphOperand],
    *,
    fold: bool = True,
    module_effects: Mapping[str, GraphEffect] | None = None,
) -> GraphOperand:
    if isinstance(operand, GraphValueRef):
        return subst.get(operand.name, operand)
    if isinstance(operand, GraphPath):
        return _replace_path_template_refs(operand, subst)
    if isinstance(operand, GraphExpr):
        rewritten = replace(
            operand,
            inputs=tuple(
                _replace_operand_refs(
                    item,
                    subst,
                    fold=fold,
                    module_effects=module_effects,
                )
                for item in operand.inputs
            ),
            attrs={
                key: _replace_operand_refs(
                    value,
                    subst,
                    fold=fold,
                    module_effects=module_effects,
                )
                for key, value in operand.attrs.items()
            },
        )
        return _fold_operand(rewritten, module_effects=module_effects) if fold else rewritten
    return operand


def _operand_path_fragment(operand: GraphOperand) -> tuple[bool, tuple[str, ...]] | None:
    if isinstance(operand, GraphPath):
        return operand.absolute, operand.parts
    if isinstance(operand, GraphValueRef):
        return False, (operand.name,)
    if isinstance(operand, GraphLiteral) and isinstance(operand.value, str | int):
        return False, (str(operand.value),)
    return None


def _replace_path_template_refs(path: GraphPath, subst: Mapping[str, GraphOperand]) -> GraphPath:
    absolute = path.absolute
    parts: list[str] = []
    changed = False
    for part in path.parts:
        names = graph_path_template_names(GraphPath(absolute=path.absolute, parts=(part,)))
        if not names:
            parts.append(part)
            continue
        if part.startswith("{") and part.endswith("}") and part[1:-1] in subst:
            replacement = _operand_path_fragment(subst[part[1:-1]])
            if replacement is not None:
                repl_absolute, repl_parts = replacement
                absolute = absolute or repl_absolute
                parts.extend(repl_parts)
                changed = True
                continue
        rewritten = part
        for name in sorted(names, key=len, reverse=True):
            if name not in subst:
                continue
            replacement = _operand_path_fragment(subst[name])
            if replacement is None:
                continue
            repl_absolute, repl_parts = replacement
            absolute = absolute or repl_absolute
            rewritten = rewritten.replace("{" + name + "}", ".".join(repl_parts))
            changed = True
        parts.append(rewritten)
    if not changed:
        return path
    return GraphPath(absolute=absolute, parts=tuple(part for part in parts if part))


def _fold_operand(
    operand: GraphOperand,
    *,
    module_effects: Mapping[str, GraphEffect] | None = None,
) -> GraphOperand:
    if not isinstance(operand, GraphExpr):
        return operand
    expr = replace(
        operand,
        inputs=tuple(_fold_operand(item, module_effects=module_effects) for item in operand.inputs),
        attrs={
            key: _fold_operand(value, module_effects=module_effects)
            for key, value in operand.attrs.items()
        },
    )
    if expr.op.name == "core.ascribe" and len(expr.inputs) == 1:
        return expr.inputs[0]
    if expr.op.name == "core.select" and len(expr.inputs) == 3 and isinstance(expr.inputs[0], GraphLiteral):
        if isinstance(expr.inputs[0].value, bool):
            selected = expr.inputs[1] if expr.inputs[0].value else expr.inputs[2]
            if _is_atomic_operand(selected) or graph_operand_effect(
                selected,
                module_effects=dict(module_effects or {}),
            ) == GraphEffect.TOTAL_PURE:
                return selected
    if expr.op.name.startswith("core.binary.") and len(expr.inputs) == 2:
        left, right = expr.inputs
        if isinstance(left, GraphLiteral) and isinstance(right, GraphLiteral):
            folded = _fold_graph_binary(expr.op.name, left, right, expr)
            if folded is not None:
                return folded
    return expr


def _rewrite_node_operands(
    node: GraphNode,
    subst: Mapping[str, GraphOperand],
    *,
    fold: bool = True,
    module_effects: Mapping[str, GraphEffect] | None = None,
) -> GraphNode:
    return replace(
        node,
        inputs=tuple(
            _replace_operand_refs(
                item,
                subst,
                fold=fold,
                module_effects=module_effects,
            )
            for item in node.inputs
        ),
        attrs={
            key: _replace_operand_refs(
                value,
                subst,
                fold=fold,
                module_effects=module_effects,
            )
            for key, value in node.attrs.items()
        },
    )


def _node_replacement(
    node: GraphNode,
    *,
    config: GraphOptimizeConfig,
    module_effects: Mapping[str, GraphEffect],
) -> GraphOperand | None:
    if len(node.outputs) != 1:
        return None
    if config.atomic_alias_cleanup and node.op.name == "core.alias" and len(node.inputs) == 1:
        return node.inputs[0]
    if config.atomic_alias_cleanup and node.op.name == "core.ascribe" and len(node.inputs) == 1:
        return node.inputs[0]
    if config.atomic_alias_cleanup and node.op.name in {"core.list", "core.tuple"}:
        return GraphExpr(
            op=node.op,
            inputs=node.inputs,
            attrs=node.attrs,
            type_expr=node.outputs[0].type_expr,
            dims=node.outputs[0].dims,
        )
    if (
        config.constant_folding
        and node.op.name == "core.select"
        and len(node.inputs) == 3
        and isinstance(node.inputs[0], GraphLiteral)
    ):
        if isinstance(node.inputs[0].value, bool):
            selected = node.inputs[1] if node.inputs[0].value else node.inputs[2]
            if _is_atomic_operand(selected) or graph_operand_effect(
                selected,
                module_effects=dict(module_effects),
            ) == GraphEffect.TOTAL_PURE:
                return selected
    if config.constant_folding and node.op.name.startswith("core.binary.") and len(node.inputs) == 2:
        left, right = node.inputs
        if isinstance(left, GraphLiteral) and isinstance(right, GraphLiteral):
            return _fold_graph_binary(node.op.name, left, right, node)
    if node.op.name == "core.tuple" and len(node.inputs) == len(node.outputs):
        return None
    return None


def _optimize_module_local(
    module: GraphModule,
    *,
    config: GraphOptimizeConfig,
    module_effects: Mapping[str, GraphEffect],
) -> GraphModule:
    before_outputs = module.outputs
    subst: dict[str, GraphOperand] = {}
    nodes: list[GraphNode] = []
    for node in module.nodes:
        rewritten = _rewrite_node_operands(
            node,
            subst,
            fold=config.constant_folding,
            module_effects=module_effects,
        )
        replacement = _node_replacement(
            rewritten,
            config=config,
            module_effects=module_effects,
        )
        if replacement is not None and len(rewritten.outputs) == 1:
            subst[rewritten.outputs[0].name] = replacement
            continue
        nodes.append(rewritten)
    outputs = tuple(
        _replace_operand_refs(
            item,
            subst,
            fold=config.constant_folding,
            module_effects=module_effects,
        )
        for item in module.outputs
    )
    module = replace(module, nodes=tuple(nodes), outputs=outputs)
    module = _returned_name_preserving_module(module, before_outputs)
    if not config.dead_temp_elimination:
        return module
    return _dead_temp_eliminate_module(module, module_effects=module_effects)


def _dead_temp_eliminate_module(
    module: GraphModule,
    *,
    module_effects: Mapping[str, GraphEffect] | None = None,
) -> GraphModule:
    live: set[str] = set()
    for output in module.outputs:
        _operand_refs(output, live)
    kept_rev: list[GraphNode] = []
    for node in reversed(module.nodes):
        output_names = {value.name for value in node.outputs}
        if output_names and not (output_names & live) and _is_total_pure_op(
            node.op.name,
            module_effects,
        ):
            continue
        live.difference_update(output_names)
        for operand in node.inputs:
            _operand_refs(operand, live)
        for operand in node.attrs.values():
            _operand_refs(operand, live)
        kept_rev.append(node)
    return replace(module, nodes=tuple(reversed(kept_rev)))


def _returned_name_preserving_module(module: GraphModule, before_outputs: tuple[GraphOperand, ...]) -> GraphModule:
    renames: dict[str, str] = {}
    defined_names = {value.name for value in module.inputs}
    for node in module.nodes:
        defined_names.update(value.name for value in node.outputs)
    for before, after in zip(before_outputs, module.outputs, strict=False):
        if not isinstance(before, GraphValueRef) or not isinstance(after, GraphValueRef):
            continue
        if before.name == after.name:
            continue
        if before.name in defined_names:
            continue
        renames[after.name] = before.name
        defined_names.add(before.name)
    if not renames:
        return module
    return replace(
        module,
        nodes=tuple(
            replace(
                node,
                inputs=tuple(_rename_operand(item, renames) for item in node.inputs),
                attrs={key: _rename_operand(value, renames) for key, value in node.attrs.items()},
                outputs=tuple(
                    replace(output, name=renames.get(output.name, output.name))
                    for output in node.outputs
                ),
            )
            for node in module.nodes
        ),
        outputs=tuple(_rename_operand(output, renames) for output in module.outputs),
    )


def _graph_call_graph(graph: GraphProgram) -> dict[str, set[str]]:
    module_names = {module.name for module in graph.modules}
    calls: dict[str, set[str]] = {module.name: set() for module in graph.modules}
    for module in graph.modules:
        for node in module.nodes:
            if node.op.name in module_names:
                calls[module.name].add(node.op.name)
            for operand in node.inputs:
                _operand_module_calls(operand, module_names, calls[module.name])
            for operand in node.attrs.values():
                _operand_module_calls(operand, module_names, calls[module.name])
        for operand in module.outputs:
            _operand_module_calls(operand, module_names, calls[module.name])
        _module_metadata_refs(module, module_names, calls[module.name])
    return calls


def _strongly_connected_components(edges: Mapping[str, set[str]]) -> list[set[str]]:
    index = 0
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[set[str]] = []

    def visit(name: str) -> None:
        nonlocal index
        indices[name] = index
        lowlinks[name] = index
        index += 1
        stack.append(name)
        on_stack.add(name)
        for target in sorted(edges.get(name, ())):
            if target not in edges:
                continue
            if target not in indices:
                visit(target)
                lowlinks[name] = min(lowlinks[name], lowlinks[target])
            elif target in on_stack:
                lowlinks[name] = min(lowlinks[name], indices[target])
        if lowlinks[name] != indices[name]:
            return
        component: set[str] = set()
        while True:
            item = stack.pop()
            on_stack.remove(item)
            component.add(item)
            if item == name:
                break
        components.append(component)

    for name in sorted(edges):
        if name not in indices:
            visit(name)
    return components


def _recursive_modules(graph: GraphProgram) -> set[str]:
    edges = _graph_call_graph(graph)
    recursive: set[str] = set()
    for component in _strongly_connected_components(edges):
        if len(component) > 1:
            recursive.update(component)
            continue
        name = next(iter(component))
        if name in edges.get(name, ()):
            recursive.add(name)
    return recursive


def prune_graph_to_main(graph: GraphProgram) -> GraphProgram:
    validate_graph_program(graph)
    calls = _graph_call_graph(graph)
    seen: set[str] = set()
    stack = [graph.main_module]
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        stack.extend(sorted(calls.get(name, ())))
    pruned = replace(graph, modules=tuple(module for module in graph.modules if module.name in seen))
    validate_graph_program(pruned)
    return pruned


def _call_counts(graph: GraphProgram) -> Counter[str]:
    module_names = {module.name for module in graph.modules}
    counts: Counter[str] = Counter()
    for module in graph.modules:
        for node in module.nodes:
            if node.op.name in module_names:
                counts[node.op.name] += 1
            for operand in node.inputs:
                _count_operand_module_calls(operand, module_names, counts)
            for operand in node.attrs.values():
                _count_operand_module_calls(operand, module_names, counts)
        for operand in module.outputs:
            _count_operand_module_calls(operand, module_names, counts)
    return counts


def _top_level_call_counts(graph: GraphProgram) -> Counter[str]:
    module_names = {module.name for module in graph.modules}
    counts: Counter[str] = Counter()
    for module in graph.modules:
        for node in module.nodes:
            if node.op.name in module_names:
                counts[node.op.name] += 1
        for output in module.outputs:
            if isinstance(output, GraphExpr) and output.op.name in module_names:
                counts[output.op.name] += 1
    return counts


def _has_safe_specialization_actual(node: GraphNode, module: GraphModule) -> bool:
    if len(node.inputs) != len(module.inputs):
        return False
    return any(_is_safe_specialization_operand(item) for item in node.inputs)


def _can_specialize_module(module: GraphModule, *, recursive_modules: set[str], main_module: str) -> bool:
    if module.name == main_module:
        return False
    if module.name in recursive_modules:
        return False
    if module.constraints:
        return False
    return True


def _specialized_module(
    module: GraphModule,
    *,
    name: str,
    call_node: GraphNode,
) -> GraphModule | None:
    if len(call_node.inputs) != len(module.inputs):
        return None
    subst: dict[str, GraphOperand] = {}
    kept_inputs: list[GraphValue] = []
    for formal, actual in zip(module.inputs, call_node.inputs, strict=True):
        if _is_safe_specialization_operand(actual):
            subst[formal.name] = actual
        else:
            kept_inputs.append(formal)
    if not subst:
        return None
    nodes = tuple(_rewrite_node_operands(node, subst) for node in module.nodes)
    outputs = tuple(_replace_operand_refs(output, subst) for output in module.outputs)
    return replace(module, name=name, inputs=tuple(kept_inputs), nodes=nodes, outputs=outputs)


def _rewrite_call_to_specialized(node: GraphNode, original: GraphModule, specialized_name: str) -> GraphNode:
    inputs = tuple(
        actual
        for formal, actual in zip(original.inputs, node.inputs, strict=True)
        if not _is_safe_specialization_operand(actual)
    )
    return replace(node, op=GraphOp(specialized_name), inputs=inputs)


def _specialize_definitions(graph: GraphProgram, *, config: GraphOptimizeConfig) -> GraphProgram:
    if config.specialize_definitions not in _SPECIALIZE_MODES:
        raise ValueError(
            "GraphOptimizeConfig.specialize_definitions must be one of: "
            + ", ".join(sorted(_SPECIALIZE_MODES))
        )
    if config.specialize_definitions == "off":
        return graph
    modules_by_name = {module.name: module for module in graph.modules}
    counts = _call_counts(graph)
    recursive = _recursive_modules(graph)
    replacements: dict[tuple[str, str], str] = {}
    new_modules: list[GraphModule] = list(graph.modules)
    used_module_names = {module.name for module in graph.modules}
    clone_index = 0
    for caller in graph.modules:
        for node in caller.nodes:
            callee = modules_by_name.get(node.op.name)
            if callee is None:
                continue
            if not _can_specialize_module(
                callee,
                recursive_modules=recursive,
                main_module=graph.main_module,
            ):
                continue
            if config.specialize_definitions == "single-callsite" and counts[callee.name] != 1:
                continue
            if not _has_safe_specialization_actual(node, callee):
                continue
            while True:
                clone_index += 1
                clone_name = f"{callee.name}__spec_{clone_index}"
                if clone_name not in used_module_names:
                    used_module_names.add(clone_name)
                    break
            specialized = _specialized_module(callee, name=clone_name, call_node=node)
            if specialized is None:
                continue
            replacements[(caller.name, node.id)] = clone_name
            new_modules.append(specialized)
    if not replacements:
        return graph
    original_by_name = modules_by_name
    rewritten_modules: list[GraphModule] = []
    for module in new_modules:
        if module.name not in {item.name for item in graph.modules}:
            rewritten_modules.append(module)
            continue
        nodes: list[GraphNode] = []
        for node in module.nodes:
            clone_name = replacements.get((module.name, node.id))
            if clone_name is not None:
                original = original_by_name[node.op.name]
                nodes.append(_rewrite_call_to_specialized(node, original, clone_name))
            else:
                nodes.append(node)
        rewritten_modules.append(replace(module, nodes=tuple(nodes)))
    specialized_graph = replace(graph, modules=tuple(rewritten_modules))
    validate_graph_program(specialized_graph)
    return specialized_graph


def _module_value_names(module: GraphModule) -> set[str]:
    names = {value.name for value in module.inputs}
    for node in module.nodes:
        names.update(value.name for value in node.outputs)
    for output in module.outputs:
        _operand_refs(output, names)
    return names


def _operand_has_core_select(operand: GraphOperand) -> bool:
    if not isinstance(operand, GraphExpr):
        return False
    if operand.op.name == "core.select":
        return True
    return any(_operand_has_core_select(item) for item in operand.inputs) or any(
        _operand_has_core_select(value) for value in operand.attrs.values()
    )


def _module_has_core_select(module: GraphModule) -> bool:
    for node in module.nodes:
        if node.op.name == "core.select":
            return True
        if any(_operand_has_core_select(item) for item in node.inputs):
            return True
        if any(_operand_has_core_select(value) for value in node.attrs.values()):
            return True
    return any(_operand_has_core_select(output) for output in module.outputs)


def _can_inline_module(
    module: GraphModule,
    *,
    module_effects: Mapping[str, GraphEffect],
    recursive_modules: set[str],
    main_module: str,
) -> bool:
    if module.name == main_module:
        return False
    if module.name in recursive_modules:
        return False
    if module.constraints:
        return False
    if _module_has_core_select(module):
        return False
    return module_effects.get(module.name) == GraphEffect.TOTAL_PURE


def _is_atomic_constant_module(module: GraphModule) -> bool:
    return (
        not module.inputs
        and not module.nodes
        and len(module.outputs) == 1
        and _is_atomic_operand(module.outputs[0])
    )


def _can_inline_call_node(node: GraphNode, callee: GraphModule) -> bool:
    if len(node.inputs) != len(callee.inputs):
        return False
    if len(node.outputs) != len(callee.outputs):
        return False
    if node.attrs:
        return False
    for actual, formal in zip(node.inputs, callee.inputs, strict=True):
        if not graph_type_compatible(graph_operand_type(actual), formal.type_expr):
            return False
    for call_output, returned in zip(node.outputs, callee.outputs, strict=True):
        if not graph_type_compatible(graph_operand_type(returned), call_output.type_expr):
            return False
    return True


def _can_inline_call_expr(expr: GraphExpr, callee: GraphModule) -> bool:
    if len(expr.inputs) != len(callee.inputs):
        return False
    if len(callee.outputs) != 1:
        return False
    if expr.attrs:
        return False
    for actual, formal in zip(expr.inputs, callee.inputs, strict=True):
        if not graph_type_compatible(graph_operand_type(actual), formal.type_expr):
            return False
    if not graph_type_compatible(graph_operand_type(callee.outputs[0]), expr.type_expr):
        return False
    return True


def _rename_operand(operand: GraphOperand, renames: Mapping[str, str]) -> GraphOperand:
    if isinstance(operand, GraphValueRef):
        return replace(operand, name=renames.get(operand.name, operand.name))
    if isinstance(operand, GraphPath):
        subst = {
            old: GraphValueRef(name=new, type_expr=TypePath())
            for old, new in renames.items()
            if old != new
        }
        return _replace_path_template_refs(operand, subst)
    if isinstance(operand, GraphExpr):
        return replace(
            operand,
            inputs=tuple(_rename_operand(item, renames) for item in operand.inputs),
            attrs={key: _rename_operand(value, renames) for key, value in operand.attrs.items()},
        )
    return operand


def _inline_safe_modules(graph: GraphProgram, *, config: GraphOptimizeConfig) -> GraphProgram:
    counts = _call_counts(graph)
    top_level_counts = _top_level_call_counts(graph)
    modules_by_name = {module.name: module for module in graph.modules}
    module_effects = infer_graph_module_effects(graph.modules)
    recursive = _recursive_modules(graph)
    inlineable = {
        module.name: module
        for module in graph.modules
        if (
            (
                counts[module.name] == 1
                and top_level_counts[module.name] == 1
                and not _is_atomic_constant_module(module)
            )
            or (
                _is_atomic_constant_module(module)
                and config.constant_dim_substitution
            )
        )
        and _can_inline_module(
            module,
            module_effects=module_effects,
            recursive_modules=recursive,
            main_module=graph.main_module,
        )
    }
    if not inlineable:
        return graph
    rewritten_modules: list[GraphModule] = []
    for module in graph.modules:
        nodes: list[GraphNode] = []
        subst: dict[str, GraphOperand] = {}
        temp_index = 0
        used_names = _module_value_names(module)

        def _inline_expr_call(expr: GraphExpr, *, prefix: str) -> GraphOperand:
            nonlocal temp_index
            callee = inlineable.get(expr.op.name)
            if callee is None or not _can_inline_call_expr(expr, callee):
                return expr
            formal_subst = {
                formal.name: actual
                for formal, actual in zip(callee.inputs, expr.inputs, strict=True)
            }
            renames: dict[str, str] = {}
            for inner in callee.nodes:
                for output in inner.outputs:
                    while True:
                        temp_index += 1
                        candidate = f"{prefix}__inl_{temp_index}_{output.name}"
                        if candidate not in used_names:
                            used_names.add(candidate)
                            renames[output.name] = candidate
                            break
            for inner in callee.nodes:
                renamed_inputs = tuple(
                    _replace_operand_refs(
                        _rename_operand(item, renames),
                        formal_subst,
                        fold=config.constant_folding,
                    )
                    for item in inner.inputs
                )
                renamed_attrs = {
                    key: _replace_operand_refs(
                        _rename_operand(value, renames),
                        formal_subst,
                        fold=config.constant_folding,
                    )
                    for key, value in inner.attrs.items()
                }
                renamed_outputs = tuple(
                    replace(output, name=renames.get(output.name, output.name))
                    for output in inner.outputs
                )
                nodes.append(
                    replace(
                        inner,
                        id=f"{module.name}:inl:output:{expr.op.name}:{inner.id}",
                        inputs=renamed_inputs,
                        attrs=renamed_attrs,
                        outputs=renamed_outputs,
                        source_module=module.name,
                    )
                )
            return _replace_operand_refs(
                _rename_operand(callee.outputs[0], renames),
                formal_subst,
                fold=config.constant_folding,
            )

        def _inline_nested_expr_calls(operand: GraphOperand, *, prefix: str) -> GraphOperand:
            if not isinstance(operand, GraphExpr):
                return operand
            rewritten = replace(
                operand,
                inputs=tuple(
                    _inline_nested_expr_calls(item, prefix=f"{prefix}_arg{index + 1}")
                    for index, item in enumerate(operand.inputs)
                ),
                attrs={
                    key: _inline_nested_expr_calls(value, prefix=f"{prefix}_{key}")
                    for key, value in operand.attrs.items()
                },
            )
            return _inline_expr_call(rewritten, prefix=prefix)

        for node in module.nodes:
            node = _rewrite_node_operands(node, subst, fold=config.constant_folding)
            node = replace(
                node,
                inputs=tuple(
                    _inline_nested_expr_calls(
                        item,
                        prefix=f"{node.outputs[0].name}_arg{index + 1}",
                    )
                    for index, item in enumerate(node.inputs)
                ),
                attrs={
                    key: _inline_nested_expr_calls(
                        value,
                        prefix=f"{node.outputs[0].name}_{key}",
                    )
                    for key, value in node.attrs.items()
                },
            )
            callee = inlineable.get(node.op.name)
            if callee is None:
                nodes.append(node)
                continue
            if not _can_inline_call_node(node, callee):
                nodes.append(node)
                continue
            formal_subst = {formal.name: actual for formal, actual in zip(callee.inputs, node.inputs, strict=True)}
            renames: dict[str, str] = {}
            for inner in callee.nodes:
                for output in inner.outputs:
                    while True:
                        temp_index += 1
                        candidate = f"{node.outputs[0].name}__inl_{temp_index}_{output.name}"
                        if candidate not in used_names:
                            used_names.add(candidate)
                            renames[output.name] = candidate
                            break
            for inner in callee.nodes:
                renamed_inputs = tuple(
                    _replace_operand_refs(_rename_operand(item, renames), formal_subst, fold=config.constant_folding)
                    for item in inner.inputs
                )
                renamed_attrs = {
                    key: _replace_operand_refs(
                        _rename_operand(value, renames),
                        formal_subst,
                        fold=config.constant_folding,
                    )
                    for key, value in inner.attrs.items()
                }
                renamed_outputs = tuple(
                    replace(output, name=renames.get(output.name, output.name))
                    for output in inner.outputs
                )
                nodes.append(
                    replace(
                        inner,
                        id=f"{module.name}:inl:{node.id}:{inner.id}",
                        inputs=renamed_inputs,
                        attrs=renamed_attrs,
                        outputs=renamed_outputs,
                        source_module=module.name,
                    )
                )
            for output, returned in zip(node.outputs, callee.outputs, strict=True):
                subst[output.name] = _replace_operand_refs(
                    _rename_operand(returned, renames),
                    formal_subst,
                    fold=config.constant_folding,
                )
        outputs = tuple(
            _inline_nested_expr_calls(
                _replace_operand_refs(
                    output,
                    subst,
                    fold=config.constant_folding,
                ),
                prefix=f"__out_{index + 1}",
            )
            for index, output in enumerate(module.outputs)
        )
        rewritten_modules.append(replace(module, nodes=tuple(nodes), outputs=outputs))
    inlined = replace(graph, modules=tuple(rewritten_modules))
    inlined = prune_graph_to_main(inlined)
    validate_graph_program(inlined)
    return inlined


def optimize_graph_program(
    graph: GraphProgram,
    *,
    config: GraphOptimizeConfig | None = None,
) -> GraphProgram:
    config = config or GraphOptimizeConfig()
    validate_graph_program(graph)
    current = prune_graph_to_main(graph) if config.prune_to_main else graph
    for _ in range(config.max_iterations):
        if config.constant_dim_substitution:
            current = _substitute_atomic_constant_dims_local(current)
        before = current
        module_effects = infer_graph_module_effects(current.modules)
        current = replace(
            current,
            modules=tuple(
                _optimize_module_local(module, config=config, module_effects=module_effects)
                for module in current.modules
            ),
        )
        current = _specialize_definitions(current, config=config)
        if config.inline_safe:
            current = _inline_safe_modules(current, config=config)
        if config.prune_to_main:
            current = prune_graph_to_main(current)
        validate_graph_program(current)
        if current == before:
            return current
    return current


__all__ = ["GraphOptimizeConfig", "optimize_graph_program", "prune_graph_to_main"]
