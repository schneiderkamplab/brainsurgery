from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, replace

from ...ops import get_op_type_rule
from ..ast import (
    Constraint,
    ConstraintAtom,
    ConstraintOperand,
    DimExprBinary,
    DimToken,
    TypeAny,
    TypeBool,
    TypeDim,
    TypeExpr,
    TypeFloat,
    TypeInt,
    TypeList,
    TypeNamed,
    TypeNull,
    TypeOptional,
    TypeTensor,
    TypeTuple,
    dim_token_names,
)
from ..typecheck_shared import _PrimitiveTypeHelpers
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
from .domain import GraphDomainFact, GraphDomainKind, infer_main_module_domain_facts
from .effects import GraphEffect, graph_node_effect, graph_op_effect, graph_operand_effect, infer_graph_module_effects
from .substitute import (
    UnsupportedConstraintSubstitution,
    replace_constraint_refs,
    rename_operand,
    replace_operand_refs,
    substitute_dim_token,
    substitute_graph_module_dims,
    substitute_graph_node_dims,
    substitute_graph_operand_dims,
    substitute_type_expr,
)


@dataclass(frozen=True)
class GraphOptimizeConfig:
    prune_to_main: bool = True
    atomic_alias_cleanup: bool = True
    dead_temp_elimination: bool = True
    constant_folding: bool = True
    constant_dim_substitution: bool = True
    common_subexpression_elimination: bool = True
    specialize_definitions: str = "single-callsite"
    inline_safe: bool = True
    max_iterations: int = 64


_SPECIALIZE_MODES = {"off", "single-callsite", "monomorphize"}


def _is_atomic_operand(operand: GraphOperand) -> bool:
    return isinstance(operand, GraphValueRef | GraphLiteral | GraphPath)


def _path_has_template(path: GraphPath) -> bool:
    return any("{" in part or "}" in part for part in path.parts)


def _is_safe_specialization_operand(operand: GraphOperand) -> bool:
    if isinstance(operand, GraphPath):
        return True
    if isinstance(operand, GraphLiteral):
        return True
    return False


def _is_safe_callsite_specialization_operand(
    operand: GraphOperand,
    *,
    global_symbol_names: set[str],
) -> bool:
    return _is_safe_specialization_operand(operand) or (
        isinstance(operand, GraphValueRef) and operand.name in global_symbol_names
    ) or (
        isinstance(operand, GraphExpr)
        and operand.op.name in global_symbol_names
        and not operand.inputs
        and not operand.attrs
    )


def _is_safe_shared_specialization_operand(
    operand: GraphOperand,
    *,
    global_symbol_names: set[str],
) -> bool:
    if _is_safe_specialization_operand(operand):
        return True
    if isinstance(operand, GraphLiteral):
        return True
    if isinstance(operand, GraphValueRef) and operand.name in global_symbol_names:
        return True
    return (
        isinstance(operand, GraphExpr)
        and operand.op.name in global_symbol_names
        and not operand.inputs
        and not operand.attrs
    )


def _canonical_specialization_operand(
    operand: GraphOperand,
    *,
    global_symbol_names: set[str],
) -> GraphOperand:
    if (
        isinstance(operand, GraphExpr)
        and operand.op.name in global_symbol_names
        and not operand.inputs
        and not operand.attrs
    ):
        return GraphValueRef(
            name=operand.op.name,
            type_expr=operand.type_expr,
            dims=operand.dims,
        )
    return operand


def _specialization_actual_matches_formal(actual: GraphOperand, formal: GraphValue) -> bool:
    actual_type = graph_operand_type(actual)
    if formal.optional and isinstance(actual_type, TypeNull):
        return True
    return graph_type_compatible(actual_type, formal.type_expr)


def _domain_fact_specialization_operand(
    fact: GraphDomainFact | None,
    formal: GraphValue,
) -> GraphOperand | None:
    if fact is None or fact.kind == GraphDomainKind.UNKNOWN:
        return None
    if fact.kind == GraphDomainKind.NULL:
        candidate: GraphOperand = GraphLiteral(None, TypeNull())
    elif fact.kind == GraphDomainKind.LITERAL:
        candidate = GraphLiteral(fact.value, _literal_fact_type(fact.value, formal.type_expr))
    elif fact.kind == GraphDomainKind.PATH and isinstance(fact.value, GraphPath):
        candidate = fact.value
    else:
        return None
    if _specialization_actual_matches_formal(candidate, formal):
        return candidate
    return None


def _is_total_pure_op(op_name: str, module_effects: Mapping[str, GraphEffect] | None = None) -> bool:
    if module_effects is not None and op_name in module_effects:
        return module_effects[op_name] == GraphEffect.TOTAL_PURE
    return graph_op_effect(op_name) == GraphEffect.TOTAL_PURE


def _is_total_pure_node(
    node: GraphNode,
    module_effects: Mapping[str, GraphEffect] | None = None,
) -> bool:
    return graph_node_effect(node, module_effects=dict(module_effects or {})) == GraphEffect.TOTAL_PURE


def _literal_like(value: object, type_like: GraphOperand | GraphNode | GraphExpr) -> GraphLiteral:
    type_expr = getattr(type_like, "type_expr")
    return GraphLiteral(value=value, type_expr=type_expr)


def _bool_literal(value: bool) -> GraphLiteral:
    return GraphLiteral(value=value, type_expr=TypeBool())


def _literal_fact_type(value: object, formal_type: TypeExpr) -> TypeExpr:
    if isinstance(formal_type, TypeOptional):
        inner = formal_type.inner
        if value is not None and graph_type_compatible(inner, formal_type):
            return inner
    if isinstance(value, bool):
        return TypeBool()
    if type(value) is int and isinstance(formal_type, TypeOptional):
        inner = formal_type.inner
        if isinstance(inner, TypeDim | TypeInt):
            return inner
    if isinstance(value, float):
        return TypeFloat()
    if value is None:
        return TypeNull()
    return formal_type


def _validate_optimizer_graph(graph: GraphProgram, *, phase: str) -> None:
    try:
        validate_graph_program(graph)
        modules_by_name = {module.name: module for module in graph.modules}
        for module in graph.modules:
            _validate_optimizer_module_metadata(module, modules_by_name=modules_by_name)
    except ValueError as exc:
        raise ValueError(f"graph optimizer phase {phase!r} produced invalid graph: {exc}") from exc


def _validate_optimizer_module_metadata(
    module: GraphModule,
    *,
    modules_by_name: Mapping[str, GraphModule],
) -> None:
    dim_values: dict[str, DimToken] = {}
    for value in module.inputs:
        _require_value_dims_match_type(value, context=f"module {module.name!r} input")
    for node in module.nodes:
        if (
            isinstance(node.type_expr, TypeTensor)
            and node.dims is not None
            and not _optimizer_dims_metadata_compatible(node.type_expr.dims, node.dims)
        ):
            raise ValueError(
                f"node {node.id!r} has stale dims metadata: "
                f"type has {node.type_expr.dims!r}, dims has {node.dims!r}"
            )
        for output in node.outputs:
            _require_value_dims_match_type(output, context=f"node {node.id!r} output")
        for operand in (*node.inputs, *node.attrs.values()):
            _validate_optimizer_operand_metadata(operand, context=f"node {node.id!r} operand")
            _validate_optimizer_nested_call_results(
                operand,
                modules_by_name=modules_by_name,
                dim_values=dim_values,
                context=f"node {node.id!r} operand",
            )
        _validate_optimizer_call_result(
            node,
            modules_by_name=modules_by_name,
            dim_values=dim_values,
            context=f"node {node.id!r}",
        )
        if len(node.outputs) == 1 and isinstance(node.outputs[0].type_expr, TypeDim | TypeInt):
            dim = _operand_dim_token(
                GraphExpr(
                    op=node.op,
                    inputs=node.inputs,
                    attrs=node.attrs,
                    type_expr=node.type_expr,
                    dims=node.dims,
                ),
                dim_values,
            )
            if dim is not None:
                dim_values[node.outputs[0].name] = dim
    for output in module.outputs:
        _validate_optimizer_operand_metadata(output, context=f"module {module.name!r} return")
        _validate_optimizer_nested_call_results(
            output,
            modules_by_name=modules_by_name,
            dim_values=dim_values,
            context=f"module {module.name!r} return",
        )
    _validate_optimizer_constraints(module, modules_by_name=modules_by_name)


def _require_value_dims_match_type(value: GraphValue, *, context: str) -> None:
    if (
        isinstance(value.type_expr, TypeTensor)
        and value.dims is not None
        and not _optimizer_dims_metadata_compatible(value.type_expr.dims, value.dims)
    ):
        raise ValueError(
            f"{context} {value.name!r} has stale dims metadata: "
            f"type has {value.type_expr.dims!r}, dims has {value.dims!r}"
        )


def _validate_optimizer_operand_metadata(operand: GraphOperand, *, context: str) -> None:
    if isinstance(operand, GraphValueRef):
        if (
            isinstance(operand.type_expr, TypeTensor)
            and operand.dims is not None
            and not _optimizer_dims_metadata_compatible(operand.type_expr.dims, operand.dims)
        ):
            raise ValueError(
                f"{context} ref {operand.name!r} has stale dims metadata: "
                f"type has {operand.type_expr.dims!r}, dims has {operand.dims!r}"
            )
        return
    if isinstance(operand, GraphExpr):
        if (
            isinstance(operand.type_expr, TypeTensor)
            and operand.dims is not None
            and not _optimizer_dims_metadata_compatible(operand.type_expr.dims, operand.dims)
        ):
            raise ValueError(
                f"{context} expr {operand.op.name!r} has stale dims metadata: "
                f"type has {operand.type_expr.dims!r}, dims has {operand.dims!r}"
            )
        for item in operand.inputs:
            _validate_optimizer_operand_metadata(item, context=f"{context} input")
        for key, item in operand.attrs.items():
            _validate_optimizer_operand_metadata(item, context=f"{context} attr {key!r}")


def _validate_optimizer_nested_call_results(
    operand: GraphOperand,
    *,
    modules_by_name: Mapping[str, GraphModule],
    dim_values: Mapping[str, DimToken],
    context: str,
) -> None:
    if not isinstance(operand, GraphExpr):
        return
    _validate_optimizer_call_result(
        operand,
        modules_by_name=modules_by_name,
        dim_values=dim_values,
        context=context,
    )
    for item in operand.inputs:
        _validate_optimizer_nested_call_results(
            item,
            modules_by_name=modules_by_name,
            dim_values=dim_values,
            context=f"{context} input",
        )
    for key, item in operand.attrs.items():
        _validate_optimizer_nested_call_results(
            item,
            modules_by_name=modules_by_name,
            dim_values=dim_values,
            context=f"{context} attr {key!r}",
        )


def _validate_optimizer_call_result(
    call: GraphNode | GraphExpr,
    *,
    modules_by_name: Mapping[str, GraphModule],
    dim_values: Mapping[str, DimToken],
    context: str,
) -> None:
    callee = modules_by_name.get(call.op.name)
    if callee is None:
        return
    actuals = _call_actuals(call, callee)
    if len(actuals) != len(callee.inputs):
        raise ValueError(
            f"{context}: call to {call.op.name!r} has arity {len(actuals)}, "
            f"expected {len(callee.inputs)}"
        )
    expected_types = _instantiate_call_output_types(
        callee,
        actuals,
        len(call.outputs) if isinstance(call, GraphNode) else 1,
        dim_values=dim_values,
    )
    actual_types = (
        tuple(output.type_expr for output in call.outputs)
        if isinstance(call, GraphNode)
        else (call.type_expr,)
    )
    if len(actual_types) != len(expected_types):
        raise ValueError(
            f"{context}: call to {call.op.name!r} result arity {len(actual_types)}, "
            f"expected {len(expected_types)}"
        )
    for index, (actual_type, expected_type) in enumerate(zip(actual_types, expected_types, strict=True)):
        if not graph_type_compatible(actual_type, expected_type):
            raise ValueError(
                f"{context}: call to {call.op.name!r} result {index} has stale type "
                f"{actual_type!r}, expected {expected_type!r}"
            )


def _validate_optimizer_constraints(
    module: GraphModule,
    *,
    modules_by_name: Mapping[str, GraphModule],
) -> None:
    value_names = {value.name for value in module.inputs}
    for node in module.nodes:
        value_names.update(value.name for value in node.outputs)
    dim_symbols = _module_dim_refs(module)
    globals_or_modules = set(modules_by_name)
    allowed = value_names | dim_symbols | globals_or_modules
    for constraint in module.constraints:
        refs = _constraint_ref_names(constraint)
        if _constraint_has_callsite_guard(constraint):
            continue
        unknown = sorted(ref for ref in refs if ref not in allowed)
        if unknown:
            raise ValueError(
                f"module {module.name!r} constraint uses undefined refs: "
                + ", ".join(unknown)
            )


def _sanitize_graph_constraints(graph: GraphProgram) -> GraphProgram:
    modules_by_name = {module.name: module for module in graph.modules}
    return replace(
        graph,
        modules=tuple(
            _sanitize_module_constraints(module, modules_by_name=modules_by_name)
            for module in graph.modules
        ),
    )


def _sanitize_module_constraints(
    module: GraphModule,
    *,
    modules_by_name: Mapping[str, GraphModule],
) -> GraphModule:
    if not module.constraints:
        return module
    value_names = {value.name for value in module.inputs}
    for node in module.nodes:
        value_names.update(value.name for value in node.outputs)
    allowed = value_names | _module_constraint_dim_symbols(module)
    kept: list[Constraint] = []
    for constraint in module.constraints:
        if _constraint_is_trivially_true(constraint):
            continue
        if _constraint_has_callsite_guard(constraint):
            kept.append(constraint)
            continue
        if _constraint_ref_names(constraint) - allowed:
            continue
        kept.append(constraint)
    if len(kept) == len(module.constraints):
        return module
    return replace(module, constraints=tuple(kept))


def _module_constraint_dim_symbols(module: GraphModule) -> set[str]:
    symbols: set[str] = set()
    for value in module.inputs:
        symbols.update(_type_dim_refs(value.type_expr))
        if value.dims is not None:
            for dim in value.dims:
                symbols.update(dim_token_names(dim))
        if isinstance(value.type_expr, TypeDim):
            symbols.add(value.name)
    symbols.update(_type_dim_refs(module.return_type_expr))
    for output in module.outputs:
        symbols.update(_type_dim_refs(graph_operand_type(output)))
        if isinstance(output, GraphExpr) and output.dims is not None:
            for dim in output.dims:
                symbols.update(dim_token_names(dim))
        if isinstance(output, GraphValueRef):
            symbols.update(dim_token_names(output.name))
    for node in module.nodes:
        symbols.update(_type_dim_refs(node.type_expr))
        if node.dims is not None:
            for dim in node.dims:
                symbols.update(dim_token_names(dim))
        for output in node.outputs:
            symbols.update(_type_dim_refs(output.type_expr))
            if output.dims is not None:
                for dim in output.dims:
                    symbols.update(dim_token_names(dim))
            if isinstance(output.type_expr, TypeDim):
                symbols.add(output.name)
    return symbols


def _optimizer_dims_metadata_compatible(
    type_dims: tuple[DimToken, ...],
    metadata_dims: tuple[DimToken, ...],
) -> bool:
    if type_dims == metadata_dims:
        return True
    if any(isinstance(dim, str) and dim.startswith("..") for dim in (*type_dims, *metadata_dims)):
        return True
    if len(type_dims) != len(metadata_dims):
        return False
    for type_dim, metadata_dim in zip(type_dims, metadata_dims, strict=True):
        if type_dim == metadata_dim:
            continue
        if type(type_dim) is int or type(metadata_dim) is int:
            return False
    return True


def _graph_operand_key(operand: GraphOperand) -> object:
    if isinstance(operand, GraphValueRef):
        return ("ref", operand.name)
    if isinstance(operand, GraphLiteral):
        return ("lit", operand.value, _graph_cse_type_key(operand.type_expr))
    if isinstance(operand, GraphPath):
        return ("path", operand.absolute, operand.parts)
    return (
        "expr",
        operand.op.name,
        tuple(_graph_operand_key(item) for item in operand.inputs),
        tuple(sorted((key, _graph_operand_key(value)) for key, value in operand.attrs.items())),
    )


def _graph_cse_type_key(type_expr: TypeExpr | None) -> object:
    if isinstance(type_expr, TypeDim | TypeInt):
        return "DimOrInt"
    return type_expr


def _graph_node_cse_key(node: GraphNode) -> object:
    return (
        node.op.name,
        tuple(_graph_operand_key(item) for item in node.inputs),
        tuple(sorted((key, _graph_operand_key(value)) for key, value in node.attrs.items())),
        len(node.outputs),
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
    row_map: dict[str, tuple[DimToken, ...]] | None = None,
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
        variadic_dim = formal_dims[variadic_index]
        if (
            row_map is not None
            and isinstance(variadic_dim, str)
            and not any(isinstance(dim, str) and dim.startswith("..") for dim in actual_dims)
        ):
            row_end = len(actual_dims) - len(suffix) if suffix else len(actual_dims)
            row_map.setdefault(variadic_dim, actual_dims[len(prefix) : row_end])
        pairs.extend(zip(prefix, actual_dims[: len(prefix)], strict=False))
        if suffix:
            pairs.extend(zip(suffix, actual_dims[-len(suffix) :], strict=False))
    for formal_dim, actual_dim in pairs:
        if isinstance(formal_dim, str) and not formal_dim.startswith(".."):
            if any(isinstance(name, str) and name.startswith("..") for name in dim_token_names(actual_dim)):
                continue
            dim_map.setdefault(formal_dim, actual_dim)


def _bind_type_dim_map(
    formal: TypeExpr,
    actual: TypeExpr,
    dim_map: dict[str, DimToken],
    row_map: dict[str, tuple[DimToken, ...]] | None = None,
) -> None:
    if isinstance(formal, TypeTensor) and isinstance(actual, TypeTensor):
        _bind_dim_sequence_map(formal.dims, actual.dims, dim_map, row_map=row_map)
        return
    if isinstance(formal, TypeNamed) and isinstance(actual, TypeNamed):
        _bind_dim_sequence_map(formal.args, actual.args, dim_map)
        return
    if isinstance(formal, TypeOptional) and isinstance(actual, TypeOptional):
        _bind_type_dim_map(formal.inner, actual.inner, dim_map, row_map=row_map)
        return
    if isinstance(formal, TypeList) and isinstance(actual, TypeList):
        _bind_type_dim_map(formal.item, actual.item, dim_map, row_map=row_map)
        return
    if isinstance(formal, TypeTuple) and isinstance(actual, TypeTuple):
        for formal_item, actual_item in zip(formal.items, actual.items, strict=False):
            _bind_type_dim_map(formal_item, actual_item, dim_map, row_map=row_map)


def _operand_dim_token(
    operand: GraphOperand,
    dim_values: Mapping[str, DimToken] | None = None,
) -> DimToken | None:
    if isinstance(operand, GraphLiteral) and type(operand.value) is int:
        return operand.value
    if isinstance(operand, GraphValueRef):
        if not isinstance(operand.type_expr, TypeDim | TypeInt):
            return None
        if dim_values is not None and operand.name in dim_values:
            return dim_values[operand.name]
        return operand.name
    if (
        isinstance(operand, GraphExpr)
        and not operand.inputs
        and not operand.attrs
        and isinstance(operand.type_expr, TypeDim | TypeInt)
    ):
        if dim_values is not None and operand.op.name in dim_values:
            return dim_values[operand.op.name]
        return operand.op.name
    if (
        isinstance(operand, GraphExpr)
        and operand.op.name.startswith("core.binary.")
        and len(operand.inputs) == 2
        and isinstance(operand.type_expr, TypeDim | TypeInt)
    ):
        op = operand.op.name.removeprefix("core.binary.")
        if op not in {"+", "-", "*", "/"}:
            return None
        left = _operand_dim_token(operand.inputs[0], dim_values)
        right = _operand_dim_token(operand.inputs[1], dim_values)
        if left is None or right is None:
            return None
        return substitute_dim_token(DimExprBinary(op=op, left=left, right=right), {})
    return None


def _dim_token_operand(dim: DimToken, type_expr: TypeExpr | None = None) -> GraphOperand:
    scalar_type = type_expr if isinstance(type_expr, TypeDim | TypeInt) else TypeDim()
    if type(dim) is int:
        return GraphLiteral(value=dim, type_expr=scalar_type)
    if isinstance(dim, str):
        return GraphValueRef(name=dim, type_expr=scalar_type)
    if isinstance(dim, DimExprBinary):
        return GraphExpr(
            op=GraphOp(f"core.binary.{dim.op}"),
            inputs=(
                _dim_token_operand(dim.left, TypeDim()),
                _dim_token_operand(dim.right, TypeDim()),
            ),
            attrs={},
            type_expr=scalar_type,
        )
    return GraphValueRef(name=str(dim), type_expr=scalar_type)


def _fold_dim_binary_operand(
    op_name: str,
    left: GraphOperand,
    right: GraphOperand,
    *,
    type_expr: TypeExpr,
    dim_values: Mapping[str, DimToken],
) -> GraphOperand | None:
    op = op_name.removeprefix("core.binary.")
    if op not in {"+", "-", "*", "/"} or not isinstance(type_expr, TypeDim | TypeInt):
        return None
    left_dim = _operand_dim_token(left, dim_values)
    right_dim = _operand_dim_token(right, dim_values)
    if left_dim is None or right_dim is None:
        return None
    original = DimExprBinary(op=op, left=left_dim, right=right_dim)
    simplified = substitute_dim_token(original, {})
    if simplified == original:
        return None
    return _dim_token_operand(simplified, type_expr)


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


def _call_dim_subst(
    callee: GraphModule,
    actuals: tuple[GraphOperand, ...],
    *,
    dim_values: Mapping[str, DimToken] | None = None,
) -> dict[str, DimToken]:
    dim_map: dict[str, DimToken] = {}
    for formal, actual in zip(callee.inputs, actuals, strict=False):
        _bind_type_dim_map(formal.type_expr, graph_operand_type(actual), dim_map)
        _bind_value_dim_map(formal, actual, dim_map, dim_values=dim_values)
    return dim_map


def _call_type_substitutions(
    callee: GraphModule,
    actuals: tuple[GraphOperand, ...],
    *,
    dim_values: Mapping[str, DimToken] | None = None,
) -> tuple[dict[str, DimToken], dict[str, tuple[DimToken, ...]]]:
    dim_map: dict[str, DimToken] = {}
    row_map: dict[str, tuple[DimToken, ...]] = {}
    for formal, actual in zip(callee.inputs, actuals, strict=False):
        _bind_type_dim_map(
            formal.type_expr,
            graph_operand_type(actual),
            dim_map,
            row_map=row_map,
        )
        _bind_value_dim_map(formal, actual, dim_map, dim_values=dim_values)
    return dim_map, row_map


def _substitute_type_expr_graph(
    type_expr: TypeExpr,
    *,
    dim_map: Mapping[str, DimToken],
    row_map: Mapping[str, tuple[DimToken, ...]],
) -> TypeExpr:
    if isinstance(type_expr, TypeTensor):
        dims: list[DimToken] = []
        for dim in type_expr.dims:
            if isinstance(dim, str) and dim in row_map:
                dims.extend(row_map[dim])
                continue
            dims.append(substitute_dim_token(dim, dim_map))
        return TypeTensor(base=type_expr.base, dims=tuple(dims))
    if isinstance(type_expr, TypeOptional):
        return TypeOptional(
            _substitute_type_expr_graph(type_expr.inner, dim_map=dim_map, row_map=row_map)
        )
    if isinstance(type_expr, TypeList):
        return TypeList(
            _substitute_type_expr_graph(type_expr.item, dim_map=dim_map, row_map=row_map)
        )
    if isinstance(type_expr, TypeTuple):
        return TypeTuple(
            tuple(
                _substitute_type_expr_graph(item, dim_map=dim_map, row_map=row_map)
                for item in type_expr.items
            )
        )
    return substitute_type_expr(type_expr, dim_map)


def _instantiate_call_output_types(
    callee: GraphModule,
    actuals: tuple[GraphOperand, ...],
    output_count: int,
    *,
    dim_values: Mapping[str, DimToken] | None = None,
) -> tuple[TypeExpr, ...]:
    dim_map, row_map = _call_type_substitutions(
        callee,
        actuals,
        dim_values=dim_values,
    )
    return tuple(
        _substitute_type_expr_graph(type_expr, dim_map=dim_map, row_map=row_map)
        for type_expr in _module_output_types_for_arity(callee, output_count)
    )


def _dim_specificity_score(dim: DimToken) -> int:
    if type(dim) is int:
        return 4
    if isinstance(dim, str):
        return 0 if dim.startswith("..") else 1
    if isinstance(dim, DimExprBinary):
        if any(isinstance(name, str) for name in dim_token_names(dim)):
            return 1
        return _dim_specificity_score(dim.left) + _dim_specificity_score(dim.right)
    return 1


def _type_specificity_score(type_expr: TypeExpr) -> int:
    if isinstance(type_expr, TypeAny):
        return 0
    if isinstance(type_expr, TypeTensor):
        score = 2
        for dim in type_expr.dims:
            score += _dim_specificity_score(dim)
        return score
    if isinstance(type_expr, TypeOptional):
        return 1 + _type_specificity_score(type_expr.inner)
    if isinstance(type_expr, TypeList):
        return 1 + _type_specificity_score(type_expr.item)
    if isinstance(type_expr, TypeTuple):
        return 1 + sum(_type_specificity_score(item) for item in type_expr.items)
    if isinstance(type_expr, TypeNamed):
        return 2 + 2 * len(type_expr.args)
    return 2


def _type_dims(type_expr: TypeExpr) -> tuple[DimToken, ...] | None:
    return type_expr.dims if isinstance(type_expr, TypeTensor) else None


def _graph_operand_dim_token_for_type_rule(
    operand: GraphOperand,
    dim_values: Mapping[str, DimToken] | None,
) -> DimToken | None:
    if isinstance(operand, GraphLiteral):
        if type(operand.value) is int:
            return operand.value
        if isinstance(operand.value, str) and isinstance(operand.type_expr, TypeDim | TypeInt):
            return operand.value
        return None
    if isinstance(operand, GraphValueRef):
        if dim_values is not None and operand.name in dim_values:
            return dim_values[operand.name]
        if isinstance(operand.type_expr, TypeDim | TypeInt):
            return operand.name
        return None
    if isinstance(operand, GraphExpr):
        if operand.op.name in {"core.alias", "core.ascribe"} and len(operand.inputs) == 1:
            return _graph_operand_dim_token_for_type_rule(operand.inputs[0], dim_values)
        if operand.op.name.startswith("core.binary.") and len(operand.inputs) == 2:
            op = operand.op.name.removeprefix("core.binary.")
            if op in {"+", "-", "*", "/"}:
                left = _graph_operand_dim_token_for_type_rule(operand.inputs[0], dim_values)
                right = _graph_operand_dim_token_for_type_rule(operand.inputs[1], dim_values)
                if left is not None and right is not None:
                    return substitute_dim_token(DimExprBinary(op=op, left=left, right=right), {})
        if not operand.inputs and not operand.attrs and isinstance(operand.type_expr, TypeDim | TypeInt):
            return operand.op.name
    return None


def _infer_primitive_graph_type(
    op_name: str,
    inputs: tuple[GraphOperand, ...],
    attrs: Mapping[str, GraphOperand],
    *,
    dim_values: Mapping[str, DimToken] | None,
) -> TypeExpr | None:
    type_rule = get_op_type_rule(op_name[1:] if op_name.startswith("_") else op_name)
    if type_rule is None:
        return None
    inferred = type_rule(
        arg_types=tuple(graph_operand_type(item) for item in inputs),
        kwarg_types={key: graph_operand_type(value) for key, value in attrs.items()},
        args=inputs,
        kwargs=dict(attrs),
        helpers=_PrimitiveTypeHelpers(
            type_dims=_type_dims,
            expr_to_dim_token=lambda value: _graph_operand_dim_token_for_type_rule(value, dim_values)
            if isinstance(value, GraphValueRef | GraphLiteral | GraphPath | GraphExpr)
            else None,
            type_tensor=lambda *, dims: TypeTensor(base="Tensor", dims=tuple(dims)),
            resolve_name_expr=lambda name: GraphValueRef(
                name=name,
                type_expr=TypeDim(),
            )
            if dim_values is not None and name in dim_values
            else None,
            broadcast_tensor_dims=lambda left, right: _broadcast_graph_dims(left, right),
            dim_equivalent=lambda left, right: substitute_dim_token(left, {}) == substitute_dim_token(right, {}),
        ),
    )
    return inferred if isinstance(inferred, TypeExpr) else None


def _broadcast_graph_dim(left: DimToken, right: DimToken) -> DimToken | None:
    left = substitute_dim_token(left, {})
    right = substitute_dim_token(right, {})
    if left == right:
        return left
    if left == 1:
        return right
    if right == 1:
        return left
    if isinstance(left, str) and left.startswith(".."):
        return right
    if isinstance(right, str) and right.startswith(".."):
        return left
    if isinstance(left, str) and not isinstance(right, int):
        return left
    if isinstance(right, str) and not isinstance(left, int):
        return right
    if isinstance(left, str):
        return left
    if isinstance(right, str):
        return right
    return None


def _broadcast_graph_dims(
    left: tuple[DimToken, ...] | None,
    right: tuple[DimToken, ...] | None,
) -> tuple[DimToken, ...] | None:
    if left is None:
        return right
    if right is None:
        return left
    max_rank = max(len(left), len(right))
    left_full = (1,) * (max_rank - len(left)) + left
    right_full = (1,) * (max_rank - len(right)) + right
    dims: list[DimToken] = []
    for left_dim, right_dim in zip(left_full, right_full, strict=True):
        merged = _broadcast_graph_dim(left_dim, right_dim)
        if merged is None:
            return None
        dims.append(merged)
    return tuple(dims)


def _core_binary_result_type(
    op: str,
    left: TypeExpr,
    right: TypeExpr,
) -> TypeExpr | None:
    left_dims = _type_dims(left)
    right_dims = _type_dims(right)
    if op in {"==", "!=", "<", "<=", ">", ">="}:
        if isinstance(left, TypeNull) or isinstance(right, TypeNull):
            return TypeBool()
        dims = _broadcast_graph_dims(left_dims, right_dims)
        return TypeTensor(base="Tensor", dims=dims) if dims is not None else TypeBool()
    if op not in {"+", "-", "*", "/"}:
        return None
    if left_dims is not None or right_dims is not None:
        dims = _broadcast_graph_dims(left_dims, right_dims)
        if dims is not None:
            return TypeTensor(base="Tensor", dims=dims)
    if isinstance(left, TypeFloat) or isinstance(right, TypeFloat):
        return TypeFloat()
    if isinstance(left, TypeDim) or isinstance(right, TypeDim):
        return TypeDim()
    if isinstance(left, TypeInt) and isinstance(right, TypeInt):
        return TypeInt()
    return None


def _dim_token_uses_any_name(dim: DimToken, names: set[str]) -> bool:
    if isinstance(dim, str):
        return dim in names
    if isinstance(dim, DimExprBinary):
        return any(name in names for name in dim_token_names(dim) if isinstance(name, str))
    return False


def _more_specific_compatible_type(
    existing: TypeExpr,
    refreshed: TypeExpr,
    *,
    preferred_dim_names: set[str] | None = None,
) -> TypeExpr:
    if (
        isinstance(existing, TypeTensor)
        and isinstance(refreshed, TypeTensor)
        and len(existing.dims) == len(refreshed.dims)
        and graph_type_compatible(existing, refreshed)
    ):
        if any(old == 1 and new != 1 for old, new in zip(existing.dims, refreshed.dims, strict=True)):
            return refreshed
        if any(
            isinstance(old, str)
            and old.startswith("..")
            and not (isinstance(new, str) and new.startswith(".."))
            for old, new in zip(existing.dims, refreshed.dims, strict=True)
        ):
            return refreshed
    existing_score = _type_specificity_score(existing)
    refreshed_score = _type_specificity_score(refreshed)
    if graph_type_compatible(existing, refreshed) and existing_score > refreshed_score:
        return existing
    preferred_dim_names = preferred_dim_names or set()
    if (
        preferred_dim_names
        and graph_type_compatible(existing, refreshed)
        and existing_score == refreshed_score
        and isinstance(existing, TypeTensor)
        and isinstance(refreshed, TypeTensor)
        and len(existing.dims) == len(refreshed.dims)
    ):
        for existing_dim, refreshed_dim in zip(existing.dims, refreshed.dims, strict=True):
            existing_preferred = _dim_token_uses_any_name(existing_dim, preferred_dim_names)
            refreshed_preferred = _dim_token_uses_any_name(refreshed_dim, preferred_dim_names)
            if existing_preferred and not refreshed_preferred:
                return existing
            if refreshed_preferred and not existing_preferred:
                return refreshed
    return refreshed


def _select_result_type(existing: TypeExpr, true_type: TypeExpr, false_type: TypeExpr) -> TypeExpr:
    if graph_type_compatible(true_type, existing) and graph_type_compatible(false_type, existing):
        return existing
    if isinstance(true_type, TypeNull):
        return TypeOptional(false_type) if not isinstance(false_type, TypeOptional) else false_type
    if isinstance(false_type, TypeNull):
        return TypeOptional(true_type) if not isinstance(true_type, TypeOptional) else true_type
    branch_type = (
        _more_specific_compatible_type(true_type, false_type)
        if graph_type_compatible(true_type, false_type)
        else true_type
        if graph_type_compatible(true_type, existing)
        else false_type
        if graph_type_compatible(false_type, existing)
        else existing
    )
    return (
        _more_specific_compatible_type(existing, branch_type)
        if graph_type_compatible(existing, branch_type)
        else branch_type
    )


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
    preferred_dim_names = set(globals_env)
    callee = modules_by_name.get(operand.op.name)
    if callee is None:
        primitive_type = _infer_primitive_graph_type(
            operand.op.name,
            inputs,
            attrs,
            dim_values=dim_values,
        )
        if primitive_type is not None:
            result_type = _more_specific_compatible_type(
                operand.type_expr,
                primitive_type,
                preferred_dim_names=preferred_dim_names,
            )
            return replace(
                operand,
                inputs=inputs,
                attrs=attrs,
                type_expr=result_type,
                dims=result_type.dims if isinstance(result_type, TypeTensor) else operand.dims,
            )
        if operand.op.name.startswith("core.binary.") and len(inputs) == 2:
            op = operand.op.name.removeprefix("core.binary.")
            result_type = _core_binary_result_type(
                op,
                graph_operand_type(inputs[0]),
                graph_operand_type(inputs[1]),
            )
            if result_type is not None:
                result_type = _more_specific_compatible_type(
                    operand.type_expr,
                    result_type,
                    preferred_dim_names=preferred_dim_names,
                )
                return replace(
                    operand,
                    inputs=inputs,
                    attrs=attrs,
                    type_expr=result_type,
                    dims=result_type.dims if isinstance(result_type, TypeTensor) else operand.dims,
                )
        if operand.op.name == "core.select" and len(inputs) == 3:
            result_type = _select_result_type(
                operand.type_expr,
                graph_operand_type(inputs[1]),
                graph_operand_type(inputs[2]),
            )
            return replace(
                operand,
                inputs=inputs,
                attrs=attrs,
                type_expr=result_type,
                dims=result_type.dims if isinstance(result_type, TypeTensor) else operand.dims,
            )
        return replace(operand, inputs=inputs, attrs=attrs)
    call = replace(operand, inputs=inputs, attrs=attrs)
    result_types = _instantiate_call_output_types(
        callee,
        _call_actuals(call, callee),
        1,
        dim_values=dim_values,
    )
    if len(result_types) != 1:
        return replace(call, type_expr=TypeTuple(result_types))
    result_type = _more_specific_compatible_type(
        call.type_expr,
        result_types[0],
        preferred_dim_names=preferred_dim_names,
    )
    dims = result_type.dims if isinstance(result_type, TypeTensor) else None
    return replace(call, type_expr=result_type, dims=dims)


def _refresh_graph_module_types(
    module: GraphModule,
    *,
    globals_env: Mapping[str, GraphValue],
    modules_by_name: Mapping[str, GraphModule],
    global_dim_values: Mapping[str, DimToken] | None = None,
) -> GraphModule:
    env = {value.name: value for value in module.inputs}
    shadowed_dim_names = _module_signature_dim_refs(module)
    shadowed_global_dim_names = shadowed_dim_names & set(global_dim_values or {})
    dim_values: dict[str, DimToken] = {
        name: value
        for name, value in (global_dim_values or {}).items()
        if name not in shadowed_dim_names
    }
    preferred_dim_names = set(globals_env)
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
            call = replace(node, inputs=inputs, attrs=attrs)
            output_types = _instantiate_call_output_types(
                callee,
                _call_actuals(call, callee),
                len(node.outputs),
                dim_values=dim_values,
            )
            output_types = tuple(
                _more_specific_compatible_type(
                    node.outputs[index].type_expr,
                    output_type,
                    preferred_dim_names=preferred_dim_names,
                )
                if index < len(node.outputs)
                else output_type
                for index, output_type in enumerate(output_types)
            )
            type_expr = output_types[0] if len(output_types) == 1 else TypeTuple(output_types)
        elif node.op.name == "core.tuple":
            output_types = tuple(graph_operand_type(item) for item in inputs)
            type_expr = output_types[0] if len(output_types) == 1 else TypeTuple(output_types)
        elif node.op.name in {"core.alias", "core.ascribe"} and len(inputs) == 1:
            input_type = graph_operand_type(inputs[0])
            output_types = _result_types(input_type, len(node.outputs))
            type_expr = output_types[0] if len(output_types) == 1 else TypeTuple(output_types)
        elif node.op.name.startswith("core.binary.") and len(inputs) == 2:
            op = node.op.name.removeprefix("core.binary.")
            binary_type = _core_binary_result_type(
                op,
                graph_operand_type(inputs[0]),
                graph_operand_type(inputs[1]),
            )
            if binary_type is not None:
                type_expr = _more_specific_compatible_type(
                    type_expr,
                    binary_type,
                    preferred_dim_names=preferred_dim_names,
                )
                output_types = (type_expr,)
        elif node.op.name == "core.select" and len(inputs) == 3:
            type_expr = _select_result_type(
                type_expr,
                graph_operand_type(inputs[1]),
                graph_operand_type(inputs[2]),
            )
            output_types = _result_types(type_expr, len(node.outputs))
        else:
            primitive_type = _infer_primitive_graph_type(
                node.op.name,
                inputs,
                attrs,
                dim_values=dim_values,
            )
            if primitive_type is not None:
                if (
                    len(node.outputs) > 1
                    and isinstance(type_expr, TypeTuple)
                    and isinstance(primitive_type, TypeList)
                    and len(type_expr.items) == len(node.outputs)
                ):
                    # Primitive type rules for list-returning ops can be less
                    # precise than a destructuring call site. Keep the refined
                    # tuple shape rather than widening back to List[item].
                    type_expr = type_expr
                else:
                    type_expr = _more_specific_compatible_type(
                        type_expr,
                        primitive_type,
                        preferred_dim_names=preferred_dim_names,
                    )
                output_types = _result_types(type_expr, len(node.outputs))
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
            elif node.op.name.startswith("core.binary.") and isinstance(
                output.type_expr,
                TypeDim | TypeInt,
            ):
                dim = _operand_dim_token(
                    GraphExpr(
                        op=rewritten.op,
                        inputs=rewritten.inputs,
                        attrs=rewritten.attrs,
                        type_expr=rewritten.type_expr,
                        dims=rewritten.dims,
                    ),
                    dim_values,
                )
                if dim is not None:
                    dim_values[output.name] = dim
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
    current = graph
    for _ in range(16):
        modules_by_name = {module.name: module for module in current.modules}
        global_dim_values = _atomic_int_constant_dims(current)
        globals_env = {
            module.name: GraphValue(
                name=module.name,
                type_expr=_module_output_types(module)[0],
                dims=None,
            )
            for module in current.modules
            if not module.inputs and len(module.outputs) == 1
        }
        refreshed = replace(
            current,
            modules=tuple(
                _refresh_graph_module_types(
                    module,
                    globals_env=globals_env,
                    modules_by_name=modules_by_name,
                    global_dim_values=global_dim_values,
                )
                for module in current.modules
            ),
        )
        if refreshed == current:
            return current
        current = refreshed
    raise RuntimeError("graph type refresh did not converge after 16 iterations")


def _atomic_literal_constants(graph: GraphProgram) -> dict[str, GraphLiteral]:
    modules_by_name = {module.name: module for module in graph.modules}
    evaluating: set[str] = set()
    memo: dict[str, GraphLiteral] = {}

    def eval_operand(operand: GraphOperand, env: Mapping[str, GraphLiteral]) -> GraphLiteral | None:
        if isinstance(operand, GraphLiteral):
            return operand
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
            return replace(values[0], type_expr=operand.type_expr)
        if operand.op.name == "core.select" and len(values) == 3 and isinstance(values[0].value, bool):
            selected = values[1] if values[0].value else values[2]
            replacement = _select_fold_replacement(selected, operand.type_expr)
            return replacement if isinstance(replacement, GraphLiteral) else None
        if operand.op.name.startswith("core.binary.") and len(values) == 2:
            left, right = values
            return _fold_graph_binary(operand.op.name, left, right, operand)
        if operand.op.name in modules_by_name and not operand.inputs and not operand.attrs:
            return eval_module(operand.op.name)
        return None

    def eval_module(name: str) -> GraphLiteral | None:
        if name in memo:
            return memo[name]
        if name in evaluating:
            return None
        module = modules_by_name.get(name)
        if module is None or module.inputs or len(module.outputs) != 1:
            return None
        evaluating.add(name)
        env: dict[str, GraphLiteral] = {}
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
            if not isinstance(value, GraphLiteral):
                evaluating.remove(name)
                return None
            env[node.outputs[0].name] = value
        value = eval_operand(module.outputs[0], env)
        evaluating.remove(name)
        if isinstance(value, GraphLiteral):
            memo[name] = value
            return value
        return None

    for module in graph.modules:
        value = eval_module(module.name)
        if value is not None:
            memo[module.name] = value
    return memo


def _atomic_int_constant_dims(graph: GraphProgram) -> dict[str, DimToken]:
    return {
        name: literal.value
        for name, literal in _atomic_literal_constants(graph).items()
        if type(literal.value) is int
    }


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


def _type_dim_refs(type_expr: TypeExpr | None) -> set[str]:
    if type_expr is None:
        return set()
    if isinstance(type_expr, TypeTensor):
        refs: set[str] = set()
        for dim in type_expr.dims:
            refs.update(dim_token_names(dim))
        return refs
    if isinstance(type_expr, TypeNamed):
        refs: set[str] = set()
        for dim in type_expr.args:
            refs.update(dim_token_names(dim))
        return refs
    if isinstance(type_expr, TypeOptional):
        return _type_dim_refs(type_expr.inner)
    if isinstance(type_expr, TypeList):
        return _type_dim_refs(type_expr.item)
    if isinstance(type_expr, TypeTuple):
        refs: set[str] = set()
        for item in type_expr.items:
            refs.update(_type_dim_refs(item))
        return refs
    return set()


def _dims_dim_refs(dims: tuple[DimToken, ...] | None) -> set[str]:
    refs: set[str] = set()
    for dim in dims or ():
        refs.update(dim_token_names(dim))
    return refs


@dataclass
class _ModuleFreeSymbols:
    value_refs: set[str]
    path_refs: set[str]
    type_dim_refs: set[str]
    term_dim_refs: set[str]
    constraint_refs: set[str]


def _collect_operand_free_symbols(operand: GraphOperand, symbols: _ModuleFreeSymbols) -> None:
    symbols.type_dim_refs.update(_type_dim_refs(graph_operand_type(operand)))
    if isinstance(operand, GraphValueRef):
        symbols.value_refs.add(operand.name)
        if isinstance(operand.type_expr, TypeDim):
            symbols.term_dim_refs.add(operand.name)
        symbols.type_dim_refs.update(_dims_dim_refs(operand.dims))
        return
    if isinstance(operand, GraphPath):
        symbols.path_refs.update(graph_path_template_names(operand))
        return
    if isinstance(operand, GraphExpr):
        symbols.type_dim_refs.update(_dims_dim_refs(operand.dims))
        for item in operand.inputs:
            _collect_operand_free_symbols(item, symbols)
        for item in operand.attrs.values():
            _collect_operand_free_symbols(item, symbols)


def _collect_module_free_symbols(module: GraphModule) -> _ModuleFreeSymbols:
    symbols = _ModuleFreeSymbols(
        value_refs=set(),
        path_refs=set(),
        type_dim_refs=set(),
        term_dim_refs=set(),
        constraint_refs=set(),
    )
    for value in module.inputs:
        symbols.type_dim_refs.update(_type_dim_refs(value.type_expr))
        symbols.type_dim_refs.update(_dims_dim_refs(value.dims))
    for node in module.nodes:
        symbols.type_dim_refs.update(_type_dim_refs(node.type_expr))
        symbols.type_dim_refs.update(_dims_dim_refs(node.dims))
        for operand in (*node.inputs, *node.attrs.values()):
            _collect_operand_free_symbols(operand, symbols)
        for value in node.outputs:
            symbols.type_dim_refs.update(_type_dim_refs(value.type_expr))
            symbols.type_dim_refs.update(_dims_dim_refs(value.dims))
    for output in module.outputs:
        _collect_operand_free_symbols(output, symbols)
    symbols.type_dim_refs.update(_type_dim_refs(module.return_type_expr))
    for constraint in module.constraints:
        symbols.constraint_refs.update(_constraint_ref_names(constraint))
    return symbols


def _module_signature_dim_refs(module: GraphModule) -> set[str]:
    refs: set[str] = set()
    for value in module.inputs:
        refs.update(_type_dim_refs(value.type_expr))
        refs.update(_dims_dim_refs(value.dims))
    if module.return_type_expr is not None:
        refs.update(_type_dim_refs(module.return_type_expr))
    else:
        for output in module.outputs:
            refs.update(_type_dim_refs(graph_operand_type(output)))
            dims = getattr(output, "dims", None)
            refs.update(_dims_dim_refs(dims))
    return refs


def _module_return_dim_refs(module: GraphModule) -> set[str]:
    if module.return_type_expr is not None:
        return _type_dim_refs(module.return_type_expr)
    refs: set[str] = set()
    for output in module.outputs:
        refs.update(_type_dim_refs(graph_operand_type(output)))
        refs.update(_dims_dim_refs(getattr(output, "dims", None)))
    return refs


def _specialized_module_render_closure_safe(
    module: GraphModule,
    *,
    global_symbol_names: set[str],
) -> bool:
    input_names = {value.name for value in module.inputs}
    local_names = {
        value.name
        for node in module.nodes
        for value in node.outputs
    }
    signature_dim_refs = _module_signature_dim_refs(module)
    return_dim_refs = _module_return_dim_refs(module)
    symbols = _collect_module_free_symbols(module)

    value_bound = input_names | local_names | global_symbol_names
    dim_bound = signature_dim_refs | global_symbol_names
    if (symbols.value_refs | symbols.path_refs) - value_bound - dim_bound:
        return False
    if symbols.constraint_refs - value_bound - dim_bound:
        return False

    term_dim_bound = input_names | signature_dim_refs | return_dim_refs | global_symbol_names
    return not (symbols.term_dim_refs - term_dim_bound)


def _replace_atomic_literal_globals(
    operand: GraphOperand,
    constants: Mapping[str, GraphLiteral],
) -> GraphOperand:
    if isinstance(operand, GraphValueRef):
        return constants.get(operand.name, operand)
    if not isinstance(operand, GraphExpr):
        return operand
    if operand.op.name in constants and not operand.inputs and not operand.attrs:
        return constants[operand.op.name]
    return replace(
        operand,
        inputs=tuple(_replace_atomic_literal_globals(item, constants) for item in operand.inputs),
        attrs={
            key: _replace_atomic_literal_globals(value, constants)
            for key, value in operand.attrs.items()
        },
    )


def _substitute_atomic_constant_dims_local(graph: GraphProgram) -> GraphProgram:
    literal_constants = _atomic_literal_constants(graph)
    if not literal_constants:
        return graph
    dim_constants = {
        name: literal.value
        for name, literal in literal_constants.items()
        if type(literal.value) is int
    }
    modules: list[GraphModule] = []
    for module in graph.modules:
        shadowed_dim_names = _module_signature_dim_refs(module)
        module_dim_constants = {
            name: value
            for name, value in dim_constants.items()
            if name not in shadowed_dim_names
        }
        module_constants = {
            name: literal
            for name, literal in literal_constants.items()
            if name != module.name and name not in shadowed_dim_names
        }
        candidate_module = (
            substitute_graph_module_dims(module, module_dim_constants)
            if module_dim_constants
            else module
        )
        candidate_module = replace(
            candidate_module,
            nodes=tuple(
                replace(
                    node,
                    inputs=tuple(
                        _replace_atomic_literal_globals(item, module_constants)
                        for item in node.inputs
                    ),
                    attrs={
                        key: _replace_atomic_literal_globals(value, module_constants)
                        for key, value in node.attrs.items()
                    },
                )
                for node in candidate_module.nodes
            ),
            outputs=tuple(
                _replace_atomic_literal_globals(output, module_constants)
                for output in candidate_module.outputs
            ),
        )
        modules.append(candidate_module)
    candidate = replace(graph, modules=tuple(modules))
    if candidate == graph:
        return graph
    candidate = _refresh_graph_program_types(candidate)
    candidate = _sanitize_graph_constraints(candidate)
    try:
        _validate_optimizer_graph(candidate, phase="constant_dim_substitution")
    except ValueError:
        accepted = list(graph.modules)
        changed = False
        for index, (original_module, candidate_module) in enumerate(
            zip(graph.modules, modules, strict=True)
        ):
            if candidate_module == original_module:
                continue
            candidate_modules = list(accepted)
            candidate_modules[index] = candidate_module
            module_candidate = replace(graph, modules=tuple(candidate_modules))
            module_candidate = _refresh_graph_program_types(module_candidate)
            module_candidate = _sanitize_graph_constraints(module_candidate)
            try:
                _validate_optimizer_graph(
                    module_candidate,
                    phase="constant_dim_substitution.module",
                )
            except ValueError:
                continue
            accepted = list(module_candidate.modules)
            changed = True
        if not changed:
            return graph
        return replace(graph, modules=tuple(accepted))
    return candidate


def _simplify_symbolic_graph_dims(graph: GraphProgram) -> GraphProgram:
    candidate = replace(
        graph,
        modules=tuple(substitute_graph_module_dims(module, {}) for module in graph.modules),
    )
    candidate = _sanitize_graph_constraints(candidate)
    _validate_optimizer_graph(candidate, phase="symbolic_dim_simplification")
    return candidate


def _fold_graph_binary(op_name: str, left: GraphLiteral, right: GraphLiteral, template: GraphExpr | GraphNode) -> GraphLiteral | None:
    op = op_name.removeprefix("core.binary.")
    lval = left.value
    rval = right.value
    if isinstance(lval, bool) and isinstance(rval, bool):
        if op == "and":
            return _bool_literal(lval and rval)
        if op == "or":
            return _bool_literal(lval or rval)
        if op == "==":
            return _bool_literal(lval == rval)
        if op == "!=":
            return _bool_literal(lval != rval)
    if isinstance(lval, str) and isinstance(rval, str):
        if op == "==":
            return _bool_literal(lval == rval)
        if op == "!=":
            return _bool_literal(lval != rval)
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
            return _bool_literal(lval == rval)
        if op == "!=":
            return _bool_literal(lval != rval)
        if op == "<":
            return _bool_literal(lval < rval)
        if op == "<=":
            return _bool_literal(lval <= rval)
        if op == ">":
            return _bool_literal(lval > rval)
        if op == ">=":
            return _bool_literal(lval >= rval)
    if isinstance(lval, float) and isinstance(rval, float):
        if op == "+":
            return _literal_like(lval + rval, template)
        if op == "-":
            return _literal_like(lval - rval, template)
        if op == "*":
            return _literal_like(lval * rval, template)
        if op == "/" and rval != 0.0:
            return _literal_like(lval / rval, template)
        if op == "==":
            return _bool_literal(lval == rval)
        if op == "!=":
            return _bool_literal(lval != rval)
        if op == "<":
            return _bool_literal(lval < rval)
        if op == "<=":
            return _bool_literal(lval <= rval)
        if op == ">":
            return _bool_literal(lval > rval)
        if op == ">=":
            return _bool_literal(lval >= rval)
    if lval is None and rval is None:
        if op == "==":
            return _bool_literal(True)
        if op == "!=":
            return _bool_literal(False)
    if (lval is None) != (rval is None):
        if op == "==":
            return _bool_literal(False)
        if op == "!=":
            return _bool_literal(True)
    return None


def _is_null_literal(operand: GraphOperand) -> bool:
    return isinstance(operand, GraphLiteral) and operand.value is None


def _operand_is_statically_non_null_for_fold(operand: GraphOperand) -> bool:
    if isinstance(operand, GraphPath):
        return True
    if isinstance(operand, GraphLiteral):
        return operand.value is not None
    if isinstance(operand, GraphExpr) and not operand.inputs and not operand.attrs:
        return not isinstance(operand.type_expr, TypeAny | TypeOptional | TypeNull)
    return False


def _fold_typed_null_comparison(op_name: str, left: GraphOperand, right: GraphOperand) -> GraphLiteral | None:
    op = op_name.removeprefix("core.binary.")
    if op not in {"==", "!="}:
        return None
    if _is_null_literal(left) and _operand_is_statically_non_null_for_fold(right):
        return _bool_literal(op == "!=")
    if _is_null_literal(right) and _operand_is_statically_non_null_for_fold(left):
        return _bool_literal(op == "!=")
    return None


def _operand_domain_fact(
    operand: GraphOperand,
    local_domain_facts: Mapping[str, GraphDomainFact] | None,
) -> GraphDomainFact | None:
    if isinstance(operand, GraphLiteral):
        if operand.value is None:
            return GraphDomainFact(GraphDomainKind.NULL)
        if isinstance(operand.value, bool | int | float | str):
            return GraphDomainFact(GraphDomainKind.LITERAL, operand.value)
        return None
    if isinstance(operand, GraphPath):
        return GraphDomainFact(GraphDomainKind.PATH, operand)
    if isinstance(operand, GraphValueRef) and local_domain_facts is not None:
        return local_domain_facts.get(operand.name)
    return None


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


def _fold_domain_binary_comparison(
    op_name: str,
    left: GraphOperand,
    right: GraphOperand,
    *,
    local_domain_facts: Mapping[str, GraphDomainFact] | None,
) -> GraphLiteral | None:
    op = op_name.removeprefix("core.binary.")
    if op not in {"==", "!="}:
        return None
    left_fact = _operand_domain_fact(left, local_domain_facts)
    right_fact = _operand_domain_fact(right, local_domain_facts)
    if left_fact is None or right_fact is None:
        return None
    equality = _domain_facts_equal(left_fact, right_fact)
    if equality is None:
        return None
    return _bool_literal(equality if op == "==" else not equality)


def _domain_bool_value(
    operand: GraphOperand,
    local_domain_facts: Mapping[str, GraphDomainFact] | None,
) -> bool | None:
    if isinstance(operand, GraphLiteral) and type(operand.value) is bool:
        return operand.value
    fact = _operand_domain_fact(operand, local_domain_facts)
    if (
        fact is not None
        and fact.kind == GraphDomainKind.LITERAL
        and isinstance(fact.value, bool)
    ):
        return fact.value
    return None


def _select_fold_replacement(selected: GraphOperand, result_type: TypeExpr) -> GraphOperand | None:
    selected_type = graph_operand_type(selected)
    if selected_type == result_type:
        return selected
    if graph_type_compatible(selected_type, result_type):
        return selected
    if graph_type_compatible(result_type, selected_type):
        if isinstance(selected, GraphValueRef):
            return replace(selected, type_expr=result_type, dims=_type_dims(result_type))
        if isinstance(selected, GraphExpr):
            return replace(selected, type_expr=result_type, dims=_type_dims(result_type))
    return None


def _fold_numeric_primitive(expr: GraphExpr | GraphNode) -> GraphLiteral | None:
    if expr.attrs:
        return None
    if not all(isinstance(item, GraphLiteral) for item in expr.inputs):
        return None
    values = tuple(item.value for item in expr.inputs if isinstance(item, GraphLiteral))
    op_name = expr.op.name[1:] if expr.op.name.startswith("_") else expr.op.name
    if op_name == "sqrt" and len(values) == 1 and type(values[0]) in {int, float} and values[0] >= 0:
        return GraphLiteral(value=math.sqrt(values[0]), type_expr=expr.type_expr)
    return None


def _dim_token_to_operand(dim: DimToken) -> GraphOperand:
    if type(dim) is int:
        return GraphLiteral(value=dim, type_expr=TypeDim())
    if isinstance(dim, str):
        return GraphValueRef(name=dim, type_expr=TypeDim())
    if isinstance(dim, DimExprBinary):
        return GraphExpr(
            op=GraphOp(f"core.binary.{dim.op}"),
            inputs=(_dim_token_to_operand(dim.left), _dim_token_to_operand(dim.right)),
            attrs={},
            type_expr=TypeDim(),
        )
    return GraphValueRef(name=str(dim), type_expr=TypeDim())


def _typed_shape_operands(operand: GraphOperand) -> tuple[GraphOperand, ...] | None:
    type_expr = graph_operand_type(operand)
    if not isinstance(type_expr, TypeTensor):
        return None
    if any(isinstance(dim, str) and dim.startswith("..") for dim in type_expr.dims):
        return None
    return tuple(_dim_token_to_operand(dim) for dim in type_expr.dims)


def _literal_int(operand: GraphOperand) -> int | None:
    if isinstance(operand, GraphLiteral) and type(operand.value) is int:
        return operand.value
    return None


def _indexed_operand(items: tuple[GraphOperand, ...], index: int) -> GraphOperand | None:
    resolved = index if index >= 0 else len(items) + index
    if resolved < 0 or resolved >= len(items):
        return None
    return items[resolved]


def _shape_items_operand(
    operand: GraphOperand,
    *,
    modules_by_name: Mapping[str, GraphModule] | None,
) -> tuple[GraphOperand, ...] | None:
    if isinstance(operand, GraphExpr) and operand.op.name == "core.list":
        return operand.inputs
    if isinstance(operand, GraphExpr) and operand.op.name == "_shape" and len(operand.inputs) == 1:
        return _typed_shape_operands(operand.inputs[0])
    return None


def _shape_index_forwarder(module: GraphModule) -> tuple[int, int] | None:
    if module.is_global_binding:
        return None
    if len(module.nodes) != 2 or len(module.outputs) != 1:
        return None
    shape_node, index_node = module.nodes
    if (
        shape_node.op.name != "_shape"
        or index_node.op.name != "_list_index"
        or shape_node.attrs
        or index_node.attrs
        or len(shape_node.inputs) != 1
        or len(shape_node.outputs) != 1
        or len(index_node.inputs) != 2
        or len(index_node.outputs) != 1
    ):
        return None
    shape_input = shape_node.inputs[0]
    if not isinstance(shape_input, GraphValueRef):
        return None
    indexed_shape, index_input = index_node.inputs
    if not isinstance(indexed_shape, GraphValueRef):
        return None
    if indexed_shape.name != shape_node.outputs[0].name:
        return None
    if not isinstance(index_input, GraphValueRef):
        return None
    returned = module.outputs[0]
    if not isinstance(returned, GraphValueRef) or returned.name != index_node.outputs[0].name:
        return None
    formal_indexes = {formal.name: index for index, formal in enumerate(module.inputs)}
    tensor_index = formal_indexes.get(shape_input.name)
    dim_index = formal_indexes.get(index_input.name)
    if tensor_index is None or dim_index is None:
        return None
    return tensor_index, dim_index


def _shape_query_replacement(
    expr: GraphExpr,
    *,
    modules_by_name: Mapping[str, GraphModule] | None,
    stable_shape_values: set[str] | None = None,
    blocked_dim_ref_names: set[str] | None = None,
) -> GraphOperand | None:
    stable_shape_values = stable_shape_values or set()
    blocked_dim_ref_names = blocked_dim_ref_names or set()

    def allowed(candidate: GraphOperand | None) -> GraphOperand | None:
        if (
            isinstance(candidate, GraphValueRef)
            and candidate.name in blocked_dim_ref_names
            and isinstance(candidate.type_expr, TypeDim | TypeInt)
        ):
            return None
        return candidate

    if expr.op.name == "_shape" and len(expr.inputs) == 1:
        if not (
            isinstance(expr.inputs[0], GraphValueRef)
            and expr.inputs[0].name in stable_shape_values
        ):
            return None
        items = _typed_shape_operands(expr.inputs[0])
        if items is None:
            return None
        if any(
            isinstance(item, GraphValueRef)
            and item.name in blocked_dim_ref_names
            and isinstance(item.type_expr, TypeDim | TypeInt)
            for item in items
        ):
            return None
        return GraphExpr(
            op=GraphOp("core.list"),
            inputs=items,
            attrs={},
            type_expr=TypeList(TypeDim()),
        )
    if expr.op.name == "_list_index" and len(expr.inputs) == 2:
        items = _shape_items_operand(expr.inputs[0], modules_by_name=modules_by_name)
        index = _literal_int(expr.inputs[1])
        if items is None or index is None:
            return None
        return allowed(_indexed_operand(items, index))
    if modules_by_name is None:
        return None
    callee = modules_by_name.get(expr.op.name)
    if callee is None:
        return None
    forwarder = _shape_index_forwarder(callee)
    if forwarder is None:
        return None
    tensor_index, dim_index = forwarder
    actuals = _call_actuals(expr, callee)
    if len(actuals) <= max(tensor_index, dim_index):
        return None
    if not (
        isinstance(actuals[tensor_index], GraphValueRef)
        and actuals[tensor_index].name in stable_shape_values
    ):
        return None
    items = _typed_shape_operands(actuals[tensor_index])
    index = _literal_int(actuals[dim_index])
    if items is None or index is None:
        return None
    return allowed(_indexed_operand(items, index))


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
    if isinstance(operand, GraphValueRef):
        if operand.name in module_names:
            out.add(operand.name)
        return
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


def _operand_called_module_names(operand: GraphOperand, module_names: set[str]) -> set[str]:
    calls: set[str] = set()
    _operand_module_calls(operand, module_names, calls)
    return calls


def _find_operand_call(operand: GraphOperand, callee_name: str) -> GraphExpr | None:
    if not isinstance(operand, GraphExpr):
        return None
    if operand.op.name == callee_name:
        return operand
    for item in operand.inputs:
        found = _find_operand_call(item, callee_name)
        if found is not None:
            return found
    for item in operand.attrs.values():
        found = _find_operand_call(item, callee_name)
        if found is not None:
            return found
    return None


def _rename_module_dim_token(dim: DimToken, renames: Mapping[str, str]) -> DimToken:
    if isinstance(dim, str):
        return renames.get(dim, dim)
    if isinstance(dim, DimExprBinary):
        return DimExprBinary(
            op=dim.op,
            left=_rename_module_dim_token(dim.left, renames),
            right=_rename_module_dim_token(dim.right, renames),
        )
    return dim


def _rename_module_type_expr(type_expr: TypeExpr, renames: Mapping[str, str]) -> TypeExpr:
    if isinstance(type_expr, TypeTensor):
        return TypeTensor(
            base=type_expr.base,
            dims=tuple(_rename_module_dim_token(dim, renames) for dim in type_expr.dims),
        )
    if isinstance(type_expr, TypeNamed):
        return TypeNamed(
            name=type_expr.name,
            args=tuple(_rename_module_dim_token(dim, renames) for dim in type_expr.args),
        )
    if isinstance(type_expr, TypeOptional):
        return TypeOptional(_rename_module_type_expr(type_expr.inner, renames))
    if isinstance(type_expr, TypeList):
        return TypeList(_rename_module_type_expr(type_expr.item, renames))
    if isinstance(type_expr, TypeTuple):
        return TypeTuple(tuple(_rename_module_type_expr(item, renames) for item in type_expr.items))
    return type_expr


def _rename_module_value(value: GraphValue, renames: Mapping[str, str]) -> GraphValue:
    return replace(
        value,
        type_expr=_rename_module_type_expr(value.type_expr, renames),
        dims=(
            None
            if value.dims is None
            else tuple(_rename_module_dim_token(dim, renames) for dim in value.dims)
        ),
    )


def _rename_module_operand(operand: GraphOperand, renames: Mapping[str, str]) -> GraphOperand:
    if isinstance(operand, GraphValueRef):
        return replace(
            operand,
            name=renames.get(operand.name, operand.name),
            type_expr=_rename_module_type_expr(operand.type_expr, renames),
            dims=(
                None
                if operand.dims is None
                else tuple(_rename_module_dim_token(dim, renames) for dim in operand.dims)
            ),
        )
    if isinstance(operand, GraphLiteral):
        return replace(operand, type_expr=_rename_module_type_expr(operand.type_expr, renames))
    if isinstance(operand, GraphExpr):
        return replace(
            operand,
            op=GraphOp(renames.get(operand.op.name, operand.op.name)),
            inputs=tuple(_rename_module_operand(item, renames) for item in operand.inputs),
            attrs={key: _rename_module_operand(value, renames) for key, value in operand.attrs.items()},
            type_expr=_rename_module_type_expr(operand.type_expr, renames),
            dims=(
                None
                if operand.dims is None
                else tuple(_rename_module_dim_token(dim, renames) for dim in operand.dims)
            ),
        )
    return operand


def _rename_module_constraint_operand(
    operand: ConstraintOperand,
    renames: Mapping[str, str],
) -> ConstraintOperand:
    if isinstance(operand, tuple):
        return tuple(_rename_module_constraint_operand(item, renames) for item in operand)
    if isinstance(operand, str):
        return renames.get(operand, operand)
    if isinstance(operand, DimExprBinary):
        return _rename_module_dim_token(operand, renames)
    return operand


def _rename_module_constraint(constraint: Constraint, renames: Mapping[str, str]) -> Constraint:
    return Constraint(
        relation=constraint.relation,
        left=_rename_module_constraint_operand(constraint.left, renames),
        right=(
            None
            if constraint.right is None
            else _rename_module_constraint_operand(constraint.right, renames)
        ),
        guards=tuple(_rename_module_constraint(guard, renames) for guard in constraint.guards),
    )


def _rename_modules(graph: GraphProgram, renames: Mapping[str, str]) -> GraphProgram:
    if not renames:
        return graph
    modules: list[GraphModule] = []
    for module in graph.modules:
        modules.append(
            replace(
                module,
                name=renames.get(module.name, module.name),
                inputs=tuple(_rename_module_value(value, renames) for value in module.inputs),
                outputs=tuple(_rename_module_operand(output, renames) for output in module.outputs),
                nodes=tuple(
                    replace(
                        node,
                        op=GraphOp(renames.get(node.op.name, node.op.name)),
                        inputs=tuple(_rename_module_operand(item, renames) for item in node.inputs),
                        attrs={key: _rename_module_operand(value, renames) for key, value in node.attrs.items()},
                        outputs=tuple(_rename_module_value(output, renames) for output in node.outputs),
                        type_expr=_rename_module_type_expr(node.type_expr, renames),
                        dims=(
                            None
                            if node.dims is None
                            else tuple(_rename_module_dim_token(dim, renames) for dim in node.dims)
                        ),
                    )
                    for node in module.nodes
                ),
                return_type_expr=(
                    None
                    if module.return_type_expr is None
                    else _rename_module_type_expr(module.return_type_expr, renames)
                ),
                constraints=tuple(_rename_module_constraint(item, renames) for item in module.constraints),
            )
        )
    return replace(
        graph,
        modules=tuple(modules),
        main_module=renames.get(graph.main_module, graph.main_module),
    )


def _specialized_module_base(name: str) -> str | None:
    marker = "__spec_"
    if marker not in name:
        return None
    base, suffix = name.rsplit(marker, 1)
    if not base or not suffix.isdigit():
        return None
    return base


def _canonicalize_generated_module_names(graph: GraphProgram) -> GraphProgram:
    names = {module.name for module in graph.modules}
    generated_by_base: dict[str, list[str]] = {}
    for module in graph.modules:
        if module.name == graph.main_module or module.is_global_binding:
            continue
        base = _specialized_module_base(module.name)
        if base is None:
            continue
        generated_by_base.setdefault(base, []).append(module.name)
    renames: dict[str, str] = {}
    reserved = set(names)
    for base, generated_names in sorted(generated_by_base.items()):
        for index, generated_name in enumerate(sorted(generated_names), start=1):
            if base not in names and base not in renames.values() and generated_name == generated_names[0]:
                target = base
            else:
                suffix = 1
                while True:
                    target = f"{base}__s{suffix}"
                    if target not in reserved and target not in renames.values():
                        break
                    suffix += 1
            if target == generated_name:
                continue
            reserved.add(target)
            renames[generated_name] = target
    if not renames:
        return graph
    renamed = _rename_modules(graph, renames)
    _validate_optimizer_graph(renamed, phase="canonicalize_module_names")
    return renamed


def _is_generated_value_name(name: str) -> bool:
    return (
        name.startswith("__")
        or "__inl_" in name
        or "___" in name
        or (name.startswith("_v") and "_arg" in name)
        or (name.startswith("_v") and name[2:].isdigit())
    )


def _fresh_canonical_value_name(used: set[str], next_index: int) -> tuple[str, int]:
    index = next_index
    while True:
        candidate = f"_v{index}"
        if candidate not in used:
            used.add(candidate)
            return candidate, index + 1
        index += 1


def _rename_value(value: GraphValue, renames: Mapping[str, str]) -> GraphValue:
    return replace(
        value,
        name=renames.get(value.name, value.name),
        type_expr=_rename_module_type_expr(value.type_expr, renames),
        dims=(
            None
            if value.dims is None
            else tuple(_rename_module_dim_token(dim, renames) for dim in value.dims)
        ),
    )


def _rename_value_operand(operand: GraphOperand, renames: Mapping[str, str]) -> GraphOperand:
    if isinstance(operand, GraphValueRef):
        return replace(
            operand,
            name=renames.get(operand.name, operand.name),
            type_expr=_rename_module_type_expr(operand.type_expr, renames),
            dims=(
                None
                if operand.dims is None
                else tuple(_rename_module_dim_token(dim, renames) for dim in operand.dims)
            ),
        )
    if isinstance(operand, GraphLiteral):
        return replace(operand, type_expr=_rename_module_type_expr(operand.type_expr, renames))
    if isinstance(operand, GraphPath):
        return rename_operand(operand, renames)
    if isinstance(operand, GraphExpr):
        return replace(
            operand,
            inputs=tuple(_rename_value_operand(item, renames) for item in operand.inputs),
            attrs={key: _rename_value_operand(value, renames) for key, value in operand.attrs.items()},
            type_expr=_rename_module_type_expr(operand.type_expr, renames),
            dims=(
                None
                if operand.dims is None
                else tuple(_rename_module_dim_token(dim, renames) for dim in operand.dims)
            ),
        )
    return operand


def _rename_value_constraint_operand(
    operand: ConstraintOperand,
    renames: Mapping[str, str],
) -> ConstraintOperand:
    if isinstance(operand, tuple):
        return tuple(_rename_value_constraint_operand(item, renames) for item in operand)
    if isinstance(operand, str):
        return renames.get(operand, operand)
    if isinstance(operand, DimExprBinary):
        return _rename_module_dim_token(operand, renames)
    return operand


def _rename_value_constraint(constraint: Constraint, renames: Mapping[str, str]) -> Constraint:
    return Constraint(
        relation=constraint.relation,
        left=_rename_value_constraint_operand(constraint.left, renames),
        right=(
            None
            if constraint.right is None
            else _rename_value_constraint_operand(constraint.right, renames)
        ),
        guards=tuple(_rename_value_constraint(guard, renames) for guard in constraint.guards),
    )


def _is_zero_arg_global_call_node(
    node: GraphNode,
    *,
    global_symbol_names: set[str],
) -> bool:
    return (
        node.op.name in global_symbol_names
        and not node.inputs
        and not node.attrs
        and len(node.outputs) == 1
    )


def _fresh_hidden_global_value_name(used: set[str], next_index: int) -> tuple[str, int]:
    index = next_index
    while True:
        candidate = f"__global_{index}"
        if candidate not in used:
            used.add(candidate)
            return candidate, index + 1
        index += 1


def _canonicalize_generated_value_names_in_module(
    module: GraphModule,
    *,
    global_symbol_names: set[str],
) -> GraphModule:
    renames: dict[str, str] = {}
    used = {value.name for value in module.inputs}
    for name in module.output_names:
        if not _is_generated_value_name(name):
            used.add(name)
    for node in module.nodes:
        for output in node.outputs:
            if not _is_generated_value_name(output.name):
                used.add(output.name)
    next_hidden_global_index = 1
    for node in module.nodes:
        if not _is_zero_arg_global_call_node(node, global_symbol_names=global_symbol_names):
            continue
        output = node.outputs[0]
        target, next_hidden_global_index = _fresh_hidden_global_value_name(
            used,
            next_hidden_global_index,
        )
        if target != output.name:
            renames[output.name] = target
    next_index = 1
    for node in module.nodes:
        if _is_zero_arg_global_call_node(node, global_symbol_names=global_symbol_names):
            continue
        for output in node.outputs:
            if not _is_generated_value_name(output.name):
                continue
            target, next_index = _fresh_canonical_value_name(used, next_index)
            if target != output.name:
                renames[output.name] = target
    if not renames:
        return module
    return replace(
        module,
        inputs=tuple(_rename_value(value, renames) for value in module.inputs),
        outputs=tuple(_rename_value_operand(output, renames) for output in module.outputs),
        output_names=tuple(renames.get(name, name) for name in module.output_names),
        nodes=tuple(
            replace(
                node,
                inputs=tuple(_rename_value_operand(item, renames) for item in node.inputs),
                attrs={key: _rename_value_operand(value, renames) for key, value in node.attrs.items()},
                outputs=tuple(_rename_value(output, renames) for output in node.outputs),
                type_expr=_rename_module_type_expr(node.type_expr, renames),
                dims=(
                    None
                    if node.dims is None
                    else tuple(_rename_module_dim_token(dim, renames) for dim in node.dims)
                ),
            )
            for node in module.nodes
        ),
        return_type_expr=(
            None
            if module.return_type_expr is None
            else _rename_module_type_expr(module.return_type_expr, renames)
        ),
        constraints=tuple(_rename_value_constraint(item, renames) for item in module.constraints),
    )


def _canonicalize_generated_value_names(graph: GraphProgram) -> GraphProgram:
    global_symbol_names = {
        module.name
        for module in graph.modules
        if module.is_global_binding and not module.inputs and len(module.outputs) == 1
    }
    modules = tuple(
        _canonicalize_generated_value_names_in_module(
            module,
            global_symbol_names=global_symbol_names,
        )
        for module in graph.modules
    )
    if modules == graph.modules:
        return graph
    renamed = replace(graph, modules=modules)
    _validate_optimizer_graph(renamed, phase="canonicalize_value_names")
    return renamed


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
    modules_by_name: Mapping[str, GraphModule] | None = None,
    local_domain_facts: Mapping[str, GraphDomainFact] | None = None,
) -> GraphOperand:
    return replace_operand_refs(
        operand,
        subst,
        fold_operand=(
            None
            if not fold
            else lambda item: _fold_operand(
                item,
                module_effects=module_effects,
                modules_by_name=modules_by_name,
                local_domain_facts=local_domain_facts,
            )
        ),
    )


def _fold_operand(
    operand: GraphOperand,
    *,
    module_effects: Mapping[str, GraphEffect] | None = None,
    modules_by_name: Mapping[str, GraphModule] | None = None,
    local_domain_facts: Mapping[str, GraphDomainFact] | None = None,
) -> GraphOperand:
    if not isinstance(operand, GraphExpr):
        return operand
    expr = replace(
        operand,
        inputs=tuple(
            _fold_operand(
                item,
                module_effects=module_effects,
                modules_by_name=modules_by_name,
                local_domain_facts=local_domain_facts,
            )
            for item in operand.inputs
        ),
        attrs={
            key: _fold_operand(
                value,
                module_effects=module_effects,
                modules_by_name=modules_by_name,
                local_domain_facts=local_domain_facts,
            )
            for key, value in operand.attrs.items()
        },
    )
    shape_replacement = _shape_query_replacement(
        expr,
        modules_by_name=modules_by_name,
        stable_shape_values=set(),
    )
    if shape_replacement is not None:
        return shape_replacement
    if expr.op.name == "core.ascribe" and len(expr.inputs) == 1:
        return expr.inputs[0]
    if expr.op.name == "core.select" and len(expr.inputs) == 3:
        cond_value = _domain_bool_value(expr.inputs[0], local_domain_facts)
        if cond_value is not None:
            selected = expr.inputs[1] if cond_value else expr.inputs[2]
            replacement = _select_fold_replacement(selected, expr.type_expr)
            if replacement is not None:
                return replacement
    if expr.op.name.startswith("core.binary.") and len(expr.inputs) == 2:
        left, right = expr.inputs
        domain_fold = _fold_domain_binary_comparison(
            expr.op.name,
            left,
            right,
            local_domain_facts=local_domain_facts,
        )
        if domain_fold is not None:
            return domain_fold
        typed_null_fold = _fold_typed_null_comparison(expr.op.name, left, right)
        if typed_null_fold is not None:
            return typed_null_fold
        if isinstance(left, GraphLiteral) and isinstance(right, GraphLiteral):
            folded = _fold_graph_binary(expr.op.name, left, right, expr)
            if folded is not None:
                return folded
    if _is_total_pure_op(expr.op.name, module_effects) and (folded := _fold_numeric_primitive(expr)) is not None:
        return folded
    return expr


def _rewrite_node_operands(
    node: GraphNode,
    subst: Mapping[str, GraphOperand],
    *,
    fold: bool = True,
    module_effects: Mapping[str, GraphEffect] | None = None,
    modules_by_name: Mapping[str, GraphModule] | None = None,
    local_domain_facts: Mapping[str, GraphDomainFact] | None = None,
) -> GraphNode:
    return replace(
        node,
        inputs=tuple(
            _replace_operand_refs(
                item,
                subst,
                fold=fold,
                module_effects=module_effects,
                modules_by_name=modules_by_name,
                local_domain_facts=local_domain_facts,
            )
            for item in node.inputs
        ),
        attrs={
            key: _replace_operand_refs(
                value,
                subst,
                fold=fold,
                module_effects=module_effects,
                modules_by_name=modules_by_name,
                local_domain_facts=local_domain_facts,
            )
            for key, value in node.attrs.items()
        },
    )


def _node_replacement(
    node: GraphNode,
    *,
    config: GraphOptimizeConfig,
    module_effects: Mapping[str, GraphEffect],
    modules_by_name: Mapping[str, GraphModule],
    dim_values: Mapping[str, DimToken] | None = None,
    local_domain_facts: Mapping[str, GraphDomainFact] | None = None,
    stable_shape_values: set[str] | None = None,
    blocked_dim_ref_names: set[str] | None = None,
) -> GraphOperand | None:
    if len(node.outputs) != 1:
        return None
    if config.constant_folding:
        shape_replacement = _shape_query_replacement(
            GraphExpr(
                op=node.op,
                inputs=node.inputs,
                attrs=node.attrs,
                type_expr=node.type_expr,
                dims=node.dims,
            ),
            modules_by_name=modules_by_name,
            stable_shape_values=stable_shape_values,
            blocked_dim_ref_names=blocked_dim_ref_names,
        )
        if shape_replacement is not None:
            return shape_replacement
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
    ):
        cond_value = _domain_bool_value(node.inputs[0], local_domain_facts)
        if cond_value is not None:
            selected = node.inputs[1] if cond_value else node.inputs[2]
            selected_replacement = _select_fold_replacement(selected, node.type_expr)
            if selected_replacement is None:
                return None
            if _is_atomic_operand(selected_replacement) or graph_operand_effect(
                selected_replacement,
                module_effects=dict(module_effects),
            ) == GraphEffect.TOTAL_PURE:
                return selected_replacement
    if config.constant_folding and node.op.name.startswith("core.binary.") and len(node.inputs) == 2:
        left, right = node.inputs
        domain_fold = _fold_domain_binary_comparison(
            node.op.name,
            left,
            right,
            local_domain_facts=local_domain_facts,
        )
        if domain_fold is not None:
            return domain_fold
        typed_null_fold = _fold_typed_null_comparison(node.op.name, left, right)
        if typed_null_fold is not None:
            return typed_null_fold
        if isinstance(left, GraphLiteral) and isinstance(right, GraphLiteral):
            return _fold_graph_binary(node.op.name, left, right, node)
        dim_fold = _fold_dim_binary_operand(
            node.op.name,
            left,
            right,
            type_expr=node.type_expr,
            dim_values=dim_values or {},
        )
        if dim_fold is not None:
            return dim_fold
    if (
        config.constant_folding
        and _is_total_pure_op(node.op.name, module_effects)
        and (folded := _fold_numeric_primitive(node)) is not None
    ):
        return folded
    if node.op.name == "core.tuple" and len(node.inputs) == len(node.outputs):
        return None
    return None


def _multi_output_tuple_alias_subst(node: GraphNode) -> dict[str, GraphOperand] | None:
    if node.op.name != "core.tuple":
        return None
    if node.attrs or len(node.inputs) != len(node.outputs) or len(node.outputs) <= 1:
        return None
    if not all(_is_repackaging_operand(item) for item in node.inputs):
        return None
    return {
        output.name: input_operand
        for output, input_operand in zip(node.outputs, node.inputs, strict=True)
    }


def _is_repackaging_operand(operand: GraphOperand) -> bool:
    if _is_atomic_operand(operand):
        return True
    return (
        isinstance(operand, GraphExpr)
        and operand.op.name == "core.tuple"
        and not operand.attrs
        and all(_is_repackaging_operand(item) for item in operand.inputs)
    )


def _literal_select_selected_node(
    node: GraphNode,
    module_effects: Mapping[str, GraphEffect],
    *,
    local_domain_facts: Mapping[str, GraphDomainFact] | None = None,
) -> GraphNode | None:
    if (
        node.op.name != "core.select"
        or len(node.inputs) != 3
        or node.attrs
    ):
        return None
    cond_value = _domain_bool_value(node.inputs[0], local_domain_facts)
    if cond_value is None:
        return None
    selected = node.inputs[1] if cond_value else node.inputs[2]
    selected_replacement = _select_fold_replacement(selected, node.type_expr)
    if selected_replacement is None:
        selected_replacement = selected
    if not isinstance(selected_replacement, GraphExpr):
        return replace(
            node,
            op=GraphOp("core.alias"),
            inputs=(selected_replacement,),
            attrs={},
        )
    if graph_operand_effect(selected_replacement, module_effects=dict(module_effects)) == GraphEffect.TOTAL_PURE:
        return replace(
            node,
            op=GraphOp("core.alias"),
            inputs=(selected_replacement,),
            attrs={},
        )
    return replace(
        node,
        op=selected_replacement.op,
        inputs=selected_replacement.inputs,
        attrs=selected_replacement.attrs,
    )


def _call_output_dim_value_subst(
    module: GraphModule,
    *,
    modules_by_name: Mapping[str, GraphModule],
) -> dict[str, GraphOperand]:
    local_names = _module_value_names(module)
    candidates: dict[str, GraphOperand] = {}
    conflicts: set[str] = set()

    def add_candidate(dim_name: str, operand: GraphOperand) -> None:
        if dim_name in local_names:
            return
        existing = candidates.get(dim_name)
        if existing is None:
            candidates[dim_name] = operand
        elif existing != operand:
            conflicts.add(dim_name)

    def collect_from_dim(
        formal_dim: DimToken,
        actual_dim: DimToken,
        formal_dim_values: Mapping[str, GraphOperand],
    ) -> None:
        if isinstance(formal_dim, str) and isinstance(actual_dim, str):
            replacement = formal_dim_values.get(formal_dim)
            if replacement is not None:
                add_candidate(actual_dim, replacement)
            return
        if isinstance(formal_dim, DimExprBinary) and isinstance(actual_dim, DimExprBinary):
            if formal_dim.op != actual_dim.op:
                return
            collect_from_dim(formal_dim.left, actual_dim.left, formal_dim_values)
            collect_from_dim(formal_dim.right, actual_dim.right, formal_dim_values)

    def collect_from_type(
        formal_type: TypeExpr,
        actual_type: TypeExpr,
        formal_dim_values: Mapping[str, GraphOperand],
    ) -> None:
        if isinstance(formal_type, TypeTensor) and isinstance(actual_type, TypeTensor):
            if formal_type.base != actual_type.base or len(formal_type.dims) != len(actual_type.dims):
                return
            for formal_dim, actual_dim in zip(formal_type.dims, actual_type.dims, strict=True):
                collect_from_dim(formal_dim, actual_dim, formal_dim_values)
            return
        if isinstance(formal_type, TypeNamed) and isinstance(actual_type, TypeNamed):
            if formal_type.name != actual_type.name or len(formal_type.args) != len(actual_type.args):
                return
            for formal_dim, actual_dim in zip(formal_type.args, actual_type.args, strict=True):
                collect_from_dim(formal_dim, actual_dim, formal_dim_values)
            return
        if isinstance(formal_type, TypeOptional) and isinstance(actual_type, TypeOptional):
            collect_from_type(formal_type.inner, actual_type.inner, formal_dim_values)
            return
        if isinstance(formal_type, TypeList) and isinstance(actual_type, TypeList):
            collect_from_type(formal_type.item, actual_type.item, formal_dim_values)
            return
        if isinstance(formal_type, TypeTuple) and isinstance(actual_type, TypeTuple):
            if len(formal_type.items) != len(actual_type.items):
                return
            for formal_item, actual_item in zip(formal_type.items, actual_type.items, strict=True):
                collect_from_type(formal_item, actual_item, formal_dim_values)

    for node in module.nodes:
        callee = modules_by_name.get(node.op.name)
        if callee is None or len(node.inputs) != len(callee.inputs):
            continue
        formal_dim_values: dict[str, GraphOperand] = {}
        for formal, actual in zip(callee.inputs, node.inputs, strict=True):
            if not isinstance(formal.type_expr, TypeDim | TypeInt):
                continue
            if isinstance(actual, GraphValueRef) and isinstance(actual.type_expr, TypeDim | TypeInt):
                formal_dim_values[formal.name] = actual
            elif isinstance(actual, GraphLiteral) and type(actual.value) is int:
                formal_dim_values[formal.name] = actual
        if not formal_dim_values:
            continue
        formal_output_types = _module_output_types_for_arity(callee, len(node.outputs))
        for formal_type, output in zip(formal_output_types, node.outputs, strict=True):
            collect_from_type(formal_type, output.type_expr, formal_dim_values)
    for name in conflicts:
        candidates.pop(name, None)
    return candidates


def _optimize_module_local(
    module: GraphModule,
    *,
    config: GraphOptimizeConfig,
    module_effects: Mapping[str, GraphEffect],
    modules_by_name: Mapping[str, GraphModule] | None = None,
    local_domain_facts: Mapping[str, GraphDomainFact] | None = None,
    global_dim_values: Mapping[str, DimToken] | None = None,
    global_literals: Mapping[str, GraphLiteral] | None = None,
) -> GraphModule:
    modules_by_name = modules_by_name or {}
    global_literals = global_literals or {}
    before_outputs = module.outputs
    shadowed_dim_names = _module_signature_dim_refs(module)
    global_symbol_names = {
        name
        for name, global_module in modules_by_name.items()
        if not global_module.inputs and len(global_module.outputs) == 1
    }
    shadowed_global_dim_names = shadowed_dim_names & global_symbol_names
    subst: dict[str, GraphOperand] = _call_output_dim_value_subst(
        module,
        modules_by_name=modules_by_name,
    )
    stable_shape_values = {value.name for value in module.inputs}
    dim_values: dict[str, DimToken] = {
        name: value
        for name, value in (global_dim_values or {}).items()
        if name not in shadowed_dim_names
    }
    nodes: list[GraphNode] = []
    for node in module.nodes:
        rewritten = _rewrite_node_operands(
            node,
            subst,
            fold=config.constant_folding,
            module_effects=module_effects,
            modules_by_name=modules_by_name,
            local_domain_facts=local_domain_facts,
        )
        if global_literals:
            rewritten = replace(
                rewritten,
                inputs=tuple(
                    _replace_atomic_literal_globals(item, global_literals)
                    for item in rewritten.inputs
                ),
                attrs={
                    key: _replace_atomic_literal_globals(value, global_literals)
                    for key, value in rewritten.attrs.items()
                },
            )
        selected_node = (
            _literal_select_selected_node(
                rewritten,
                module_effects,
                local_domain_facts=local_domain_facts,
            )
            if config.constant_folding
            else None
        )
        if selected_node is not None:
            rewritten = selected_node
        if (
            len(rewritten.outputs) == 1
            and not rewritten.inputs
            and not rewritten.attrs
            and rewritten.op.name in global_literals
            and rewritten.op.name != module.name
        ):
            output_name = rewritten.outputs[0].name
            replacement = global_literals[rewritten.op.name]
            if _constraints_reference_any(module.constraints, {output_name}) and _specialize_constraints(
                module.constraints,
                {output_name: replacement},
            ) is None:
                nodes.append(rewritten)
                continue
            subst[output_name] = replacement
            if type(replacement.value) is int and isinstance(
                rewritten.outputs[0].type_expr,
                TypeDim | TypeInt,
            ):
                dim_values[output_name] = replacement.value
            continue
        if (
            len(rewritten.outputs) == 1
            and not rewritten.inputs
            and not rewritten.attrs
            and rewritten.op.name != module.name
        ):
            global_module = modules_by_name.get(rewritten.op.name)
            if (
                global_module is not None
                and global_module.is_global_binding
                and not global_module.inputs
                and len(global_module.outputs) == 1
            ):
                output_name = rewritten.outputs[0].name
                replacement = GraphValueRef(
                    name=rewritten.op.name,
                    type_expr=rewritten.outputs[0].type_expr,
                    dims=rewritten.outputs[0].dims,
                )
                if _constraints_reference_any(module.constraints, {output_name}) and _specialize_constraints(
                    module.constraints,
                    {output_name: replacement},
                ) is None:
                    nodes.append(rewritten)
                    continue
                subst[output_name] = replacement
                replacement_dim = _operand_dim_token(replacement, dim_values)
                if replacement_dim is not None:
                    dim_values[output_name] = replacement_dim
                continue
        tuple_subst = (
            _multi_output_tuple_alias_subst(rewritten)
            if config.atomic_alias_cleanup
            else None
        )
        if tuple_subst is not None:
            if _constraints_reference_any(module.constraints, set(tuple_subst)) and _specialize_constraints(
                module.constraints,
                tuple_subst,
            ) is None:
                nodes.append(rewritten)
                continue
            subst.update(tuple_subst)
            continue
        replacement = _node_replacement(
            rewritten,
            config=config,
            module_effects=module_effects,
            modules_by_name=modules_by_name,
            dim_values=dim_values,
            local_domain_facts=local_domain_facts,
            stable_shape_values=stable_shape_values,
            blocked_dim_ref_names=shadowed_global_dim_names,
        )
        if replacement is not None and len(rewritten.outputs) == 1:
            output_name = rewritten.outputs[0].name
            if _constraints_reference_any(module.constraints, {output_name}) and _specialize_constraints(
                module.constraints,
                {output_name: replacement},
            ) is None:
                nodes.append(rewritten)
                continue
            subst[output_name] = replacement
            replacement_dim = _operand_dim_token(replacement, dim_values)
            if replacement_dim is not None:
                dim_values[output_name] = replacement_dim
            continue
        nodes.append(rewritten)
        if len(rewritten.outputs) == 1 and isinstance(
            rewritten.outputs[0].type_expr, TypeDim | TypeInt
        ):
            dim_expr = _operand_dim_token(
                GraphExpr(
                    op=rewritten.op,
                    inputs=rewritten.inputs,
                    attrs=rewritten.attrs,
                    type_expr=rewritten.type_expr,
                    dims=rewritten.dims,
                ),
                dim_values,
            )
            if dim_expr is not None:
                dim_values[rewritten.outputs[0].name] = dim_expr
    outputs = tuple(
        _replace_operand_refs(
            item,
            subst,
            fold=config.constant_folding,
            module_effects=module_effects,
            modules_by_name=modules_by_name,
            local_domain_facts=local_domain_facts,
        )
        for item in module.outputs
    )
    constraints = _specialize_constraints(module.constraints, subst)
    module = replace(
        module,
        nodes=tuple(nodes),
        outputs=outputs,
        constraints=module.constraints if constraints is None else constraints,
    )
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
    live.update(_module_dim_refs(module))
    for output in module.outputs:
        _operand_refs(output, live)
    kept_rev: list[GraphNode] = []
    for node in reversed(module.nodes):
        output_names = {value.name for value in node.outputs}
        if output_names and not (output_names & live) and _is_total_pure_node(
            node,
            module_effects,
        ):
            continue
        live.difference_update(output_names)
        for operand in node.inputs:
            _operand_refs(operand, live)
        for operand in node.attrs.values():
            _operand_refs(operand, live)
        kept_rev.append(node)
    kept_nodes = tuple(reversed(kept_rev))
    value_names = {value.name for value in module.inputs}
    for node in kept_nodes:
        value_names.update(value.name for value in node.outputs)
    kept_constraints = tuple(
        constraint
        for constraint in module.constraints
        if _constraint_has_callsite_guard(constraint)
        or _constraint_ref_names(constraint) <= value_names | _module_dim_refs(module)
    )
    return replace(module, nodes=kept_nodes, constraints=kept_constraints)


def _collect_nested_total_expr_counts(
    operand: GraphOperand,
    *,
    module_effects: Mapping[str, GraphEffect],
    counts: Counter[object],
) -> None:
    if not isinstance(operand, GraphExpr):
        return
    for item in operand.inputs:
        _collect_nested_total_expr_counts(item, module_effects=module_effects, counts=counts)
    for item in operand.attrs.values():
        _collect_nested_total_expr_counts(item, module_effects=module_effects, counts=counts)
    if graph_operand_effect(operand, module_effects=dict(module_effects)) == GraphEffect.TOTAL_PURE:
        counts[_graph_operand_key(operand)] += 1


def _fresh_graph_value_name(used_names: set[str], preferred: str) -> str:
    if preferred not in used_names:
        used_names.add(preferred)
        return preferred
    index = 1
    while True:
        candidate = f"{preferred}_{index}"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        index += 1


def _hoist_repeated_nested_total_exprs_module(
    module: GraphModule,
    *,
    module_effects: Mapping[str, GraphEffect],
) -> GraphModule:
    counts: Counter[object] = Counter()
    for node in module.nodes:
        for operand in (*node.inputs, *node.attrs.values()):
            _collect_nested_total_expr_counts(
                operand,
                module_effects=module_effects,
                counts=counts,
            )
    for output in module.outputs:
        _collect_nested_total_expr_counts(
            output,
            module_effects=module_effects,
            counts=counts,
        )
    repeated = {key for key, count in counts.items() if count > 1}
    if not repeated:
        return module

    used_names = _module_value_names(module)
    emitted: dict[object, GraphValueRef] = {}
    temp_index = 0
    changed = False

    def rewrite_operand(
        operand: GraphOperand,
        *,
        inserted_nodes: list[GraphNode],
        source_id: str,
    ) -> GraphOperand:
        nonlocal changed, temp_index
        if not isinstance(operand, GraphExpr):
            return operand
        original_key = _graph_operand_key(operand)
        rewritten = replace(
            operand,
            inputs=tuple(
                rewrite_operand(item, inserted_nodes=inserted_nodes, source_id=source_id)
                for item in operand.inputs
            ),
            attrs={
                key: rewrite_operand(value, inserted_nodes=inserted_nodes, source_id=source_id)
                for key, value in operand.attrs.items()
            },
        )
        if original_key not in repeated:
            return rewritten
        existing = emitted.get(original_key)
        if existing is not None:
            changed = True
            return existing
        temp_index += 1
        name = _fresh_graph_value_name(used_names, f"__cse{temp_index}")
        value = GraphValue(name=name, type_expr=rewritten.type_expr, dims=rewritten.dims)
        ref = GraphValueRef(name=name, type_expr=value.type_expr, dims=value.dims)
        emitted[original_key] = ref
        inserted_nodes.append(
            GraphNode(
                id=f"{module.name}:nested-cse:{source_id}:{temp_index}",
                op=rewritten.op,
                inputs=rewritten.inputs,
                attrs=rewritten.attrs,
                outputs=(value,),
                source_module=module.name,
                type_expr=rewritten.type_expr,
                dims=rewritten.dims,
            )
        )
        changed = True
        return ref

    nodes: list[GraphNode] = []
    for node in module.nodes:
        inserted: list[GraphNode] = []
        rewritten = replace(
            node,
            inputs=tuple(
                rewrite_operand(item, inserted_nodes=inserted, source_id=node.id)
                for item in node.inputs
            ),
            attrs={
                key: rewrite_operand(value, inserted_nodes=inserted, source_id=node.id)
                for key, value in node.attrs.items()
            },
        )
        nodes.extend(inserted)
        nodes.append(rewritten)
    output_inserted: list[GraphNode] = []
    outputs = tuple(
        rewrite_operand(output, inserted_nodes=output_inserted, source_id="return")
        for output in module.outputs
    )
    nodes.extend(output_inserted)
    if not changed:
        return module
    return replace(module, nodes=tuple(nodes), outputs=outputs)


def _hoist_eager_nested_exprs_module(
    module: GraphModule,
    *,
    module_effects: Mapping[str, GraphEffect],
) -> GraphModule:
    used_names = _module_value_names(module)
    next_index = 1
    changed = False

    def fresh_name() -> str:
        nonlocal next_index
        name, next_index = _fresh_canonical_value_name(used_names, next_index)
        return name

    def hoist_operand(
        operand: GraphOperand,
        *,
        inserted_nodes: list[GraphNode],
        source_id: str,
        eager: bool,
    ) -> GraphOperand:
        nonlocal changed
        if not isinstance(operand, GraphExpr):
            return operand
        if operand.op.name == "core.select" and len(operand.inputs) == 3 and not operand.attrs:
            rewritten = replace(
                operand,
                inputs=(
                    hoist_operand(
                        operand.inputs[0],
                        inserted_nodes=inserted_nodes,
                        source_id=source_id,
                        eager=True,
                    ),
                    hoist_operand(
                        operand.inputs[1],
                        inserted_nodes=inserted_nodes,
                        source_id=source_id,
                        eager=False,
                    ),
                    hoist_operand(
                        operand.inputs[2],
                        inserted_nodes=inserted_nodes,
                        source_id=source_id,
                        eager=False,
                    ),
                ),
            )
        else:
            rewritten = replace(
                operand,
                inputs=tuple(
                    hoist_operand(
                        item,
                        inserted_nodes=inserted_nodes,
                        source_id=source_id,
                        eager=True,
                    )
                    for item in operand.inputs
                ),
                attrs={
                    key: hoist_operand(
                        value,
                        inserted_nodes=inserted_nodes,
                        source_id=source_id,
                        eager=True,
                    )
                    for key, value in operand.attrs.items()
                },
            )
        if (
            not eager
            or rewritten.op.name in {"core.list", "core.tuple"}
            or graph_operand_effect(rewritten, module_effects=dict(module_effects)) != GraphEffect.TOTAL_PURE
        ):
            return rewritten
        name = fresh_name()
        value = GraphValue(name=name, type_expr=rewritten.type_expr, dims=rewritten.dims)
        inserted_nodes.append(
            GraphNode(
                id=f"{module.name}:hoist:{source_id}:{name}",
                op=rewritten.op,
                inputs=rewritten.inputs,
                attrs=rewritten.attrs,
                outputs=(value,),
                source_module=module.name,
                type_expr=rewritten.type_expr,
                dims=rewritten.dims,
            )
        )
        changed = True
        return GraphValueRef(name=name, type_expr=value.type_expr, dims=value.dims)

    nodes: list[GraphNode] = []
    for node in module.nodes:
        inserted: list[GraphNode] = []
        if node.op.name == "core.select" and len(node.inputs) == 3 and not node.attrs:
            inputs = (
                hoist_operand(node.inputs[0], inserted_nodes=inserted, source_id=node.id, eager=True),
                hoist_operand(node.inputs[1], inserted_nodes=inserted, source_id=node.id, eager=False),
                hoist_operand(node.inputs[2], inserted_nodes=inserted, source_id=node.id, eager=False),
            )
            attrs = node.attrs
        else:
            inputs = tuple(
                hoist_operand(item, inserted_nodes=inserted, source_id=node.id, eager=True)
                for item in node.inputs
            )
            attrs = {
                key: hoist_operand(value, inserted_nodes=inserted, source_id=node.id, eager=True)
                for key, value in node.attrs.items()
            }
        nodes.extend(inserted)
        nodes.append(replace(node, inputs=inputs, attrs=attrs))
    if not changed:
        return module
    return replace(module, nodes=tuple(nodes))


def _hoist_eager_nested_exprs(
    graph: GraphProgram,
    *,
    module_effects: Mapping[str, GraphEffect],
) -> GraphProgram:
    modules = tuple(
        _hoist_eager_nested_exprs_module(module, module_effects=module_effects)
        for module in graph.modules
    )
    if modules == graph.modules:
        return graph
    return replace(graph, modules=modules)


def _common_subexpression_eliminate_module(
    module: GraphModule,
    *,
    module_effects: Mapping[str, GraphEffect],
    fold: bool,
) -> GraphModule:
    subst: dict[str, GraphOperand] = {}
    seen: dict[object, tuple[GraphValue, ...]] = {}
    nodes: list[GraphNode] = []
    changed = False
    for node in module.nodes:
        rewritten = _rewrite_node_operands(
            node,
            subst,
            fold=fold,
            module_effects=module_effects,
        )
        if (
            len(rewritten.outputs) == 1
            and _is_total_pure_node(rewritten, module_effects)
        ):
            key = _graph_node_cse_key(rewritten)
            previous = seen.get(key)
            if previous is not None:
                output_name = rewritten.outputs[0].name
                if previous[0].name in subst:
                    nodes.append(rewritten)
                    continue
                replacement = GraphValueRef(
                    name=previous[0].name,
                    type_expr=rewritten.outputs[0].type_expr,
                    dims=rewritten.outputs[0].dims,
                )
                replacement_refs: set[str] = set()
                _operand_refs(replacement, replacement_refs)
                if output_name in replacement_refs:
                    nodes.append(rewritten)
                    continue
                if _constraints_reference_any(module.constraints, {output_name}) and _specialize_constraints(
                    module.constraints,
                    {output_name: replacement},
                ) is None:
                    nodes.append(rewritten)
                    continue
                subst[output_name] = replacement
                changed = True
                continue
            seen[key] = rewritten.outputs
        nodes.append(rewritten)
    outputs = tuple(
        _replace_operand_refs(
            output,
            subst,
            fold=fold,
            module_effects=module_effects,
        )
        for output in module.outputs
    )
    constraints = _specialize_constraints(module.constraints, subst)
    rewritten_module = (
        module
        if not changed
        else replace(
            module,
            nodes=tuple(nodes),
            outputs=outputs,
            constraints=module.constraints if constraints is None else constraints,
        )
    )
    return _hoist_repeated_nested_total_exprs_module(
        rewritten_module,
        module_effects=module_effects,
    )


def _returned_name_preserving_module(module: GraphModule, before_outputs: tuple[GraphOperand, ...]) -> GraphModule:
    renames: dict[str, str] = {}
    input_names = {value.name for value in module.inputs}
    defined_names = set(input_names)
    node_output_names: set[str] = set()
    for node in module.nodes:
        for value in node.outputs:
            defined_names.add(value.name)
            node_output_names.add(value.name)
    for before, after in zip(before_outputs, module.outputs, strict=False):
        if not isinstance(before, GraphValueRef) or not isinstance(after, GraphValueRef):
            continue
        if before.name == after.name:
            continue
        if after.name not in node_output_names:
            continue
        if before.name in defined_names:
            continue
        renames[after.name] = before.name
        defined_names.add(before.name)
    if not renames:
        return module
    constraint_subst = {
        old: GraphValueRef(name=new, type_expr=TypeAny())
        for old, new in renames.items()
        if old != new
    }
    constraints = _specialize_constraints(module.constraints, constraint_subst)
    return replace(
        module,
        nodes=tuple(
            replace(
                node,
                inputs=tuple(rename_operand(item, renames) for item in node.inputs),
                attrs={key: rename_operand(value, renames) for key, value in node.attrs.items()},
                outputs=tuple(
                    replace(output, name=renames.get(output.name, output.name))
                    for output in node.outputs
                ),
            )
            for node in module.nodes
        ),
        outputs=tuple(rename_operand(output, renames) for output in module.outputs),
        constraints=module.constraints if constraints is None else constraints,
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
    graph = _sanitize_graph_constraints(graph)
    _validate_optimizer_graph(graph, phase="prune.input")
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
    pruned = _sanitize_graph_constraints(pruned)
    _validate_optimizer_graph(pruned, phase="prune")
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


def _top_level_calls_by_callee(graph: GraphProgram) -> dict[str, list[tuple[str, GraphNode]]]:
    module_names = {module.name for module in graph.modules}
    calls: dict[str, list[tuple[str, GraphNode]]] = {name: [] for name in module_names}
    for module in graph.modules:
        for node in module.nodes:
            if node.op.name in module_names:
                calls[node.op.name].append((module.name, node))
    return calls


def _all_calls_by_callee(graph: GraphProgram) -> dict[str, list[tuple[str, GraphNode | GraphExpr]]]:
    module_names = {module.name for module in graph.modules}
    calls: dict[str, list[tuple[str, GraphNode | GraphExpr]]] = {name: [] for name in module_names}

    def collect_operand(caller_name: str, operand: GraphOperand) -> None:
        if not isinstance(operand, GraphExpr):
            return
        if operand.op.name in module_names:
            calls[operand.op.name].append((caller_name, operand))
        for item in operand.inputs:
            collect_operand(caller_name, item)
        for item in operand.attrs.values():
            collect_operand(caller_name, item)

    for module in graph.modules:
        for node in module.nodes:
            if node.op.name in module_names:
                calls[node.op.name].append((module.name, node))
            for operand in node.inputs:
                collect_operand(module.name, operand)
            for operand in node.attrs.values():
                collect_operand(module.name, operand)
        for output in module.outputs:
            collect_operand(module.name, output)
    return calls


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


def _has_safe_specialization_actual(
    node: GraphNode,
    module: GraphModule,
    *,
    global_symbol_names: set[str],
    input_domain_facts: Mapping[str, GraphDomainFact] | None = None,
) -> bool:
    if len(node.inputs) != len(module.inputs):
        return False
    return any(
        (
            _domain_fact_specialization_operand(
                None if input_domain_facts is None else input_domain_facts.get(formal.name),
                formal,
            )
            is not None
            or _is_safe_callsite_specialization_operand(
                item,
                global_symbol_names=global_symbol_names,
            )
        )
        and _specialization_actual_matches_formal(item, formal)
        for formal, item in zip(module.inputs, node.inputs, strict=True)
    )


def _callsite_specialization_subst(
    module: GraphModule,
    inputs: tuple[GraphOperand, ...],
    *,
    global_symbol_names: set[str],
    caller_name: str | None = None,
    caller_modules: Mapping[str, GraphModule] | None = None,
    input_domain_facts: Mapping[str, GraphDomainFact] | None = None,
) -> dict[str, GraphOperand]:
    if len(inputs) != len(module.inputs):
        return {}
    subst: dict[str, GraphOperand] = {}
    for formal, actual in zip(module.inputs, inputs, strict=True):
        if not _specialization_actual_matches_formal(actual, formal):
            continue
        if _is_safe_callsite_specialization_operand(
            actual,
            global_symbol_names=global_symbol_names,
        ):
            subst[formal.name] = _canonical_specialization_operand(
                actual,
                global_symbol_names=global_symbol_names,
            )
            continue
        if caller_name is None or caller_modules is None:
            continue
        traced = _candidate_actual_from_operand(
            actual,
            caller_name=caller_name,
            caller_modules=caller_modules,
            candidates={},
            global_symbol_names=global_symbol_names,
        )
        if traced is not None and _specialization_actual_matches_formal(traced, formal):
            subst[formal.name] = _canonical_specialization_operand(
                traced,
                global_symbol_names=global_symbol_names,
            )
            continue
        domain_actual = _domain_fact_specialization_operand(
            None if input_domain_facts is None else input_domain_facts.get(formal.name),
            formal,
        )
        if domain_actual is not None:
            subst[formal.name] = _canonical_specialization_operand(
                domain_actual,
                global_symbol_names=global_symbol_names,
            )
    return subst


def _can_specialize_module(module: GraphModule, *, recursive_modules: set[str], main_module: str) -> bool:
    if module.name == main_module:
        return False
    if module.name in recursive_modules:
        return False
    return True


def _constraint_operand_ref_names(operand: ConstraintOperand) -> set[str]:
    if isinstance(operand, tuple):
        names: set[str] = set()
        for item in operand:
            names.update(_constraint_atom_ref_names(item))
        return names
    return _constraint_atom_ref_names(operand)


def _constraint_atom_ref_names(atom: ConstraintAtom) -> set[str]:
    if isinstance(atom, str):
        return {atom}
    if isinstance(atom, DimExprBinary):
        return set(dim_token_names(atom))
    return set()


def _constraint_ref_names(constraint: Constraint) -> set[str]:
    if constraint.relation == "callsite":
        return set()
    names = _constraint_operand_ref_names(constraint.left)
    if constraint.right is not None:
        names.update(_constraint_operand_ref_names(constraint.right))
    for guard in constraint.guards:
        names.update(_constraint_ref_names(guard))
    return names


def _constraint_has_callsite_guard(constraint: Constraint) -> bool:
    return any(
        guard.relation == "callsite" or _constraint_has_callsite_guard(guard)
        for guard in constraint.guards
    )


def _constraints_reference_any(
    constraints: tuple[Constraint, ...],
    names: set[str],
) -> bool:
    return any(_constraint_ref_names(constraint) & names for constraint in constraints)


def _constraint_is_trivially_true(constraint: Constraint) -> bool:
    if constraint.guards and not all(_constraint_is_trivially_true(guard) for guard in constraint.guards):
        return False
    left = constraint.left
    right = constraint.right
    if constraint.relation == "=":
        return right is not None and left == right
    if constraint.relation == "!=":
        return right is not None and _constraint_literals_comparable(left, right) and left != right
    if constraint.relation == "is_true":
        return left is True and right is None
    if constraint.relation == "is_false":
        return left is False and right is None
    if constraint.relation == "is_null":
        return left is None and right is None
    if constraint.relation == "not_null":
        return left is not None and not isinstance(left, str | tuple | DimExprBinary) and right is None
    if (
        constraint.relation in {"<", "<=", ">", ">="}
        and type(left) is int
        and type(right) is int
    ):
        return _evaluate_int_relation(left, constraint.relation, right)
    return False


def _constraint_is_trivially_false(constraint: Constraint) -> bool:
    if constraint.guards and not all(_constraint_is_trivially_true(guard) for guard in constraint.guards):
        return False
    left = constraint.left
    right = constraint.right
    if constraint.relation == "=":
        return right is not None and _constraint_literals_comparable(left, right) and left != right
    if constraint.relation == "!=":
        return right is not None and left == right
    if constraint.relation == "is_true":
        return left is False and right is None
    if constraint.relation == "is_false":
        return left is True and right is None
    if constraint.relation == "is_null":
        return left is not None and not isinstance(left, str | tuple | DimExprBinary) and right is None
    if constraint.relation == "not_null":
        return left is None and right is None
    if (
        constraint.relation in {"<", "<=", ">", ">="}
        and type(left) is int
        and type(right) is int
    ):
        return not _evaluate_int_relation(left, constraint.relation, right)
    return False


def _constraint_literals_comparable(left: ConstraintOperand, right: ConstraintOperand) -> bool:
    literal_types = (int, bool, type(None))
    return isinstance(left, literal_types) and isinstance(right, literal_types)


def _evaluate_int_relation(left: int, relation: str, right: int) -> bool:
    if relation == "<":
        return left < right
    if relation == "<=":
        return left <= right
    if relation == ">":
        return left > right
    if relation == ">=":
        return left >= right
    raise ValueError(f"unsupported int constraint relation {relation!r}")


def _specialize_constraints(
    constraints: tuple[Constraint, ...],
    subst: Mapping[str, GraphOperand],
) -> tuple[Constraint, ...] | None:
    rewritten: list[Constraint] = []
    for constraint in constraints:
        try:
            candidate = replace_constraint_refs(constraint, subst)
        except UnsupportedConstraintSubstitution:
            if _constraint_has_callsite_guard(constraint):
                continue
            return None
        if _constraint_is_trivially_false(candidate):
            return None
        if _constraint_is_trivially_true(candidate):
            continue
        rewritten.append(candidate)
    return tuple(rewritten)


def _specialized_module(
    module: GraphModule,
    *,
    name: str,
    call_node: GraphNode,
    subst_override: Mapping[str, GraphOperand] | None = None,
    global_symbol_names: set[str] | None = None,
) -> GraphModule | None:
    if len(call_node.inputs) != len(module.inputs):
        return None
    subst: dict[str, GraphOperand] = dict(subst_override or {})
    dim_subst: dict[str, DimToken] = {}
    kept_inputs: list[GraphValue] = []
    for formal, actual in zip(module.inputs, call_node.inputs, strict=True):
        if formal.name in subst:
            actual = subst[formal.name]
            subst[formal.name] = actual
            if (
                isinstance(formal.type_expr, TypeDim | TypeInt)
                and isinstance(actual, GraphLiteral)
                and type(actual.value) is int
            ):
                dim_subst[formal.name] = actual.value
            continue
        if subst_override is None and _is_safe_callsite_specialization_operand(
            actual,
            global_symbol_names=global_symbol_names or set(),
        ):
            subst[formal.name] = _canonical_specialization_operand(
                actual,
                global_symbol_names=global_symbol_names or set(),
            )
            if (
                isinstance(formal.type_expr, TypeDim | TypeInt)
                and isinstance(actual, GraphLiteral)
                and type(actual.value) is int
            ):
                dim_subst[formal.name] = actual.value
            continue
        kept_inputs.append(formal)
    if not subst:
        return None
    kept_constraints = _specialize_constraints(module.constraints, subst)
    if kept_constraints is None:
        return None
    nodes = tuple(_rewrite_node_operands(node, subst) for node in module.nodes)
    outputs = tuple(_replace_operand_refs(output, subst) for output in module.outputs)
    specialized = replace(
        module,
        name=name,
        inputs=tuple(kept_inputs),
        nodes=nodes,
        outputs=outputs,
        constraints=tuple(kept_constraints),
    )
    if dim_subst:
        dim_specialized = substitute_graph_module_dims(specialized, dim_subst)
        specialized = dim_specialized
    cleanup_config = GraphOptimizeConfig(
        prune_to_main=False,
        common_subexpression_elimination=False,
        specialize_definitions="off",
        inline_safe=False,
    )
    for _ in range(64):
        before = specialized
        specialized = _optimize_module_local(
            specialized,
            config=cleanup_config,
            module_effects={},
        )
        if specialized == before:
            break
    else:
        raise RuntimeError(f"specialized module {name!r} local cleanup did not converge")
    bound_names = (
        {value.name for value in specialized.inputs}
        | {value.name for node in specialized.nodes for value in node.outputs}
        | _module_signature_dim_refs(specialized)
        | (global_symbol_names or set())
    )
    specialized = replace(
        specialized,
        constraints=tuple(
            constraint
            for constraint in specialized.constraints
            if not (
                _constraint_has_callsite_guard(constraint)
                and not (_constraint_ref_names(constraint) <= bound_names)
            )
        ),
    )
    if not _specialized_module_render_closure_safe(
        specialized,
        global_symbol_names=global_symbol_names or set(),
    ):
        return None
    return specialized


def _specialized_candidate_valid(
    graph: GraphProgram,
    modules: tuple[GraphModule, ...],
    *,
    phase: str,
) -> bool:
    candidate = replace(graph, modules=modules)
    candidate = _sanitize_graph_constraints(candidate)
    try:
        _validate_optimizer_graph(candidate, phase=phase)
    except ValueError:
        return False
    return True


def _rewrite_call_to_specialized(node: GraphNode, original: GraphModule, specialized_name: str) -> GraphNode:
    return _rewrite_call_to_specialized_with_subst(node, original, specialized_name, None)


def _rewrite_call_to_specialized_with_subst(
    node: GraphNode,
    original: GraphModule,
    specialized_name: str,
    subst_names: set[str] | None,
) -> GraphNode:
    inputs = tuple(
        actual
        for formal, actual in zip(original.inputs, node.inputs, strict=True)
        if not (
            formal.name in subst_names
            if subst_names is not None
            else _is_safe_specialization_operand(actual)
        )
    )
    return replace(node, op=GraphOp(specialized_name), inputs=inputs)


def _rewrite_call_expr_to_specialized_with_subst(
    expr: GraphExpr,
    original: GraphModule,
    specialized_name: str,
    subst_names: set[str],
) -> GraphExpr:
    inputs = tuple(
        actual
        for formal, actual in zip(original.inputs, expr.inputs, strict=True)
        if formal.name not in subst_names
    )
    return replace(expr, op=GraphOp(specialized_name), inputs=inputs)


def _rewrite_recursive_specialized_operand(
    operand: GraphOperand,
    *,
    originals_by_name: Mapping[str, GraphModule],
    clone_names: Mapping[str, str],
    subst_names_by_module: Mapping[str, set[str]],
) -> GraphOperand:
    if not isinstance(operand, GraphExpr):
        return operand
    inputs = tuple(
        _rewrite_recursive_specialized_operand(
            item,
            originals_by_name=originals_by_name,
            clone_names=clone_names,
            subst_names_by_module=subst_names_by_module,
        )
        for item in operand.inputs
    )
    attrs = {
        key: _rewrite_recursive_specialized_operand(
            value,
            originals_by_name=originals_by_name,
            clone_names=clone_names,
            subst_names_by_module=subst_names_by_module,
        )
        for key, value in operand.attrs.items()
    }
    rewritten = replace(operand, inputs=inputs, attrs=attrs)
    clone_name = clone_names.get(rewritten.op.name)
    if clone_name is None:
        return rewritten
    original = originals_by_name[rewritten.op.name]
    subst_names = subst_names_by_module[rewritten.op.name]
    return _rewrite_call_expr_to_specialized_with_subst(
        rewritten,
        original,
        clone_name,
        subst_names,
    )


def _rewrite_specialized_nested_operand(
    operand: GraphOperand,
    *,
    replacements: Mapping[str, tuple[str, GraphModule, set[str]]],
) -> GraphOperand:
    if not isinstance(operand, GraphExpr):
        return operand
    rewritten = replace(
        operand,
        inputs=tuple(
            _rewrite_specialized_nested_operand(item, replacements=replacements)
            for item in operand.inputs
        ),
        attrs={
            key: _rewrite_specialized_nested_operand(value, replacements=replacements)
            for key, value in operand.attrs.items()
        },
    )
    replacement = replacements.get(rewritten.op.name)
    if replacement is None:
        return rewritten
    clone_name, original, subst_names = replacement
    return _rewrite_call_expr_to_specialized_with_subst(
        rewritten,
        original,
        clone_name,
        subst_names,
    )


def _rewrite_recursive_specialized_node(
    node: GraphNode,
    *,
    originals_by_name: Mapping[str, GraphModule],
    clone_names: Mapping[str, str],
    subst_names_by_module: Mapping[str, set[str]],
) -> GraphNode:
    inputs = tuple(
        _rewrite_recursive_specialized_operand(
            item,
            originals_by_name=originals_by_name,
            clone_names=clone_names,
            subst_names_by_module=subst_names_by_module,
        )
        for item in node.inputs
    )
    attrs = {
        key: _rewrite_recursive_specialized_operand(
            value,
            originals_by_name=originals_by_name,
            clone_names=clone_names,
            subst_names_by_module=subst_names_by_module,
        )
        for key, value in node.attrs.items()
    }
    rewritten = replace(node, inputs=inputs, attrs=attrs)
    clone_name = clone_names.get(rewritten.op.name)
    if clone_name is None:
        return rewritten
    original = originals_by_name[rewritten.op.name]
    subst_names = subst_names_by_module[rewritten.op.name]
    return _rewrite_call_to_specialized_with_subst(
        rewritten,
        original,
        clone_name,
        subst_names,
    )


def _shared_constant_specialization_subst(
    module: GraphModule,
    calls: list[tuple[str, GraphNode | GraphExpr]],
    *,
    global_symbol_names: set[str],
    caller_modules: Mapping[str, GraphModule],
    input_domain_facts: Mapping[str, GraphDomainFact] | None = None,
) -> dict[str, GraphOperand]:
    if not calls:
        return {}
    candidates: dict[int, GraphOperand] = {}
    for index, formal in enumerate(module.inputs):
        actuals: list[GraphOperand] = []
        for caller_name, node in calls:
            if len(node.inputs) != len(module.inputs):
                break
            actual = node.inputs[index]
            if not _specialization_actual_matches_formal(actual, formal):
                break
            if _is_safe_shared_specialization_operand(
                actual,
                global_symbol_names=global_symbol_names,
            ):
                actuals.append(
                    _canonical_specialization_operand(
                        actual,
                        global_symbol_names=global_symbol_names,
                    )
                )
                continue
            traced = _candidate_actual_from_operand(
                actual,
                caller_name=caller_name,
                caller_modules=caller_modules,
                candidates={},
                global_symbol_names=global_symbol_names,
            )
            if traced is not None and _specialization_actual_matches_formal(traced, formal):
                actuals.append(
                    _canonical_specialization_operand(
                        traced,
                        global_symbol_names=global_symbol_names,
                    )
                )
                continue
            domain_actual = _domain_fact_specialization_operand(
                None if input_domain_facts is None else input_domain_facts.get(formal.name),
                formal,
            )
            if domain_actual is not None:
                actuals.append(
                    _canonical_specialization_operand(
                        domain_actual,
                        global_symbol_names=global_symbol_names,
                    )
                )
                continue
            break
        else:
            first = actuals[0]
            if all(_graph_operand_key(actual) == _graph_operand_key(first) for actual in actuals[1:]):
                candidates[index] = first
    return {
        module.inputs[index].name: actual
        for index, actual in candidates.items()
    }


def _external_top_level_scc_calls(
    graph: GraphProgram,
    component: set[str],
) -> list[tuple[str, GraphNode]]:
    calls: list[tuple[str, GraphNode]] = []
    for module in graph.modules:
        for node in module.nodes:
            if node.op.name in component and module.name not in component:
                calls.append((module.name, node))
    return calls


def _internal_top_level_scc_calls(
    graph: GraphProgram,
    component: set[str],
) -> list[tuple[str, GraphNode | GraphExpr]]:
    calls: list[tuple[str, GraphNode | GraphExpr]] = []

    def collect_operand(caller_name: str, operand: GraphOperand) -> None:
        if not isinstance(operand, GraphExpr):
            return
        if operand.op.name in component:
            calls.append((caller_name, operand))
        for item in operand.inputs:
            collect_operand(caller_name, item)
        for item in operand.attrs.values():
            collect_operand(caller_name, item)

    for module in graph.modules:
        if module.name not in component:
            continue
        for node in module.nodes:
            if node.op.name in component:
                calls.append((module.name, node))
            for operand in node.inputs:
                collect_operand(module.name, operand)
            for operand in node.attrs.values():
                collect_operand(module.name, operand)
        for output in module.outputs:
            collect_operand(module.name, output)
    return calls


def _candidate_actual_from_operand(
    operand: GraphOperand,
    *,
    caller_name: str,
    caller_modules: Mapping[str, GraphModule],
    candidates: Mapping[tuple[str, int], GraphOperand],
    global_symbol_names: set[str],
) -> GraphOperand | None:
    if _is_safe_shared_specialization_operand(
        operand,
        global_symbol_names=global_symbol_names,
    ):
        return _canonical_specialization_operand(
            operand,
            global_symbol_names=global_symbol_names,
        )
    if not isinstance(operand, GraphValueRef):
        return None
    caller = caller_modules.get(caller_name)
    if caller is None:
        return None
    for index, formal in enumerate(caller.inputs):
        if formal.name == operand.name:
            return candidates.get((caller_name, index))
    producers = {
        output.name: node
        for node in caller.nodes
        if len(node.outputs) == 1
        for output in node.outputs
    }
    producer = producers.get(operand.name)
    if producer is None:
        return None
    if producer.op.name in {"core.alias", "core.ascribe"} and len(producer.inputs) == 1 and not producer.attrs:
        return _candidate_actual_from_operand(
            producer.inputs[0],
            caller_name=caller_name,
            caller_modules=caller_modules,
            candidates=candidates,
            global_symbol_names=global_symbol_names,
        )
    candidate = GraphExpr(
        op=producer.op,
        inputs=producer.inputs,
        attrs=producer.attrs,
        type_expr=producer.type_expr,
        dims=producer.dims,
    )
    if _is_safe_shared_specialization_operand(
        candidate,
        global_symbol_names=global_symbol_names,
    ):
        return _canonical_specialization_operand(
            candidate,
            global_symbol_names=global_symbol_names,
        )
    return None


def _recursive_scc_specialization_substs(
    graph: GraphProgram,
    component: set[str],
    *,
    global_symbol_names: set[str],
) -> dict[str, dict[str, GraphOperand]]:
    modules_by_name = {module.name: module for module in graph.modules}
    external_calls = _external_top_level_scc_calls(graph, component)
    if len(external_calls) != 1:
        return {}
    entry_caller_name, entry_call = external_calls[0]
    entry = modules_by_name[entry_call.op.name]
    if len(entry_call.inputs) != len(entry.inputs):
        return {}

    candidates: dict[tuple[str, int], GraphOperand] = {}
    for index, actual in enumerate(entry_call.inputs):
        formal = entry.inputs[index]
        if not _specialization_actual_matches_formal(actual, formal):
            continue
        candidate_actual = _candidate_actual_from_operand(
            actual,
            caller_name=entry_caller_name,
            caller_modules=modules_by_name,
            candidates=candidates,
            global_symbol_names=global_symbol_names,
        )
        if candidate_actual is not None and _specialization_actual_matches_formal(candidate_actual, formal):
            candidates[(entry.name, index)] = candidate_actual

    internal_calls = _internal_top_level_scc_calls(graph, component)
    changed = True
    while changed:
        changed = False
        for caller_name, node in internal_calls:
            callee = modules_by_name[node.op.name]
            if len(node.inputs) != len(callee.inputs):
                continue
            for index, actual in enumerate(node.inputs):
                formal = callee.inputs[index]
                if not _specialization_actual_matches_formal(actual, formal):
                    continue
                propagated = _candidate_actual_from_operand(
                    actual,
                    caller_name=caller_name,
                    caller_modules=modules_by_name,
                    candidates=candidates,
                    global_symbol_names=global_symbol_names,
                )
                if propagated is None or not _specialization_actual_matches_formal(propagated, formal):
                    continue
                key = (callee.name, index)
                existing = candidates.get(key)
                if existing is None:
                    candidates[key] = propagated
                    changed = True
                elif _graph_operand_key(existing) != _graph_operand_key(propagated):
                    del candidates[key]
                    changed = True

        for caller_name, node in internal_calls:
            callee = modules_by_name[node.op.name]
            if len(node.inputs) != len(callee.inputs):
                continue
            for index, actual in enumerate(node.inputs):
                key = (callee.name, index)
                expected = candidates.get(key)
                if expected is None:
                    continue
                formal = callee.inputs[index]
                if not _specialization_actual_matches_formal(actual, formal):
                    del candidates[key]
                    changed = True
                    continue
                propagated = _candidate_actual_from_operand(
                    actual,
                    caller_name=caller_name,
                    caller_modules=modules_by_name,
                    candidates=candidates,
                    global_symbol_names=global_symbol_names,
                )
                if (
                    propagated is None
                    or not _specialization_actual_matches_formal(propagated, formal)
                    or _graph_operand_key(propagated) != _graph_operand_key(expected)
                ):
                    del candidates[key]
                    changed = True

    substs: dict[str, dict[str, GraphOperand]] = {}
    for module_name in component:
        module = modules_by_name[module_name]
        module_subst = {
            formal.name: candidates[(module_name, index)]
            for index, formal in enumerate(module.inputs)
            if (module_name, index) in candidates
        }
        if module_subst:
            substs[module_name] = module_subst
    return substs


def _specialize_recursive_sccs(graph: GraphProgram, *, config: GraphOptimizeConfig) -> GraphProgram:
    if config.specialize_definitions == "off":
        return graph
    modules_by_name = {module.name: module for module in graph.modules}
    edges = _graph_call_graph(graph)
    global_symbol_names = {
        module.name
        for module in graph.modules
        if _is_global_symbol_module(module)
    }
    used_module_names = {module.name for module in graph.modules}
    clone_index = 0
    cloned_modules: list[GraphModule] = []
    external_replacements: dict[tuple[str, str], tuple[str, GraphModule, set[str]]] = {}

    for component in _strongly_connected_components(edges):
        if graph.main_module in component:
            continue
        recursive = len(component) > 1 or any(name in edges.get(name, ()) for name in component)
        if not recursive:
            continue
        substs = _recursive_scc_specialization_substs(
            graph,
            component,
            global_symbol_names=global_symbol_names,
        )
        if not substs:
            continue
        clone_names: dict[str, str] = {}
        for module_name in sorted(component):
            if module_name not in substs:
                continue
            while True:
                clone_index += 1
                clone_name = f"{module_name}__spec_{clone_index}"
                if clone_name not in used_module_names:
                    used_module_names.add(clone_name)
                    clone_names[module_name] = clone_name
                    break
        if not clone_names:
            continue
        specialized_by_original: dict[str, GraphModule] = {}
        for module_name, clone_name in clone_names.items():
            module = modules_by_name[module_name]
            fake_inputs = tuple(
                substs[module_name].get(
                    formal.name,
                    GraphValueRef(formal.name, formal.type_expr, formal.dims),
                )
                for formal in module.inputs
            )
            fake_call = GraphNode(
                id=f"{module_name}:recursive_specialization",
                op=GraphOp(module_name),
                inputs=fake_inputs,
                attrs={},
                outputs=(),
                source_module=module_name,
                type_expr=module.return_type_expr or TypeAny(),
            )
            specialized = _specialized_module(
                module,
                name=clone_name,
                call_node=fake_call,
                subst_override=substs[module_name],
                global_symbol_names=global_symbol_names,
            )
            if specialized is None:
                specialized_by_original.clear()
                break
            specialized_by_original[module_name] = specialized
        if not specialized_by_original:
            continue
        subst_names_by_module = {name: set(subst) for name, subst in substs.items()}
        rewritten_specialized: list[GraphModule] = []
        for module_name, specialized in specialized_by_original.items():
            rewritten_nodes = tuple(
                _rewrite_recursive_specialized_node(
                    node,
                    originals_by_name=modules_by_name,
                    clone_names=clone_names,
                    subst_names_by_module=subst_names_by_module,
                )
                for node in specialized.nodes
            )
            rewritten_outputs = tuple(
                _rewrite_recursive_specialized_operand(
                    output,
                    originals_by_name=modules_by_name,
                    clone_names=clone_names,
                    subst_names_by_module=subst_names_by_module,
                )
                for output in specialized.outputs
            )
            rewritten_specialized.append(
                replace(
                    specialized,
                    nodes=rewritten_nodes,
                    outputs=rewritten_outputs,
                )
            )
        cloned_modules.extend(rewritten_specialized)
        for caller_name, node in _external_top_level_scc_calls(graph, component):
            clone_name = clone_names.get(node.op.name)
            if clone_name is None:
                continue
            external_replacements[(caller_name, node.id)] = (
                clone_name,
                modules_by_name[node.op.name],
                subst_names_by_module[node.op.name],
            )

    if not cloned_modules:
        return graph
    rewritten_modules: list[GraphModule] = []
    for module in graph.modules:
        nodes: list[GraphNode] = []
        for node in module.nodes:
            replacement_info = external_replacements.get((module.name, node.id))
            if replacement_info is None:
                nodes.append(node)
                continue
            clone_name, original, subst_names = replacement_info
            nodes.append(
                _rewrite_call_to_specialized_with_subst(
                    node,
                    original,
                    clone_name,
                    subst_names,
                )
            )
        rewritten_modules.append(replace(module, nodes=tuple(nodes)))
    specialized_graph = replace(graph, modules=tuple((*rewritten_modules, *cloned_modules)))
    specialized_graph = _refresh_graph_program_types(specialized_graph)
    module_effects = infer_graph_module_effects(specialized_graph.modules)
    specialized_modules_by_name = {module.name: module for module in specialized_graph.modules}
    global_literals = _atomic_literal_constants(specialized_graph)
    global_dim_values = _atomic_int_constant_dims(specialized_graph)
    specialized_graph = replace(
        specialized_graph,
        modules=tuple(
            _optimize_module_local(
                module,
                config=config,
                module_effects=module_effects,
                modules_by_name=specialized_modules_by_name,
                global_dim_values=global_dim_values,
                global_literals=global_literals,
            )
            for module in specialized_graph.modules
        ),
    )
    specialized_graph = _sanitize_graph_constraints(specialized_graph)
    _validate_optimizer_graph(specialized_graph, phase="recursive_specialize")
    return specialized_graph


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
    top_level_calls = _top_level_calls_by_callee(graph)
    all_calls = _all_calls_by_callee(graph)
    recursive = _recursive_modules(graph)
    global_symbol_names = {
        module.name
        for module in graph.modules
        if _is_global_symbol_module(module)
    }
    domain_analysis = infer_main_module_domain_facts(graph)
    replacements: dict[tuple[str, str], str] = {}
    replacement_subst_names: dict[tuple[str, str], set[str] | None] = {}
    nested_replacements: dict[str, tuple[str, GraphModule, set[str]]] = {}
    new_modules: list[GraphModule] = list(graph.modules)
    used_module_names = {module.name for module in graph.modules}
    clone_index = 0
    for callee in graph.modules:
        if not _can_specialize_module(
            callee,
            recursive_modules=recursive,
            main_module=graph.main_module,
        ):
            continue
        calls = all_calls.get(callee.name, [])
        if len(calls) < 2:
            continue
        subst = _shared_constant_specialization_subst(
            callee,
            calls,
            global_symbol_names=global_symbol_names,
            caller_modules=modules_by_name,
            input_domain_facts=domain_analysis.module_input_facts.get(callee.name),
        )
        if not subst:
            continue
        while True:
            clone_index += 1
            clone_name = f"{callee.name}__spec_{clone_index}"
            if clone_name not in used_module_names:
                used_module_names.add(clone_name)
                break
        representative = calls[0][1]
        specialized = _specialized_module(
            callee,
            name=clone_name,
            call_node=representative,
            subst_override=subst,
            global_symbol_names=global_symbol_names,
        )
        if specialized is None:
            continue
        if not _specialized_candidate_valid(
            graph,
            tuple((*new_modules, specialized)),
            phase="specialize.candidate.shared",
        ):
            continue
        new_modules.append(specialized)
        subst_names = set(subst)
        for caller_name, node in calls:
            if isinstance(node, GraphNode):
                replacements[(caller_name, node.id)] = clone_name
                replacement_subst_names[(caller_name, node.id)] = subst_names
            else:
                nested_replacements[callee.name] = (clone_name, callee, subst_names)
    for caller in graph.modules:
        for node in caller.nodes:
            if (caller.name, node.id) in replacements:
                continue
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
            subst = _callsite_specialization_subst(
                callee,
                node.inputs,
                global_symbol_names=global_symbol_names,
                caller_name=caller.name,
                caller_modules=modules_by_name,
                input_domain_facts=domain_analysis.module_input_facts.get(callee.name),
            )
            if not subst:
                continue
            if not _has_safe_specialization_actual(
                node,
                callee,
                global_symbol_names=global_symbol_names,
                input_domain_facts=domain_analysis.module_input_facts.get(callee.name),
            ):
                continue
            while True:
                clone_index += 1
                clone_name = f"{callee.name}__spec_{clone_index}"
                if clone_name not in used_module_names:
                    used_module_names.add(clone_name)
                    break
            specialized = _specialized_module(
                callee,
                name=clone_name,
                call_node=node,
                subst_override=subst,
                global_symbol_names=global_symbol_names,
            )
            if specialized is None:
                continue
            if not _specialized_candidate_valid(
                graph,
                tuple((*new_modules, specialized)),
                phase="specialize.candidate.callsite",
            ):
                continue
            replacements[(caller.name, node.id)] = clone_name
            replacement_subst_names[(caller.name, node.id)] = set(subst)
            new_modules.append(specialized)
    for caller in graph.modules:
        for node in caller.nodes:
            for operand in (*node.inputs, *node.attrs.values()):
                for callee_name in sorted(_operand_called_module_names(operand, set(modules_by_name))):
                    if callee_name in nested_replacements:
                        continue
                    callee = modules_by_name[callee_name]
                    if not _can_specialize_module(
                        callee,
                        recursive_modules=recursive,
                        main_module=graph.main_module,
                    ):
                        continue
                    if counts[callee.name] != 1:
                        continue
                    expr = _find_operand_call(operand, callee.name)
                    if expr is None or expr.attrs:
                        continue
                    subst = _callsite_specialization_subst(
                        callee,
                        expr.inputs,
                        global_symbol_names=global_symbol_names,
                        caller_name=caller.name,
                        caller_modules=modules_by_name,
                        input_domain_facts=domain_analysis.module_input_facts.get(callee.name),
                    )
                    if not subst:
                        continue
                    while True:
                        clone_index += 1
                        clone_name = f"{callee.name}__spec_{clone_index}"
                        if clone_name not in used_module_names:
                            used_module_names.add(clone_name)
                            break
                    fake_call = GraphNode(
                        id=f"{caller.name}:nested_specialize:{callee.name}",
                        op=GraphOp(callee.name),
                        inputs=expr.inputs,
                        attrs={},
                        outputs=(),
                        source_module=caller.name,
                        type_expr=expr.type_expr,
                        dims=expr.dims,
                    )
                    specialized = _specialized_module(
                        callee,
                        name=clone_name,
                        call_node=fake_call,
                        subst_override=subst,
                        global_symbol_names=global_symbol_names,
                    )
                    if specialized is None:
                        continue
                    if not _specialized_candidate_valid(
                        graph,
                        tuple((*new_modules, specialized)),
                        phase="specialize.candidate.nested",
                    ):
                        continue
                    nested_replacements[callee.name] = (clone_name, callee, set(subst))
                    new_modules.append(specialized)
        for output in caller.outputs:
            for callee_name in sorted(_operand_called_module_names(output, set(modules_by_name))):
                if callee_name in nested_replacements:
                    continue
                callee = modules_by_name[callee_name]
                if not _can_specialize_module(
                    callee,
                    recursive_modules=recursive,
                    main_module=graph.main_module,
                ):
                    continue
                if counts[callee.name] != 1:
                    continue
                expr = _find_operand_call(output, callee.name)
                if expr is None or expr.attrs:
                    continue
                subst = _callsite_specialization_subst(
                    callee,
                    expr.inputs,
                    global_symbol_names=global_symbol_names,
                    caller_name=caller.name,
                    caller_modules=modules_by_name,
                )
                if not subst:
                    continue
                while True:
                    clone_index += 1
                    clone_name = f"{callee.name}__spec_{clone_index}"
                    if clone_name not in used_module_names:
                        used_module_names.add(clone_name)
                        break
                fake_call = GraphNode(
                    id=f"{caller.name}:nested_specialize:{callee.name}",
                    op=GraphOp(callee.name),
                    inputs=expr.inputs,
                    attrs={},
                    outputs=(),
                    source_module=caller.name,
                    type_expr=expr.type_expr,
                    dims=expr.dims,
                )
                specialized = _specialized_module(
                    callee,
                    name=clone_name,
                    call_node=fake_call,
                    subst_override=subst,
                    global_symbol_names=global_symbol_names,
                )
                if specialized is None:
                    continue
                if not _specialized_candidate_valid(
                    graph,
                    tuple((*new_modules, specialized)),
                    phase="specialize.candidate.output",
                ):
                    continue
                nested_replacements[callee.name] = (clone_name, callee, set(subst))
                new_modules.append(specialized)
    if not replacements:
        if not nested_replacements:
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
                nodes.append(
                    _rewrite_call_to_specialized_with_subst(
                        node,
                        original,
                        clone_name,
                        replacement_subst_names.get((module.name, node.id)),
                    )
                )
            else:
                nodes.append(
                    replace(
                        node,
                        inputs=tuple(
                            _rewrite_specialized_nested_operand(
                                item,
                                replacements=nested_replacements,
                            )
                            for item in node.inputs
                        ),
                        attrs={
                            key: _rewrite_specialized_nested_operand(
                                value,
                                replacements=nested_replacements,
                            )
                            for key, value in node.attrs.items()
                        },
                    )
                )
        rewritten_modules.append(
            replace(
                module,
                nodes=tuple(nodes),
                outputs=tuple(
                    _rewrite_specialized_nested_operand(
                        output,
                        replacements=nested_replacements,
                    )
                    for output in module.outputs
                ),
            )
        )
    specialized_graph = replace(graph, modules=tuple(rewritten_modules))
    module_effects = infer_graph_module_effects(specialized_graph.modules)
    specialized_modules_by_name = {module.name: module for module in specialized_graph.modules}
    global_literals = _atomic_literal_constants(specialized_graph)
    global_dim_values = _atomic_int_constant_dims(specialized_graph)
    specialized_graph = replace(
        specialized_graph,
        modules=tuple(
            _optimize_module_local(
                module,
                config=config,
                module_effects=module_effects,
                modules_by_name=specialized_modules_by_name,
                global_dim_values=global_dim_values,
                global_literals=global_literals,
            )
            for module in specialized_graph.modules
        ),
    )
    specialized_graph = _refresh_graph_program_types(specialized_graph)
    specialized_graph = _sanitize_graph_constraints(specialized_graph)
    try:
        _validate_optimizer_graph(specialized_graph, phase="specialize")
    except ValueError:
        return graph
    return specialized_graph


def _specialize_definitions_to_fixpoint(
    graph: GraphProgram,
    *,
    config: GraphOptimizeConfig,
) -> GraphProgram:
    current = graph
    for iteration in range(config.max_iterations):
        before = current
        current = _specialize_recursive_sccs(current, config=config)
        _validate_optimizer_graph(current, phase=f"recursive_specialize.fixpoint.{iteration}")
        current = _specialize_definitions(current, config=config)
        _validate_optimizer_graph(current, phase=f"specialize.fixpoint.{iteration}")
        module_effects = infer_graph_module_effects(current.modules)
        modules_by_name = {module.name: module for module in current.modules}
        global_literals = _atomic_literal_constants(current)
        global_dim_values = _atomic_int_constant_dims(current)
        cleanup_candidate = replace(
            current,
            modules=tuple(
                _optimize_module_local(
                    module,
                    config=config,
                    module_effects=module_effects,
                    modules_by_name=modules_by_name,
                    global_dim_values=global_dim_values,
                    global_literals=global_literals,
                )
                for module in current.modules
            ),
        )
        cleanup_candidate = _refresh_graph_program_types(cleanup_candidate)
        cleanup_candidate = _sanitize_graph_constraints(cleanup_candidate)
        try:
            _validate_optimizer_graph(cleanup_candidate, phase=f"specialize.cleanup.{iteration}")
        except ValueError:
            pass
        else:
            current = cleanup_candidate
        if current == before:
            return current
    raise RuntimeError(
        f"graph specialization did not converge after {config.max_iterations} iterations"
    )


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


def _select_branches_do_not_use_local_outputs(
    operand: GraphOperand,
    *,
    local_output_names: set[str],
) -> bool:
    if not isinstance(operand, GraphExpr):
        return True
    if operand.op.name == "core.select":
        if len(operand.inputs) != 3:
            return False
        for branch in operand.inputs[1:]:
            refs: set[str] = set()
            _operand_refs(branch, refs)
            if refs & local_output_names:
                return False
    return all(
        _select_branches_do_not_use_local_outputs(item, local_output_names=local_output_names)
        for item in operand.inputs
    ) and all(
        _select_branches_do_not_use_local_outputs(value, local_output_names=local_output_names)
        for value in operand.attrs.values()
    )


def _module_select_branches_do_not_use_local_outputs(module: GraphModule) -> bool:
    local_output_names = {
        output.name
        for node in module.nodes
        for output in node.outputs
    }
    for node in module.nodes:
        if node.op.name == "core.select":
            if len(node.inputs) != 3:
                return False
            for branch in node.inputs[1:]:
                refs: set[str] = set()
                _operand_refs(branch, refs)
                if refs & local_output_names:
                    return False
        if not all(
            _select_branches_do_not_use_local_outputs(item, local_output_names=local_output_names)
            for item in node.inputs
        ):
            return False
        if not all(
            _select_branches_do_not_use_local_outputs(value, local_output_names=local_output_names)
            for value in node.attrs.values()
        ):
            return False
    return all(
        _select_branches_do_not_use_local_outputs(output, local_output_names=local_output_names)
        for output in module.outputs
    )


def _can_inline_module(
    module: GraphModule,
    *,
    module_effects: Mapping[str, GraphEffect],
    recursive_modules: set[str],
    main_module: str,
    allow_control_select: bool,
) -> bool:
    if module.name == main_module:
        return False
    if module.name in recursive_modules:
        return False
    if _forwarding_node(module) is not None:
        return True
    forwarding_expr = _forwarding_expr(module)
    if forwarding_expr is not None:
        return not _operand_has_core_select(forwarding_expr)
    if len(module.nodes) > 1 and _module_has_tensor_values(module):
        return False
    if module.is_global_binding and not _is_atomic_constant_module(module):
        return False
    if _module_has_core_select(module):
        if not allow_control_select:
            return False
        if not _module_select_branches_do_not_use_local_outputs(module):
            return False
    return module_effects.get(module.name) == GraphEffect.TOTAL_PURE


def _module_uses_runtime_shape_queries(module: GraphModule) -> bool:
    for node in module.nodes:
        if node.op.name in {"_shape", "Tensor.size"}:
            return True
        for operand in (*node.inputs, *node.attrs.values()):
            if _operand_uses_runtime_shape_queries(operand):
                return True
    for output in module.outputs:
        if _operand_uses_runtime_shape_queries(output):
            return True
    return False


def _module_has_tensor_values(module: GraphModule) -> bool:
    def has_tensor_type(type_expr: TypeExpr | None) -> bool:
        if isinstance(type_expr, TypeTensor):
            return True
        if isinstance(type_expr, TypeOptional):
            return has_tensor_type(type_expr.inner)
        if isinstance(type_expr, TypeList):
            return has_tensor_type(type_expr.item)
        if isinstance(type_expr, TypeTuple):
            return any(has_tensor_type(item) for item in type_expr.items)
        return False

    if any(has_tensor_type(value.type_expr) for value in module.inputs):
        return True
    if has_tensor_type(module.return_type_expr):
        return True
    for node in module.nodes:
        if has_tensor_type(node.type_expr):
            return True
        if any(has_tensor_type(value.type_expr) for value in node.outputs):
            return True
        if any(has_tensor_type(graph_operand_type(operand)) for operand in (*node.inputs, *node.attrs.values())):
            return True
    return any(has_tensor_type(graph_operand_type(output)) for output in module.outputs)


def _operand_uses_runtime_shape_queries(operand: GraphOperand) -> bool:
    if not isinstance(operand, GraphExpr):
        return False
    if operand.op.name in {"_shape", "Tensor.size"}:
        return True
    return any(_operand_uses_runtime_shape_queries(item) for item in operand.inputs) or any(
        _operand_uses_runtime_shape_queries(item) for item in operand.attrs.values()
    )


def _is_atomic_constant_module(module: GraphModule) -> bool:
    return (
        not module.inputs
        and not module.nodes
        and len(module.outputs) == 1
        and _is_atomic_operand(module.outputs[0])
    )


def _is_global_symbol_module(module: GraphModule) -> bool:
    return not module.inputs and len(module.outputs) == 1


def _promote_total_pure_zero_arg_modules_to_globals(
    graph: GraphProgram,
    *,
    module_effects: Mapping[str, GraphEffect],
) -> GraphProgram:
    modules: list[GraphModule] = []
    changed = False
    for module in graph.modules:
        should_promote = (
            module.name != graph.main_module
            and not module.is_global_binding
            and not module.inputs
            and len(module.outputs) == 1
            and module_effects.get(module.name) == GraphEffect.TOTAL_PURE
        )
        if should_promote:
            modules.append(replace(module, is_global_binding=True))
            changed = True
        else:
            modules.append(module)
    if not changed:
        return graph
    promoted = replace(graph, modules=tuple(modules))
    _validate_optimizer_graph(promoted, phase="promote_zero_arg_globals")
    return promoted


def _can_inline_call_node(node: GraphNode, callee: GraphModule) -> bool:
    if len(node.inputs) != len(callee.inputs):
        return False
    if len(node.outputs) != len(callee.outputs):
        return False
    if node.attrs:
        return False
    for actual, formal in zip(node.inputs, callee.inputs, strict=True):
        if formal.optional and isinstance(graph_operand_type(actual), TypeNull):
            continue
        if not graph_type_compatible(graph_operand_type(actual), formal.type_expr):
            return False
    expected_output_types = _instantiate_call_output_types(
        callee,
        node.inputs,
        len(node.outputs),
    )
    for call_output, expected in zip(node.outputs, expected_output_types, strict=True):
        if not (
            graph_type_compatible(expected, call_output.type_expr)
            or graph_type_compatible(call_output.type_expr, expected)
        ):
            return False
    return True


def _dim_subst_uses_caller_local_values(
    callee: GraphModule,
    actuals: tuple[GraphOperand, ...],
    *,
    caller_local_values: set[str],
) -> bool:
    dim_subst = _call_dim_subst(callee, actuals)
    for dim in dim_subst.values():
        if dim_token_names(dim) & caller_local_values:
            return True
    return False


def _can_inline_forwarded_call_node(node: GraphNode, callee: GraphModule, inner: GraphNode) -> bool:
    if len(node.inputs) != len(callee.inputs):
        return False
    if node.attrs:
        return False
    for actual, formal in zip(node.inputs, callee.inputs, strict=True):
        if formal.optional and isinstance(graph_operand_type(actual), TypeNull):
            continue
        if not graph_type_compatible(graph_operand_type(actual), formal.type_expr):
            return False
    expected_output_types = _instantiate_call_output_types(
        callee,
        node.inputs,
        len(node.outputs),
    )
    if len(expected_output_types) == len(node.outputs) and all(
        graph_type_compatible(expected, output.type_expr)
        or graph_type_compatible(output.type_expr, expected)
        for expected, output in zip(expected_output_types, node.outputs, strict=True)
    ):
        return True
    forwarded = _rewrite_forwarded_call_node(node, callee, inner, fold=True)
    return graph_type_compatible(forwarded.type_expr, node.type_expr) or graph_type_compatible(
        node.type_expr,
        forwarded.type_expr,
    )


def _can_inline_call_expr(expr: GraphExpr, callee: GraphModule) -> bool:
    if len(expr.inputs) != len(callee.inputs):
        return False
    if len(callee.outputs) != 1:
        return False
    if expr.attrs:
        return False
    for actual, formal in zip(expr.inputs, callee.inputs, strict=True):
        if formal.optional and isinstance(graph_operand_type(actual), TypeNull):
            continue
        if not graph_type_compatible(graph_operand_type(actual), formal.type_expr):
            return False
    expected_output_types = _instantiate_call_output_types(callee, expr.inputs, 1)
    if len(expected_output_types) != 1:
        return False
    expected = expected_output_types[0]
    if not (
        graph_type_compatible(expected, expr.type_expr)
        or graph_type_compatible(expr.type_expr, expected)
    ):
        return False
    return True


def _can_inline_forwarded_call_expr(expr: GraphExpr, callee: GraphModule, inner: GraphNode) -> bool:
    if len(expr.inputs) != len(callee.inputs):
        return False
    if expr.attrs:
        return False
    for actual, formal in zip(expr.inputs, callee.inputs, strict=True):
        if formal.optional and isinstance(graph_operand_type(actual), TypeNull):
            continue
        if not graph_type_compatible(graph_operand_type(actual), formal.type_expr):
            return False
    if len(callee.outputs) == 1 and graph_type_compatible(graph_operand_type(callee.outputs[0]), expr.type_expr):
        return True
    forwarded = _rewrite_forwarded_call_expr(expr, callee, inner, fold=True)
    return graph_type_compatible(graph_operand_type(forwarded), expr.type_expr) or graph_type_compatible(
        expr.type_expr,
        graph_operand_type(forwarded),
    )


def _is_single_callsite_inline_candidate(
    module: GraphModule,
    counts: Counter[str],
    top_level_counts: Counter[str],
) -> bool:
    return (
        counts[module.name] == 1
        and top_level_counts[module.name] <= 1
        and not _is_atomic_constant_module(module)
        and (
            not module.is_global_binding
            or _forwarding_expr(module) is not None
        )
    )


def _forwarding_node(module: GraphModule) -> GraphNode | None:
    if module.is_global_binding:
        return None
    if len(module.nodes) != 1:
        return None
    inner = module.nodes[0]
    if len(module.outputs) != len(inner.outputs):
        return None
    for returned, output in zip(module.outputs, inner.outputs, strict=True):
        if not isinstance(returned, GraphValueRef) or returned.name != output.name:
            return None
    return inner


def _forwarding_expr(module: GraphModule) -> GraphExpr | None:
    if module.is_global_binding:
        return None
    if module.nodes or len(module.outputs) != 1:
        return None
    output = module.outputs[0]
    if isinstance(output, GraphExpr):
        return output
    return None


def _rewrite_inlined_node(
    inner: GraphNode,
    *,
    module_name: str,
    node_id: str,
    renames: Mapping[str, str],
    formal_subst: Mapping[str, GraphOperand],
    dim_subst: Mapping[str, DimToken],
    fold: bool,
) -> GraphNode:
    renamed_outputs = tuple(
        replace(output, name=renames.get(output.name, output.name))
        for output in inner.outputs
    )
    rewritten = replace(
        inner,
        id=node_id,
        inputs=tuple(
            _replace_operand_refs(
                rename_operand(item, renames),
                formal_subst,
                fold=fold,
            )
            for item in inner.inputs
        ),
        attrs={
            key: _replace_operand_refs(
                rename_operand(value, renames),
                formal_subst,
                fold=fold,
            )
            for key, value in inner.attrs.items()
        },
        outputs=renamed_outputs,
        source_module=module_name,
    )
    return substitute_graph_node_dims(rewritten, dim_subst) if dim_subst else rewritten


def _rewrite_forwarded_call_node(
    node: GraphNode,
    callee: GraphModule,
    inner: GraphNode,
    *,
    fold: bool,
) -> GraphNode:
    formal_subst = {
        formal.name: actual
        for formal, actual in zip(callee.inputs, node.inputs, strict=True)
    }
    dim_subst = _call_dim_subst(callee, node.inputs)
    rewritten = replace(
        node,
        op=inner.op,
        inputs=tuple(
            _replace_operand_refs(item, formal_subst, fold=fold)
            for item in inner.inputs
        ),
        attrs={
            key: _replace_operand_refs(value, formal_subst, fold=fold)
            for key, value in inner.attrs.items()
        },
    )
    rewritten = substitute_graph_node_dims(rewritten, dim_subst) if dim_subst else rewritten
    return replace(
        rewritten,
        outputs=node.outputs,
        type_expr=node.type_expr,
        dims=node.dims,
        source_module=node.source_module,
    )


def _rewrite_forwarded_call_expr(
    expr: GraphExpr,
    callee: GraphModule,
    inner: GraphNode,
    *,
    fold: bool,
) -> GraphExpr:
    formal_subst = {
        formal.name: actual
        for formal, actual in zip(callee.inputs, expr.inputs, strict=True)
    }
    dim_subst = _call_dim_subst(callee, expr.inputs)
    rewritten = GraphExpr(
        op=inner.op,
        inputs=tuple(
            _replace_operand_refs(item, formal_subst, fold=fold)
            for item in inner.inputs
        ),
        attrs={
            key: _replace_operand_refs(value, formal_subst, fold=fold)
            for key, value in inner.attrs.items()
        },
        type_expr=expr.type_expr,
        dims=expr.dims,
    )
    rewritten = substitute_graph_operand_dims(rewritten, dim_subst) if dim_subst else rewritten
    return replace(
        rewritten,
        type_expr=expr.type_expr,
        dims=expr.dims,
    )


def _rewrite_forwarded_expr_call_node(
    node: GraphNode,
    callee: GraphModule,
    expr: GraphExpr,
    *,
    fold: bool,
) -> GraphNode:
    formal_subst = {
        formal.name: actual
        for formal, actual in zip(callee.inputs, node.inputs, strict=True)
    }
    dim_subst = _call_dim_subst(callee, node.inputs)
    rewritten_expr = _replace_operand_refs(expr, formal_subst, fold=fold)
    rewritten_expr = substitute_graph_operand_dims(rewritten_expr, dim_subst) if dim_subst else rewritten_expr
    if not isinstance(rewritten_expr, GraphExpr):
        return replace(
            node,
            op=GraphOp("core.alias"),
            inputs=(rewritten_expr,),
            attrs={},
            outputs=node.outputs,
            type_expr=node.type_expr,
            dims=node.dims,
            source_module=node.source_module,
        )
    return replace(
        node,
        op=rewritten_expr.op,
        inputs=rewritten_expr.inputs,
        attrs=rewritten_expr.attrs,
        outputs=node.outputs,
        type_expr=node.type_expr,
        dims=node.dims,
        source_module=node.source_module,
    )


def _rewrite_forwarded_expr_call_expr(
    expr: GraphExpr,
    callee: GraphModule,
    forwarded: GraphExpr,
    *,
    fold: bool,
) -> GraphExpr:
    formal_subst = {
        formal.name: actual
        for formal, actual in zip(callee.inputs, expr.inputs, strict=True)
    }
    dim_subst = _call_dim_subst(callee, expr.inputs)
    rewritten = _replace_operand_refs(forwarded, formal_subst, fold=fold)
    rewritten = substitute_graph_operand_dims(rewritten, dim_subst) if dim_subst else rewritten
    if not isinstance(rewritten, GraphExpr):
        return GraphExpr(
            op=GraphOp("core.alias"),
            inputs=(rewritten,),
            attrs={},
            type_expr=expr.type_expr,
            dims=expr.dims,
        )
    return replace(
        rewritten,
        type_expr=expr.type_expr,
        dims=expr.dims,
    )


def _rewrite_inlined_return(
    returned: GraphOperand,
    *,
    renames: Mapping[str, str],
    formal_subst: Mapping[str, GraphOperand],
    dim_subst: Mapping[str, DimToken],
    fold: bool,
) -> GraphOperand:
    rewritten = _replace_operand_refs(
        rename_operand(returned, renames),
        formal_subst,
        fold=fold,
    )
    return substitute_graph_operand_dims(rewritten, dim_subst) if dim_subst else rewritten


def _inlined_constraints(
    callee: GraphModule,
    *,
    renames: Mapping[str, str],
    formal_subst: Mapping[str, GraphOperand],
) -> tuple[Constraint, ...] | None:
    if not callee.constraints:
        return ()
    subst: dict[str, GraphOperand] = dict(formal_subst)
    subst.update(
        {
            old: GraphValueRef(name=new, type_expr=TypeAny())
            for old, new in renames.items()
            if old != new
        }
    )
    return _specialize_constraints(callee.constraints, subst)


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
            _forwarding_node(module) is not None
            or _forwarding_expr(module) is not None
            or _is_single_callsite_inline_candidate(module, counts, top_level_counts)
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
            allow_control_select=_is_single_callsite_inline_candidate(
                module,
                counts,
                top_level_counts,
            ),
        )
    }
    if not inlineable:
        return graph
    rewritten_modules: list[GraphModule] = []
    for module in graph.modules:
        nodes: list[GraphNode] = []
        subst: dict[str, GraphOperand] = {}
        constraints: list[Constraint] = list(module.constraints)
        temp_index = 0
        used_names = _module_value_names(module)
        caller_local_values = used_names - {value.name for value in module.inputs}

        def _inline_expr_call(expr: GraphExpr, *, prefix: str) -> GraphOperand:
            nonlocal temp_index
            callee = inlineable.get(expr.op.name)
            if callee is None or not _can_inline_call_expr(expr, callee):
                return expr
            if _dim_subst_uses_caller_local_values(
                callee,
                expr.inputs,
                caller_local_values=caller_local_values,
            ):
                return expr
            formal_subst = {
                formal.name: actual
                for formal, actual in zip(callee.inputs, expr.inputs, strict=True)
            }
            dim_subst = _call_dim_subst(callee, expr.inputs)
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
            inlined_constraints = _inlined_constraints(
                callee,
                renames=renames,
                formal_subst=formal_subst,
            )
            if inlined_constraints is None:
                return expr
            for inner in callee.nodes:
                nodes.append(
                    _rewrite_inlined_node(
                        inner,
                        module_name=module.name,
                        node_id=f"{module.name}:inl:output:{expr.op.name}:{inner.id}",
                        renames=renames,
                        formal_subst=formal_subst,
                        dim_subst=dim_subst,
                        fold=config.constant_folding,
                    )
                )
            constraints.extend(inlined_constraints)
            return _rewrite_inlined_return(
                callee.outputs[0],
                renames=renames,
                formal_subst=formal_subst,
                dim_subst=dim_subst,
                fold=config.constant_folding,
            )

        def _inline_nested_expr_calls(
            operand: GraphOperand,
            *,
            prefix: str,
            allow_general_inline: bool = True,
            expected_type: TypeExpr | None = None,
        ) -> GraphOperand:
            if not isinstance(operand, GraphExpr):
                return operand
            forwarding_callee = inlineable.get(operand.op.name)
            forwarding = _forwarding_node(forwarding_callee) if forwarding_callee is not None else None
            forwarded_expr = _forwarding_expr(forwarding_callee) if forwarding_callee is not None else None
            if (
                forwarding_callee is not None
                and forwarding is not None
                and _can_inline_forwarded_call_expr(operand, forwarding_callee, forwarding)
                and not _dim_subst_uses_caller_local_values(
                    forwarding_callee,
                    operand.inputs,
                    caller_local_values=caller_local_values,
                )
            ):
                forwarded = _rewrite_forwarded_call_expr(
                    operand,
                    forwarding_callee,
                    forwarding,
                    fold=config.constant_folding,
                )
                if allow_general_inline or graph_type_compatible(
                    graph_operand_type(forwarded),
                    expected_type or operand.type_expr,
                ):
                    return forwarded
            if (
                forwarding_callee is not None
                and forwarded_expr is not None
                and _can_inline_call_expr(operand, forwarding_callee)
                and not _dim_subst_uses_caller_local_values(
                    forwarding_callee,
                    operand.inputs,
                    caller_local_values=caller_local_values,
                )
            ):
                forwarded = _rewrite_forwarded_expr_call_expr(
                    operand,
                    forwarding_callee,
                    forwarded_expr,
                    fold=config.constant_folding,
                )
                if allow_general_inline or graph_type_compatible(
                    graph_operand_type(forwarded),
                    expected_type or operand.type_expr,
                ):
                    return forwarded
            if operand.op.name == "core.select" and len(operand.inputs) == 3 and not operand.attrs:
                return replace(
                    operand,
                    inputs=(
                        _inline_nested_expr_calls(
                            operand.inputs[0],
                            prefix=f"{prefix}_cond",
                            allow_general_inline=allow_general_inline,
                        ),
                        _inline_nested_expr_calls(
                            operand.inputs[1],
                            prefix=f"{prefix}_then",
                            allow_general_inline=False,
                            expected_type=operand.type_expr,
                        ),
                        _inline_nested_expr_calls(
                            operand.inputs[2],
                            prefix=f"{prefix}_else",
                            allow_general_inline=False,
                            expected_type=operand.type_expr,
                        ),
                    ),
                )
            rewritten = replace(
                operand,
                inputs=tuple(
                    _inline_nested_expr_calls(
                        item,
                        prefix=f"{prefix}_arg{index + 1}",
                        allow_general_inline=allow_general_inline,
                    )
                    for index, item in enumerate(operand.inputs)
                ),
                attrs={
                    key: _inline_nested_expr_calls(
                        value,
                        prefix=f"{prefix}_{key}",
                        allow_general_inline=allow_general_inline,
                    )
                    for key, value in operand.attrs.items()
                },
            )
            if not allow_general_inline:
                return rewritten
            return _inline_expr_call(rewritten, prefix=prefix)

        for node in module.nodes:
            node = _rewrite_node_operands(node, subst, fold=config.constant_folding)
            if node.op.name == "core.select" and len(node.inputs) == 3 and not node.attrs:
                node = replace(
                    node,
                    inputs=(
                        _inline_nested_expr_calls(
                            node.inputs[0],
                            prefix=f"{node.outputs[0].name}_cond",
                        ),
                        _inline_nested_expr_calls(
                            node.inputs[1],
                            prefix=f"{node.outputs[0].name}_then",
                            allow_general_inline=False,
                            expected_type=node.type_expr,
                        ),
                        _inline_nested_expr_calls(
                            node.inputs[2],
                            prefix=f"{node.outputs[0].name}_else",
                            allow_general_inline=False,
                            expected_type=node.type_expr,
                        ),
                    ),
                )
            else:
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
            forwarding = _forwarding_node(callee)
            unsafe_local_dim_subst = _dim_subst_uses_caller_local_values(
                callee,
                node.inputs,
                caller_local_values=caller_local_values,
            )
            if (
                forwarding is not None
                and not unsafe_local_dim_subst
                and _can_inline_forwarded_call_node(node, callee, forwarding)
            ):
                inlined_constraints = _inlined_constraints(
                    callee,
                    renames={
                        output.name: call_output.name
                        for output, call_output in zip(forwarding.outputs, node.outputs, strict=False)
                    },
                    formal_subst={
                        formal.name: actual
                        for formal, actual in zip(callee.inputs, node.inputs, strict=True)
                    },
                )
                if inlined_constraints is not None:
                    nodes.append(
                        _rewrite_forwarded_call_node(
                            node,
                            callee,
                            forwarding,
                            fold=config.constant_folding,
                        )
                    )
                    constraints.extend(inlined_constraints)
                    continue
            forwarded_expr = _forwarding_expr(callee)
            if (
                forwarded_expr is not None
                and not unsafe_local_dim_subst
                and _can_inline_call_node(node, callee)
            ):
                inlined_constraints = _inlined_constraints(
                    callee,
                    renames={},
                    formal_subst={
                        formal.name: actual
                        for formal, actual in zip(callee.inputs, node.inputs, strict=True)
                    },
                )
                if inlined_constraints is not None:
                    nodes.append(
                        _rewrite_forwarded_expr_call_node(
                            node,
                            callee,
                            forwarded_expr,
                            fold=config.constant_folding,
                        )
                    )
                    constraints.extend(inlined_constraints)
                    continue
            if not _can_inline_call_node(node, callee):
                nodes.append(node)
                continue
            if unsafe_local_dim_subst:
                nodes.append(node)
                continue
            formal_subst = {formal.name: actual for formal, actual in zip(callee.inputs, node.inputs, strict=True)}
            dim_subst = _call_dim_subst(callee, node.inputs)
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
            inlined_constraints = _inlined_constraints(
                callee,
                renames=renames,
                formal_subst=formal_subst,
            )
            if inlined_constraints is None:
                nodes.append(node)
                continue
            for inner in callee.nodes:
                nodes.append(
                    _rewrite_inlined_node(
                        inner,
                        module_name=module.name,
                        node_id=f"{module.name}:inl:{node.id}:{inner.id}",
                        renames=renames,
                        formal_subst=formal_subst,
                        dim_subst=dim_subst,
                        fold=config.constant_folding,
                    )
                )
            constraints.extend(inlined_constraints)
            for output, returned in zip(node.outputs, callee.outputs, strict=True):
                subst[output.name] = _rewrite_inlined_return(
                    returned,
                    renames=renames,
                    formal_subst=formal_subst,
                    dim_subst=dim_subst,
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
        rewritten_modules.append(
            replace(
                module,
                nodes=tuple(nodes),
                outputs=outputs,
                constraints=tuple(constraints),
            )
        )
    inlined = replace(graph, modules=tuple(rewritten_modules))
    inlined = _refresh_graph_program_types(inlined)
    inlined = _sanitize_graph_constraints(inlined)
    try:
        _validate_optimizer_graph(inlined, phase="inline.candidate")
    except ValueError:
        accepted = list(graph.modules)
        changed = False
        for index, (original_module, rewritten_module) in enumerate(
            zip(graph.modules, rewritten_modules, strict=True)
        ):
            if rewritten_module == original_module:
                continue
            candidate_modules = list(accepted)
            candidate_modules[index] = rewritten_module
            candidate = replace(graph, modules=tuple(candidate_modules))
            candidate = _refresh_graph_program_types(candidate)
            candidate = _sanitize_graph_constraints(candidate)
            try:
                _validate_optimizer_graph(candidate, phase="inline.candidate.module")
            except ValueError:
                continue
            accepted = list(candidate.modules)
            changed = True
        if not changed:
            return graph
        inlined = replace(graph, modules=tuple(accepted))
    inlined = prune_graph_to_main(inlined)
    inlined = _sanitize_graph_constraints(inlined)
    _validate_optimizer_graph(inlined, phase="inline")
    return inlined


def optimize_graph_program(
    graph: GraphProgram,
    *,
    config: GraphOptimizeConfig | None = None,
) -> GraphProgram:
    config = config or GraphOptimizeConfig()
    graph = _sanitize_graph_constraints(graph)
    _validate_optimizer_graph(graph, phase="input")
    current = prune_graph_to_main(graph) if config.prune_to_main else graph
    _validate_optimizer_graph(current, phase="initial_prune" if config.prune_to_main else "initial")
    for _ in range(config.max_iterations):
        before = current
        if config.constant_dim_substitution:
            current = _substitute_atomic_constant_dims_local(current)
            _validate_optimizer_graph(current, phase="constant_dim_substitution")
        if config.constant_folding:
            current = _simplify_symbolic_graph_dims(current)
        module_effects = infer_graph_module_effects(current.modules)
        current = _promote_total_pure_zero_arg_modules_to_globals(
            current,
            module_effects=module_effects,
        )
        _validate_optimizer_graph(current, phase="promote_zero_arg_globals")
        module_effects = infer_graph_module_effects(current.modules)
        modules_by_name = {module.name: module for module in current.modules}
        domain_analysis = infer_main_module_domain_facts(current)
        global_literals = _atomic_literal_constants(current)
        global_dim_values = _atomic_int_constant_dims(current)
        cleanup_candidate = replace(
            current,
            modules=tuple(
                _optimize_module_local(
                    module,
                    config=config,
                    module_effects=module_effects,
                    modules_by_name=modules_by_name,
                    local_domain_facts=domain_analysis.module_local_facts.get(module.name),
                    global_dim_values=global_dim_values,
                    global_literals=global_literals,
                )
                for module in current.modules
            ),
        )
        cleanup_candidate = _refresh_graph_program_types(cleanup_candidate)
        cleanup_candidate = _sanitize_graph_constraints(cleanup_candidate)
        try:
            _validate_optimizer_graph(cleanup_candidate, phase="local_cleanup")
        except ValueError:
            pass
        else:
            current = cleanup_candidate
        if config.common_subexpression_elimination:
            module_effects = infer_graph_module_effects(current.modules)
            candidate = replace(
                current,
                modules=tuple(
                    _common_subexpression_eliminate_module(
                        module,
                        module_effects=module_effects,
                        fold=config.constant_folding,
                    )
                    for module in current.modules
                ),
            )
            candidate = _refresh_graph_program_types(candidate)
            candidate = _sanitize_graph_constraints(candidate)
            try:
                _validate_optimizer_graph(candidate, phase="common_subexpression_elimination")
            except ValueError:
                pass
            else:
                current = candidate
        current = _specialize_definitions_to_fixpoint(current, config=config)
        _validate_optimizer_graph(current, phase="specialize")
        if config.inline_safe:
            current = _inline_safe_modules(current, config=config)
            _validate_optimizer_graph(current, phase="inline")
            if config.common_subexpression_elimination:
                module_effects = infer_graph_module_effects(current.modules)
                candidate = replace(
                    current,
                    modules=tuple(
                        _common_subexpression_eliminate_module(
                            module,
                            module_effects=module_effects,
                            fold=config.constant_folding,
                        )
                        for module in current.modules
                    ),
                )
                candidate = _refresh_graph_program_types(candidate)
                candidate = _sanitize_graph_constraints(candidate)
                try:
                    _validate_optimizer_graph(candidate, phase="post_inline_cse")
                except ValueError:
                    pass
                else:
                    current = candidate
        if config.constant_folding:
            module_effects = infer_graph_module_effects(current.modules)
            hoist_candidate = _hoist_eager_nested_exprs(current, module_effects=module_effects)
            hoist_candidate = _refresh_graph_program_types(hoist_candidate)
            hoist_candidate = _sanitize_graph_constraints(hoist_candidate)
            try:
                _validate_optimizer_graph(hoist_candidate, phase="hoist_eager_nested_exprs")
            except ValueError:
                pass
            else:
                current = hoist_candidate
        if config.prune_to_main:
            current = prune_graph_to_main(current)
            _validate_optimizer_graph(current, phase="prune")
        current = _canonicalize_generated_module_names(current)
        _validate_optimizer_graph(current, phase="canonicalize_module_names")
        current = _canonicalize_generated_value_names(current)
        _validate_optimizer_graph(current, phase="canonicalize_value_names")
        _validate_optimizer_graph(current, phase="iteration")
        if current == before:
            return current
    raise RuntimeError(
        f"graph optimizer did not converge after {config.max_iterations} iterations"
    )


__all__ = ["GraphOptimizeConfig", "optimize_graph_program", "prune_graph_to_main"]
