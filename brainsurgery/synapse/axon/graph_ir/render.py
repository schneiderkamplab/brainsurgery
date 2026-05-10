from __future__ import annotations

from dataclasses import replace

from ..ast import (
    AxonBind,
    AxonDefinition,
    AxonExpr,
    AxonExprAscribe,
    AxonExprBinary,
    AxonExprBool,
    AxonExprCall,
    AxonExprFloat,
    AxonExprInt,
    AxonExprList,
    AxonExprName,
    AxonExprNull,
    AxonExprPath,
    AxonExprString,
    AxonExprTernary,
    AxonExprTuple,
    AxonFile,
    AxonParam,
    AxonReturn,
    DimToken,
    TypeAny,
    TypeExpr,
    TypePath,
    TypeTensor,
    TypeTuple,
)
from ..validate import validate_flat_axon_file, validate_typed_axon_file
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


def _arity(type_expr: TypeExpr) -> int:
    return len(type_expr.items) if isinstance(type_expr, TypeTuple) else 1


def _dims(type_expr: TypeExpr) -> tuple[DimToken, ...] | None:
    return tuple(type_expr.dims) if isinstance(type_expr, TypeTensor) else None


def _typed(
    expr: AxonExpr,
    type_expr: TypeExpr,
    dims: tuple[DimToken, ...] | None = None,
    *,
    arity: int | None = None,
) -> AxonExpr:
    return replace(
        expr,
        inferred_type=type_expr,
        inferred_arity=_arity(type_expr) if arity is None else arity,
        inferred_dims=dims if dims is not None else _dims(type_expr),
    )


def _literal_to_expr(literal: GraphLiteral) -> AxonExpr:
    value = literal.value
    if isinstance(value, bool):
        expr: AxonExpr = AxonExprBool(value=value)
    elif isinstance(value, int):
        expr = AxonExprInt(value=value)
    elif isinstance(value, float):
        expr = AxonExprFloat(value=value)
    elif isinstance(value, str):
        expr = AxonExprString(value=value)
    elif value is None:
        expr = AxonExprNull()
    else:
        raise TypeError(f"unsupported graph literal value: {value!r}")
    return _typed(expr, literal.type_expr)


def _path_to_expr(path: GraphPath) -> AxonExpr:
    return _typed(AxonExprPath(absolute=path.absolute, parts=path.parts), TypePath())


def _operand_to_expr(operand: GraphOperand) -> AxonExpr:
    if isinstance(operand, GraphValueRef):
        return _typed(AxonExprName(name=operand.name), operand.type_expr, operand.dims)
    if isinstance(operand, GraphLiteral):
        return _literal_to_expr(operand)
    if isinstance(operand, GraphPath):
        return _path_to_expr(operand)
    if isinstance(operand, GraphExpr):
        return _graph_expr_to_expr(operand)
    raise TypeError(f"unsupported graph operand: {type(operand).__name__}")


def _graph_expr_to_expr(expr: GraphExpr) -> AxonExpr:
    return _node_expr_to_expr(
        op_name=expr.op.name,
        inputs=expr.inputs,
        attrs=expr.attrs,
        type_expr=expr.type_expr,
        dims=expr.dims,
    )


def _node_expr_to_expr(
    *,
    op_name: str,
    inputs: tuple[GraphOperand, ...],
    attrs: dict[str, GraphOperand],
    type_expr: TypeExpr,
    dims: tuple[DimToken, ...] | None,
) -> AxonExpr:
    input_exprs = tuple(_operand_to_expr(item) for item in inputs)
    attr_exprs = {key: _operand_to_expr(value) for key, value in attrs.items()}
    if op_name.startswith("core.binary."):
        if len(input_exprs) != 2 or attr_exprs:
            raise ValueError(f"invalid graph binary expression {op_name!r}")
        return _typed(
            AxonExprBinary(op=op_name.removeprefix("core.binary."), left=input_exprs[0], right=input_exprs[1]),
            type_expr,
            dims,
        )
    if op_name == "core.select":
        if len(input_exprs) != 3 or attr_exprs:
            raise ValueError("invalid graph select expression")
        return _typed(
            AxonExprTernary(
                cond=input_exprs[0],
                true_expr=input_exprs[1],
                false_expr=input_exprs[2],
            ),
            type_expr,
            dims,
        )
    if op_name == "core.alias":
        if len(input_exprs) != 1 or attr_exprs:
            raise ValueError("invalid graph alias expression")
        return _typed(input_exprs[0], type_expr, dims)
    if op_name == "core.ascribe":
        if len(input_exprs) != 1 or attr_exprs:
            raise ValueError("invalid graph ascribe expression")
        return _typed(
            AxonExprAscribe(expr=input_exprs[0], type_expr=type_expr),
            type_expr,
            dims,
        )
    if op_name == "core.list":
        return _typed(AxonExprList(items=input_exprs), type_expr, dims)
    if op_name == "core.tuple":
        return _typed(AxonExprTuple(items=input_exprs), type_expr, dims)
    return _typed(
        AxonExprCall(callee=op_name, args=input_exprs, kwargs=attr_exprs),
        type_expr,
        dims,
    )


def _input_to_param(value: GraphValue) -> AxonParam:
    return AxonParam(
        name=value.name,
        optional=value.optional,
        type_expr=value.type_expr,
        default_expr=None,
    )


def _module_return_type(module: GraphModule) -> TypeExpr:
    if module.return_type_expr is not None:
        return module.return_type_expr
    if not module.outputs:
        return TypeAny()
    if len(module.outputs) == 1:
        return _operand_to_expr(module.outputs[0]).inferred_type or TypeAny()
    return TypeTuple(tuple(_operand_to_expr(output).inferred_type or TypeAny() for output in module.outputs))


def graph_module_to_axon_definition(module: GraphModule) -> AxonDefinition:
    statements: list[AxonBind | AxonReturn] = []
    for node in module.nodes:
        expr = _node_expr_to_expr(
            op_name=node.op.name,
            inputs=node.inputs,
            attrs=node.attrs,
            type_expr=node.type_expr,
            dims=node.dims,
        )
        expr = _typed(
            expr,
            node.type_expr,
            node.dims,
            arity=len(node.outputs) if len(node.outputs) > 1 else None,
        )
        statements.append(AxonBind(targets=tuple(output.name for output in node.outputs), expr=expr))
    statements.append(
        AxonReturn(values=tuple(_operand_to_expr(output) for output in module.outputs))
    )
    return AxonDefinition(
        name=module.name,
        path_param=None,
        params=tuple(_input_to_param(value) for value in module.inputs),
        returns=module.output_names,
        statements=tuple(statements),
        return_type_expr=_module_return_type(module),
        constraints=module.constraints,
    )


def graph_program_to_axon_file(program: GraphProgram) -> AxonFile:
    validate_graph_program(program)
    axon = AxonFile(
        modules=tuple(graph_module_to_axon_definition(module) for module in program.modules),
        imports=(),
        imported_members={},
        exports=(),
        pragmas=dict(program.pragmas),
        type_aliases={},
    )
    validate_flat_axon_file(axon, main_module=program.main_module)
    validate_typed_axon_file(axon, main_module=program.main_module)
    return axon


__all__ = ["graph_module_to_axon_definition", "graph_program_to_axon_file"]
