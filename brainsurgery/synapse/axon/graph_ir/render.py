from __future__ import annotations

from dataclasses import replace

from brainsurgery.synapse.ops import get_op_parameter_names

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
    GraphNode,
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
    if attr_exprs:
        param_names = get_op_parameter_names(op_name[1:] if op_name.startswith("_") else op_name)
        if param_names is None:
            raise ValueError(
                f"cannot render graph op {op_name!r} with attrs as Axon: no parameter metadata"
            )
        positional: list[AxonExpr] = list(input_exprs)
        for name in param_names[len(input_exprs) :]:
            value = attr_exprs.pop(name, None)
            if value is not None:
                positional.append(value)
                continue
            if attr_exprs:
                break
        if attr_exprs:
            names = ", ".join(sorted(attr_exprs))
            raise ValueError(
                f"cannot render graph op {op_name!r} with non-positional attrs: {names}"
            )
        input_exprs = tuple(positional)
    return _typed(AxonExprCall(callee=op_name, args=input_exprs, kwargs={}), type_expr, dims)


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


def _count_operand_refs(operand: GraphOperand, counts: dict[str, int]) -> None:
    if isinstance(operand, GraphValueRef):
        counts[operand.name] = counts.get(operand.name, 0) + 1
        return
    if isinstance(operand, GraphExpr):
        for item in operand.inputs:
            _count_operand_refs(item, counts)
        for item in operand.attrs.values():
            _count_operand_refs(item, counts)


def _replace_operand_refs(operand: GraphOperand, subst: dict[str, GraphOperand]) -> GraphOperand:
    if isinstance(operand, GraphValueRef):
        return subst.get(operand.name, operand)
    if isinstance(operand, GraphExpr):
        return replace(
            operand,
            inputs=tuple(_replace_operand_refs(item, subst) for item in operand.inputs),
            attrs={key: _replace_operand_refs(value, subst) for key, value in operand.attrs.items()},
        )
    return operand


def _node_inline_operand(node: GraphNode) -> GraphOperand | None:
    if len(node.outputs) != 1:
        return None
    if node.op.name == "core.alias" and len(node.inputs) == 1 and not node.attrs:
        return node.inputs[0]
    if node.op.name in {"core.list", "core.tuple"} and not node.attrs:
        return GraphExpr(
            op=node.op,
            inputs=node.inputs,
            attrs={},
            type_expr=node.type_expr,
            dims=node.dims,
        )
    return None


def _is_generated_graph_name(name: str) -> bool:
    if name.startswith("__") or name.startswith("_v"):
        return True
    base, marker, suffix = name.rpartition("__g")
    return bool(base and marker and suffix.isdigit())


def _graph_name_rank(name: str) -> int:
    if not _is_generated_graph_name(name):
        return 0
    base, marker, suffix = name.rpartition("__g")
    if marker and base and suffix.isdigit() and not base.startswith(("_", "__")):
        return 1
    return 2


def _better_name(candidate: str, current: str) -> bool:
    return _graph_name_rank(candidate) < _graph_name_rank(current)


def _can_inline_returned_alias(node: GraphNode, inline_operand: GraphOperand) -> bool:
    if node.op.name != "core.alias" or len(node.outputs) != 1:
        return False
    output_name = node.outputs[0].name
    if not isinstance(inline_operand, GraphValueRef):
        return False
    return _better_name(inline_operand.name, output_name)


def _maybe_retarget_return_alias(
    node: GraphNode,
    *,
    rendered_nodes: list[GraphNode],
    ref_counts: dict[str, int],
    returned_names: set[str],
) -> bool:
    if (
        node.op.name != "core.alias"
        or len(node.outputs) != 1
        or len(node.inputs) != 1
        or node.attrs
        or not isinstance(node.inputs[0], GraphValueRef)
    ):
        return False
    output = node.outputs[0]
    source = node.inputs[0]
    if output.name not in returned_names or ref_counts.get(output.name, 0) != 1:
        return False
    if not _better_name(output.name, source.name):
        return False
    for idx in range(len(rendered_nodes) - 1, -1, -1):
        producer = rendered_nodes[idx]
        if len(producer.outputs) == 1 and producer.outputs[0].name == source.name:
            rendered_nodes[idx] = replace(
                producer,
                outputs=(replace(producer.outputs[0], name=output.name),),
            )
            return True
    return False


def _render_nodes_and_outputs(
    module: GraphModule,
) -> tuple[tuple[GraphNode, ...], tuple[GraphOperand, ...]]:
    ref_counts: dict[str, int] = {}
    for node in module.nodes:
        for item in node.inputs:
            _count_operand_refs(item, ref_counts)
        for item in node.attrs.values():
            _count_operand_refs(item, ref_counts)
    for output in module.outputs:
        _count_operand_refs(output, ref_counts)

    returned_names = {
        output.name for output in module.outputs if isinstance(output, GraphValueRef)
    }
    subst: dict[str, GraphOperand] = {}
    rendered_nodes: list[GraphNode] = []
    for node in module.nodes:
        rewritten = replace(
            node,
            inputs=tuple(_replace_operand_refs(item, subst) for item in node.inputs),
            attrs={key: _replace_operand_refs(value, subst) for key, value in node.attrs.items()},
        )
        if (
            len(rewritten.outputs) == 1
            and ref_counts.get(rewritten.outputs[0].name, 0) == 1
        ):
            inline_operand = _node_inline_operand(rewritten)
            if inline_operand is not None and (
                rewritten.outputs[0].name not in returned_names
                or _can_inline_returned_alias(rewritten, inline_operand)
            ):
                subst[rewritten.outputs[0].name] = inline_operand
                continue
        if _maybe_retarget_return_alias(
            rewritten,
            rendered_nodes=rendered_nodes,
            ref_counts=ref_counts,
            returned_names=returned_names,
        ):
            continue
        rendered_nodes.append(rewritten)

    rendered_outputs = tuple(_replace_operand_refs(output, subst) for output in module.outputs)
    return tuple(rendered_nodes), rendered_outputs


def graph_module_to_axon_definition(module: GraphModule) -> AxonDefinition:
    statements: list[AxonBind | AxonReturn] = []
    nodes, outputs = _render_nodes_and_outputs(module)
    used_names = {value.name for value in module.inputs}
    for node in nodes:
        for output in node.outputs:
            used_names.add(output.name)

    def fresh_temp(base: str) -> str:
        candidate = base
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        index = 1
        while True:
            candidate = f"{base}_{index}"
            if candidate not in used_names:
                used_names.add(candidate)
                return candidate
            index += 1

    def atomicize_operand(operand: GraphOperand, *, base: str) -> GraphOperand:
        if not isinstance(operand, GraphExpr):
            return operand
        if operand.op.name == "core.ascribe" and len(operand.inputs) == 1 and not operand.attrs:
            inner = operand.inputs[0]
            if not isinstance(inner, GraphExpr):
                return inner
        if operand.op.name == "core.select" and len(operand.inputs) == 3 and not operand.attrs:
            return replace(
                operand,
                inputs=(
                    atomicize_operand(operand.inputs[0], base=f"{base}_cond"),
                    operand.inputs[1],
                    operand.inputs[2],
                ),
            )
        inputs = tuple(
            atomicize_operand(item, base=f"{base}_arg{idx + 1}")
            for idx, item in enumerate(operand.inputs)
        )
        attrs = {
            key: atomicize_operand(value, base=f"{base}_{key}")
            for key, value in operand.attrs.items()
        }
        expr = _node_expr_to_expr(
            op_name=operand.op.name,
            inputs=inputs,
            attrs=attrs,
            type_expr=operand.type_expr,
            dims=operand.dims,
        )
        temp = fresh_temp(base)
        statements.append(AxonBind(targets=(temp,), expr=expr))
        return GraphValueRef(name=temp, type_expr=operand.type_expr, dims=operand.dims)

    for node in nodes:
        output_base = node.outputs[0].name if node.outputs else node.id.replace(":", "_")
        if node.op.name == "core.select" and len(node.inputs) == 3:
            node_inputs = (
                atomicize_operand(node.inputs[0], base=f"{output_base}_cond"),
                node.inputs[1],
                node.inputs[2],
            )
        else:
            node_inputs = tuple(
                atomicize_operand(item, base=f"{output_base}_arg{idx + 1}")
                for idx, item in enumerate(node.inputs)
            )
        node_attrs = {
            key: atomicize_operand(value, base=f"{output_base}_{key}")
            for key, value in node.attrs.items()
        }
        expr = _node_expr_to_expr(
            op_name=node.op.name,
            inputs=node_inputs,
            attrs=node_attrs,
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
    outputs = tuple(
        atomicize_operand(output, base=f"__return_{idx + 1}")
        for idx, output in enumerate(outputs)
    )
    statements.append(
        AxonReturn(values=tuple(_operand_to_expr(output) for output in outputs))
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
