from __future__ import annotations

from dataclasses import replace

from brainsurgery.synapse.ops import get_op_parameter_names

from ..ast import (
    AxonBind,
    AxonDefinition,
    AxonExpr,
    AxonExprAscribe,
    AxonExprBinary,
    AxonExprBind,
    AxonExprBool,
    AxonExprCall,
    AxonExprDo,
    AxonExprFloat,
    AxonExprIf,
    AxonExprInt,
    AxonExprLambda,
    AxonExprList,
    AxonExprName,
    AxonExprNull,
    AxonExprParen,
    AxonExprPipe,
    AxonExprPath,
    AxonExprString,
    AxonExprTernary,
    AxonExprTuple,
    AxonExprTyping,
    AxonFile,
    AxonParam,
    AxonRepeat,
    AxonReturn,
    AxonYield,
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
from ..validate import validate_flat_axon_file, validate_typed_axon_file
from .core import (
    GraphExpr,
    GraphLiteral,
    GraphModule,
    GraphNode,
    GraphOp,
    GraphOperand,
    GraphPath,
    GraphProgram,
    GraphValue,
    GraphValueRef,
    graph_operand_type,
    graph_path_template_names,
    validate_graph_program,
)
from .domain import (
    GraphDomainFact,
    GraphDomainInterval,
    GraphDomainKind,
    infer_main_module_domain_facts,
)
from .effects import GraphEffect, UsageClass, graph_operand_effect, graph_operand_usage


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


def _operand_to_expr(
    operand: GraphOperand,
    *,
    local_names: set[str] | None = None,
    zero_arg_modules: set[str] | None = None,
) -> AxonExpr:
    if isinstance(operand, GraphValueRef):
        return _typed(AxonExprName(name=operand.name), operand.type_expr, operand.dims)
    if isinstance(operand, GraphLiteral):
        return _literal_to_expr(operand)
    if isinstance(operand, GraphPath):
        return _path_to_expr(operand)
    if isinstance(operand, GraphExpr):
        return _graph_expr_to_expr(
            operand,
            local_names=local_names,
            zero_arg_modules=zero_arg_modules,
        )
    raise TypeError(f"unsupported graph operand: {type(operand).__name__}")


def _graph_expr_to_expr(
    expr: GraphExpr,
    *,
    local_names: set[str] | None = None,
    zero_arg_modules: set[str] | None = None,
) -> AxonExpr:
    return _node_expr_to_expr(
        op_name=expr.op.name,
        inputs=expr.inputs,
        attrs=expr.attrs,
        type_expr=expr.type_expr,
        dims=expr.dims,
        local_names=local_names,
        zero_arg_modules=zero_arg_modules,
    )


def _node_expr_to_expr(
    *,
    op_name: str,
    inputs: tuple[GraphOperand, ...],
    attrs: dict[str, GraphOperand],
    type_expr: TypeExpr,
    dims: tuple[DimToken, ...] | None,
    local_names: set[str] | None = None,
    zero_arg_modules: set[str] | None = None,
) -> AxonExpr:
    input_exprs = tuple(
        _operand_to_expr(
            item,
            local_names=local_names,
            zero_arg_modules=zero_arg_modules,
        )
        for item in inputs
    )
    attr_exprs = {
        key: _operand_to_expr(
            value,
            local_names=local_names,
            zero_arg_modules=zero_arg_modules,
        )
        for key, value in attrs.items()
    }
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
        return module.outputs[0].type_expr if hasattr(module.outputs[0], "type_expr") else TypeAny()
    return TypeTuple(
        tuple(
            output.type_expr if hasattr(output, "type_expr") else TypeAny()
            for output in module.outputs
        )
    )


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


def _rename_render_dim_token(dim: DimToken, renames: dict[str, str]) -> DimToken:
    if isinstance(dim, str):
        return renames.get(dim, dim)
    if isinstance(dim, DimExprBinary):
        return DimExprBinary(
            op=dim.op,
            left=_rename_render_dim_token(dim.left, renames),
            right=_rename_render_dim_token(dim.right, renames),
        )
    return dim


def _rename_render_type_expr(type_expr: TypeExpr | None, renames: dict[str, str]) -> TypeExpr | None:
    if isinstance(type_expr, TypeTensor):
        return TypeTensor(
            base=type_expr.base,
            dims=tuple(_rename_render_dim_token(dim, renames) for dim in type_expr.dims),
        )
    if isinstance(type_expr, TypeNamed):
        return TypeNamed(
            name=type_expr.name,
            args=tuple(_rename_render_dim_token(dim, renames) for dim in type_expr.args),
        )
    if isinstance(type_expr, TypeOptional):
        return TypeOptional(_rename_render_type_expr(type_expr.inner, renames))
    if isinstance(type_expr, TypeList):
        return TypeList(_rename_render_type_expr(type_expr.item, renames))
    if isinstance(type_expr, TypeTuple):
        return TypeTuple(tuple(_rename_render_type_expr(item, renames) for item in type_expr.items))
    return type_expr


def _rename_render_expr_metadata(expr: AxonExpr, renames: dict[str, str]) -> AxonExpr:
    return replace(
        expr,
        inferred_type=_rename_render_type_expr(expr.inferred_type, renames),
        inferred_dims=(
            None
            if expr.inferred_dims is None
            else tuple(_rename_render_dim_token(dim, renames) for dim in expr.inferred_dims)
        ),
    )


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
    if "__inl_" in name or "___" in name:
        return True
    base, marker, suffix = name.rpartition("__g")
    return bool(base and marker and suffix.isdigit())


def _is_generated_render_temp_base(name: str) -> bool:
    return (
        _is_generated_graph_name(name)
        or "_arg" in name
        or "_cond" in name
        or name.startswith("return_")
    )


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


def _rename_render_expr(expr: AxonExpr, renames: dict[str, str]) -> AxonExpr:
    if isinstance(expr, AxonExprName):
        return _rename_render_expr_metadata(
            replace(expr, name=renames.get(expr.name, expr.name)),
            renames,
        )
    if isinstance(expr, AxonExprList):
        return _rename_render_expr_metadata(
            replace(expr, items=tuple(_rename_render_expr(item, renames) for item in expr.items)),
            renames,
        )
    if isinstance(expr, AxonExprTuple):
        return _rename_render_expr_metadata(
            replace(expr, items=tuple(_rename_render_expr(item, renames) for item in expr.items)),
            renames,
        )
    if isinstance(expr, AxonExprCall):
        return _rename_render_expr_metadata(
            replace(
                expr,
                args=tuple(_rename_render_expr(arg, renames) for arg in expr.args),
                kwargs={
                    key: _rename_render_expr(value, renames) if isinstance(value, AxonExprTyping) else value
                    for key, value in expr.kwargs.items()
                },
            ),
            renames,
        )
    if isinstance(expr, AxonExprPipe):
        return _rename_render_expr_metadata(
            replace(
                expr,
                value=_rename_render_expr(expr.value, renames),
                stages=tuple(_rename_render_expr(stage, renames) for stage in expr.stages),
            ),
            renames,
        )
    if isinstance(expr, AxonExprBind):
        body_renames = dict(renames)
        body_renames.pop(expr.var, None)
        return _rename_render_expr_metadata(
            replace(
                expr,
                value=_rename_render_expr(expr.value, renames),
                body=_rename_render_expr(expr.body, body_renames),
            ),
            renames,
        )
    if isinstance(expr, AxonExprIf):
        return _rename_render_expr_metadata(
            replace(
                expr,
                cond=_rename_render_expr(expr.cond, renames),
                true_expr=_rename_render_expr(expr.true_expr, renames),
                false_expr=_rename_render_expr(expr.false_expr, renames),
            ),
            renames,
        )
    if isinstance(expr, AxonExprTernary):
        return _rename_render_expr_metadata(
            replace(
                expr,
                cond=_rename_render_expr(expr.cond, renames),
                true_expr=_rename_render_expr(expr.true_expr, renames),
                false_expr=_rename_render_expr(expr.false_expr, renames),
            ),
            renames,
        )
    if isinstance(expr, AxonExprBinary):
        return _rename_render_expr_metadata(
            replace(
                expr,
                left=_rename_render_expr(expr.left, renames),
                right=_rename_render_expr(expr.right, renames),
            ),
            renames,
        )
    if isinstance(expr, AxonExprLambda):
        body_renames = dict(renames)
        body_renames.pop(expr.var, None)
        return _rename_render_expr_metadata(
            replace(expr, body=_rename_render_expr(expr.body, body_renames)),
            renames,
        )
    if isinstance(expr, AxonExprParen):
        return _rename_render_expr_metadata(
            replace(expr, inner=_rename_render_expr(expr.inner, renames)),
            renames,
        )
    if isinstance(expr, AxonExprAscribe):
        return _rename_render_expr_metadata(
            replace(
                expr,
                expr=_rename_render_expr(expr.expr, renames),
                type_expr=_rename_render_type_expr(expr.type_expr, renames),
            ),
            renames,
        )
    if isinstance(expr, AxonExprDo):
        return _rename_render_expr_metadata(
            replace(expr, body=_canonicalize_render_value_names(expr.body)),
            renames,
        )
    return _rename_render_expr_metadata(expr, renames)


def _rename_render_stmt(
    stmt: AxonBind | AxonRepeat | AxonReturn | AxonYield,
    renames: dict[str, str],
) -> AxonBind | AxonRepeat | AxonReturn | AxonYield:
    if isinstance(stmt, AxonBind):
        return replace(stmt, expr=_rename_render_expr(stmt.expr, renames))
    if isinstance(stmt, AxonRepeat):
        return replace(
            stmt,
            from_expr=_rename_render_expr(stmt.from_expr, renames),
            to_expr=_rename_render_expr(stmt.to_expr, renames),
            step_expr=_rename_render_expr(stmt.step_expr, renames),
            body=tuple(_rename_render_stmt(item, renames) for item in stmt.body),
            targets=tuple(renames.get(target, target) for target in stmt.targets)
            if stmt.targets is not None
            else None,
            carry=tuple(renames.get(name, name) for name in stmt.carry)
            if stmt.carry is not None
            else None,
        )
    return replace(
        stmt,
        values=tuple(_rename_render_expr(value, renames) for value in stmt.values),
    )


def _canonicalize_render_value_names(
    statements: tuple[AxonBind | AxonRepeat | AxonReturn | AxonYield, ...],
) -> tuple[AxonBind | AxonRepeat | AxonReturn | AxonYield, ...]:
    used: set[str] = set()
    for stmt in statements:
        if isinstance(stmt, AxonBind):
            for target in stmt.targets:
                if not _is_generated_render_temp_base(target):
                    used.add(target)
    renames: dict[str, str] = {}
    next_index = 1
    rewritten: list[AxonBind | AxonRepeat | AxonReturn | AxonYield] = []
    for stmt in statements:
        if isinstance(stmt, AxonBind):
            expr = _rename_render_expr(stmt.expr, renames)
            targets: list[str] = []
            for target in stmt.targets:
                if _is_generated_render_temp_base(target):
                    while True:
                        candidate = f"_v{next_index}"
                        next_index += 1
                        if candidate not in used:
                            break
                    used.add(candidate)
                    renames[target] = candidate
                    targets.append(candidate)
                else:
                    used.add(target)
                    targets.append(target)
            rewritten.append(replace(stmt, targets=tuple(targets), expr=expr))
        elif isinstance(stmt, AxonRepeat):
            repeat_renames = dict(renames)
            if stmt.targets is not None:
                for target in stmt.targets:
                    if _is_generated_render_temp_base(target):
                        while True:
                            candidate = f"_v{next_index}"
                            next_index += 1
                            if candidate not in used:
                                break
                        used.add(candidate)
                        repeat_renames[target] = candidate
                    else:
                        used.add(target)
            rewritten.append(
                replace(
                    stmt,
                    from_expr=_rename_render_expr(stmt.from_expr, renames),
                    to_expr=_rename_render_expr(stmt.to_expr, renames),
                    step_expr=_rename_render_expr(stmt.step_expr, renames),
                    body=_canonicalize_render_value_names(
                        tuple(_rename_render_stmt(item, repeat_renames) for item in stmt.body)
                    ),
                    targets=tuple(repeat_renames.get(target, target) for target in stmt.targets)
                    if stmt.targets is not None
                    else None,
                    carry=tuple(repeat_renames.get(name, name) for name in stmt.carry)
                    if stmt.carry is not None
                    else None,
                )
            )
            renames.update(repeat_renames)
        else:
            rewritten.append(
                replace(
                    stmt,
                    values=tuple(_rename_render_expr(value, renames) for value in stmt.values),
                )
            )
    return tuple(rewritten)


def graph_module_to_axon_definition(
    module: GraphModule,
    *,
    zero_arg_modules: set[str] | None = None,
    global_value_modules: set[str] | None = None,
) -> AxonDefinition:
    statements: list[AxonBind | AxonReturn] = []
    nodes, outputs = _render_nodes_and_outputs(module)
    input_names = {value.name for value in module.inputs}
    used_names = set(input_names)
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

    def fresh_generated_temp() -> str:
        index = 1
        while True:
            candidate = f"_v{index}"
            if candidate not in used_names:
                used_names.add(candidate)
                return candidate
            index += 1

    def fresh_render_temp(base: str) -> str:
        if _is_generated_render_temp_base(base):
            return fresh_generated_temp()
        return fresh_temp(base)

    bound_dim_values: set[str] = set()

    def collect_type_dim_values(type_expr: TypeExpr | None) -> None:
        if isinstance(type_expr, TypeTensor):
            for dim in type_expr.dims:
                bound_dim_values.update(name for name in dim_token_names(dim) if name.isidentifier())
            return
        if isinstance(type_expr, TypeOptional):
            collect_type_dim_values(type_expr.inner)
            return
        if isinstance(type_expr, TypeList):
            collect_type_dim_values(type_expr.item)
            return
        if isinstance(type_expr, TypeTuple):
            for item in type_expr.items:
                collect_type_dim_values(item)
            return
        if isinstance(type_expr, TypeNamed):
            for dim in type_expr.args:
                bound_dim_values.update(name for name in dim_token_names(dim) if name.isidentifier())

    for value in module.inputs:
        collect_type_dim_values(value.type_expr)
    collect_type_dim_values(module.return_type_expr)

    required_dim_values: set[str] = set()

    def collect_term_dim_value_refs(operand: GraphOperand) -> None:
        if isinstance(operand, GraphValueRef):
            if isinstance(operand.type_expr, TypeDim | TypeInt):
                required_dim_values.add(operand.name)
            return
        if not isinstance(operand, GraphExpr):
            return
        for item in operand.inputs:
            collect_term_dim_value_refs(item)
        for item in operand.attrs.values():
            collect_term_dim_value_refs(item)

    for node in nodes:
        for item in node.inputs:
            collect_term_dim_value_refs(item)
        for item in node.attrs.values():
            collect_term_dim_value_refs(item)
    for output in outputs:
        collect_term_dim_value_refs(output)

    module_ref_subst: dict[str, GraphOperand] = {}

    def bind_output_existential_dims(output: GraphValue) -> None:
        if not isinstance(output.type_expr, TypeTensor):
            return
        output_ref = GraphValueRef(
            name=output.name,
            type_expr=output.type_expr,
            dims=output.dims,
        )
        for index, dim in enumerate(output.type_expr.dims):
            if (
                not isinstance(dim, str)
                or not dim.isidentifier()
                or dim not in required_dim_values
                or dim in bound_dim_values
                or dim in input_names
                or dim in (zero_arg_modules or set())
            ):
                continue
            if dim.isupper() and dim.startswith("NUM_") and len(dim) > 4:
                base = dim[4:].lower()
            else:
                base = dim.lower() if dim.isupper() else dim
            target = fresh_render_temp(base)
            shape_name = fresh_render_temp(f"{target}_shape")
            shape_ref = GraphValueRef(
                name=shape_name,
                type_expr=TypeList(TypeDim()),
                dims=None,
            )
            statements.append(
                AxonBind(
                    targets=(shape_name,),
                    expr=_node_expr_to_expr(
                        op_name="_shape",
                        inputs=(output_ref,),
                        attrs={},
                        type_expr=TypeList(TypeDim()),
                        dims=None,
                        local_names=used_names,
                        zero_arg_modules=zero_arg_modules,
                    ),
                )
            )
            statements.append(
                AxonBind(
                    targets=(target,),
                    expr=_node_expr_to_expr(
                        op_name="_list_index",
                        inputs=(
                            shape_ref,
                            GraphLiteral(value=index, type_expr=TypeInt()),
                        ),
                        attrs={},
                        type_expr=TypeDim(),
                        dims=None,
                        local_names=used_names,
                        zero_arg_modules=zero_arg_modules,
                    ),
                )
            )
            if target != dim:
                module_ref_subst[dim] = GraphValueRef(
                    name=target,
                    type_expr=TypeDim(),
                    dims=None,
                )
            bound_dim_values.add(dim)

    def zero_arg_global_ref(operand: GraphOperand) -> GraphOperand:
        if (
            zero_arg_modules is not None
            and isinstance(operand, GraphExpr)
            and operand.op.name in zero_arg_modules
            and not operand.inputs
            and not operand.attrs
            and operand.op.name not in input_names
        ):
            return GraphValueRef(
                name=operand.op.name,
                type_expr=operand.type_expr,
                dims=operand.dims,
            )
        return operand

    def atomicize_operand(operand: GraphOperand, *, base: str) -> GraphOperand:
        operand = _replace_operand_refs(operand, module_ref_subst)
        operand = zero_arg_global_ref(operand)
        if (
            zero_arg_modules is not None
            and isinstance(operand, GraphValueRef)
            and operand.name in zero_arg_modules
            and operand.name not in (global_value_modules or set())
            and operand.name not in input_names
        ):
            operand = GraphExpr(
                op=GraphOp(operand.name),
                inputs=(),
                attrs={},
                type_expr=operand.type_expr,
                dims=operand.dims,
            )
        if not isinstance(operand, GraphExpr):
            return operand
        if (
            zero_arg_modules is not None
            and operand.op.name == "core.ascribe"
            and len(operand.inputs) == 1
            and not operand.attrs
        ):
            input_operand = zero_arg_global_ref(operand.inputs[0])
            if (
                isinstance(input_operand, GraphValueRef)
                and input_operand.name in zero_arg_modules
                and input_operand.name not in input_names
            ):
                return replace(operand, inputs=(input_operand,))
        if operand.op.name in {"core.list", "core.tuple"} and not operand.attrs:
            inputs = tuple(
                atomicize_operand(item, base=f"{base}_arg{idx + 1}")
                for idx, item in enumerate(operand.inputs)
            )
            if all(not isinstance(item, GraphExpr) for item in inputs):
                return replace(operand, inputs=inputs)
        if operand.op.name == "core.select" and len(operand.inputs) == 3 and not operand.attrs:
            return replace(
                operand,
                inputs=(
                    atomicize_operand(operand.inputs[0], base=f"{base}_cond"),
                    prepare_lazy_branch_operand(operand.inputs[1], base=f"{base}_true"),
                    prepare_lazy_branch_operand(operand.inputs[2], base=f"{base}_false"),
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
            local_names=used_names,
            zero_arg_modules=zero_arg_modules,
        )
        temp = fresh_render_temp(base)
        statements.append(AxonBind(targets=(temp,), expr=expr))
        return GraphValueRef(name=temp, type_expr=operand.type_expr, dims=operand.dims)

    def prepare_lazy_branch_operand(operand: GraphOperand, *, base: str) -> GraphOperand:
        operand = _replace_operand_refs(operand, module_ref_subst)
        operand = zero_arg_global_ref(operand)
        if not isinstance(operand, GraphExpr):
            return operand
        if operand.op.name == "core.select" and len(operand.inputs) == 3 and not operand.attrs:
            return replace(
                operand,
                inputs=(
                    atomicize_operand(operand.inputs[0], base=f"{base}_cond"),
                    prepare_lazy_branch_operand(operand.inputs[1], base=f"{base}_true"),
                    prepare_lazy_branch_operand(operand.inputs[2], base=f"{base}_false"),
                ),
            )
        inputs: list[GraphOperand] = []
        for idx, item in enumerate(operand.inputs):
            if isinstance(item, GraphExpr) and (
                graph_operand_effect(item) != GraphEffect.TOTAL_PURE
                or graph_operand_usage(item) != UsageClass.UNRESTRICTED
            ):
                inputs.append(item)
            else:
                inputs.append(atomicize_operand(item, base=f"{base}_arg{idx + 1}"))
        attrs: dict[str, GraphOperand] = {}
        for key, value in operand.attrs.items():
            if isinstance(value, GraphExpr) and (
                graph_operand_effect(value) != GraphEffect.TOTAL_PURE
                or graph_operand_usage(value) != UsageClass.UNRESTRICTED
            ):
                attrs[key] = value
            else:
                attrs[key] = atomicize_operand(value, base=f"{base}_{key}")
        return replace(operand, inputs=tuple(inputs), attrs=attrs)

    for node in nodes:
        output_base = node.outputs[0].name if node.outputs else node.id.replace(":", "_")
        rewritten_node = replace(
            node,
            inputs=tuple(_replace_operand_refs(item, module_ref_subst) for item in node.inputs),
            attrs={
                key: _replace_operand_refs(value, module_ref_subst)
                for key, value in node.attrs.items()
            },
        )
        if rewritten_node.op.name == "core.repeat":
            callee_attr = rewritten_node.attrs.get("callee")
            var_attr = rewritten_node.attrs.get("var")
            arg_count_attr = rewritten_node.attrs.get("arg_count")
            carry_count_attr = rewritten_node.attrs.get("carry_count")
            if (
                not isinstance(callee_attr, GraphLiteral)
                or not isinstance(callee_attr.value, str)
                or not isinstance(var_attr, GraphLiteral)
                or not isinstance(var_attr.value, str)
                or not isinstance(arg_count_attr, GraphLiteral)
                or type(arg_count_attr.value) is not int
                or not isinstance(carry_count_attr, GraphLiteral)
                or type(carry_count_attr.value) is not int
            ):
                raise ValueError("cannot render malformed core.repeat node")
            arg_count = arg_count_attr.value
            carry_count = carry_count_attr.value
            carry_names: list[str] = []
            for index in range(carry_count):
                carry_attr = rewritten_node.attrs.get(f"carry_{index}")
                if not isinstance(carry_attr, GraphLiteral) or not isinstance(carry_attr.value, str):
                    raise ValueError("cannot render core.repeat without carry names")
                carry_input = rewritten_node.inputs[3 + index]
                if isinstance(carry_input, GraphValueRef):
                    carry_names.append(carry_input.name)
                else:
                    carry_names.append(carry_attr.value)
            target_names = tuple(output.name for output in rewritten_node.outputs)
            args: list[AxonExpr] = []
            for index in range(arg_count):
                role_attr = rewritten_node.attrs.get(f"arg_{index}")
                if not isinstance(role_attr, GraphLiteral) or not isinstance(role_attr.value, str):
                    raise ValueError("cannot render malformed core.repeat arg role")
                role = role_attr.value
                if role == "iter":
                    args.append(_typed(AxonExprName(var_attr.value), TypeInt(), None))
                elif role.startswith("carry:"):
                    carry_index = int(role.removeprefix("carry:"))
                    args.append(
                        _typed(
                            AxonExprName(carry_names[carry_index]),
                            graph_operand_type(rewritten_node.inputs[3 + carry_index]),
                            rewritten_node.outputs[carry_index].dims
                            if carry_index < len(rewritten_node.outputs)
                            else None,
                        )
                    )
                elif role.startswith("input:"):
                    input_index = int(role.removeprefix("input:"))
                    args.append(
                        _operand_to_expr(
                            atomicize_operand(rewritten_node.inputs[input_index], base=f"{output_base}_arg{index + 1}"),
                            local_names=used_names,
                            zero_arg_modules=zero_arg_modules,
                        )
                    )
                else:
                    raise ValueError(f"cannot render core.repeat arg role {role!r}")
            yield_expr = _typed(
                AxonExprCall(callee=callee_attr.value, args=tuple(args), kwargs={}),
                rewritten_node.type_expr,
                rewritten_node.dims,
                arity=len(rewritten_node.outputs) if len(rewritten_node.outputs) > 1 else None,
            )
            statements.append(
                AxonRepeat(
                    name=None,
                    var=var_attr.value,
                    to_expr=_operand_to_expr(
                        atomicize_operand(rewritten_node.inputs[1], base=f"{output_base}_to"),
                        local_names=used_names,
                        zero_arg_modules=zero_arg_modules,
                    ),
                    from_expr=_operand_to_expr(
                        atomicize_operand(rewritten_node.inputs[0], base=f"{output_base}_from"),
                        local_names=used_names,
                        zero_arg_modules=zero_arg_modules,
                    ),
                    step_expr=_operand_to_expr(
                        atomicize_operand(rewritten_node.inputs[2], base=f"{output_base}_step"),
                        local_names=used_names,
                        zero_arg_modules=zero_arg_modules,
                    ),
                    body=(AxonYield(values=(yield_expr,)),),
                    targets=target_names,
                    carry=tuple(carry_names),
                )
            )
            continue
        if (
            zero_arg_modules is not None
            and rewritten_node.op.name in {"core.alias", "core.ascribe"}
            and len(rewritten_node.inputs) == 1
            and not rewritten_node.attrs
            and len(rewritten_node.outputs) == 1
            and isinstance(rewritten_node.inputs[0], GraphValueRef)
            and rewritten_node.inputs[0].name in (global_value_modules or set())
            and rewritten_node.inputs[0].name not in input_names
        ):
            input_operand: GraphOperand = rewritten_node.inputs[0]
            if rewritten_node.op.name == "core.ascribe":
                input_operand = GraphExpr(
                    op=GraphOp("core.ascribe"),
                    inputs=(input_operand,),
                    attrs={},
                    type_expr=rewritten_node.type_expr,
                    dims=rewritten_node.dims,
                )
            output = rewritten_node.outputs[0]
            module_ref_subst[output.name] = input_operand
            continue
        if (
            zero_arg_modules is not None
            and rewritten_node.op.name in (global_value_modules or set())
            and not rewritten_node.inputs
            and not rewritten_node.attrs
            and len(rewritten_node.outputs) == 1
        ):
            output = rewritten_node.outputs[0]
            module_ref_subst[output.name] = GraphValueRef(
                name=rewritten_node.op.name,
                type_expr=output.type_expr,
                dims=output.dims,
            )
            continue
        if (
            zero_arg_modules is not None
            and rewritten_node.op.name in (global_value_modules or set())
            and not rewritten_node.inputs
            and not rewritten_node.attrs
            and len(rewritten_node.outputs) == 1
            and rewritten_node.op.name in module_ref_subst
        ):
            module_ref_subst[rewritten_node.outputs[0].name] = module_ref_subst[
                rewritten_node.op.name
            ]
            continue
        if rewritten_node.op.name == "core.select" and len(rewritten_node.inputs) == 3:
            node_inputs = (
                atomicize_operand(rewritten_node.inputs[0], base=f"{output_base}_cond"),
                prepare_lazy_branch_operand(rewritten_node.inputs[1], base=f"{output_base}_true"),
                prepare_lazy_branch_operand(rewritten_node.inputs[2], base=f"{output_base}_false"),
            )
        else:
            node_inputs = tuple(
                atomicize_operand(item, base=f"{output_base}_arg{idx + 1}")
                for idx, item in enumerate(rewritten_node.inputs)
            )
        node_attrs = {
            key: atomicize_operand(value, base=f"{output_base}_{key}")
            for key, value in rewritten_node.attrs.items()
        }
        expr = _node_expr_to_expr(
            op_name=rewritten_node.op.name,
            inputs=node_inputs,
            attrs=node_attrs,
            type_expr=rewritten_node.type_expr,
            dims=rewritten_node.dims,
            local_names=used_names,
            zero_arg_modules=zero_arg_modules,
        )
        expr = _typed(
            expr,
            rewritten_node.type_expr,
            rewritten_node.dims,
            arity=len(rewritten_node.outputs) if len(rewritten_node.outputs) > 1 else None,
        )
        statements.append(AxonBind(targets=tuple(output.name for output in rewritten_node.outputs), expr=expr))
        for output in rewritten_node.outputs:
            if isinstance(output.type_expr, TypeDim | TypeInt):
                bound_dim_values.add(output.name)
        for output in rewritten_node.outputs:
            bind_output_existential_dims(output)
        if (
            zero_arg_modules is not None
            and rewritten_node.op.name in (global_value_modules or set())
            and not rewritten_node.inputs
            and not rewritten_node.attrs
            and len(rewritten_node.outputs) == 1
        ):
            output = rewritten_node.outputs[0]
            module_ref_subst[rewritten_node.op.name] = GraphValueRef(
                name=output.name,
                type_expr=output.type_expr,
                dims=output.dims,
            )
    outputs = tuple(
        atomicize_operand(
            _replace_operand_refs(output, module_ref_subst),
            base=f"__return_{idx + 1}",
        )
        for idx, output in enumerate(outputs)
    )
    statements.append(
        AxonReturn(
            values=tuple(
                _operand_to_expr(
                    output,
                    local_names=used_names,
                    zero_arg_modules=zero_arg_modules,
                )
                for output in outputs
            )
        )
    )
    statements = list(_canonicalize_render_value_names(tuple(statements)))
    return AxonDefinition(
        name=module.name,
        path_param=None,
        params=tuple(_input_to_param(value) for value in module.inputs),
        returns=module.output_names,
        statements=tuple(statements),
        return_type_expr=_module_return_type(module),
        constraints=module.constraints,
        is_global_binding=module.is_global_binding,
    )


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


def _operand_module_refs(operand: GraphOperand, module_names: set[str], out: set[str]) -> None:
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
        _operand_module_refs(item, module_names, out)
    for item in operand.attrs.values():
        _operand_module_refs(item, module_names, out)


def _module_dependency_refs(module: GraphModule, module_names: set[str]) -> set[str]:
    refs: set[str] = set()
    for value in module.inputs:
        _type_module_refs(value.type_expr, module_names, refs)
        if value.dims is not None:
            for dim in value.dims:
                _dim_module_refs(dim, module_names, refs)
    for node in module.nodes:
        if node.op.name in module_names:
            refs.add(node.op.name)
        if node.op.name == "core.repeat":
            callee = node.attrs.get("callee")
            if isinstance(callee, GraphLiteral) and isinstance(callee.value, str) and callee.value in module_names:
                refs.add(callee.value)
        _type_module_refs(node.type_expr, module_names, refs)
        if node.dims is not None:
            for dim in node.dims:
                _dim_module_refs(dim, module_names, refs)
        for value in node.outputs:
            _type_module_refs(value.type_expr, module_names, refs)
            if value.dims is not None:
                for dim in value.dims:
                    _dim_module_refs(dim, module_names, refs)
        for operand in (*node.inputs, *node.attrs.values()):
            _operand_module_refs(operand, module_names, refs)
            _type_module_refs(graph_operand_type(operand), module_names, refs)
    for output in module.outputs:
        _operand_module_refs(output, module_names, refs)
        _type_module_refs(graph_operand_type(output), module_names, refs)
    _type_module_refs(module.return_type_expr, module_names, refs)
    for constraint in module.constraints:
        _constraint_module_refs(constraint, module_names, refs)
    refs.discard(module.name)
    return refs


def _reachable_dependency_ordered_modules(program: GraphProgram) -> tuple[GraphModule, ...]:
    modules_by_name = {module.name: module for module in program.modules}
    module_names = set(modules_by_name)
    global_module_names = {
        module.name
        for module in program.modules
        if module.is_global_binding and module.name != program.main_module
    }
    edges = {
        module.name: _module_dependency_refs(module, module_names)
        for module in program.modules
    }

    reachable: set[str] = set()
    stack = [program.main_module]
    while stack:
        name = stack.pop()
        if name in reachable:
            continue
        reachable.add(name)
        stack.extend(sorted(edges.get(name, ()), reverse=True))

    index = 0
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    active_stack: list[str] = []
    on_stack: set[str] = set()
    component_by_name: dict[str, int] = {}
    components: list[tuple[str, ...]] = []

    def visit(name: str) -> None:
        nonlocal index
        indices[name] = index
        lowlinks[name] = index
        index += 1
        active_stack.append(name)
        on_stack.add(name)
        for target in sorted(edges.get(name, ()) & reachable):
            if target not in indices:
                visit(target)
                lowlinks[name] = min(lowlinks[name], lowlinks[target])
            elif target in on_stack:
                lowlinks[name] = min(lowlinks[name], indices[target])
        if lowlinks[name] != indices[name]:
            return
        component: list[str] = []
        while True:
            item = active_stack.pop()
            on_stack.remove(item)
            component.append(item)
            if item == name:
                break
        component_id = len(components)
        ordered_component = tuple(sorted(component))
        for item in ordered_component:
            component_by_name[item] = component_id
        components.append(ordered_component)

    if program.main_module in module_names:
        visit(program.main_module)

    component_edges: dict[int, set[int]] = {idx: set() for idx in range(len(components))}
    for name in reachable:
        source_component = component_by_name[name]
        for target in edges.get(name, ()) & reachable:
            target_component = component_by_name[target]
            if target_component != source_component:
                component_edges[source_component].add(target_component)

    remaining = set(component_edges)
    ordered_component_ids: list[int] = []

    def component_sort_key(component_id: int) -> tuple[bool, tuple[str, ...]]:
        names = components[component_id]
        return (not all(name in global_module_names for name in names), names)

    while remaining:
        ready = [
            component_id
            for component_id in remaining
            if not (component_edges[component_id] & remaining)
        ]
        if not ready:
            ready = list(remaining)
        component_id = min(ready, key=component_sort_key)
        remaining.remove(component_id)
        ordered_component_ids.append(component_id)

    return tuple(
        modules_by_name[name]
        for component_id in ordered_component_ids
        for name in sorted(
            components[component_id],
            key=lambda item: (item not in global_module_names, item),
        )
    )


def graph_program_to_axon_file(program: GraphProgram) -> AxonFile:
    validate_graph_program(program)
    ordered_modules = _reachable_dependency_ordered_modules(program)
    zero_arg_modules = {module.name for module in program.modules if not module.inputs}
    global_value_modules = {
        module.name
        for module in program.modules
        if module.is_global_binding and not module.inputs
    }
    axon = AxonFile(
        modules=tuple(
            graph_module_to_axon_definition(
                module,
                zero_arg_modules=zero_arg_modules,
                global_value_modules=global_value_modules,
            )
            for module in ordered_modules
        ),
        imports=(),
        imported_members={},
        exports=(),
        pragmas=dict(program.pragmas),
        type_aliases={},
    )
    validate_flat_axon_file(axon, main_module=program.main_module)
    validate_typed_axon_file(axon, main_module=program.main_module)
    return axon


def _format_domain_fact(fact: GraphDomainFact) -> str:
    if fact.kind == GraphDomainKind.UNKNOWN:
        return "unknown"
    if fact.kind == GraphDomainKind.NULL:
        return "null"
    if fact.kind == GraphDomainKind.NOT_NULL:
        return "not_null"
    if fact.kind == GraphDomainKind.LITERAL:
        return repr(fact.value)
    if fact.kind == GraphDomainKind.INTERVAL and isinstance(fact.value, GraphDomainInterval):
        lower = "-inf" if fact.value.lower is None else repr(fact.value.lower)
        upper = "+inf" if fact.value.upper is None else repr(fact.value.upper)
        return f"[{lower},{upper}]"
    if fact.kind == GraphDomainKind.GLOBAL_VALUE:
        return str(fact.value)
    if fact.kind == GraphDomainKind.PATH:
        path = fact.value
        if isinstance(path, GraphPath):
            prefix = "@@" if path.absolute else "@"
            return prefix + ".".join(path.parts)
    return str(fact.value)


def graph_domain_definition_comments(program: GraphProgram) -> dict[str, tuple[str, ...]]:
    analysis = infer_main_module_domain_facts(program)
    comments: dict[str, tuple[str, ...]] = {}
    for module in program.modules:
        parts: list[str] = []
        input_facts = analysis.module_input_facts.get(module.name, {})
        input_parts = [
            f"{name}={_format_domain_fact(fact)}"
            for name, fact in sorted(input_facts.items())
            if fact.kind != GraphDomainKind.UNKNOWN
        ]
        if input_parts:
            parts.append("inputs " + ", ".join(input_parts))
        local_facts = analysis.module_local_facts.get(module.name, {})
        local_parts = [
            f"{name}={_format_domain_fact(fact)}"
            for name, fact in sorted(local_facts.items())
            if name not in input_facts and fact.kind != GraphDomainKind.UNKNOWN
        ]
        if local_parts:
            parts.append("locals " + ", ".join(local_parts[:24]))
        output_facts = analysis.module_output_facts.get(module.name, ())
        output_parts = [
            f"out{index}={_format_domain_fact(fact)}"
            for index, fact in enumerate(output_facts)
            if fact.kind != GraphDomainKind.UNKNOWN
        ]
        if output_parts:
            parts.append("outputs " + ", ".join(output_parts))
        if parts:
            comments[module.name] = tuple(f"domain: {part}" for part in parts)
    return comments


__all__ = [
    "graph_domain_definition_comments",
    "graph_module_to_axon_definition",
    "graph_program_to_axon_file",
]
