from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import TypeAlias

from ..ast import (
    AxonBind,
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
    AxonDefinition,
    AxonParam,
    AxonRepeat,
    AxonReturn,
    AxonYield,
    Constraint,
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
    TypePath,
    TypeString,
    TypeTensor,
    TypeTuple,
    TypeVar,
    ast_equal,
    render_type,
)
from ..entrypoint import resolve_main_module
from ..ast.types import dim_token_names
from ..validate import (
    validate_closed_axon_file,
    validate_flat_axon_file,
    validate_typed_axon_file,
)


@dataclass(frozen=True)
class GraphPath:
    absolute: bool
    parts: tuple[str, ...]


@dataclass(frozen=True)
class GraphLiteral:
    value: bool | int | float | str | None
    type_expr: TypeExpr


@dataclass(frozen=True)
class GraphValue:
    name: str
    type_expr: TypeExpr
    dims: tuple[DimToken, ...] | None = None
    optional: bool = False


@dataclass(frozen=True)
class GraphValueRef:
    name: str
    type_expr: TypeExpr
    dims: tuple[DimToken, ...] | None = None


@dataclass(frozen=True)
class GraphOp:
    name: str


@dataclass(frozen=True)
class GraphExpr:
    op: GraphOp
    inputs: tuple["GraphOperand", ...]
    attrs: dict[str, "GraphAttr"]
    type_expr: TypeExpr
    dims: tuple[DimToken, ...] | None = None


GraphOperand: TypeAlias = GraphValueRef | GraphLiteral | GraphPath | GraphExpr
GraphAttr: TypeAlias = GraphOperand


_PATH_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass(frozen=True)
class GraphNode:
    id: str
    op: GraphOp
    inputs: tuple[GraphOperand, ...]
    attrs: dict[str, GraphAttr]
    outputs: tuple[GraphValue, ...]
    source_module: str
    type_expr: TypeExpr
    dims: tuple[DimToken, ...] | None = None


@dataclass(frozen=True)
class GraphModule:
    name: str
    inputs: tuple[GraphValue, ...]
    outputs: tuple[GraphOperand, ...]
    output_names: tuple[str, ...]
    nodes: tuple[GraphNode, ...]
    return_type_expr: TypeExpr | None = None
    constraints: tuple[Constraint, ...] = ()
    is_global_binding: bool = False


@dataclass(frozen=True)
class GraphPackedParameter:
    output: GraphPath
    inputs: tuple[GraphPath, ...]
    dim: int
    remove_inputs: bool = True


@dataclass(frozen=True)
class GraphProgram:
    modules: tuple[GraphModule, ...]
    main_module: str
    pragmas: dict[str, object]
    packed_parameters: tuple[GraphPackedParameter, ...] = ()


@dataclass
class _GraphLowerCtx:
    module_name: str
    nodes: list[GraphNode]
    env: dict[str, str]
    used: set[str]
    next_temp: int = 0
    next_node: int = 0

    def temp(self) -> str:
        while True:
            self.next_temp += 1
            name = f"__g{self.next_temp}"
            if name not in self.used:
                return name

    def node_id(self) -> str:
        self.next_node += 1
        return f"{self.module_name}:{self.next_node}"

    def define_target(self, source_name: str) -> str:
        if source_name not in self.used:
            graph_name = source_name
        else:
            index = 1
            while True:
                graph_name = f"{source_name}__g{index}"
                if graph_name not in self.used:
                    break
                index += 1
        self.used.add(graph_name)
        self.env[source_name] = graph_name
        return graph_name


def _expr_type(expr: AxonExpr) -> TypeExpr:
    if expr.inferred_type is None:
        raise ValueError("graph IR lowering requires typed Axon expressions")
    return expr.inferred_type


def _literal_expr(expr: AxonExpr) -> GraphLiteral | None:
    if isinstance(expr, AxonExprInt):
        type_expr = expr.inferred_type or TypeInt()
        return GraphLiteral(value=expr.value, type_expr=type_expr)
    if isinstance(expr, AxonExprFloat):
        type_expr = expr.inferred_type or TypeFloat()
        return GraphLiteral(value=expr.value, type_expr=type_expr)
    if isinstance(expr, AxonExprBool):
        type_expr = expr.inferred_type or TypeBool()
        return GraphLiteral(value=expr.value, type_expr=type_expr)
    if isinstance(expr, AxonExprNull):
        type_expr = expr.inferred_type or TypeNull()
        return GraphLiteral(value=None, type_expr=type_expr)
    if isinstance(expr, AxonExprString):
        type_expr = expr.inferred_type or TypeString()
        return GraphLiteral(value=expr.value, type_expr=type_expr)
    return None


def _expr_to_operand(expr: AxonExpr, ctx: _GraphLowerCtx | None = None) -> GraphOperand:
    if isinstance(expr, AxonExprAscribe):
        return GraphExpr(
            op=GraphOp("core.ascribe"),
            inputs=(_expr_to_operand(expr.expr, ctx),),
            attrs={},
            type_expr=_expr_type(expr),
            dims=expr.inferred_dims,
        )
    literal = _literal_expr(expr)
    if literal is not None:
        return literal
    if isinstance(expr, AxonExprName):
        name = ctx.env.get(expr.name, expr.name) if ctx is not None else expr.name
        return GraphValueRef(
            name=name,
            type_expr=_expr_type(expr),
            dims=expr.inferred_dims,
        )
    if isinstance(expr, AxonExprPath):
        return GraphPath(absolute=expr.absolute, parts=expr.parts)
    if isinstance(expr, AxonExprList | AxonExprTuple):
        return GraphExpr(
            op=GraphOp("core.list" if isinstance(expr, AxonExprList) else "core.tuple"),
            inputs=tuple(_expr_to_operand(item, ctx) for item in expr.items),
            attrs={},
            type_expr=_expr_type(expr),
            dims=expr.inferred_dims,
        )
    raise ValueError(
        f"graph IR lowering requires flat atomic operands, got {type(expr).__name__}"
    )


def _graph_ref(value: GraphValue) -> GraphValueRef:
    return GraphValueRef(name=value.name, type_expr=_value_ref_type(value), dims=value.dims)


def _value_ref_type(value: GraphValue) -> TypeExpr:
    if value.optional and not isinstance(value.type_expr, TypeOptional):
        return TypeOptional(value.type_expr)
    return value.type_expr


def _kwarg_to_attr(value: object, ctx: _GraphLowerCtx | None = None) -> GraphAttr:
    if isinstance(value, AxonExpr):
        if ctx is None:
            return _expr_to_operand(value, ctx)
        return _lower_expr_to_lazy_operand(value, ctx)
    if isinstance(value, bool):
        return GraphLiteral(value=value, type_expr=TypeAny())
    if isinstance(value, int):
        return GraphLiteral(value=value, type_expr=TypeAny())
    if isinstance(value, float):
        return GraphLiteral(value=value, type_expr=TypeAny())
    if isinstance(value, str):
        return GraphLiteral(value=value, type_expr=TypeAny())
    if value is None:
        return GraphLiteral(value=None, type_expr=TypeAny())
    if isinstance(value, list):
        return GraphExpr(
            op=GraphOp("core.list"),
            inputs=tuple(_kwarg_to_attr(item, ctx) for item in value),
            attrs={},
            type_expr=TypeList(TypeAny()),
        )
    raise ValueError(f"unsupported graph IR attr value {value!r}")


def _param_to_value(param: AxonParam) -> GraphValue:
    if param.type_expr is None:
        raise ValueError(f"graph IR lowering requires typed parameter {param.name!r}")
    if param.default_expr is not None:
        raise ValueError(
            "graph IR lowering requires elaborated Axon input; "
            f"parameter {param.name!r} still has a default"
        )
    return GraphValue(
        name=param.name,
        type_expr=param.type_expr,
        dims=tuple(param.type_expr.dims) if isinstance(param.type_expr, TypeTensor) else None,
        optional=param.optional,
    )


def _target_types(expr: AxonExpr, target_count: int) -> tuple[TypeExpr, ...]:
    inferred = _expr_type(expr)
    if target_count == 1:
        return (inferred,)
    if isinstance(inferred, TypeTuple) and len(inferred.items) == target_count:
        return inferred.items
    if isinstance(inferred, TypeList):
        return tuple(inferred.item for _ in range(target_count))
    return tuple(TypeAny() for _ in range(target_count))


def _target_dims(expr: AxonExpr, target_count: int) -> tuple[tuple[DimToken, ...] | None, ...]:
    if target_count == 1:
        return (expr.inferred_dims,)
    inferred = _expr_type(expr)
    if isinstance(inferred, TypeTuple):
        return tuple(None for _ in range(target_count))
    if isinstance(inferred, TypeList):
        return tuple(expr.inferred_dims for _ in range(target_count))
    return tuple(None for _ in range(target_count))


def _lower_expr_to_operand(expr: AxonExpr, ctx: _GraphLowerCtx) -> GraphOperand:
    if isinstance(expr, AxonExprAscribe):
        return GraphExpr(
            op=GraphOp("core.ascribe"),
            inputs=(_lower_expr_to_operand(expr.expr, ctx),),
            attrs={},
            type_expr=_expr_type(expr),
            dims=expr.inferred_dims,
        )
    if isinstance(
        expr,
        (
            AxonExprName,
            AxonExprInt,
            AxonExprFloat,
            AxonExprBool,
            AxonExprNull,
            AxonExprString,
            AxonExprPath,
        ),
    ):
        return _expr_to_operand(expr, ctx)
    if isinstance(expr, AxonExprList | AxonExprTuple):
        return GraphExpr(
            op=GraphOp("core.list" if isinstance(expr, AxonExprList) else "core.tuple"),
            inputs=tuple(_lower_expr_to_operand(item, ctx) for item in expr.items),
            attrs={},
            type_expr=_expr_type(expr),
            dims=expr.inferred_dims,
        )
    target = ctx.temp()
    node = _node_for_expr(
        expr,
        targets=(target,),
        node_id=ctx.node_id(),
        module_name=ctx.module_name,
        ctx=ctx,
    )
    ctx.nodes.append(node)
    if len(node.outputs) != 1:
        raise ValueError("graph IR inline expression produced multiple values")
    return _graph_ref(node.outputs[0])


def _lower_expr_to_lazy_operand(expr: AxonExpr, ctx: _GraphLowerCtx) -> GraphOperand:
    if isinstance(expr, AxonExprAscribe):
        return GraphExpr(
            op=GraphOp("core.ascribe"),
            inputs=(_lower_expr_to_lazy_operand(expr.expr, ctx),),
            attrs={},
            type_expr=_expr_type(expr),
            dims=expr.inferred_dims,
        )
    if isinstance(
        expr,
        (
            AxonExprName,
            AxonExprInt,
            AxonExprFloat,
            AxonExprBool,
            AxonExprNull,
            AxonExprString,
            AxonExprPath,
        ),
    ):
        return _expr_to_operand(expr, ctx)
    if isinstance(expr, AxonExprList | AxonExprTuple):
        return GraphExpr(
            op=GraphOp("core.list" if isinstance(expr, AxonExprList) else "core.tuple"),
            inputs=tuple(_lower_expr_to_lazy_operand(item, ctx) for item in expr.items),
            attrs={},
            type_expr=_expr_type(expr),
            dims=expr.inferred_dims,
        )
    if isinstance(expr, AxonExprBinary):
        return GraphExpr(
            op=GraphOp(f"core.binary.{expr.op}"),
            inputs=(
                _lower_expr_to_lazy_operand(expr.left, ctx),
                _lower_expr_to_lazy_operand(expr.right, ctx),
            ),
            attrs={},
            type_expr=_expr_type(expr),
            dims=expr.inferred_dims,
        )
    if isinstance(expr, AxonExprTernary):
        return GraphExpr(
            op=GraphOp("core.select"),
            inputs=(
                _lower_expr_to_lazy_operand(expr.cond, ctx),
                _lower_expr_to_lazy_operand(expr.true_expr, ctx),
                _lower_expr_to_lazy_operand(expr.false_expr, ctx),
            ),
            attrs={},
            type_expr=_expr_type(expr),
            dims=expr.inferred_dims,
        )
    if isinstance(expr, AxonExprCall):
        if expr.kwargs:
            raise ValueError(
                f"graph IR lowering requires elaborated Axon; call to {expr.callee!r} still has kwargs"
            )
        return GraphExpr(
            op=GraphOp(expr.callee),
            inputs=tuple(_lower_expr_to_lazy_operand(arg, ctx) for arg in expr.args),
            attrs={},
            type_expr=_expr_type(expr),
            dims=expr.inferred_dims,
        )
    raise ValueError(f"unsupported lazy graph expression {type(expr).__name__}")


def _node_for_expr(
    expr: AxonExpr,
    *,
    targets: tuple[str, ...],
    node_id: str,
    module_name: str,
    ctx: _GraphLowerCtx,
) -> GraphNode:
    types = _target_types(expr, len(targets))
    dims = _target_dims(expr, len(targets))
    core_expr = expr.expr if isinstance(expr, AxonExprAscribe) else expr

    def _outputs() -> tuple[GraphValue, ...]:
        graph_targets = tuple(ctx.define_target(target) for target in targets)
        return tuple(
            GraphValue(name=target, type_expr=type_expr, dims=dim)
            for target, type_expr, dim in zip(graph_targets, types, dims, strict=True)
        )

    if isinstance(core_expr, AxonExprCall):
        if core_expr.kwargs:
            raise ValueError(
                f"graph IR lowering requires elaborated Axon; call to {core_expr.callee!r} still has kwargs"
            )
        inputs = tuple(_lower_expr_to_operand(arg, ctx) for arg in core_expr.args)
        return GraphNode(
            id=node_id,
            op=GraphOp(core_expr.callee),
            inputs=inputs,
            attrs={},
            outputs=_outputs(),
            source_module=module_name,
            type_expr=_expr_type(expr),
            dims=expr.inferred_dims,
        )
    if isinstance(core_expr, AxonExprBinary):
        inputs = (
            _lower_expr_to_operand(core_expr.left, ctx),
            _lower_expr_to_operand(core_expr.right, ctx),
        )
        return GraphNode(
            id=node_id,
            op=GraphOp(f"core.binary.{core_expr.op}"),
            inputs=inputs,
            attrs={},
            outputs=_outputs(),
            source_module=module_name,
            type_expr=_expr_type(expr),
            dims=expr.inferred_dims,
        )
    if isinstance(core_expr, AxonExprTernary):
        inputs = (
            _lower_expr_to_operand(core_expr.cond, ctx),
            _lower_expr_to_lazy_operand(core_expr.true_expr, ctx),
            _lower_expr_to_lazy_operand(core_expr.false_expr, ctx),
        )
        return GraphNode(
            id=node_id,
            op=GraphOp("core.select"),
            inputs=inputs,
            attrs={},
            outputs=_outputs(),
            source_module=module_name,
            type_expr=_expr_type(expr),
            dims=expr.inferred_dims,
        )
    if isinstance(core_expr, AxonExprList | AxonExprTuple):
        inputs = tuple(_lower_expr_to_operand(item, ctx) for item in core_expr.items)
        return GraphNode(
            id=node_id,
            op=GraphOp("core.list" if isinstance(core_expr, AxonExprList) else "core.tuple"),
            inputs=inputs,
            attrs={},
            outputs=_outputs(),
            source_module=module_name,
            type_expr=_expr_type(expr),
            dims=expr.inferred_dims,
        )
    inputs = (_lower_expr_to_operand(core_expr, ctx),)
    return GraphNode(
        id=node_id,
        op=GraphOp("core.alias"),
        inputs=inputs,
        attrs={},
        outputs=_outputs(),
        source_module=module_name,
        type_expr=_expr_type(expr),
        dims=expr.inferred_dims,
    )


def _repeat_yield_expr(stmt: AxonRepeat) -> AxonExpr:
    if len(stmt.body) != 1 or not isinstance(stmt.body[0], AxonYield):
        raise ValueError("graph IR lowering requires flat repeat bodies to be a single yield")
    if len(stmt.body[0].values) != 1:
        raise ValueError("graph IR lowering requires flat repeat yields to contain one expression")
    return stmt.body[0].values[0]


def _ascribed_call(expr: AxonExpr) -> AxonExprCall | None:
    if isinstance(expr, AxonExprAscribe):
        return _ascribed_call(expr.expr)
    if isinstance(expr, AxonExprCall):
        return expr
    return None


def _node_for_repeat(
    stmt: AxonRepeat,
    *,
    node_id: str,
    module_name: str,
    ctx: _GraphLowerCtx,
) -> GraphNode:
    if stmt.targets is None:
        raise ValueError("graph IR lowering requires flat repeat targets")
    yield_expr = _repeat_yield_expr(stmt)
    call = _ascribed_call(yield_expr)
    if call is None:
        raise ValueError("graph IR lowering requires flat repeat body to yield a helper call")
    carry_names = tuple(stmt.carry or ())
    if len(stmt.targets) != len(carry_names):
        raise ValueError("graph IR lowering requires repeat targets to match carry arity")

    types = _target_types(yield_expr, len(stmt.targets))
    dims = _target_dims(yield_expr, len(stmt.targets))
    carry_index = {name: index for index, name in enumerate(carry_names)}
    inputs: list[GraphOperand] = [
        _lower_expr_to_operand(stmt.from_expr, ctx),
        _lower_expr_to_operand(stmt.to_expr, ctx),
        _lower_expr_to_operand(stmt.step_expr, ctx),
    ]
    inputs.extend(
        GraphValueRef(name=ctx.env.get(name, name), type_expr=tp, dims=dim)
        for name, tp, dim in zip(carry_names, types, dims, strict=True)
    )
    attrs: dict[str, GraphAttr] = {
        "callee": GraphLiteral(call.callee, TypeString()),
        "var": GraphLiteral(stmt.var, TypeString()),
        "arg_count": GraphLiteral(len(call.args), TypeInt()),
        "carry_count": GraphLiteral(len(carry_names), TypeInt()),
    }
    for index, name in enumerate(carry_names):
        attrs[f"carry_{index}"] = GraphLiteral(name, TypeString())
    for index, arg in enumerate(call.args):
        if isinstance(arg, AxonExprName) and arg.name == stmt.var:
            role = "iter"
        elif isinstance(arg, AxonExprName) and arg.name in carry_index:
            role = f"carry:{carry_index[arg.name]}"
        else:
            input_index = len(inputs)
            inputs.append(_lower_expr_to_operand(arg, ctx))
            role = f"input:{input_index}"
        attrs[f"arg_{index}"] = GraphLiteral(role, TypeString())

    graph_targets = tuple(ctx.define_target(target) for target in stmt.targets)
    outputs = tuple(
        GraphValue(name=target, type_expr=type_expr, dims=dim)
        for target, type_expr, dim in zip(graph_targets, types, dims, strict=True)
    )
    return GraphNode(
        id=node_id,
        op=GraphOp("core.repeat"),
        inputs=tuple(inputs),
        attrs=attrs,
        outputs=outputs,
        source_module=module_name,
        type_expr=_expr_type(yield_expr),
        dims=yield_expr.inferred_dims,
    )


def _module_to_graph(module: AxonDefinition) -> GraphModule:
    inputs = tuple(_param_to_value(param) for param in module.params)
    input_names = {value.name for value in inputs}
    ctx = _GraphLowerCtx(
        module_name=module.name,
        nodes=[],
        env={name: name for name in input_names},
        used=set(input_names),
    )
    outputs: tuple[GraphOperand, ...] | None = None
    for stmt in module.statements:
        if isinstance(stmt, AxonBind):
            ctx.nodes.append(
                _node_for_expr(
                    stmt.expr,
                    targets=stmt.targets,
                    node_id=ctx.node_id(),
                    module_name=module.name,
                    ctx=ctx,
                )
            )
            continue
        if isinstance(stmt, AxonRepeat):
            ctx.nodes.append(
                _node_for_repeat(
                    stmt,
                    node_id=ctx.node_id(),
                    module_name=module.name,
                    ctx=ctx,
                )
            )
            continue
        if isinstance(stmt, AxonReturn):
            outputs = tuple(_lower_expr_to_operand(value, ctx) for value in stmt.values)
            continue
        raise ValueError(
            f"graph IR lowering requires flat typed Axon; "
            f"unexpected {type(stmt).__name__} in {module.name!r}"
        )
    if outputs is None:
        raise ValueError(f"graph IR module {module.name!r} has no return")
    return GraphModule(
        name=module.name,
        inputs=inputs,
        outputs=outputs,
        output_names=module.returns,
        nodes=tuple(ctx.nodes),
        return_type_expr=module.return_type_expr,
        constraints=module.constraints or (),
        is_global_binding=module.is_global_binding,
    )


def lower_axon_program_to_graph_ir(
    program: AxonFile,
    *,
    main_module: str | None = None,
) -> GraphProgram:
    validate_closed_axon_file(program, main_module=main_module)
    validate_flat_axon_file(program, main_module=main_module)
    validate_typed_axon_file(program, main_module=main_module)
    if not program.modules:
        raise ValueError("Axon program must contain at least one module")
    by_name = {module.name: module for module in program.modules}
    if len(by_name) != len(program.modules):
        raise ValueError("Axon program contains duplicate module names")
    resolved_main = resolve_main_module(program, main_module=main_module)
    if resolved_main not in by_name:
        raise ValueError(f"Unknown main module: {resolved_main!r}")
    graph = GraphProgram(
        modules=tuple(_module_to_graph(module) for module in program.modules),
        main_module=resolved_main,
        pragmas=dict(program.pragmas),
    )
    graph = _sanitize_graph_constraints_for_validation(graph)
    validate_graph_program(graph)
    return graph


def _type_dim_names(type_expr: TypeExpr) -> set[str]:
    from ..ast import TypeNamed, TypeOptional, TypeTensor

    if isinstance(type_expr, TypeTensor):
        names: set[str] = set()
        for dim in type_expr.dims:
            names.update(dim_token_names(dim))
        return names
    if isinstance(type_expr, TypeTuple):
        tuple_names: set[str] = set()
        for item in type_expr.items:
            tuple_names.update(_type_dim_names(item))
        return tuple_names
    if isinstance(type_expr, TypeList):
        return _type_dim_names(type_expr.item)
    if isinstance(type_expr, TypeOptional):
        return _type_dim_names(type_expr.inner)
    if isinstance(type_expr, TypeNamed):
        named_names: set[str] = set()
        for dim in type_expr.args:
            named_names.update(dim_token_names(dim))
        return named_names
    return set()


def _value_dim_names(value: GraphValue | GraphValueRef) -> set[str]:
    names: set[str] = set()
    names.update(_type_dim_names(value.type_expr))
    if value.dims is not None:
        for dim in value.dims:
            names.update(dim_token_names(dim))
    if isinstance(value.type_expr, TypeDim):
        names.add(value.name)
    return names


def _module_boundary_dim_symbols(module: GraphModule) -> set[str]:
    names: set[str] = set()
    for value in module.inputs:
        names.update(_value_dim_names(value))
    if module.return_type_expr is not None:
        names.update(_type_dim_names(module.return_type_expr))
    for output in module.outputs:
        if isinstance(output, GraphValueRef):
            names.update(_value_dim_names(output))
        elif isinstance(output, GraphExpr):
            names.update(_type_dim_names(output.type_expr))
            if output.dims is not None:
                for dim in output.dims:
                    names.update(dim_token_names(dim))
    return names


def _constraint_operand_names(operand: object) -> set[str]:
    if isinstance(operand, str):
        return {operand}
    if isinstance(operand, int | bool) or operand is None:
        return set()
    if isinstance(operand, DimExprBinary):
        return dim_token_names(operand)
    if isinstance(operand, tuple):
        names: set[str] = set()
        for item in operand:
            names.update(_constraint_operand_names(item))
        return names
    return set()


def _constraint_names(constraint: Constraint) -> set[str]:
    names = _constraint_operand_names(constraint.left)
    if constraint.right is not None:
        names.update(_constraint_operand_names(constraint.right))
    for guard in constraint.guards:
        names.update(_constraint_names(guard))
    return names


def _constraint_has_callsite_guard(constraint: Constraint) -> bool:
    return any(
        guard.relation == "callsite" or _constraint_has_callsite_guard(guard)
        for guard in constraint.guards
    )


def _constraint_is_trivial_identity(constraint: Constraint) -> bool:
    return (
        constraint.relation == "="
        and constraint.right is not None
        and constraint.left == constraint.right
    )


def _constraint_allowed_names_for_module(
    module: GraphModule,
    *,
    global_values: dict[str, GraphValue],
    modules_by_name: dict[str, GraphModule],
) -> set[str]:
    value_names = {value.name for value in module.inputs}
    dim_symbols = _module_boundary_dim_symbols(module)
    for value in global_values.values():
        if isinstance(value.type_expr, TypeDim):
            dim_symbols.add(value.name)
        dim_symbols.update(_type_dim_names(value.type_expr))
    for node in module.nodes:
        value_names.update(output.name for output in node.outputs)
        dim_symbols.update(_type_dim_names(node.type_expr))
        if node.dims is not None:
            for dim in node.dims:
                dim_symbols.update(dim_token_names(dim))
        for output in node.outputs:
            dim_symbols.update(_value_dim_names(output))
    return value_names | set(global_values) | set(modules_by_name) | dim_symbols


def _sanitize_graph_constraints_for_validation(program: GraphProgram) -> GraphProgram:
    """Drop stale constraint metadata that no longer closes over a graph module.

    Constraints are optimization/debug metadata. They must never make an otherwise
    well-formed graph invalid after earlier stages rewrite or remove temporaries.
    Callsite-guarded constraints are preserved because graph validation treats
    them as interprocedural facts rather than local module facts.
    """

    modules_by_name = {module.name: module for module in program.modules}
    global_values = {
        module.name: GraphValue(
            name=module.name,
            type_expr=_module_output_types(module)[0],
            dims=None,
        )
        for module in program.modules
        if not module.inputs and len(module.outputs) == 1
    }
    modules: list[GraphModule] = []
    for module in program.modules:
        if not module.constraints:
            modules.append(module)
            continue
        allowed = _constraint_allowed_names_for_module(
            module,
            global_values=global_values,
            modules_by_name=modules_by_name,
        )
        kept: list[Constraint] = []
        for constraint in module.constraints:
            if _constraint_is_trivial_identity(constraint):
                continue
            if _constraint_has_callsite_guard(constraint):
                kept.append(constraint)
                continue
            if _constraint_names(constraint) - allowed:
                continue
            kept.append(constraint)
        modules.append(replace(module, constraints=tuple(kept)))
    return replace(program, modules=tuple(modules))


def _validate_dim_names(
    names: set[str],
    *,
    dim_symbols: set[str],
    context: str,
) -> None:
    unknown = sorted(
        name for name in names if name not in dim_symbols and not name.startswith("..")
    )
    if unknown:
        raise ValueError(
            f"{context} uses unbound dim symbol(s): {', '.join(repr(name) for name in unknown)}"
        )


def _validate_type_dim_closure(
    type_expr: TypeExpr,
    *,
    dim_symbols: set[str],
    context: str,
) -> None:
    _validate_dim_names(_type_dim_names(type_expr), dim_symbols=dim_symbols, context=context)


def _validate_value_dim_closure(
    value: GraphValue | GraphValueRef,
    *,
    dim_symbols: set[str],
    context: str,
) -> None:
    _validate_type_dim_closure(value.type_expr, dim_symbols=dim_symbols, context=context)
    if value.dims is not None:
        names: set[str] = set()
        for dim in value.dims:
            names.update(dim_token_names(dim))
        _validate_dim_names(names, dim_symbols=dim_symbols, context=f"{context} dims")


def _validate_operand_dim_closure(
    operand: GraphOperand,
    *,
    dim_symbols: set[str],
    context: str,
    typevar_names: set[str] | None = None,
    defined_names: set[str] | None = None,
) -> None:
    typevar_names = typevar_names or set()
    defined_names = defined_names or set()
    if isinstance(operand, GraphLiteral):
        _validate_type_dim_closure(operand.type_expr, dim_symbols=dim_symbols, context=context)
        return
    if isinstance(operand, GraphPath):
        return
    if isinstance(operand, GraphValueRef):
        local_dim_symbols = set(dim_symbols)
        if operand.name in typevar_names or operand.name in defined_names:
            local_dim_symbols.update(_value_dim_names(operand))
        _validate_value_dim_closure(operand, dim_symbols=local_dim_symbols, context=context)
        return
    if isinstance(operand, GraphExpr):
        local_dim_symbols = set(dim_symbols)
        local_dim_symbols.update(_type_dim_names(operand.type_expr))
        if operand.dims is not None:
            for dim in operand.dims:
                local_dim_symbols.update(dim_token_names(dim))
        _validate_type_dim_closure(
            operand.type_expr,
            dim_symbols=local_dim_symbols,
            context=context,
        )
        if operand.dims is not None:
            names: set[str] = set()
            for dim in operand.dims:
                names.update(dim_token_names(dim))
            _validate_dim_names(names, dim_symbols=local_dim_symbols, context=f"{context} dims")
        for index, item in enumerate(operand.inputs):
            _validate_operand_dim_closure(
                item,
                dim_symbols=local_dim_symbols,
                context=f"{context} input {index}",
                typevar_names=typevar_names,
                defined_names=defined_names,
            )
        for key, item in operand.attrs.items():
            _validate_operand_dim_closure(
                item,
                dim_symbols=local_dim_symbols,
                context=f"{context} attr {key!r}",
                typevar_names=typevar_names,
                defined_names=defined_names,
            )
        return
    raise TypeError(f"unsupported graph operand {operand!r}")


def _dims_metadata_compatible(
    type_dims: tuple[DimToken, ...],
    metadata_dims: tuple[DimToken, ...],
) -> bool:
    if type_dims == metadata_dims:
        return True
    if any(_is_variadic_dim(dim) for dim in (*type_dims, *metadata_dims)):
        return _dim_sequence_compatible(metadata_dims, type_dims)
    if len(type_dims) != len(metadata_dims):
        return False
    return all(type_dim == metadata_dim for type_dim, metadata_dim in zip(type_dims, metadata_dims, strict=True))


def _require_value_metadata_coherent(value: GraphValue | GraphValueRef, *, context: str) -> None:
    if (
        isinstance(value.type_expr, TypeTensor)
        and value.dims is not None
        and not _dims_metadata_compatible(value.type_expr.dims, value.dims)
    ):
        raise ValueError(
            f"{context} {value.name!r} has stale dims metadata: "
            f"type has {value.type_expr.dims!r}, dims has {value.dims!r}"
        )


def _require_operand_metadata_coherent(operand: GraphOperand, *, context: str) -> None:
    if isinstance(operand, GraphValueRef):
        _require_value_metadata_coherent(operand, context=f"{context} ref")
        return
    if isinstance(operand, GraphExpr):
        if (
            isinstance(operand.type_expr, TypeTensor)
            and operand.dims is not None
            and not _dims_metadata_compatible(operand.type_expr.dims, operand.dims)
        ):
            raise ValueError(
                f"{context} expr {operand.op.name!r} has stale dims metadata: "
                f"type has {operand.type_expr.dims!r}, dims has {operand.dims!r}"
            )
        for index, item in enumerate(operand.inputs):
            _require_operand_metadata_coherent(item, context=f"{context} input {index}")
        for key, item in operand.attrs.items():
            _require_operand_metadata_coherent(item, context=f"{context} attr {key!r}")


def _validate_operand_defined(
    operand: GraphOperand,
    *,
    defined: set[str],
    dim_symbols: set[str],
    context: str,
) -> None:
    if isinstance(operand, GraphValueRef):
        if operand.name in defined:
            return
        if isinstance(operand.type_expr, TypeDim) and operand.name in dim_symbols:
            return
        raise ValueError(f"{context} uses undefined value {operand.name!r}")
    if isinstance(operand, GraphPath):
        for name in graph_path_template_names(operand):
            if name in defined or name in dim_symbols:
                continue
            raise ValueError(f"{context} path template uses undefined value {name!r}")
        return
    if isinstance(operand, GraphExpr):
        for item in operand.inputs:
            _validate_operand_defined(
                item, defined=defined, dim_symbols=dim_symbols, context=context
            )
        for item in operand.attrs.values():
            _validate_operand_defined(
                item, defined=defined, dim_symbols=dim_symbols, context=context
            )


@lru_cache(maxsize=200_000)
def _type_compatible(actual: TypeExpr, expected: TypeExpr) -> bool:
    if (
        isinstance(actual, TypeAny | TypeVar)
        or isinstance(expected, TypeAny | TypeVar)
        or _is_type_variable_like(actual)
        or _is_type_variable_like(expected)
    ):
        return True
    if isinstance(actual, TypeInt) and isinstance(expected, TypeDim):
        return True
    if isinstance(actual, TypeDim) and isinstance(expected, TypeInt):
        return True
    if isinstance(actual, TypeDim) and isinstance(expected, TypeFloat):
        return True
    if isinstance(actual, TypeInt) and isinstance(expected, TypeFloat):
        return True
    if ast_equal(actual, expected):
        return True
    if isinstance(expected, TypeOptional):
        if isinstance(actual, TypeNull):
            return True
        return _type_compatible(actual, expected.inner)
    if isinstance(actual, TypeOptional) and isinstance(expected, TypeOptional):
        return _type_compatible(actual.inner, expected.inner)
    if isinstance(actual, TypeOptional):
        return _type_compatible(actual.inner, expected)
    if isinstance(actual, TypeTensor) and isinstance(expected, TypeTensor):
        return actual.base == expected.base and _dim_sequence_compatible(actual.dims, expected.dims)
    if isinstance(actual, TypeNamed) and isinstance(expected, TypeNamed):
        return actual.name == expected.name and _dim_sequence_compatible(actual.args, expected.args)
    if isinstance(actual, TypeList) and isinstance(expected, TypeList):
        return _type_compatible(actual.item, expected.item)
    if isinstance(actual, TypeTuple) and isinstance(expected, TypeTuple):
        return len(actual.items) == len(expected.items) and all(
            _type_compatible(actual_item, expected_item)
            for actual_item, expected_item in zip(actual.items, expected.items, strict=True)
        )
    return False


def _is_type_variable_like(type_expr: TypeExpr) -> bool:
    return isinstance(type_expr, TypeNamed) and not type_expr.args and (
        type_expr.name.startswith("_") or type_expr.name[:1].isupper()
    )


@lru_cache(maxsize=200_000)
def _simplify_dim_token(dim: DimToken) -> DimToken:
    if isinstance(dim, DimExprBinary):
        left = _simplify_dim_token(dim.left)
        right = _simplify_dim_token(dim.right)
        if type(left) is int and type(right) is int:
            if dim.op == "+":
                return left + right
            if dim.op == "-":
                return left - right
            if dim.op == "*":
                return left * right
            if dim.op == "/" and right != 0 and left % right == 0:
                return left // right
        if dim.op == "+":
            if right == 0:
                return left
            if left == 0:
                return right
        if dim.op == "-":
            if right == 0:
                return left
            if left == right:
                return 0
            if isinstance(left, DimExprBinary) and left.op == "+":
                if left.left == right:
                    return left.right
                if left.right == right:
                    return left.left
            if isinstance(left, DimExprBinary) and left.op == "*":
                if left.left == right and isinstance(left.right, int):
                    remaining = left.right - 1
                    if remaining == 0:
                        return 0
                    if remaining == 1:
                        return right
                    return DimExprBinary(op="*", left=remaining, right=right)
                if left.right == right and isinstance(left.left, int):
                    remaining = left.left - 1
                    if remaining == 0:
                        return 0
                    if remaining == 1:
                        return right
                    return DimExprBinary(op="*", left=remaining, right=right)
        if dim.op == "*":
            if right == 1:
                return left
            if left == 1:
                return right
            if right == 0 or left == 0:
                return 0
            if isinstance(right, DimExprBinary) and right.op == "/" and right.right == left:
                return right.left
            if isinstance(left, DimExprBinary) and left.op == "/" and left.right == right:
                return left.left
        if dim.op == "/":
            if right == 1:
                return left
            if isinstance(left, DimExprBinary) and left.op == "*" and left.right == right:
                return left.left
            if isinstance(left, DimExprBinary) and left.op == "*" and left.left == right:
                return left.right
        return DimExprBinary(op=dim.op, left=left, right=right)
    return dim


@lru_cache(maxsize=200_000)
def _dim_token_compatible(actual: DimToken, expected: DimToken) -> bool:
    actual = _simplify_dim_token(actual)
    expected = _simplify_dim_token(expected)
    if ast_equal(actual, expected):
        return True
    if dim_token_names(actual) or dim_token_names(expected):
        return True
    if isinstance(actual, str) and actual.startswith(".."):
        return True
    if isinstance(expected, str) and expected.startswith(".."):
        return True
    if isinstance(actual, str) and isinstance(expected, str):
        return True
    if isinstance(actual, str):
        return True
    if isinstance(expected, str):
        return True
    if isinstance(actual, DimExprBinary) and isinstance(expected, DimExprBinary):
        return (
            actual.op == expected.op
            and _dim_token_compatible(actual.left, expected.left)
            and _dim_token_compatible(actual.right, expected.right)
        )
    return False


def _is_variadic_dim(dim: DimToken) -> bool:
    return isinstance(dim, str) and dim.startswith("..")


def _dim_sequence_compatible(
    actual: tuple[DimToken, ...],
    expected: tuple[DimToken, ...],
) -> bool:
    if not any(_is_variadic_dim(dim) for dim in actual + expected):
        return len(actual) == len(expected) and all(
            _dim_token_compatible(actual_dim, expected_dim)
            for actual_dim, expected_dim in zip(actual, expected, strict=True)
        )
    if len(expected) == 1 and _is_variadic_dim(expected[0]):
        return True
    if len(actual) == 1 and _is_variadic_dim(actual[0]):
        return True
    expected_variadic = next(
        (index for index, dim in enumerate(expected) if _is_variadic_dim(dim)),
        None,
    )
    if expected_variadic is not None:
        prefix = expected[:expected_variadic]
        suffix = expected[expected_variadic + 1 :]
        if len(actual) < len(prefix) + len(suffix):
            return False
        return _dim_sequence_compatible(actual[: len(prefix)], prefix) and _dim_sequence_compatible(
            actual[len(actual) - len(suffix) :] if suffix else (),
            suffix,
        )
    actual_variadic = next(
        (index for index, dim in enumerate(actual) if _is_variadic_dim(dim)),
        None,
    )
    if actual_variadic is not None:
        prefix = actual[:actual_variadic]
        suffix = actual[actual_variadic + 1 :]
        if len(expected) < len(prefix) + len(suffix):
            return False
        return _dim_sequence_compatible(prefix, expected[: len(prefix)]) and _dim_sequence_compatible(
            suffix,
            expected[len(expected) - len(suffix) :] if suffix else (),
        )
    return False


def _require_type_compatible(
    actual: TypeExpr,
    expected: TypeExpr,
    *,
    context: str,
) -> None:
    if not _type_compatible(actual, expected):
        raise ValueError(
            f"{context}: expected {render_type(expected)}, got {render_type(actual)}"
        )


def _require_actual_compatible_with_formal(
    actual: TypeExpr,
    formal: GraphValue,
    *,
    context: str,
) -> None:
    if formal.optional and isinstance(actual, TypeNull):
        return
    _require_type_compatible(actual, formal.type_expr, context=context)


def _bind_dim_substitution_from_types(
    formal: TypeExpr,
    actual: TypeExpr,
    subst: dict[str, DimToken],
) -> None:
    if isinstance(formal, TypeTensor) and isinstance(actual, TypeTensor) and formal.base == actual.base:
        _bind_dim_substitution_from_sequences(formal.dims, actual.dims, subst)
        return
    if isinstance(formal, TypeNamed) and isinstance(actual, TypeNamed) and formal.name == actual.name:
        _bind_dim_substitution_from_sequences(formal.args, actual.args, subst)
        return
    if isinstance(formal, TypeList) and isinstance(actual, TypeList):
        _bind_dim_substitution_from_types(formal.item, actual.item, subst)
        return
    if isinstance(formal, TypeOptional) and isinstance(actual, TypeOptional):
        _bind_dim_substitution_from_types(formal.inner, actual.inner, subst)
        return
    if isinstance(formal, TypeTuple) and isinstance(actual, TypeTuple) and len(formal.items) == len(actual.items):
        for formal_item, actual_item in zip(formal.items, actual.items, strict=True):
            _bind_dim_substitution_from_types(formal_item, actual_item, subst)


def _bind_dim_substitution_from_sequences(
    formal_dims: tuple[DimToken, ...],
    actual_dims: tuple[DimToken, ...],
    subst: dict[str, DimToken],
) -> None:
    if len(formal_dims) != len(actual_dims):
        return
    for formal_dim, actual_dim in zip(formal_dims, actual_dims, strict=True):
        _bind_dim_substitution_from_dim(formal_dim, actual_dim, subst)


def _bind_dim_substitution_from_dim(
    formal_dim: DimToken,
    actual_dim: DimToken,
    subst: dict[str, DimToken],
) -> None:
    if isinstance(formal_dim, str):
        existing = subst.get(formal_dim)
        if existing is None:
            subst[formal_dim] = actual_dim
        elif existing != actual_dim:
            # Keep the original binding; the type compatibility check reports shape mismatch.
            return
        return
    if isinstance(formal_dim, DimExprBinary) and isinstance(actual_dim, DimExprBinary):
        if formal_dim.op != actual_dim.op:
            return
        _bind_dim_substitution_from_dim(formal_dim.left, actual_dim.left, subst)
        _bind_dim_substitution_from_dim(formal_dim.right, actual_dim.right, subst)


def _operand_dim_token_for_validation(
    operand: GraphOperand,
    dim_values: Mapping[str, DimToken],
) -> DimToken | None:
    if isinstance(operand, GraphLiteral) and type(operand.value) is int:
        return operand.value
    if isinstance(operand, GraphValueRef):
        if operand.name in dim_values:
            return dim_values[operand.name]
        if isinstance(operand.type_expr, TypeDim | TypeInt):
            return operand.name
        return None
    if (
        isinstance(operand, GraphExpr)
        and operand.op.name.startswith("core.binary.")
        and len(operand.inputs) == 2
        and isinstance(operand.type_expr, TypeDim | TypeInt)
    ):
        op = operand.op.name.removeprefix("core.binary.")
        if op not in {"+", "-", "*", "/"}:
            return None
        left = _operand_dim_token_for_validation(operand.inputs[0], dim_values)
        right = _operand_dim_token_for_validation(operand.inputs[1], dim_values)
        if left is None or right is None:
            return None
        return _substitute_dim_token(DimExprBinary(op=op, left=left, right=right), {})
    return None


def _call_dim_substitution(
    callee: GraphModule,
    actuals: tuple[GraphOperand, ...],
    dim_values: Mapping[str, DimToken] | None = None,
) -> dict[str, DimToken]:
    subst: dict[str, DimToken] = {}
    dim_values = dim_values or {}
    for formal, actual in zip(callee.inputs, actuals, strict=True):
        if isinstance(formal.type_expr, TypeDim):
            actual_dim = _operand_dim_token_for_validation(actual, dim_values)
            if actual_dim is not None:
                subst[formal.name] = actual_dim
        _bind_dim_substitution_from_types(formal.type_expr, _declared_operand_type(actual), subst)
    return subst


def _substitute_dim_token(dim: DimToken, subst: dict[str, DimToken]) -> DimToken:
    if isinstance(dim, str):
        return subst.get(dim, dim)
    if isinstance(dim, DimExprBinary):
        left = _substitute_dim_token(dim.left, subst)
        right = _substitute_dim_token(dim.right, subst)
        return _simplify_dim_token(DimExprBinary(op=dim.op, left=left, right=right))
    return dim


def _substitute_type_expr(type_expr: TypeExpr, subst: dict[str, DimToken]) -> TypeExpr:
    if isinstance(type_expr, TypeTensor):
        return TypeTensor(
            type_expr.base,
            tuple(_substitute_dim_token(dim, subst) for dim in type_expr.dims),
        )
    if isinstance(type_expr, TypeNamed):
        return TypeNamed(
            type_expr.name,
            tuple(_substitute_dim_token(dim, subst) for dim in type_expr.args),
        )
    if isinstance(type_expr, TypeList):
        return TypeList(_substitute_type_expr(type_expr.item, subst))
    if isinstance(type_expr, TypeOptional):
        return TypeOptional(_substitute_type_expr(type_expr.inner, subst))
    if isinstance(type_expr, TypeTuple):
        return TypeTuple(tuple(_substitute_type_expr(item, subst) for item in type_expr.items))
    return type_expr


def _instantiated_module_output_types(
    callee: GraphModule,
    actuals: tuple[GraphOperand, ...],
    output_count: int,
    dim_values: Mapping[str, DimToken] | None = None,
) -> tuple[TypeExpr, ...]:
    subst = _call_dim_substitution(callee, actuals, dim_values=dim_values)
    if callee.return_type_expr is not None:
        raw_types = _result_types(callee.return_type_expr, output_count)
    else:
        raw_types = _module_output_types(callee)
    return tuple(_substitute_type_expr(type_expr, subst) for type_expr in raw_types)


def _declared_operand_type(operand: GraphOperand) -> TypeExpr:
    if isinstance(operand, GraphLiteral):
        return operand.type_expr
    if isinstance(operand, GraphPath):
        return TypePath()
    if isinstance(operand, GraphValueRef):
        return operand.type_expr
    if isinstance(operand, GraphExpr):
        return operand.type_expr
    raise TypeError(f"unsupported graph operand {operand!r}")


def graph_path_template_names(path: GraphPath) -> set[str]:
    names: set[str] = set()
    for part in path.parts:
        names.update(match.group(1) for match in _PATH_PLACEHOLDER_RE.finditer(part))
    return names


def graph_operand_type(operand: GraphOperand) -> TypeExpr:
    return _declared_operand_type(operand)


def graph_type_compatible(actual: TypeExpr, expected: TypeExpr) -> bool:
    return _type_compatible(actual, expected)


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
    return tuple(_declared_operand_type(output) for output in module.outputs)


def _call_actuals_for_callee(
    *,
    op_name: str,
    inputs: tuple[GraphOperand, ...],
    attrs: dict[str, GraphAttr],
    callee: GraphModule,
    context: str,
) -> tuple[GraphOperand, ...]:
    formals = callee.inputs
    if len(inputs) > len(formals):
        raise ValueError(
            f"{context}: call to {op_name!r} passes {len(inputs)} positional args "
            f"to {len(formals)} parameters"
        )
    actuals: list[GraphOperand | None] = [None] * len(formals)
    for index, operand in enumerate(inputs):
        actuals[index] = operand
    formal_by_name = {formal.name: index for index, formal in enumerate(formals)}
    for key, operand in attrs.items():
        index = formal_by_name.get(key)
        if index is None:
            raise ValueError(f"{context}: call to {op_name!r} passes unknown kwarg {key!r}")
        if actuals[index] is not None:
            raise ValueError(f"{context}: call to {op_name!r} passes duplicate arg {key!r}")
        actuals[index] = operand
    missing = [formal.name for formal, actual in zip(formals, actuals, strict=True) if actual is None]
    if missing:
        raise ValueError(
            f"{context}: call to {op_name!r} is missing args: {', '.join(missing)}"
        )
    return tuple(actual for actual in actuals if actual is not None)


def _validate_core_expr_contract(
    expr: GraphExpr,
    input_types: tuple[TypeExpr, ...],
    *,
    context: str,
) -> None:
    op_name = expr.op.name
    if op_name == "core.alias":
        if len(input_types) != 1:
            raise ValueError(f"{context}: core.alias expects one input")
        _require_type_compatible(input_types[0], expr.type_expr, context=context)
        return
    if op_name == "core.ascribe":
        if len(input_types) != 1:
            raise ValueError(f"{context}: core.ascribe expects one input")
        return
    if op_name == "core.select":
        if len(input_types) != 3:
            raise ValueError(f"{context}: core.select expects three inputs")
        _require_type_compatible(input_types[0], TypeBool(), context=f"{context} condition")
        selected_type: TypeExpr | None = None
        if isinstance(expr.inputs[0], GraphLiteral) and isinstance(expr.inputs[0].value, bool):
            selected_type = input_types[1] if expr.inputs[0].value else input_types[2]
        if selected_type is not None:
            _require_type_compatible(selected_type, expr.type_expr, context=f"{context} selected branch")
            return
        _require_type_compatible(input_types[1], expr.type_expr, context=f"{context} true branch")
        _require_type_compatible(input_types[2], expr.type_expr, context=f"{context} false branch")
        return
    if op_name == "core.repeat":
        if len(input_types) < 3:
            raise ValueError(f"{context}: core.repeat expects at least from/to/step inputs")
        _require_type_compatible(input_types[0], TypeInt(), context=f"{context} from")
        _require_type_compatible(input_types[1], TypeInt(), context=f"{context} to")
        _require_type_compatible(input_types[2], TypeInt(), context=f"{context} step")
        return
    if op_name.startswith("core.binary."):
        if len(input_types) != 2:
            raise ValueError(f"{context}: {op_name} expects two inputs")
        return
    if op_name == "core.tuple":
        if isinstance(expr.type_expr, TypeTuple) and len(expr.type_expr.items) == len(input_types):
            for index, (actual, expected) in enumerate(zip(input_types, expr.type_expr.items, strict=True)):
                _require_type_compatible(actual, expected, context=f"{context} tuple item {index}")
        return
    if op_name == "core.list":
        if isinstance(expr.type_expr, TypeList):
            for index, actual in enumerate(input_types):
                _require_type_compatible(actual, expr.type_expr.item, context=f"{context} list item {index}")
        return


def _repeat_attr_string(node: GraphNode, key: str, *, context: str) -> str:
    value = node.attrs.get(key)
    if not isinstance(value, GraphLiteral) or not isinstance(value.value, str):
        raise ValueError(f"{context}: core.repeat attr {key!r} must be a string literal")
    return value.value


def _repeat_attr_int(node: GraphNode, key: str, *, context: str) -> int:
    value = node.attrs.get(key)
    if not isinstance(value, GraphLiteral) or type(value.value) is not int:
        raise ValueError(f"{context}: core.repeat attr {key!r} must be an int literal")
    return value.value


def _validate_core_repeat_node(
    node: GraphNode,
    *,
    env: dict[str, GraphValue],
    globals_env: dict[str, GraphValue],
    dim_symbols: set[str],
    modules_by_name: dict[str, GraphModule],
    context: str,
) -> None:
    if len(node.inputs) < 3:
        raise ValueError(f"{context}: core.repeat expects at least from/to/step inputs")
    callee_name = _repeat_attr_string(node, "callee", context=context)
    arg_count = _repeat_attr_int(node, "arg_count", context=context)
    carry_count = _repeat_attr_int(node, "carry_count", context=context)
    if carry_count != len(node.outputs):
        raise ValueError(f"{context}: core.repeat carry/output arity mismatch")
    if len(node.inputs) < 3 + carry_count:
        raise ValueError(f"{context}: core.repeat missing carry inputs")
    callee = modules_by_name.get(callee_name)
    if callee is None:
        raise ValueError(f"{context}: core.repeat references unknown callee {callee_name!r}")
    if len(callee.inputs) != arg_count:
        raise ValueError(f"{context}: core.repeat call arity mismatch for {callee_name!r}")
    actuals: list[GraphOperand] = []
    for index in range(arg_count):
        role = _repeat_attr_string(node, f"arg_{index}", context=context)
        if role == "iter":
            actuals.append(GraphLiteral(0, TypeInt()))
            continue
        if role.startswith("carry:"):
            carry_index = int(role.removeprefix("carry:"))
            if carry_index < 0 or carry_index >= carry_count:
                raise ValueError(f"{context}: invalid core.repeat carry role {role!r}")
            actuals.append(node.inputs[3 + carry_index])
            continue
        if role.startswith("input:"):
            input_index = int(role.removeprefix("input:"))
            if input_index < 0 or input_index >= len(node.inputs):
                raise ValueError(f"{context}: invalid core.repeat input role {role!r}")
            actuals.append(node.inputs[input_index])
            continue
        raise ValueError(f"{context}: invalid core.repeat arg role {role!r}")
    for formal, actual in zip(callee.inputs, actuals, strict=True):
        actual_type = _operand_type_checked(
            actual,
            env=env,
            globals_env=globals_env,
            dim_symbols=dim_symbols,
            modules_by_name=modules_by_name,
            context=f"{context} arg {formal.name!r}",
        )
        _require_actual_compatible_with_formal(
            actual_type,
            formal,
            context=f"{context} arg {formal.name!r}",
        )
    expected_types = _instantiated_module_output_types(callee, tuple(actuals), len(node.outputs))
    if len(expected_types) != len(node.outputs):
        raise ValueError(f"{context}: core.repeat output arity mismatch")
    for index, (output, expected) in enumerate(zip(node.outputs, expected_types, strict=True)):
        _require_type_compatible(
            output.type_expr,
            expected,
            context=f"{context} output {index}",
        )


def _operand_type_checked(
    operand: GraphOperand,
    *,
    env: dict[str, GraphValue],
    globals_env: dict[str, GraphValue],
    dim_symbols: set[str],
    modules_by_name: dict[str, GraphModule],
    context: str,
) -> TypeExpr:
    if isinstance(operand, GraphLiteral):
        return operand.type_expr
    if isinstance(operand, GraphPath):
        for name in graph_path_template_names(operand):
            if name in set(env) | set(globals_env) or name in dim_symbols:
                continue
            raise ValueError(f"{context} path template uses undefined value {name!r}")
        return TypePath()
    if isinstance(operand, GraphValueRef):
        value = env.get(operand.name) or globals_env.get(operand.name)
        if value is not None:
            value_type = _value_ref_type(value)
            _require_type_compatible(operand.type_expr, value_type, context=context)
            if (
                operand.dims is not None
                and value.dims is not None
                and not _dim_sequence_compatible(operand.dims, value.dims)
            ):
                raise ValueError(
                    f"{context}: stale dims for {operand.name!r}: "
                    f"expected {value.dims!r}, got {operand.dims!r}"
                )
            return value_type
        if isinstance(operand.type_expr, TypeDim) and operand.name in dim_symbols:
            return operand.type_expr
        raise ValueError(f"{context} uses undefined value {operand.name!r}")
    if isinstance(operand, GraphExpr):
        input_types = tuple(
            _operand_type_checked(
                item,
                env=env,
                globals_env=globals_env,
                dim_symbols=dim_symbols,
                modules_by_name=modules_by_name,
                context=f"{context} operand",
            )
            for item in operand.inputs
        )
        for key, item in operand.attrs.items():
            _operand_type_checked(
                item,
                env=env,
                globals_env=globals_env,
                dim_symbols=dim_symbols,
                modules_by_name=modules_by_name,
                context=f"{context} attr {key!r}",
            )
        callee = modules_by_name.get(operand.op.name)
        if callee is not None:
            actuals = _call_actuals_for_callee(
                op_name=operand.op.name,
                inputs=operand.inputs,
                attrs=operand.attrs,
                callee=callee,
                context=context,
            )
            for formal, actual in zip(callee.inputs, actuals, strict=True):
                actual_type = _operand_type_checked(
                    actual,
                    env=env,
                    globals_env=globals_env,
                    dim_symbols=dim_symbols,
                    modules_by_name=modules_by_name,
                    context=f"{context} arg {formal.name!r}",
                )
                _require_actual_compatible_with_formal(
                    actual_type,
                    formal,
                    context=f"{context} arg {formal.name!r}",
                )
            expected_output_count = len(callee.outputs)
            if expected_output_count != 1 and not (
                isinstance(operand.type_expr, TypeTuple)
                and len(operand.type_expr.items) == expected_output_count
            ):
                raise ValueError(
                    f"{context}: call to {operand.op.name!r} cannot be used as "
                    f"single expression with {expected_output_count} results"
                )
            expected_types = _instantiated_module_output_types(callee, actuals, expected_output_count)
            expected_type = (
                expected_types[0] if expected_output_count == 1 else TypeTuple(expected_types)
            )
            _require_type_compatible(operand.type_expr, expected_type, context=f"{context} call result")
            return operand.type_expr
        if operand.op.name.startswith("core."):
            _validate_core_expr_contract(operand, input_types, context=context)
        return operand.type_expr
    raise TypeError(f"unsupported graph operand {operand!r}")


def _validate_graph_module(
    module: GraphModule,
    *,
    global_values: dict[str, GraphValue] | None = None,
    modules_by_name: dict[str, GraphModule] | None = None,
) -> None:
    env = {value.name: value for value in module.inputs}
    defined = set(env)
    globals_env = dict(global_values or {})
    dim_values: dict[str, DimToken] = {}
    dim_symbols = _module_boundary_dim_symbols(module)
    for value in globals_env.values():
        if isinstance(value.type_expr, TypeDim):
            dim_symbols.add(value.name)
        dim_symbols.update(_type_dim_names(value.type_expr))
    if len(defined) != len(module.inputs):
        raise ValueError(f"graph IR module {module.name!r} has duplicate inputs")
    modules_by_name = dict(modules_by_name or {})
    typevar_names = {
        name for name, value in env.items() if isinstance(value.type_expr, TypeVar)
    }
    all_value_names = set(defined) | set(globals_env)
    all_constraint_dim_symbols = set(dim_symbols)
    for node in module.nodes:
        all_value_names.update(output.name for output in node.outputs)
        all_constraint_dim_symbols.update(_type_dim_names(node.type_expr))
        if node.dims is not None:
            for dim in node.dims:
                all_constraint_dim_symbols.update(dim_token_names(dim))
        for output in node.outputs:
            all_constraint_dim_symbols.update(_value_dim_names(output))
    for value in module.inputs:
        _require_value_metadata_coherent(value, context=f"graph IR module {module.name!r} input")
        _validate_value_dim_closure(
            value,
            dim_symbols=dim_symbols | _value_dim_names(value),
            context=f"graph IR module {module.name!r} input {value.name!r}",
        )
    if module.return_type_expr is not None:
        _validate_type_dim_closure(
            module.return_type_expr,
            dim_symbols=dim_symbols,
            context=f"graph IR module {module.name!r} return type",
        )
    for constraint in module.constraints:
        if constraint.relation == "=" and constraint.left == constraint.right:
            continue
        if any(guard.relation == "callsite" for guard in constraint.guards):
            continue
        _validate_dim_names(
            _constraint_names(constraint),
            dim_symbols=all_constraint_dim_symbols | all_value_names,
            context=f"graph IR module {module.name!r} constraint",
        )
    for node in module.nodes:
        node_local_dim_symbols = set(dim_symbols)
        node_local_dim_symbols.update(_type_dim_names(node.type_expr))
        if node.dims is not None:
            for dim in node.dims:
                node_local_dim_symbols.update(dim_token_names(dim))
        for output in node.outputs:
            node_local_dim_symbols.update(_value_dim_names(output))
        for operand in node.inputs:
            _validate_operand_defined(
                operand,
                defined=defined | set(globals_env),
                dim_symbols=dim_symbols,
                context=f"graph IR node {node.id!r}",
            )
            _require_operand_metadata_coherent(operand, context=f"graph IR node {node.id!r} input")
            _validate_operand_dim_closure(
                operand,
                dim_symbols=dim_symbols,
                context=f"graph IR node {node.id!r} input",
                typevar_names=typevar_names,
                defined_names=defined | set(globals_env),
            )
        for operand in node.attrs.values():
            _validate_operand_defined(
                operand,
                defined=defined | set(globals_env),
                dim_symbols=dim_symbols,
                context=f"graph IR node {node.id!r}",
            )
            _require_operand_metadata_coherent(operand, context=f"graph IR node {node.id!r} attr")
            _validate_operand_dim_closure(
                operand,
                dim_symbols=dim_symbols,
                context=f"graph IR node {node.id!r} attr",
                typevar_names=typevar_names,
                defined_names=defined | set(globals_env),
            )
        _validate_type_dim_closure(
            node.type_expr,
            dim_symbols=node_local_dim_symbols,
            context=f"graph IR node {node.id!r} type",
        )
        if (
            isinstance(node.type_expr, TypeTensor)
            and node.dims is not None
            and not _dims_metadata_compatible(node.type_expr.dims, node.dims)
        ):
            raise ValueError(
                f"graph IR node {node.id!r} has stale dims metadata: "
                f"type has {node.type_expr.dims!r}, dims has {node.dims!r}"
            )
        if node.dims is not None:
            names: set[str] = set()
            for dim in node.dims:
                names.update(dim_token_names(dim))
            _validate_dim_names(
                names,
                dim_symbols=node_local_dim_symbols,
                context=f"graph IR node {node.id!r} dims",
            )
        input_types = tuple(
            _operand_type_checked(
                operand,
                env=env,
                globals_env=globals_env,
                dim_symbols=dim_symbols,
                modules_by_name=modules_by_name,
                context=f"graph IR node {node.id!r} input",
            )
            for operand in node.inputs
        )
        for key, operand in node.attrs.items():
            _operand_type_checked(
                operand,
                env=env,
                globals_env=globals_env,
                dim_symbols=dim_symbols,
                modules_by_name=modules_by_name,
                context=f"graph IR node {node.id!r} attr {key!r}",
            )
        expected_node_types = _result_types(node.type_expr, len(node.outputs))
        callee = modules_by_name.get(node.op.name)
        if callee is not None:
            actuals = _call_actuals_for_callee(
                op_name=node.op.name,
                inputs=node.inputs,
                attrs=node.attrs,
                callee=callee,
                context=f"graph IR node {node.id!r}",
            )
            for formal, actual in zip(callee.inputs, actuals, strict=True):
                actual_type = _operand_type_checked(
                    actual,
                    env=env,
                    globals_env=globals_env,
                    dim_symbols=dim_symbols,
                    modules_by_name=modules_by_name,
                    context=f"graph IR node {node.id!r} arg {formal.name!r}",
                )
                _require_actual_compatible_with_formal(
                    actual_type,
                    formal,
                    context=f"graph IR node {node.id!r} arg {formal.name!r}",
                )
            expected_callee_types = _instantiated_module_output_types(
                callee,
                actuals,
                len(node.outputs),
                dim_values=dim_values,
            )
            if len(node.outputs) != len(expected_callee_types):
                raise ValueError(
                    f"graph IR node {node.id!r}: call to {node.op.name!r} result arity mismatch, "
                    f"expected {len(expected_callee_types)}, got {len(node.outputs)}"
                )
            for index, (output, expected_type) in enumerate(
                zip(node.outputs, expected_callee_types, strict=True)
            ):
                _require_type_compatible(
                    output.type_expr,
                    expected_type,
                    context=f"graph IR node {node.id!r} call result {index} stale type",
                )
        elif node.op.name == "core.repeat":
            _validate_core_repeat_node(
                node,
                env=env,
                globals_env=globals_env,
                dim_symbols=dim_symbols,
                modules_by_name=modules_by_name,
                context=f"graph IR node {node.id!r}",
            )
        elif node.op.name.startswith("core."):
            _validate_core_expr_contract(
                GraphExpr(
                    op=node.op,
                    inputs=node.inputs,
                    attrs=node.attrs,
                    type_expr=node.type_expr,
                    dims=node.dims,
                ),
                input_types,
                context=f"graph IR node {node.id!r}",
            )
        for output_index, output in enumerate(node.outputs):
            if output.name in defined:
                raise ValueError(
                    f"graph IR node {node.id!r} redefines value {output.name!r}"
                )
            _require_value_metadata_coherent(output, context=f"graph IR node {node.id!r} output")
            local_dim_symbols = dim_symbols | ({output.name} if isinstance(output.type_expr, TypeDim) else set())
            _validate_value_dim_closure(
                output,
                dim_symbols=node_local_dim_symbols | local_dim_symbols,
                context=f"graph IR node {node.id!r} output {output.name!r}",
            )
            if output_index < len(expected_node_types):
                expected_output_type = expected_node_types[output_index]
                _require_type_compatible(
                    output.type_expr,
                    expected_output_type,
                    context=f"graph IR node {node.id!r} output {output.name!r}",
                )
            defined.add(output.name)
            env[output.name] = output
            if isinstance(output.type_expr, TypeDim):
                dim_symbols.add(output.name)
            dim_symbols.update(_value_dim_names(output))
            if len(node.outputs) == 1 and isinstance(output.type_expr, TypeDim | TypeInt):
                dim_value = _operand_dim_token_for_validation(
                    GraphExpr(
                        op=node.op,
                        inputs=node.inputs,
                        attrs=node.attrs,
                        type_expr=node.type_expr,
                        dims=node.dims,
                    ),
                    dim_values,
                )
                if dim_value is not None:
                    dim_values[output.name] = dim_value
    for operand in module.outputs:
        _validate_operand_defined(
            operand,
            defined=defined | set(globals_env),
            dim_symbols=dim_symbols,
            context=f"graph IR module {module.name!r} return",
        )
        _require_operand_metadata_coherent(operand, context=f"graph IR module {module.name!r} return")
        _validate_operand_dim_closure(
            operand,
            dim_symbols=dim_symbols,
            context=f"graph IR module {module.name!r} return",
            typevar_names=typevar_names,
            defined_names=defined | set(globals_env),
        )
    output_types = tuple(
        _operand_type_checked(
            operand,
            env=env,
            globals_env=globals_env,
            dim_symbols=dim_symbols,
            modules_by_name=modules_by_name,
            context=f"graph IR module {module.name!r} return",
        )
        for operand in module.outputs
    )
    expected_return_types = _module_output_types(module)
    if len(output_types) != len(expected_return_types):
        raise ValueError(
            f"graph IR module {module.name!r}: return arity mismatch, "
            f"expected {len(expected_return_types)}, got {len(output_types)}"
        )
    for index, (actual, expected) in enumerate(zip(output_types, expected_return_types, strict=True)):
        _require_type_compatible(
            actual,
            expected,
            context=f"graph IR module {module.name!r} return {index}",
        )


def validate_graph_program(program: GraphProgram) -> None:
    names = [module.name for module in program.modules]
    if len(set(names)) != len(names):
        raise ValueError("graph IR program has duplicate module names")
    if program.main_module not in set(names):
        raise ValueError(f"graph IR main module {program.main_module!r} is missing")
    modules_by_name = {module.name: module for module in program.modules}
    global_values = {
        module.name: GraphValue(
            name=module.name,
            type_expr=_module_output_types(module)[0],
            dims=None,
        )
        for module in program.modules
        if not module.inputs and len(module.outputs) == 1
    }
    for module in program.modules:
        _validate_graph_module(
            module,
            global_values=global_values,
            modules_by_name=modules_by_name,
        )


__all__ = [
    "GraphAttr",
    "GraphLiteral",
    "GraphExpr",
    "GraphModule",
    "GraphNode",
    "GraphOperand",
    "GraphOp",
    "GraphPath",
    "GraphProgram",
    "GraphValue",
    "GraphValueRef",
    "graph_operand_type",
    "graph_path_template_names",
    "graph_type_compatible",
    "lower_axon_program_to_graph_ir",
    "validate_graph_program",
]
