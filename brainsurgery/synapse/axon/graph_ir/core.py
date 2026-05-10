from __future__ import annotations

from dataclasses import dataclass
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
    AxonReturn,
    Constraint,
    DimToken,
    TypeAny,
    TypeBool,
    TypeDim,
    TypeExpr,
    TypeFloat,
    TypeInt,
    TypeList,
    TypeNull,
    TypeString,
    TypeTensor,
    TypeTuple,
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
    default: "GraphOperand | None" = None


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


GraphOperand: TypeAlias = (
    GraphValueRef | GraphLiteral | GraphPath | GraphExpr | tuple["GraphOperand", ...]
)
GraphAttr: TypeAlias = GraphOperand


@dataclass(frozen=True)
class GraphNode:
    id: str
    op: GraphOp
    inputs: tuple[GraphOperand, ...]
    attrs: dict[str, GraphAttr]
    outputs: tuple[GraphValue, ...]
    source_module: str


@dataclass(frozen=True)
class GraphModule:
    name: str
    inputs: tuple[GraphValue, ...]
    outputs: tuple[GraphOperand, ...]
    output_names: tuple[str, ...]
    nodes: tuple[GraphNode, ...]
    constraints: tuple[Constraint, ...] = ()


@dataclass(frozen=True)
class GraphProgram:
    modules: tuple[GraphModule, ...]
    main_module: str
    pragmas: dict[str, object]


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
        return _expr_to_operand(expr.expr, ctx)
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
        return tuple(_expr_to_operand(item, ctx) for item in expr.items)
    raise ValueError(
        f"graph IR lowering requires flat atomic operands, got {type(expr).__name__}"
    )


def _graph_ref(value: GraphValue) -> GraphValueRef:
    return GraphValueRef(name=value.name, type_expr=value.type_expr, dims=value.dims)


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
        return tuple(_kwarg_to_attr(item, ctx) for item in value)
    raise ValueError(f"unsupported graph IR attr value {value!r}")


def _param_to_value(param: AxonParam) -> GraphValue:
    if param.type_expr is None:
        raise ValueError(f"graph IR lowering requires typed parameter {param.name!r}")
    return GraphValue(
        name=param.name,
        type_expr=param.type_expr,
        dims=tuple(param.type_expr.dims) if isinstance(param.type_expr, TypeTensor) else None,
        optional=param.optional or param.default_expr is not None,
        default=_expr_to_operand(param.default_expr) if param.default_expr is not None else None,
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
        return _lower_expr_to_operand(expr.expr, ctx)
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
        return tuple(_lower_expr_to_operand(item, ctx) for item in expr.items)
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
        return _lower_expr_to_lazy_operand(expr.expr, ctx)
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
        return GraphExpr(
            op=GraphOp(expr.callee),
            inputs=tuple(_lower_expr_to_lazy_operand(arg, ctx) for arg in expr.args),
            attrs={key: _kwarg_to_attr(value, ctx) for key, value in expr.kwargs.items()},
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
    if isinstance(expr, AxonExprAscribe):
        return _node_for_expr(
            expr.expr,
            targets=targets,
            node_id=node_id,
            module_name=module_name,
            ctx=ctx,
        )
    types = _target_types(expr, len(targets))
    dims = _target_dims(expr, len(targets))

    def _outputs() -> tuple[GraphValue, ...]:
        graph_targets = tuple(ctx.define_target(target) for target in targets)
        return tuple(
            GraphValue(name=target, type_expr=type_expr, dims=dim)
            for target, type_expr, dim in zip(graph_targets, types, dims, strict=True)
        )

    if isinstance(expr, AxonExprCall):
        inputs = tuple(_lower_expr_to_operand(arg, ctx) for arg in expr.args)
        attrs = {key: _kwarg_to_attr(value, ctx) for key, value in expr.kwargs.items()}
        return GraphNode(
            id=node_id,
            op=GraphOp(expr.callee),
            inputs=inputs,
            attrs=attrs,
            outputs=_outputs(),
            source_module=module_name,
        )
    if isinstance(expr, AxonExprBinary):
        inputs = (
            _lower_expr_to_operand(expr.left, ctx),
            _lower_expr_to_operand(expr.right, ctx),
        )
        return GraphNode(
            id=node_id,
            op=GraphOp(f"core.binary.{expr.op}"),
            inputs=inputs,
            attrs={},
            outputs=_outputs(),
            source_module=module_name,
        )
    if isinstance(expr, AxonExprTernary):
        inputs = (
            _lower_expr_to_operand(expr.cond, ctx),
            _lower_expr_to_lazy_operand(expr.true_expr, ctx),
            _lower_expr_to_lazy_operand(expr.false_expr, ctx),
        )
        return GraphNode(
            id=node_id,
            op=GraphOp("core.select"),
            inputs=inputs,
            attrs={},
            outputs=_outputs(),
            source_module=module_name,
        )
    if isinstance(expr, AxonExprList | AxonExprTuple):
        inputs = tuple(_lower_expr_to_operand(item, ctx) for item in expr.items)
        return GraphNode(
            id=node_id,
            op=GraphOp("core.list" if isinstance(expr, AxonExprList) else "core.tuple"),
            inputs=inputs,
            attrs={},
            outputs=_outputs(),
            source_module=module_name,
        )
    inputs = (_lower_expr_to_operand(expr, ctx),)
    return GraphNode(
        id=node_id,
        op=GraphOp("core.alias"),
        inputs=inputs,
        attrs={},
        outputs=_outputs(),
        source_module=module_name,
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
        if isinstance(stmt, AxonReturn):
            outputs = tuple(_lower_expr_to_operand(value, ctx) for value in stmt.values)
            continue
        raise ValueError(
            f"graph IR lowering requires backend-required flat Axon; "
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
        constraints=module.constraints or (),
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


def _module_dim_symbols(module: GraphModule) -> set[str]:
    names: set[str] = set()
    for value in module.inputs:
        names.update(_type_dim_names(value.type_expr))
        if value.dims is not None:
            for dim in value.dims:
                names.update(dim_token_names(dim))
    for node in module.nodes:
        for value in node.outputs:
            names.update(_type_dim_names(value.type_expr))
            if value.dims is not None:
                for dim in value.dims:
                    names.update(dim_token_names(dim))
    return names


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
    if isinstance(operand, tuple):
        for item in operand:
            _validate_operand_defined(
                item, defined=defined, dim_symbols=dim_symbols, context=context
            )
    if isinstance(operand, GraphExpr):
        for item in operand.inputs:
            _validate_operand_defined(
                item, defined=defined, dim_symbols=dim_symbols, context=context
            )
        for item in operand.attrs.values():
            _validate_operand_defined(
                item, defined=defined, dim_symbols=dim_symbols, context=context
            )


def _validate_graph_module(module: GraphModule, *, global_names: set[str] | None = None) -> None:
    defined = {value.name for value in module.inputs}
    globals_defined = set(global_names or ())
    dim_symbols = _module_dim_symbols(module)
    if len(defined) != len(module.inputs):
        raise ValueError(f"graph IR module {module.name!r} has duplicate inputs")
    for node in module.nodes:
        for operand in node.inputs:
            _validate_operand_defined(
                operand,
                defined=defined | globals_defined,
                dim_symbols=dim_symbols,
                context=f"graph IR node {node.id!r}",
            )
        for operand in node.attrs.values():
            _validate_operand_defined(
                operand,
                defined=defined | globals_defined,
                dim_symbols=dim_symbols,
                context=f"graph IR node {node.id!r}",
            )
        for output in node.outputs:
            if output.name in defined:
                raise ValueError(
                    f"graph IR node {node.id!r} redefines value {output.name!r}"
                )
            defined.add(output.name)
    for operand in module.outputs:
        _validate_operand_defined(
            operand,
            defined=defined | globals_defined,
            dim_symbols=dim_symbols,
            context=f"graph IR module {module.name!r} return",
        )


def validate_graph_program(program: GraphProgram) -> None:
    names = [module.name for module in program.modules]
    if len(set(names)) != len(names):
        raise ValueError("graph IR program has duplicate module names")
    if program.main_module not in set(names):
        raise ValueError(f"graph IR main module {program.main_module!r} is missing")
    global_names = {
        module.name
        for module in program.modules
        if not module.inputs and len(module.outputs) == 1
    }
    for module in program.modules:
        _validate_graph_module(module, global_names=global_names)


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
    "lower_axon_program_to_graph_ir",
    "validate_graph_program",
]
