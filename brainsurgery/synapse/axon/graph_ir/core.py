from __future__ import annotations

import re
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
        inputs = tuple(_lower_expr_to_operand(arg, ctx) for arg in core_expr.args)
        attrs = {key: _kwarg_to_attr(value, ctx) for key, value in core_expr.kwargs.items()}
        return GraphNode(
            id=node_id,
            op=GraphOp(core_expr.callee),
            inputs=inputs,
            attrs=attrs,
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


def _dim_token_compatible(actual: DimToken, expected: DimToken) -> bool:
    if ast_equal(actual, expected):
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
            _require_type_compatible(operand.type_expr, value.type_expr, context=context)
            if (
                operand.dims is not None
                and value.dims is not None
                and not _dim_sequence_compatible(operand.dims, value.dims)
            ):
                raise ValueError(
                    f"{context}: stale dims for {operand.name!r}: "
                    f"expected {value.dims!r}, got {operand.dims!r}"
                )
            return value.type_expr
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
    dim_symbols = _module_dim_symbols(module)
    if len(defined) != len(module.inputs):
        raise ValueError(f"graph IR module {module.name!r} has duplicate inputs")
    modules_by_name = dict(modules_by_name or {})
    for node in module.nodes:
        for operand in node.inputs:
            _validate_operand_defined(
                operand,
                defined=defined | set(globals_env),
                dim_symbols=dim_symbols,
                context=f"graph IR node {node.id!r}",
            )
        for operand in node.attrs.values():
            _validate_operand_defined(
                operand,
                defined=defined | set(globals_env),
                dim_symbols=dim_symbols,
                context=f"graph IR node {node.id!r}",
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
            if output_index < len(expected_node_types):
                expected_output_type = expected_node_types[output_index]
                _require_type_compatible(
                    output.type_expr,
                    expected_output_type,
                    context=f"graph IR node {node.id!r} output {output.name!r}",
                )
            defined.add(output.name)
            env[output.name] = output
    for operand in module.outputs:
        _validate_operand_defined(
            operand,
            defined=defined | set(globals_env),
            dim_symbols=dim_symbols,
            context=f"graph IR module {module.name!r} return",
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
