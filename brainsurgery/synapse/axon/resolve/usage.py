from __future__ import annotations

from dataclasses import dataclass

from ..ast.nodes import (
    AxonBind,
    AxonExpr,
    AxonExprBinary,
    AxonExprBind,
    AxonExprCall,
    AxonExprDo,
    AxonExprIf,
    AxonExprLambda,
    AxonExprList,
    AxonExprName,
    AxonExprParen,
    AxonExprPipe,
    AxonExprTernary,
    AxonExprTuple,
    AxonModule,
    AxonRepeat,
    AxonReturn,
    AxonScopeBind,
    AxonStatement,
    AxonYield,
)
from ..ast.types import TypeExpr, TypeList, TypeNamed, TypeOptional, TypeTensor, TypeTuple, TypeVar


@dataclass(frozen=True)
class ImportUsage:
    qualified_namespaces: frozenset[str]
    unqualified_symbols: frozenset[str]


def _track_import_usage_expr(
    expr: AxonExpr,
    *,
    bound_names: set[str],
    qualified_namespaces: set[str],
    unqualified_symbols: set[str],
) -> None:
    if isinstance(expr, AxonExprName):
        if "." in expr.name:
            namespace, _ = expr.name.rsplit(".", 1)
            qualified_namespaces.add(namespace)
        elif expr.name not in bound_names:
            unqualified_symbols.add(expr.name)
        return
    if isinstance(expr, AxonExprParen):
        _track_import_usage_expr(
            expr.inner,
            bound_names=bound_names,
            qualified_namespaces=qualified_namespaces,
            unqualified_symbols=unqualified_symbols,
        )
        return
    if isinstance(expr, AxonExprList):
        for item in expr.items:
            _track_import_usage_expr(
                item,
                bound_names=bound_names,
                qualified_namespaces=qualified_namespaces,
                unqualified_symbols=unqualified_symbols,
            )
        return
    if isinstance(expr, AxonExprTuple):
        for item in expr.items:
            _track_import_usage_expr(
                item,
                bound_names=bound_names,
                qualified_namespaces=qualified_namespaces,
                unqualified_symbols=unqualified_symbols,
            )
        return
    if isinstance(expr, AxonExprPipe):
        _track_import_usage_expr(
            expr.value,
            bound_names=bound_names,
            qualified_namespaces=qualified_namespaces,
            unqualified_symbols=unqualified_symbols,
        )
        for stage in expr.stages:
            _track_import_usage_expr(
                stage,
                bound_names=bound_names,
                qualified_namespaces=qualified_namespaces,
                unqualified_symbols=unqualified_symbols,
            )
        return
    if isinstance(expr, AxonExprBind):
        _track_import_usage_expr(
            expr.value,
            bound_names=bound_names,
            qualified_namespaces=qualified_namespaces,
            unqualified_symbols=unqualified_symbols,
        )
        nested_bound = set(bound_names)
        nested_bound.add(expr.var)
        _track_import_usage_expr(
            expr.body,
            bound_names=nested_bound,
            qualified_namespaces=qualified_namespaces,
            unqualified_symbols=unqualified_symbols,
        )
        return
    if isinstance(expr, AxonExprIf | AxonExprTernary):
        for subexpr in (expr.cond, expr.true_expr, expr.false_expr):
            _track_import_usage_expr(
                subexpr,
                bound_names=bound_names,
                qualified_namespaces=qualified_namespaces,
                unqualified_symbols=unqualified_symbols,
            )
        return
    if isinstance(expr, AxonExprBinary):
        for subexpr in (expr.left, expr.right):
            _track_import_usage_expr(
                subexpr,
                bound_names=bound_names,
                qualified_namespaces=qualified_namespaces,
                unqualified_symbols=unqualified_symbols,
            )
        return
    if isinstance(expr, AxonExprCall):
        base = expr.callee.split("@", 1)[0]
        if "." in base:
            namespace, _ = base.rsplit(".", 1)
            qualified_namespaces.add(namespace)
        elif base not in bound_names:
            unqualified_symbols.add(base)
        for arg in expr.args:
            _track_import_usage_expr(
                arg,
                bound_names=bound_names,
                qualified_namespaces=qualified_namespaces,
                unqualified_symbols=unqualified_symbols,
            )
        for kwarg in expr.kwargs.values():
            if isinstance(kwarg, AxonExpr):
                _track_import_usage_expr(
                    kwarg,
                    bound_names=bound_names,
                    qualified_namespaces=qualified_namespaces,
                    unqualified_symbols=unqualified_symbols,
                )
        return
    if isinstance(expr, AxonExprLambda):
        nested_bound = set(bound_names)
        nested_bound.add(expr.var)
        _track_import_usage_expr(
            expr.body,
            bound_names=nested_bound,
            qualified_namespaces=qualified_namespaces,
            unqualified_symbols=unqualified_symbols,
        )
        return
    if isinstance(expr, AxonExprDo):
        _track_import_usage_statements(
            expr.body,
            bound_names=set(bound_names),
            qualified_namespaces=qualified_namespaces,
            unqualified_symbols=unqualified_symbols,
        )


def _track_import_usage_statements(
    statements: tuple[AxonStatement, ...],
    *,
    bound_names: set[str],
    qualified_namespaces: set[str],
    unqualified_symbols: set[str],
) -> None:
    local_bound = set(bound_names)
    for stmt in statements:
        if isinstance(stmt, AxonBind):
            _track_import_usage_expr(
                stmt.expr,
                bound_names=local_bound,
                qualified_namespaces=qualified_namespaces,
                unqualified_symbols=unqualified_symbols,
            )
            for target in stmt.targets:
                if target != "_":
                    local_bound.add(target)
            continue
        if isinstance(stmt, AxonReturn | AxonYield):
            for value in stmt.values:
                _track_import_usage_expr(
                    value,
                    bound_names=local_bound,
                    qualified_namespaces=qualified_namespaces,
                    unqualified_symbols=unqualified_symbols,
                )
            continue
        if isinstance(stmt, AxonRepeat):
            for subexpr in (stmt.from_expr, stmt.to_expr, stmt.step_expr):
                _track_import_usage_expr(
                    subexpr,
                    bound_names=local_bound,
                    qualified_namespaces=qualified_namespaces,
                    unqualified_symbols=unqualified_symbols,
                )
            loop_bound = set(local_bound)
            loop_bound.add(stmt.var)
            if stmt.carry:
                for name in stmt.carry:
                    if name != "_":
                        loop_bound.add(name)
            _track_import_usage_statements(
                stmt.body,
                bound_names=loop_bound,
                qualified_namespaces=qualified_namespaces,
                unqualified_symbols=unqualified_symbols,
            )
            if stmt.targets:
                for target in stmt.targets:
                    if target != "_":
                        local_bound.add(target)
            continue
        if isinstance(stmt, AxonScopeBind):
            for kwarg in stmt.kwargs.values():
                if isinstance(kwarg, AxonExpr):
                    _track_import_usage_expr(
                        kwarg,
                        bound_names=local_bound,
                        qualified_namespaces=qualified_namespaces,
                        unqualified_symbols=unqualified_symbols,
                    )
            _track_import_usage_statements(
                stmt.body,
                bound_names=set(local_bound),
                qualified_namespaces=qualified_namespaces,
                unqualified_symbols=unqualified_symbols,
            )
            for target in stmt.targets:
                if target != "_":
                    local_bound.add(target)


def _track_import_usage_type(
    type_expr: TypeExpr | None,
    *,
    bound_names: set[str],
    qualified_namespaces: set[str],
    unqualified_symbols: set[str],
) -> None:
    if type_expr is None:
        return
    root = type_expr.inner if isinstance(type_expr, TypeOptional) else type_expr
    if isinstance(root, TypeNamed):
        if "." in root.name:
            namespace, _ = root.name.rsplit(".", 1)
            qualified_namespaces.add(namespace)
        elif root.name not in bound_names:
            unqualified_symbols.add(root.name)
        return
    if isinstance(root, TypeVar):
        return
    if isinstance(root, TypeList):
        _track_import_usage_type(
            root.item,
            bound_names=bound_names,
            qualified_namespaces=qualified_namespaces,
            unqualified_symbols=unqualified_symbols,
        )
        return
    if isinstance(root, TypeTuple):
        for item in root.items:
            _track_import_usage_type(
                item,
                bound_names=bound_names,
                qualified_namespaces=qualified_namespaces,
                unqualified_symbols=unqualified_symbols,
            )
        return
    if isinstance(root, TypeTensor):
        return


def collect_import_usage(modules: tuple[AxonModule, ...]) -> ImportUsage:
    qualified_namespaces: set[str] = set()
    unqualified_symbols: set[str] = set()
    for module in modules:
        bound_names = {param.name for param in module.params}
        bound_names.update(name for name in module.path_params if isinstance(name, str))
        if isinstance(module.path_param, str):
            bound_names.add(module.path_param)
        for param in module.params:
            _track_import_usage_type(
                param.type_expr,
                bound_names=bound_names,
                qualified_namespaces=qualified_namespaces,
                unqualified_symbols=unqualified_symbols,
            )
        _track_import_usage_type(
            module.return_type_expr,
            bound_names=bound_names,
            qualified_namespaces=qualified_namespaces,
            unqualified_symbols=unqualified_symbols,
        )
        _track_import_usage_statements(
            module.statements,
            bound_names=bound_names,
            qualified_namespaces=qualified_namespaces,
            unqualified_symbols=unqualified_symbols,
        )
    return ImportUsage(
        qualified_namespaces=frozenset(qualified_namespaces),
        unqualified_symbols=frozenset(unqualified_symbols),
    )


__all__ = ["ImportUsage", "collect_import_usage"]
