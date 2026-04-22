from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path

from .ast_validation import validate_axon_program
from .grammar import ParsedProgramSource, parse_program_source
from .syntax_validation import validate_parsed_program_source
from .types import (
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


@dataclass(frozen=True)
class _LoadedSyntaxFile:
    path: Path
    namespace: str | None
    parsed_source: ParsedProgramSource


@dataclass(frozen=True)
class _ImportUsage:
    qualified_namespaces: frozenset[str]
    unqualified_symbols: frozenset[str]


def _collect_expr_names(expr: AxonExpr) -> set[str]:
    names: set[str] = set()
    stack: list[AxonExpr] = [expr]
    while stack:
        current = stack.pop()
        if isinstance(current, AxonExprName):
            names.add(current.name)
            continue
        if isinstance(current, AxonExprParen):
            stack.append(current.inner)
            continue
        if isinstance(current, AxonExprList):
            stack.extend(list(current.items))
            continue
        if isinstance(current, AxonExprTuple):
            stack.extend(list(current.items))
            continue
        if isinstance(current, AxonExprPipe):
            stack.append(current.value)
            stack.extend(list(current.stages))
            continue
        if isinstance(current, AxonExprBind):
            stack.append(current.value)
            stack.append(current.body)
            continue
        if isinstance(current, AxonExprIf | AxonExprTernary):
            stack.append(current.cond)
            stack.append(current.true_expr)
            stack.append(current.false_expr)
            continue
        if isinstance(current, AxonExprBinary):
            stack.append(current.left)
            stack.append(current.right)
            continue
        if isinstance(current, AxonExprCall):
            stack.extend(list(current.args))
            for kwarg in current.kwargs.values():
                if isinstance(kwarg, AxonExpr):
                    stack.append(kwarg)
            continue
        if isinstance(current, AxonExprLambda):
            stack.append(current.body)
            continue
        if isinstance(current, AxonExprDo):
            for stmt in current.body:
                if isinstance(stmt, AxonBind):
                    stack.append(stmt.expr)
                elif isinstance(stmt, AxonReturn):
                    stack.extend(list(stmt.values))
                elif isinstance(stmt, AxonRepeat):
                    stack.append(stmt.from_expr)
                    stack.append(stmt.to_expr)
                    stack.append(stmt.step_expr)
                    for body_stmt in stmt.body:
                        if isinstance(body_stmt, AxonBind):
                            stack.append(body_stmt.expr)
                        elif isinstance(body_stmt, AxonReturn):
                            stack.extend(list(body_stmt.values))
                elif isinstance(stmt, AxonScopeBind):
                    for kwarg in stmt.kwargs.values():
                        if isinstance(kwarg, AxonExpr):
                            stack.append(kwarg)
                    for body_stmt in stmt.body:
                        if isinstance(body_stmt, AxonBind):
                            stack.append(body_stmt.expr)
                        elif isinstance(body_stmt, AxonReturn):
                            stack.extend(list(body_stmt.values))
            continue
    return names


def _collect_constant_closure(
    *,
    constants: dict[str, AxonExpr],
    seed_names: set[str],
) -> dict[str, AxonExpr]:
    closure: dict[str, AxonExpr] = {}
    order_index = {name: idx for idx, name in enumerate(constants.keys())}
    visiting: set[str] = set()
    visited: set[str] = set()

    def _visit(name: str) -> None:
        if name in visited:
            return
        if name in visiting:
            return
        expr = constants.get(name)
        if expr is None:
            return
        visiting.add(name)
        deps = [dep for dep in _collect_expr_names(expr) if dep in constants]
        deps.sort(key=lambda dep: order_index.get(dep, 10**9))
        for dep in deps:
            _visit(dep)
        visiting.remove(name)
        visited.add(name)
        closure[name] = expr

    ordered_seeds = [name for name in constants.keys() if name in seed_names]
    for name in ordered_seeds:
        _visit(name)
    return closure


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
        _track_import_usage_expr(
            expr.cond,
            bound_names=bound_names,
            qualified_namespaces=qualified_namespaces,
            unqualified_symbols=unqualified_symbols,
        )
        _track_import_usage_expr(
            expr.true_expr,
            bound_names=bound_names,
            qualified_namespaces=qualified_namespaces,
            unqualified_symbols=unqualified_symbols,
        )
        _track_import_usage_expr(
            expr.false_expr,
            bound_names=bound_names,
            qualified_namespaces=qualified_namespaces,
            unqualified_symbols=unqualified_symbols,
        )
        return
    if isinstance(expr, AxonExprBinary):
        _track_import_usage_expr(
            expr.left,
            bound_names=bound_names,
            qualified_namespaces=qualified_namespaces,
            unqualified_symbols=unqualified_symbols,
        )
        _track_import_usage_expr(
            expr.right,
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
            _track_import_usage_expr(
                stmt.from_expr,
                bound_names=local_bound,
                qualified_namespaces=qualified_namespaces,
                unqualified_symbols=unqualified_symbols,
            )
            _track_import_usage_expr(
                stmt.to_expr,
                bound_names=local_bound,
                qualified_namespaces=qualified_namespaces,
                unqualified_symbols=unqualified_symbols,
            )
            _track_import_usage_expr(
                stmt.step_expr,
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


def _collect_import_usage(modules: tuple[AxonModule, ...]) -> _ImportUsage:
    qualified_namespaces: set[str] = set()
    unqualified_symbols: set[str] = set()
    for module in modules:
        bound_names = {param.name for param in module.params}
        bound_names.update(name for name in module.path_params if isinstance(name, str))
        if isinstance(module.path_param, str):
            bound_names.add(module.path_param)
        _track_import_usage_statements(
            module.statements,
            bound_names=bound_names,
            qualified_namespaces=qualified_namespaces,
            unqualified_symbols=unqualified_symbols,
        )
    return _ImportUsage(
        qualified_namespaces=frozenset(qualified_namespaces),
        unqualified_symbols=frozenset(unqualified_symbols),
    )


def _apply_namespace(
    modules: tuple[AxonModule, ...], namespace: str | None
) -> tuple[AxonModule, ...]:
    if not namespace:
        return modules
    local_module_names = {module.name for module in modules if "." not in module.name}

    def _qualify_callee(callee: str) -> str:
        base, sep, suffix = callee.partition("@")
        if "." in base or base not in local_module_names:
            return callee
        qualified = f"{namespace}.{base}"
        return f"{qualified}{sep}{suffix}" if sep else qualified

    def _rewrite_expr(expr: AxonExpr) -> AxonExpr:
        if isinstance(expr, AxonExprCall):
            return AxonExprCall(
                callee=_qualify_callee(expr.callee),
                args=tuple(_rewrite_expr(arg) for arg in expr.args),
                kwargs={
                    key: _rewrite_expr(value) if isinstance(value, AxonExpr) else value
                    for key, value in expr.kwargs.items()
                },
            )
        if isinstance(expr, AxonExprPipe):
            return AxonExprPipe(
                value=_rewrite_expr(expr.value),
                stages=tuple(_rewrite_expr(stage) for stage in expr.stages),
            )
        if isinstance(expr, AxonExprBind):
            return AxonExprBind(
                value=_rewrite_expr(expr.value),
                var=expr.var,
                body=_rewrite_expr(expr.body),
            )
        if isinstance(expr, AxonExprIf):
            return AxonExprIf(
                cond=_rewrite_expr(expr.cond),
                true_expr=_rewrite_expr(expr.true_expr),
                false_expr=_rewrite_expr(expr.false_expr),
            )
        if isinstance(expr, AxonExprTernary):
            return AxonExprTernary(
                cond=_rewrite_expr(expr.cond),
                true_expr=_rewrite_expr(expr.true_expr),
                false_expr=_rewrite_expr(expr.false_expr),
            )
        if isinstance(expr, AxonExprBinary):
            return AxonExprBinary(
                op=expr.op,
                left=_rewrite_expr(expr.left),
                right=_rewrite_expr(expr.right),
            )
        if isinstance(expr, AxonExprLambda):
            return AxonExprLambda(var=expr.var, body=_rewrite_expr(expr.body))
        if isinstance(expr, AxonExprParen):
            return AxonExprParen(inner=_rewrite_expr(expr.inner))
        if isinstance(expr, AxonExprList):
            return AxonExprList(items=tuple(_rewrite_expr(item) for item in expr.items))
        if isinstance(expr, AxonExprTuple):
            return AxonExprTuple(items=tuple(_rewrite_expr(item) for item in expr.items))
        if isinstance(expr, AxonExprDo):
            return AxonExprDo(body=_rewrite_statements(expr.body), inline=expr.inline)
        return expr

    def _rewrite_statements(
        statements: tuple[AxonStatement, ...],
    ) -> tuple[AxonStatement, ...]:
        rewritten: list[AxonStatement] = []
        for stmt in statements:
            if isinstance(stmt, AxonBind):
                rewritten.append(AxonBind(targets=stmt.targets, expr=_rewrite_expr(stmt.expr)))
                continue
            if isinstance(stmt, AxonReturn):
                rewritten.append(AxonReturn(values=tuple(_rewrite_expr(v) for v in stmt.values)))
                continue
            if isinstance(stmt, AxonRepeat):
                rewritten.append(
                    AxonRepeat(
                        name=stmt.name,
                        var=stmt.var,
                        to_expr=_rewrite_expr(stmt.to_expr),
                        from_expr=_rewrite_expr(stmt.from_expr),
                        step_expr=_rewrite_expr(stmt.step_expr),
                        body=_rewrite_statements(stmt.body),
                        targets=stmt.targets,
                        carry=stmt.carry,
                    )
                )
                continue
            if isinstance(stmt, AxonYield):
                rewritten.append(AxonYield(values=tuple(_rewrite_expr(v) for v in stmt.values)))
                continue
            if isinstance(stmt, AxonScopeBind):
                rewritten.append(
                    AxonScopeBind(
                        targets=stmt.targets,
                        prefix=_qualify_callee(stmt.prefix),
                        body=_rewrite_statements(stmt.body),
                        kwargs={
                            key: _rewrite_expr(value) if isinstance(value, AxonExpr) else value
                            for key, value in stmt.kwargs.items()
                        },
                    )
                )
                continue
            rewritten.append(stmt)
        return tuple(rewritten)

    namespaced: list[AxonModule] = []
    for module in modules:
        if "." in module.name:
            namespaced.append(module)
            continue
        namespaced.append(
            AxonModule(
                name=f"{namespace}.{module.name}",
                path_param=module.path_param,
                path_params=module.path_params,
                params=module.params,
                returns=module.returns,
                statements=_rewrite_statements(module.statements),
                imports=module.imports,
                imported_members=module.imported_members,
                exports=module.exports,
                symbols=module.symbols,
                pragmas=module.pragmas,
                type_aliases=module.type_aliases,
                return_type_expr=module.return_type_expr,
                return_shape=module.return_shape,
            )
        )
    return tuple(namespaced)


def _warn_unused_imports(
    *,
    file_path: Path,
    parsed_source: ParsedProgramSource,
    usage: _ImportUsage,
    enabled: bool = True,
) -> None:
    if not enabled:
        return
    imported_members = parsed_source.imported_members
    for namespace in parsed_source.imports:
        members = imported_members.get(namespace)
        if members:
            for member in members:
                if member not in usage.unqualified_symbols:
                    warnings.warn(
                        f"{file_path}: unused unqualified import {namespace}.{member}",
                        stacklevel=2,
                    )
            continue
        if namespace not in usage.qualified_namespaces:
            warnings.warn(
                f"{file_path}: unused qualified import {namespace}",
                stacklevel=2,
            )


def _rewrite_imported_member_refs(
    modules: tuple[AxonModule, ...],
    *,
    imported_members: dict[str, tuple[str, ...]] | None,
) -> tuple[AxonModule, ...]:
    if not imported_members:
        return modules
    providers: dict[str, tuple[str, ...]] = {}
    for namespace, members in imported_members.items():
        for member in members:
            prev = providers.get(member, ())
            if namespace not in prev:
                providers[member] = (*prev, namespace)

    def _qualify_member(member: str, module_name: str) -> str:
        namespaces = providers.get(member, ())
        if not namespaces:
            return member
        if len(namespaces) > 1:
            choices = ", ".join(sorted(namespaces))
            raise ValueError(
                f"Axon import resolution failed in module {module_name!r}: ambiguous imported member "
                f"{member!r}; found in namespaces: {choices}"
            )
        return f"{namespaces[0]}.{member}"

    def _rewrite_expr(expr: AxonExpr, *, bound_names: set[str], module_name: str) -> AxonExpr:
        if isinstance(expr, AxonExprName):
            if "." in expr.name or expr.name in bound_names:
                return expr
            qualified = _qualify_member(expr.name, module_name)
            if qualified == expr.name:
                return expr
            return AxonExprName(name=qualified)
        if isinstance(expr, AxonExprCall):
            base, sep, suffix = expr.callee.partition("@")
            if "." not in base and base not in bound_names:
                qualified = _qualify_member(base, module_name)
                callee = f"{qualified}{sep}{suffix}" if sep else qualified
            else:
                callee = expr.callee
            return AxonExprCall(
                callee=callee,
                args=tuple(
                    _rewrite_expr(arg, bound_names=bound_names, module_name=module_name)
                    for arg in expr.args
                ),
                kwargs={
                    key: (
                        _rewrite_expr(value, bound_names=bound_names, module_name=module_name)
                        if isinstance(value, AxonExpr)
                        else value
                    )
                    for key, value in expr.kwargs.items()
                },
            )
        if isinstance(expr, AxonExprPipe):
            return AxonExprPipe(
                value=_rewrite_expr(expr.value, bound_names=bound_names, module_name=module_name),
                stages=tuple(
                    _rewrite_expr(stage, bound_names=bound_names, module_name=module_name)
                    for stage in expr.stages
                ),
            )
        if isinstance(expr, AxonExprBind):
            value = _rewrite_expr(expr.value, bound_names=bound_names, module_name=module_name)
            nested_bound = set(bound_names)
            nested_bound.add(expr.var)
            body = _rewrite_expr(expr.body, bound_names=nested_bound, module_name=module_name)
            return AxonExprBind(value=value, var=expr.var, body=body)
        if isinstance(expr, AxonExprIf):
            return AxonExprIf(
                cond=_rewrite_expr(expr.cond, bound_names=bound_names, module_name=module_name),
                true_expr=_rewrite_expr(
                    expr.true_expr, bound_names=bound_names, module_name=module_name
                ),
                false_expr=_rewrite_expr(
                    expr.false_expr, bound_names=bound_names, module_name=module_name
                ),
            )
        if isinstance(expr, AxonExprTernary):
            return AxonExprTernary(
                cond=_rewrite_expr(expr.cond, bound_names=bound_names, module_name=module_name),
                true_expr=_rewrite_expr(
                    expr.true_expr, bound_names=bound_names, module_name=module_name
                ),
                false_expr=_rewrite_expr(
                    expr.false_expr, bound_names=bound_names, module_name=module_name
                ),
            )
        if isinstance(expr, AxonExprBinary):
            return AxonExprBinary(
                op=expr.op,
                left=_rewrite_expr(expr.left, bound_names=bound_names, module_name=module_name),
                right=_rewrite_expr(expr.right, bound_names=bound_names, module_name=module_name),
            )
        if isinstance(expr, AxonExprLambda):
            nested_bound = set(bound_names)
            nested_bound.add(expr.var)
            return AxonExprLambda(
                var=expr.var,
                body=_rewrite_expr(expr.body, bound_names=nested_bound, module_name=module_name),
            )
        if isinstance(expr, AxonExprParen):
            return AxonExprParen(
                inner=_rewrite_expr(expr.inner, bound_names=bound_names, module_name=module_name)
            )
        if isinstance(expr, AxonExprList):
            return AxonExprList(
                items=tuple(
                    _rewrite_expr(item, bound_names=bound_names, module_name=module_name)
                    for item in expr.items
                )
            )
        if isinstance(expr, AxonExprTuple):
            return AxonExprTuple(
                items=tuple(
                    _rewrite_expr(item, bound_names=bound_names, module_name=module_name)
                    for item in expr.items
                )
            )
        if isinstance(expr, AxonExprDo):
            return AxonExprDo(
                body=_rewrite_statements(
                    expr.body, bound_names=set(bound_names), module_name=module_name
                ),
                inline=expr.inline,
            )
        return expr

    def _rewrite_statements(
        statements: tuple[AxonStatement, ...],
        *,
        bound_names: set[str],
        module_name: str,
    ) -> tuple[AxonStatement, ...]:
        rewritten: list[AxonStatement] = []
        local_bound = set(bound_names)
        for stmt in statements:
            if isinstance(stmt, AxonBind):
                rewritten.append(
                    AxonBind(
                        targets=stmt.targets,
                        expr=_rewrite_expr(
                            stmt.expr,
                            bound_names=local_bound,
                            module_name=module_name,
                        ),
                    )
                )
                for target in stmt.targets:
                    if target != "_":
                        local_bound.add(target)
                continue
            if isinstance(stmt, AxonReturn):
                rewritten.append(
                    AxonReturn(
                        values=tuple(
                            _rewrite_expr(value, bound_names=local_bound, module_name=module_name)
                            for value in stmt.values
                        )
                    )
                )
                continue
            if isinstance(stmt, AxonYield):
                rewritten.append(
                    AxonYield(
                        values=tuple(
                            _rewrite_expr(value, bound_names=local_bound, module_name=module_name)
                            for value in stmt.values
                        )
                    )
                )
                continue
            if isinstance(stmt, AxonRepeat):
                loop_bound = set(local_bound)
                loop_bound.add(stmt.var)
                if stmt.carry:
                    for name in stmt.carry:
                        if name != "_":
                            loop_bound.add(name)
                rewritten.append(
                    AxonRepeat(
                        name=stmt.name,
                        var=stmt.var,
                        to_expr=_rewrite_expr(
                            stmt.to_expr, bound_names=local_bound, module_name=module_name
                        ),
                        from_expr=_rewrite_expr(
                            stmt.from_expr, bound_names=local_bound, module_name=module_name
                        ),
                        step_expr=_rewrite_expr(
                            stmt.step_expr, bound_names=local_bound, module_name=module_name
                        ),
                        body=_rewrite_statements(
                            stmt.body,
                            bound_names=loop_bound,
                            module_name=module_name,
                        ),
                        targets=stmt.targets,
                        carry=stmt.carry,
                    )
                )
                if stmt.targets:
                    for target in stmt.targets:
                        if target != "_":
                            local_bound.add(target)
                continue
            if isinstance(stmt, AxonScopeBind):
                rewritten.append(
                    AxonScopeBind(
                        targets=stmt.targets,
                        prefix=stmt.prefix,
                        body=_rewrite_statements(
                            stmt.body,
                            bound_names=set(local_bound),
                            module_name=module_name,
                        ),
                        kwargs={
                            key: (
                                _rewrite_expr(
                                    value, bound_names=local_bound, module_name=module_name
                                )
                                if isinstance(value, AxonExpr)
                                else value
                            )
                            for key, value in stmt.kwargs.items()
                        },
                    )
                )
                for target in stmt.targets:
                    if target != "_":
                        local_bound.add(target)
                continue
            rewritten.append(stmt)
        return tuple(rewritten)

    out: list[AxonModule] = []
    for module in modules:
        bound_names = {param.name for param in module.params}
        bound_names.update(name for name in module.path_params if isinstance(name, str))
        if isinstance(module.path_param, str):
            bound_names.add(module.path_param)
        out.append(
            AxonModule(
                name=module.name,
                path_param=module.path_param,
                path_params=module.path_params,
                params=module.params,
                returns=module.returns,
                statements=_rewrite_statements(
                    module.statements,
                    bound_names=bound_names,
                    module_name=module.name,
                ),
                imports=module.imports,
                imported_members=module.imported_members,
                exports=module.exports,
                symbols=module.symbols,
                pragmas=module.pragmas,
                type_aliases=module.type_aliases,
                return_type_expr=module.return_type_expr,
                return_shape=module.return_shape,
            )
        )
    return tuple(out)


def _collect_called_modules(modules: tuple[AxonModule, ...]) -> dict[str, set[str]]:
    module_names = {module.name for module in modules}
    calls: dict[str, set[str]] = {}

    for module in modules:
        imported_targets: dict[str, tuple[str, ...]] = {}
        if module.imported_members:
            for namespace, members in module.imported_members.items():
                for member in members:
                    qualified = f"{namespace}.{member}"
                    if qualified not in module_names:
                        continue
                    prev = imported_targets.get(member, ())
                    if qualified not in prev:
                        imported_targets[member] = (*prev, qualified)

        def _add_candidates(name: str, *, bound_names: set[str], acc: set[str]) -> None:
            if name in bound_names:
                return
            if name in module_names:
                acc.add(name)
            if "." not in name:
                for qualified in imported_targets.get(name, ()):
                    acc.add(qualified)

        def _track_expr(
            expr: AxonExpr,
            *,
            bound_names: set[str],
            acc: set[str],
        ) -> None:
            if isinstance(expr, AxonExprName):
                _add_candidates(expr.name, bound_names=bound_names, acc=acc)
                return
            if isinstance(expr, AxonExprCall):
                base = expr.callee.split("@", 1)[0]
                _add_candidates(base, bound_names=bound_names, acc=acc)
                for arg in expr.args:
                    _track_expr(arg, bound_names=bound_names, acc=acc)
                for kwarg in expr.kwargs.values():
                    if isinstance(kwarg, AxonExpr):
                        _track_expr(kwarg, bound_names=bound_names, acc=acc)
                return
            if isinstance(expr, AxonExprParen):
                _track_expr(expr.inner, bound_names=bound_names, acc=acc)
                return
            if isinstance(expr, AxonExprList | AxonExprTuple):
                for item in expr.items:
                    _track_expr(item, bound_names=bound_names, acc=acc)
                return
            if isinstance(expr, AxonExprPipe):
                _track_expr(expr.value, bound_names=bound_names, acc=acc)
                for stage in expr.stages:
                    _track_expr(stage, bound_names=bound_names, acc=acc)
                return
            if isinstance(expr, AxonExprBind):
                _track_expr(expr.value, bound_names=bound_names, acc=acc)
                nested_bound = set(bound_names)
                nested_bound.add(expr.var)
                _track_expr(expr.body, bound_names=nested_bound, acc=acc)
                return
            if isinstance(expr, AxonExprIf | AxonExprTernary):
                _track_expr(expr.cond, bound_names=bound_names, acc=acc)
                _track_expr(expr.true_expr, bound_names=bound_names, acc=acc)
                _track_expr(expr.false_expr, bound_names=bound_names, acc=acc)
                return
            if isinstance(expr, AxonExprBinary):
                _track_expr(expr.left, bound_names=bound_names, acc=acc)
                _track_expr(expr.right, bound_names=bound_names, acc=acc)
                return
            if isinstance(expr, AxonExprLambda):
                nested_bound = set(bound_names)
                nested_bound.add(expr.var)
                _track_expr(expr.body, bound_names=nested_bound, acc=acc)
                return
            if isinstance(expr, AxonExprDo):
                _track_statements(expr.body, bound_names=set(bound_names), acc=acc)

        def _track_statements(
            statements: tuple[AxonStatement, ...],
            *,
            bound_names: set[str],
            acc: set[str],
        ) -> None:
            local_bound = set(bound_names)
            for stmt in statements:
                if isinstance(stmt, AxonBind):
                    _track_expr(stmt.expr, bound_names=local_bound, acc=acc)
                    for target in stmt.targets:
                        if target != "_":
                            local_bound.add(target)
                    continue
                if isinstance(stmt, AxonReturn | AxonYield):
                    for value in stmt.values:
                        _track_expr(value, bound_names=local_bound, acc=acc)
                    continue
                if isinstance(stmt, AxonRepeat):
                    _track_expr(stmt.from_expr, bound_names=local_bound, acc=acc)
                    _track_expr(stmt.to_expr, bound_names=local_bound, acc=acc)
                    _track_expr(stmt.step_expr, bound_names=local_bound, acc=acc)
                    loop_bound = set(local_bound)
                    loop_bound.add(stmt.var)
                    if stmt.carry:
                        for name in stmt.carry:
                            if name != "_":
                                loop_bound.add(name)
                    _track_statements(stmt.body, bound_names=loop_bound, acc=acc)
                    if stmt.targets:
                        for target in stmt.targets:
                            if target != "_":
                                local_bound.add(target)
                    continue
                if isinstance(stmt, AxonScopeBind):
                    for kwarg in stmt.kwargs.values():
                        if isinstance(kwarg, AxonExpr):
                            _track_expr(kwarg, bound_names=local_bound, acc=acc)
                    _track_statements(stmt.body, bound_names=set(local_bound), acc=acc)
                    for target in stmt.targets:
                        if target != "_":
                            local_bound.add(target)

        bound_names = {param.name for param in module.params}
        bound_names.update(name for name in module.path_params if isinstance(name, str))
        if isinstance(module.path_param, str):
            bound_names.add(module.path_param)
        acc: set[str] = set()
        _track_statements(module.statements, bound_names=bound_names, acc=acc)
        calls[module.name] = acc
    return calls


def _prune_to_reachable_modules(
    modules: tuple[AxonModule, ...],
    *,
    root_module_names: tuple[str, ...],
) -> tuple[AxonModule, ...]:
    calls = _collect_called_modules(modules)
    reachable: set[str] = set()
    stack = [name for name in root_module_names if name in calls]
    while stack:
        name = stack.pop()
        if name in reachable:
            continue
        reachable.add(name)
        for dep in calls.get(name, ()):
            if dep not in reachable:
                stack.append(dep)
    return tuple(module for module in modules if module.name in reachable)


def _axon_search_paths() -> tuple[Path, ...]:
    raw = os.environ.get("AXON_PATH", "")
    if not raw:
        return ()
    out: list[Path] = []
    seen: set[Path] = set()
    for part in raw.split(os.pathsep):
        stripped = part.strip()
        if not stripped:
            continue
        candidate = Path(stripped).expanduser().resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        out.append(candidate)
    return tuple(out)


def _resolve_import_path(
    base_file: Path, import_name: str, builtins_dir: Path, search_paths: tuple[Path, ...]
) -> Path:
    rel = Path(*import_name.split(".")).with_suffix(".axon")
    local_candidate = (base_file.parent / rel).resolve()
    if local_candidate.exists():
        return local_candidate
    search_candidates: list[Path] = []
    for search_root in search_paths:
        candidate = (search_root / rel).resolve()
        search_candidates.append(candidate)
        if candidate.exists():
            return candidate
    builtin_candidate = (builtins_dir / rel).resolve()
    if builtin_candidate.exists():
        return builtin_candidate
    tried = [
        str(local_candidate),
        *(str(path) for path in search_candidates),
        str(builtin_candidate),
    ]
    raise FileNotFoundError(
        f"Axon import {import_name!r} not found from {base_file}: tried {', '.join(tried)}"
    )


def load_axon_program_from_path(path: Path) -> tuple[AxonModule, ...]:
    root = path.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Axon file not found: {root}")
    if not root.is_file():
        raise ValueError(f"Axon import root must be a file: {root}")

    seen_paths: set[Path] = set()
    visiting: list[Path] = []
    ordered_files: list[_LoadedSyntaxFile] = []

    builtins_dir = (Path(__file__).resolve().parents[1] / "builtins").resolve()
    search_paths = _axon_search_paths()

    def _load_file_syntax(file_path: Path, *, namespace: str | None = None) -> None:
        resolved = file_path.resolve()
        if resolved in seen_paths:
            return
        if resolved in visiting:
            cycle = " -> ".join(str(p) for p in [*visiting, resolved])
            raise ValueError(f"Cyclic Axon imports detected: {cycle}")
        visiting.append(resolved)

        source = resolved.read_text(encoding="utf-8")
        parsed_source = parse_program_source(source)
        validate_parsed_program_source(parsed_source)
        for import_name in sorted(parsed_source.imports):
            dep = _resolve_import_path(resolved, import_name, builtins_dir, search_paths)
            _load_file_syntax(dep, namespace=import_name)
        ordered_files.append(
            _LoadedSyntaxFile(path=resolved, namespace=namespace, parsed_source=parsed_source)
        )
        seen_paths.add(resolved)
        visiting.pop()

    _load_file_syntax(root)

    # Local import avoids a parser<->import_loader import cycle at module import time.
    from .parser import build_axon_modules_from_parsed_source

    ordered_modules: list[AxonModule] = []
    loaded_by_namespace: dict[str, _LoadedSyntaxFile] = {
        loaded.namespace: loaded for loaded in ordered_files if loaded.namespace is not None
    }
    root_module_names: tuple[str, ...] = ()
    for loaded in ordered_files:
        effective_imported_members: dict[str, tuple[str, ...]] = dict(
            loaded.parsed_source.imported_members
        )
        effective_extra_imports: list[str] = []
        for namespace in loaded.parsed_source.imports:
            if namespace in effective_imported_members:
                continue
            dep = loaded_by_namespace.get(namespace)
            if dep is None or not dep.parsed_source.exports:
                continue
            effective_imported_members[namespace] = dep.parsed_source.exports
        imported_constants: dict[str, AxonExpr] = {}
        imported_constant_imports: list[str] = []
        for namespace, members in effective_imported_members.items():
            dep = loaded_by_namespace.get(namespace)
            if dep is None:
                continue
            dep_constants = dep.parsed_source.constants
            requested = {member for member in members if member in dep_constants}
            closure = _collect_constant_closure(constants=dep_constants, seed_names=requested)
            if closure:
                for dep_import in dep.parsed_source.imports:
                    if dep_import not in imported_constant_imports:
                        imported_constant_imports.append(dep_import)
            for name, expr in closure.items():
                imported_constants.setdefault(name, expr)

        modules = build_axon_modules_from_parsed_source(
            loaded.parsed_source,
            validate=False,
            extra_constants=imported_constants if imported_constants else None,
            extra_imports=tuple(
                dict.fromkeys([*imported_constant_imports, *effective_extra_imports])
            )
            if (imported_constant_imports or effective_extra_imports)
            else None,
            extra_imported_members=effective_imported_members
            if effective_imported_members
            else None,
        )
        usage = _collect_import_usage(modules)
        _warn_unused_imports(
            file_path=loaded.path,
            parsed_source=loaded.parsed_source,
            usage=usage,
            enabled=builtins_dir not in loaded.path.parents,
        )
        linked_modules = _rewrite_imported_member_refs(
            modules,
            imported_members=effective_imported_members if effective_imported_members else None,
        )
        namespaced_modules = _apply_namespace(linked_modules, loaded.namespace)
        if loaded.path == root and loaded.namespace is None:
            root_module_names = tuple(module.name for module in namespaced_modules)
        ordered_modules.extend(namespaced_modules)

    out = tuple(ordered_modules)
    if root_module_names:
        out = _prune_to_reachable_modules(out, root_module_names=root_module_names)
    validate_axon_program(out)
    return out


__all__ = ["load_axon_program_from_path"]
