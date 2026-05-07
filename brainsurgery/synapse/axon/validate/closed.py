from __future__ import annotations

import re
from pathlib import Path
from typing import AbstractSet, Protocol

from ..ast.nodes import (
    AxonBind,
    AxonCond,
    AxonExpr,
    AxonExprAscribe,
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
    AxonDefinition,
    AxonRepeat,
    AxonReturn,
    AxonScopeBind,
    AxonStatement,
    AxonYield,
)
from ..ast.source import AxonFile
from ..ast.types import (
    DimExprBinary,
    DimToken,
    TypeAliasDef,
    TypeExpr,
    TypeList,
    TypeNamed,
    TypeOptional,
    TypeTensor,
    TypeTuple,
    TypeVar,
    dim_token_names,
)
from ..entrypoint import pragma_main_module
from .ast import validate_axon_program
from .diagnostics import ValidationDiagnostic


class _ImportUsageLike(Protocol):
    @property
    def qualified_namespaces(self) -> AbstractSet[str]: ...

    @property
    def unqualified_symbols(self) -> AbstractSet[str]: ...


_PATH_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _collect_type_dim_names(
    type_expr: TypeExpr | None,
    *,
    type_aliases: dict[str, TypeAliasDef],
    stack: tuple[str, ...] = (),
) -> set[str]:
    if type_expr is None:
        return set()
    root = type_expr.inner if isinstance(type_expr, TypeOptional) else type_expr
    if isinstance(root, TypeTensor):
        out: set[str] = set()
        for dim in root.dims:
            out.update(name for name in dim_token_names(dim) if name.isidentifier())
        return out
    if isinstance(root, TypeTuple):
        tuple_out: set[str] = set()
        for item in root.items:
            tuple_out.update(_collect_type_dim_names(item, type_aliases=type_aliases, stack=stack))
        return tuple_out
    if isinstance(root, TypeList):
        return _collect_type_dim_names(root.item, type_aliases=type_aliases, stack=stack)
    if isinstance(root, TypeNamed):
        direct_args = {arg for arg in root.args if isinstance(arg, str)}
        alias_def = _lookup_type_alias(root.name, type_aliases=type_aliases)
        if alias_def is None or root.name in stack:
            return direct_args
        subst = _match_type_alias_dims(alias_def.params, root.args)
        if subst is None:
            return direct_args
        return direct_args | _collect_type_dim_names(
            _substitute_type_alias_dims(alias_def.value, subst=subst),
            type_aliases=type_aliases,
            stack=(*stack, root.name),
        )
    if isinstance(root, TypeVar):
        return set()
    return set()


def _substitute_type_alias_dims(
    tp: TypeExpr, *, subst: dict[str, DimToken | tuple[DimToken, ...]]
) -> TypeExpr:
    def _sub_dim(dim: DimToken) -> tuple[DimToken, ...]:
        if isinstance(dim, str):
            mapped = subst.get(dim)
            if mapped is None:
                return (dim,)
            if isinstance(mapped, tuple):
                return mapped
            return (mapped,)
        if isinstance(dim, int):
            return (dim,)
        if isinstance(dim, DimExprBinary):
            left = _sub_dim(dim.left)
            right = _sub_dim(dim.right)
            if len(left) == 1 and len(right) == 1:
                return (DimExprBinary(op=dim.op, left=left[0], right=right[0]),)
            raise ValueError(
                "variadic type alias dimension cannot appear inside dimension arithmetic"
            )
        raise TypeError(f"unsupported dim token {dim!r}")

    if isinstance(tp, TypeOptional):
        return TypeOptional(inner=_substitute_type_alias_dims(tp.inner, subst=subst))
    if isinstance(tp, TypeList):
        return TypeList(item=_substitute_type_alias_dims(tp.item, subst=subst))
    if isinstance(tp, TypeTuple):
        return TypeTuple(
            items=tuple(_substitute_type_alias_dims(item, subst=subst) for item in tp.items)
        )
    if isinstance(tp, TypeTensor):
        return TypeTensor(
            base=tp.base,
            dims=tuple(item for dim in tp.dims for item in _sub_dim(dim)),
        )
    if isinstance(tp, TypeNamed):
        return TypeNamed(
            name=tp.name,
            args=tuple(item for dim in tp.args for item in _sub_dim(dim)),
        )
    if isinstance(tp, TypeVar):
        return tp
    return tp


def _match_type_alias_dims(
    params: tuple[str, ...], args: tuple[DimToken, ...]
) -> dict[str, DimToken | tuple[DimToken, ...]] | None:
    variadic_idx = next((idx for idx, param in enumerate(params) if param.startswith("..")), None)
    if variadic_idx is None:
        if len(args) != len(params):
            return None
        return {name: dim for name, dim in zip(params, args, strict=True)}
    fixed_after = len(params) - variadic_idx - 1
    if len(args) < variadic_idx + fixed_after:
        return None
    subst: dict[str, DimToken | tuple[DimToken, ...]] = {}
    for name, dim in zip(params[:variadic_idx], args[:variadic_idx], strict=True):
        subst[name] = dim
    variadic_end = len(args) - fixed_after
    subst[params[variadic_idx]] = tuple(args[variadic_idx:variadic_end])
    if fixed_after:
        for name, dim in zip(params[variadic_idx + 1 :], args[variadic_end:], strict=True):
            subst[name] = dim
    return subst


def _lookup_type_alias(name: str, *, type_aliases: dict[str, TypeAliasDef]) -> TypeAliasDef | None:
    alias_def = type_aliases.get(name)
    if alias_def is not None:
        return alias_def
    if "." in name:
        return type_aliases.get(name.rsplit(".", 1)[1])
    return None


def _validate_type_expr(
    tp: TypeExpr | None,
    *,
    type_aliases: dict[str, TypeAliasDef],
    owner: str,
) -> None:
    if tp is None:
        return
    if isinstance(tp, TypeOptional):
        _validate_type_expr(tp.inner, type_aliases=type_aliases, owner=owner)
        return
    if isinstance(tp, TypeList):
        _validate_type_expr(tp.item, type_aliases=type_aliases, owner=owner)
        return
    if isinstance(tp, TypeTuple):
        for item in tp.items:
            _validate_type_expr(item, type_aliases=type_aliases, owner=owner)
        return
    if isinstance(tp, TypeNamed):
        alias_def = _lookup_type_alias(tp.name, type_aliases=type_aliases)
        if alias_def is None:
            if "." not in tp.name:
                return
            raise ValueError(
                f"Axon closed validation failed in {owner!r}: unknown type alias {tp.name!r}"
            )
        if _match_type_alias_dims(alias_def.params, tp.args) is None:
            raise ValueError(
                f"Axon closed validation failed in {owner!r}: type alias {tp.name!r} "
                f"expects {len(alias_def.params)} args, got {len(tp.args)}"
            )
        return
    if isinstance(tp, TypeVar):
        return


def _validate_type_aliases(type_aliases: dict[str, TypeAliasDef]) -> None:
    for name, alias in type_aliases.items():
        _validate_type_expr(alias.value, type_aliases=type_aliases, owner=f"type alias {name}")


def _call_base_name(callee: str) -> str:
    indexes = [idx for idx in (callee.find("@"), callee.find("::")) if idx >= 0]
    if not indexes:
        return callee.strip()
    return callee[: min(indexes)].strip()


def _call_surface(callee: str) -> str:
    indexes = [idx for idx in (callee.find("@"), callee.find("::")) if idx >= 0]
    if not indexes:
        return ""
    return callee[min(indexes) :]


def _validate_expr_closure(
    expr: AxonExpr,
    *,
    type_aliases: dict[str, TypeAliasDef],
    available_modules: set[str],
    available_values: set[str],
    bound_names: set[str],
    module: AxonDefinition,
) -> None:
    def _check_name(name: str) -> None:
        if name.startswith("_"):
            return
        if name in bound_names or name in available_modules or name in available_values:
            return
        raise ValueError(
            f"Axon closed validation failed in module {module.name!r}: unresolved name {name!r}"
        )

    def _check_path_placeholders(text: str) -> None:
        for match in _PATH_PLACEHOLDER_RE.finditer(text):
            _check_name(match.group(1))

    if isinstance(expr, AxonExprName):
        _check_name(_call_base_name(expr.name))
        _check_path_placeholders(_call_surface(expr.name))
        return
    if isinstance(expr, AxonExprCall):
        _check_name(_call_base_name(expr.callee))
        _check_path_placeholders(_call_surface(expr.callee))
        for arg in expr.args:
            _validate_expr_closure(
                arg,
                type_aliases=type_aliases,
                available_modules=available_modules,
                available_values=available_values,
                bound_names=bound_names,
                module=module,
            )
        for value in expr.kwargs.values():
            if isinstance(value, AxonExpr):
                _validate_expr_closure(
                    value,
                    type_aliases=type_aliases,
                    available_modules=available_modules,
                    available_values=available_values,
                    bound_names=bound_names,
                    module=module,
                )
        return
    if isinstance(expr, AxonExprPipe):
        _validate_expr_closure(
            expr.value,
            type_aliases=type_aliases,
            available_modules=available_modules,
            available_values=available_values,
            bound_names=bound_names,
            module=module,
        )
        for stage in expr.stages:
            _validate_expr_closure(
                stage,
                type_aliases=type_aliases,
                available_modules=available_modules,
                available_values=available_values,
                bound_names=bound_names,
                module=module,
            )
        return
    if isinstance(expr, AxonExprBind):
        _validate_expr_closure(
            expr.value,
            type_aliases=type_aliases,
            available_modules=available_modules,
            available_values=available_values,
            bound_names=bound_names,
            module=module,
        )
        nested_bound = set(bound_names)
        nested_bound.add(expr.var)
        _validate_expr_closure(
            expr.body,
            type_aliases=type_aliases,
            available_modules=available_modules,
            available_values=available_values,
            bound_names=nested_bound,
            module=module,
        )
        return
    if isinstance(expr, AxonExprIf | AxonExprTernary):
        for child in (expr.cond, expr.true_expr, expr.false_expr):
            _validate_expr_closure(
                child,
                type_aliases=type_aliases,
                available_modules=available_modules,
                available_values=available_values,
                bound_names=bound_names,
                module=module,
            )
        return
    if isinstance(expr, AxonExprBinary):
        for child in (expr.left, expr.right):
            _validate_expr_closure(
                child,
                type_aliases=type_aliases,
                available_modules=available_modules,
                available_values=available_values,
                bound_names=bound_names,
                module=module,
            )
        return
    if isinstance(expr, AxonExprLambda):
        nested_bound = set(bound_names)
        nested_bound.add(expr.var)
        _validate_expr_closure(
            expr.body,
            type_aliases=type_aliases,
            available_modules=available_modules,
            available_values=available_values,
            bound_names=nested_bound,
            module=module,
        )
        return
    if isinstance(expr, AxonExprAscribe):
        _validate_type_expr(
            expr.type_expr,
            type_aliases=type_aliases,
            owner=f"expression in {module.name}",
        )
        _validate_expr_closure(
            expr.expr,
            type_aliases=type_aliases,
            available_modules=available_modules,
            available_values=available_values,
            bound_names=bound_names,
            module=module,
        )
        return
    if isinstance(expr, AxonExprParen):
        _validate_expr_closure(
            expr.inner,
            type_aliases=type_aliases,
            available_modules=available_modules,
            available_values=available_values,
            bound_names=bound_names,
            module=module,
        )
        return
    if isinstance(expr, AxonExprList | AxonExprTuple):
        for item in expr.items:
            _validate_expr_closure(
                item,
                type_aliases=type_aliases,
                available_modules=available_modules,
                available_values=available_values,
                bound_names=bound_names,
                module=module,
            )
        return
    if isinstance(expr, AxonExprDo):
        _validate_stmt_closure(
            expr.body,
            type_aliases=type_aliases,
            available_modules=available_modules,
            available_values=available_values,
            bound_names=set(bound_names),
            module=module,
        )
        return
    if hasattr(expr, "parts"):
        for part in getattr(expr, "parts", ()):
            for match in _PATH_PLACEHOLDER_RE.finditer(part):
                _check_name(match.group(1))


def _validate_stmt_closure(
    stmts: tuple[AxonStatement, ...],
    *,
    type_aliases: dict[str, TypeAliasDef],
    available_modules: set[str],
    available_values: set[str],
    bound_names: set[str],
    module: AxonDefinition,
) -> None:
    def _collect_bound_names_after(
        branch_stmts: tuple[AxonStatement, ...],
        *,
        starting_bound: set[str],
    ) -> set[str]:
        collected = set(starting_bound)
        for branch_stmt in branch_stmts:
            if isinstance(branch_stmt, AxonBind):
                collected.update(name for name in branch_stmt.targets if name != "_")
            elif isinstance(branch_stmt, AxonScopeBind):
                collected.update(name for name in branch_stmt.targets if name != "_")
            elif isinstance(branch_stmt, AxonRepeat):
                collected.update(name for name in (branch_stmt.targets or ()) if name != "_")
            elif isinstance(branch_stmt, AxonCond):
                true_after = _collect_bound_names_after(
                    branch_stmt.true_body, starting_bound=set(collected)
                )
                false_after = _collect_bound_names_after(
                    branch_stmt.false_body, starting_bound=set(collected)
                )
                collected.update(true_after & false_after)
        return collected

    local_bound = set(bound_names)
    for stmt in stmts:
        if isinstance(stmt, AxonBind):
            _validate_expr_closure(
                stmt.expr,
                type_aliases=type_aliases,
                available_modules=available_modules,
                available_values=available_values,
                bound_names=local_bound,
                module=module,
            )
            local_bound.update(name for name in stmt.targets if name != "_")
            continue
        if isinstance(stmt, AxonReturn | AxonYield):
            for value in stmt.values:
                _validate_expr_closure(
                    value,
                    type_aliases=type_aliases,
                    available_modules=available_modules,
                    available_values=available_values,
                    bound_names=local_bound,
                    module=module,
                )
            continue
        if isinstance(stmt, AxonCond):
            _validate_expr_closure(
                stmt.cond,
                type_aliases=type_aliases,
                available_modules=available_modules,
                available_values=available_values,
                bound_names=local_bound,
                module=module,
            )
            _validate_stmt_closure(
                stmt.true_body,
                type_aliases=type_aliases,
                available_modules=available_modules,
                available_values=available_values,
                bound_names=set(local_bound),
                module=module,
            )
            _validate_stmt_closure(
                stmt.false_body,
                type_aliases=type_aliases,
                available_modules=available_modules,
                available_values=available_values,
                bound_names=set(local_bound),
                module=module,
            )
            true_after = _collect_bound_names_after(stmt.true_body, starting_bound=set(local_bound))
            false_after = _collect_bound_names_after(
                stmt.false_body, starting_bound=set(local_bound)
            )
            local_bound.update(true_after & false_after)
            continue
        if isinstance(stmt, AxonRepeat):
            nested_bound = set(local_bound)
            nested_bound.add(stmt.var)
            nested_bound.update(name for name in (stmt.targets or ()) if name != "_")
            nested_bound.update(name for name in (stmt.carry or ()) if name != "_")
            for expr in (stmt.from_expr, stmt.to_expr, stmt.step_expr):
                _validate_expr_closure(
                    expr,
                    type_aliases=type_aliases,
                    available_modules=available_modules,
                    available_values=available_values,
                    bound_names=local_bound,
                    module=module,
                )
            _validate_stmt_closure(
                stmt.body,
                type_aliases=type_aliases,
                available_modules=available_modules,
                available_values=available_values,
                bound_names=nested_bound,
                module=module,
            )
            continue
        if isinstance(stmt, AxonScopeBind):
            nested_bound = set(local_bound)
            nested_bound.update(name for name in stmt.targets if name != "_")
            for raw_value in stmt.kwargs.values():
                if isinstance(raw_value, AxonExpr):
                    _validate_expr_closure(
                        raw_value,
                        type_aliases=type_aliases,
                        available_modules=available_modules,
                        available_values=available_values,
                        bound_names=local_bound,
                        module=module,
                    )
            for part in stmt.prefix.parts:
                for match in _PATH_PLACEHOLDER_RE.finditer(part):
                    name = match.group(1)
                    if name not in local_bound and name not in available_values:
                        raise ValueError(
                            f"Axon closed validation failed in module {module.name!r}: "
                            f"unresolved scope placeholder {name!r}"
                        )
            _validate_stmt_closure(
                stmt.body,
                type_aliases=type_aliases,
                available_modules=available_modules,
                available_values=available_values,
                bound_names=nested_bound,
                module=module,
            )
            local_bound.update(name for name in stmt.targets if name != "_")


def validate_closed_axon_file(ast: AxonFile, *, main_module: str | None = None) -> None:
    pragma_main = pragma_main_module(ast)
    if pragma_main is not None and pragma_main not in {module.name for module in ast.modules}:
        raise ValueError(f"Axon closed validation failed: MAIN pragma references unknown definition {pragma_main!r}")
    validate_axon_program(
        ast.modules,
        main_module=main_module if main_module is not None else pragma_main,
    )
    if ast.imports:
        raise ValueError("Axon closed validation failed: closed AST must not carry file imports")
    if ast.imported_members:
        raise ValueError(
            "Axon closed validation failed: closed AST must not carry imported members"
        )
    if ast.exports:
        raise ValueError("Axon closed validation failed: closed AST must not carry file exports")
    for module in ast.modules:
        if module.imports:
            raise ValueError(
                f"Axon closed validation failed in module {module.name!r}: module imports must be empty"
            )
        if module.imported_members:
            raise ValueError(
                f"Axon closed validation failed in module {module.name!r}: module imported_members must be empty"
            )
        if module.exports:
            raise ValueError(
                f"Axon closed validation failed in module {module.name!r}: module exports must be empty"
            )
    _validate_type_aliases(ast.type_aliases)
    available_modules = {module.name for module in ast.modules}
    available_values = available_modules
    for module in ast.modules:
        for param in module.params:
            _validate_type_expr(param.type_expr, type_aliases=ast.type_aliases, owner=module.name)
        _validate_type_expr(
            module.return_type_expr, type_aliases=ast.type_aliases, owner=module.name
        )
        bound_names = {param.name for param in module.params}
        bound_names.update(name for name in module.path_params if isinstance(name, str))
        if isinstance(module.path_param, str):
            bound_names.add(module.path_param)
        for param in module.params:
            bound_names.update(
                _collect_type_dim_names(param.type_expr, type_aliases=ast.type_aliases)
            )
        bound_names.update(
            _collect_type_dim_names(module.return_type_expr, type_aliases=ast.type_aliases)
        )
        if module.body_expr is not None:
            _validate_expr_closure(
                module.body_expr,
                type_aliases=ast.type_aliases,
                available_modules=available_modules,
                available_values=available_values,
                bound_names=bound_names,
                module=module,
            )
        _validate_stmt_closure(
            module.statements,
            type_aliases=ast.type_aliases,
            available_modules=available_modules,
            available_values=available_values,
            bound_names=bound_names,
            module=module,
        )


def warn_unused_import_diagnostics(
    *,
    file_path: Path,
    ast: AxonFile,
    usage: _ImportUsageLike,
    enabled: bool = True,
) -> list[ValidationDiagnostic]:
    if not enabled:
        return []
    out: list[ValidationDiagnostic] = []
    for namespace in ast.imports:
        members = ast.imported_members.get(namespace)
        if members:
            for member in members:
                if member not in usage.unqualified_symbols:
                    out.append(
                        ValidationDiagnostic(
                            level="warning",
                            message=f"unused unqualified import {namespace}.{member}",
                            file_path=file_path,
                        )
                    )
            continue
        if namespace not in usage.qualified_namespaces:
            out.append(
                ValidationDiagnostic(
                    level="warning",
                    message=f"unused qualified import {namespace}",
                    file_path=file_path,
                )
            )
    return out


def warn_unused_definitions(
    *,
    all_module_names: tuple[str, ...],
    root_entrypoint: str | None,
    reachable_modules: set[str],
    all_value_names: set[str],
    reachable_values: set[str],
    module_sources: dict[str, Path],
    value_sources: dict[str, Path],
    builtins_dir: Path,
) -> list[ValidationDiagnostic]:
    out: list[ValidationDiagnostic] = []
    for module_name in all_module_names:
        if module_name == root_entrypoint:
            continue
        if module_name not in reachable_modules:
            source = module_sources.get(module_name)
            if source is not None and builtins_dir in source.parents:
                continue
            out.append(
                ValidationDiagnostic(
                    level="warning",
                    message=f"unused definition {module_name}",
                    file_path=source,
                )
            )
    for value_name in sorted(all_value_names):
        if value_name not in reachable_values:
            source = value_sources.get(value_name)
            if source is not None and builtins_dir in source.parents:
                continue
            out.append(
                ValidationDiagnostic(
                    level="warning",
                    message=f"unused definition {value_name}",
                    file_path=source,
                )
            )
    return out


__all__ = [
    "validate_closed_axon_file",
    "warn_unused_definitions",
    "warn_unused_import_diagnostics",
]
