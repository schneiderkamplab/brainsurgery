from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ..ast.nodes import (
    AxonBind,
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
    AxonExprPath,
    AxonExprPipe,
    AxonExprTernary,
    AxonExprTuple,
    AxonKwargValue,
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
from ..load import LoadedAxonFile, LoadedAxonProgram
from ..validate import (
    ValidationDiagnostic,
    validate_axon_program,
    validate_closed_axon_file,
    warn_unused_definitions,
)

_PATH_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _split_callable_surface_name(name: str) -> tuple[str, str]:
    indexes = [idx for idx in (name.find("@"), name.find("::")) if idx >= 0]
    if not indexes:
        return name, ""
    idx = min(indexes)
    return name[:idx], name[idx:]


def _path_placeholder_names(text: str) -> set[str]:
    return {match.group(1) for match in _PATH_PLACEHOLDER_RE.finditer(text)}


def _collect_surface_placeholder_refs(
    surface: str,
    *,
    bound_names: set[str],
    value_names: set[str],
) -> set[str]:
    return {
        name
        for name in _path_placeholder_names(surface)
        if name not in bound_names and name in value_names
    }


def _build_surface_modules(
    ast: AxonFile,
    *,
    validate: bool = False,
    extra_imported_members: dict[str, tuple[str, ...]] | None = None,
) -> tuple[AxonDefinition, ...]:
    modules: list[AxonDefinition] = []
    for module in ast.modules:
        if isinstance(module.body_expr, AxonExprDo) and not module.body_expr.inline:
            statements = module.body_expr.body
        elif isinstance(module.body_expr, AxonExpr):
            statements = (AxonReturn(values=(module.body_expr,)),)
        else:
            statements = module.statements
        modules.append(
            AxonDefinition(
                name=module.name,
                path_param=module.path_param,
                path_params=module.path_params,
                params=module.params,
                returns=module.returns,
                statements=statements,
                body_expr=None,
                imports=ast.imports,
                imported_members=extra_imported_members or dict(ast.imported_members) or None,
                exports=ast.exports,
                symbols=None,
                pragmas=None,
                type_aliases=dict(ast.type_aliases) or None,
                return_type_expr=module.return_type_expr,
            )
        )
    out = tuple(modules)
    if validate:
        validate_axon_program(out)
    return out


@dataclass(frozen=True)
class _LoadedSyntaxFile:
    path: Path
    namespace: str | None
    ast: AxonFile
    effective_imported_members: dict[str, tuple[str, ...]]


def _effective_imported_members(
    loaded: LoadedAxonFile, by_namespace: dict[str, LoadedAxonFile]
) -> dict[str, tuple[str, ...]]:
    effective: dict[str, tuple[str, ...]] = dict(loaded.ast.imported_members)
    for namespace in loaded.ast.imports:
        if namespace in effective:
            continue
        dep = by_namespace.get(namespace)
        if dep is None or not dep.ast.exports:
            continue
        effective[namespace] = dep.ast.exports
    return effective


def _local_module_names(ast: AxonFile) -> tuple[str, ...]:
    return tuple(module.name for module in ast.modules)


def _collect_type_dim_names(
    type_expr: TypeExpr | None,
    *,
    type_aliases: dict[str, TypeAliasDef] | None = None,
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
        alias_map = type_aliases if isinstance(type_aliases, dict) else {}
        alias_def = alias_map.get(root.name)
        if alias_def is None or root.name in stack:
            return set(arg for arg in root.args if isinstance(arg, str))
        if len(root.args) != len(alias_def.params):
            return set(arg for arg in root.args if isinstance(arg, str))
        subst = {name: arg for name, arg in zip(alias_def.params, root.args, strict=True)}
        instantiated = alias_def.value
        if subst:
            instantiated = _substitute_alias_dims(instantiated, subst=subst)
        return _collect_type_dim_names(
            instantiated, type_aliases=alias_map, stack=(*stack, root.name)
        )
    if isinstance(root, TypeVar):
        return set()
    return set()


def _substitute_alias_dims(tp: TypeExpr, *, subst: dict[str, DimToken]) -> TypeExpr:
    def _sub_dim(dim: DimToken) -> DimToken:
        if isinstance(dim, str):
            return subst.get(dim, dim)
        if isinstance(dim, int):
            return dim
        if isinstance(dim, DimExprBinary):
            return DimExprBinary(op=dim.op, left=_sub_dim(dim.left), right=_sub_dim(dim.right))
        raise TypeError(f"unsupported dim token {dim!r}")

    if isinstance(tp, TypeOptional):
        return TypeOptional(inner=_substitute_alias_dims(tp.inner, subst=subst))
    if isinstance(tp, TypeList):
        return TypeList(item=_substitute_alias_dims(tp.item, subst=subst))
    if isinstance(tp, TypeTuple):
        return TypeTuple(
            items=tuple(_substitute_alias_dims(item, subst=subst) for item in tp.items)
        )
    if isinstance(tp, TypeTensor):
        return TypeTensor(base=tp.base, dims=tuple(_sub_dim(dim) for dim in tp.dims))
    if isinstance(tp, TypeNamed) and tp.args:
        return TypeNamed(name=tp.name, args=tuple(_sub_dim(dim) for dim in tp.args))
    return tp


def _module_bound_names(
    module: AxonDefinition,
    *,
    type_aliases: dict[str, TypeAliasDef] | None = None,
    include_type_dims: bool = True,
) -> set[str]:
    bound_names = {param.name for param in module.params}
    bound_names.update(name for name in module.path_params if isinstance(name, str))
    if isinstance(module.path_param, str):
        bound_names.add(module.path_param)
    if include_type_dims:
        for param in module.params:
            bound_names.update(_collect_type_dim_names(param.type_expr, type_aliases=type_aliases))
        bound_names.update(_collect_type_dim_names(module.return_type_expr, type_aliases=type_aliases))
    return bound_names


def _module_type_definition_refs(
    module: AxonDefinition,
    *,
    definition_names: set[str],
    type_aliases: dict[str, TypeAliasDef] | None = None,
) -> set[str]:
    refs: set[str] = set()
    for param in module.params:
        refs.update(
            name
            for name in _collect_type_dim_names(param.type_expr, type_aliases=type_aliases)
            if name in definition_names
        )
    refs.update(
        name
        for name in _collect_type_dim_names(module.return_type_expr, type_aliases=type_aliases)
        if name in definition_names
    )
    return refs


def _merge_type_aliases(
    existing: dict[str, TypeAliasDef],
    incoming: dict[str, TypeAliasDef],
    *,
    file_path: Path,
) -> None:
    for name, value in incoming.items():
        prev = existing.get(name)
        if prev is not None and prev != value:
            raise ValueError(f"{file_path}: conflicting type alias {name!r}")
        existing.setdefault(name, value)


def _build_unqualified_target_map(
    *,
    loaded: _LoadedSyntaxFile,
    local_modules: tuple[str, ...],
    namespace_to_loaded: dict[str, _LoadedSyntaxFile],
) -> dict[str, tuple[str, ...]]:
    targets: dict[str, list[str]] = {}

    for module_name in local_modules:
        canonical = f"{loaded.namespace}.{module_name}" if loaded.namespace else module_name
        targets.setdefault(module_name, []).append(canonical)

    for namespace, members in loaded.effective_imported_members.items():
        dep = namespace_to_loaded[namespace]
        dep_modules = set(_local_module_names(dep.ast))
        dep_aliases = set(dep.ast.type_aliases)
        dep_namespaces = set(dep.ast.imports) | set(dep.effective_imported_members)
        for member in members:
            in_module = member in dep_modules
            in_alias = member in dep_aliases
            active_kinds = sum(int(flag) for flag in (in_module, in_alias))
            if active_kinds > 1:
                raise ValueError(
                    f"{loaded.path}: imported member {namespace}.{member} resolves to multiple definition kinds"
                )
            if active_kinds == 0:
                if member in dep_namespaces:
                    targets.setdefault(member, []).append(member)
                    continue
                raise ValueError(
                    f"{loaded.path}: imported member {namespace}.{member} is unresolved"
                )
            if in_module:
                canonical = f"{namespace}.{member}"
            else:
                canonical = member
            targets.setdefault(member, []).append(canonical)
    return {name: tuple(dict.fromkeys(values)) for name, values in targets.items()}


def _build_qualified_target_map(
    *,
    loaded: _LoadedSyntaxFile,
    namespace_to_loaded: dict[str, _LoadedSyntaxFile],
) -> dict[str, str]:
    targets: dict[str, str] = {}
    for namespace in loaded.ast.imports:
        dep = namespace_to_loaded[namespace]
        dep_modules = set(_local_module_names(dep.ast))
        dep_aliases = set(dep.ast.type_aliases)
        dep_namespaces = set(dep.ast.imports) | set(dep.effective_imported_members)
        for member in dep.ast.exports:
            in_module = member in dep_modules
            in_alias = member in dep_aliases
            active_kinds = sum(int(flag) for flag in (in_module, in_alias))
            if active_kinds > 1:
                raise ValueError(
                    f"{loaded.path}: imported member {namespace}.{member} resolves to multiple definition kinds"
                )
            if active_kinds == 0:
                if member in dep_namespaces:
                    targets[f"{namespace}.{member}"] = member
                    continue
                continue
            if in_module:
                canonical = f"{namespace}.{member}"
            else:
                canonical = member
            targets[f"{namespace}.{member}"] = canonical
    return targets


def _rewrite_expr(
    expr: AxonExpr,
    *,
    unqualified_targets: dict[str, tuple[str, ...]],
    qualified_targets: dict[str, str],
    bound_names: set[str],
    module_name: str,
) -> AxonExpr:
    def _resolve_unqualified(name: str) -> str:
        targets = unqualified_targets.get(name, ())
        if not targets:
            return name
        if len(targets) > 1:
            choices = ", ".join(sorted(targets))
            raise ValueError(
                f"Axon import resolution failed in module {module_name!r}: ambiguous imported member "
                f"{name!r}; canonical targets: {choices}"
            )
        return targets[0]

    def _resolve_name(name: str) -> str:
        if name in qualified_targets:
            return qualified_targets[name]
        if "." in name or name in bound_names:
            return name
        return _resolve_unqualified(name)

    def _resolve_callable_surface_name(name: str) -> str:
        base, suffix = _split_callable_surface_name(name)
        resolved_base = _resolve_name(base)
        return f"{resolved_base}{suffix}"

    def _rewrite_kwarg_value(value: AxonKwargValue) -> AxonKwargValue:
        if isinstance(value, AxonExpr):
            return _rewrite_expr(
                value,
                unqualified_targets=unqualified_targets,
                qualified_targets=qualified_targets,
                bound_names=bound_names,
                module_name=module_name,
            )
        return value

    if isinstance(expr, AxonExprName):
        resolved = _resolve_callable_surface_name(expr.name)
        return expr if resolved == expr.name else AxonExprName(name=resolved)
    if isinstance(expr, AxonExprCall):
        callee = _resolve_callable_surface_name(expr.callee)
        return AxonExprCall(
            callee=callee,
            args=tuple(
                _rewrite_expr(
                    arg,
                    unqualified_targets=unqualified_targets,
                    qualified_targets=qualified_targets,
                    bound_names=bound_names,
                    module_name=module_name,
                )
                for arg in expr.args
            ),
            kwargs={key: _rewrite_kwarg_value(value) for key, value in expr.kwargs.items()},
        )
    if isinstance(expr, AxonExprPipe):
        return AxonExprPipe(
            value=_rewrite_expr(
                expr.value,
                unqualified_targets=unqualified_targets,
                qualified_targets=qualified_targets,
                bound_names=bound_names,
                module_name=module_name,
            ),
            stages=tuple(
                _rewrite_expr(
                    stage,
                    unqualified_targets=unqualified_targets,
                    qualified_targets=qualified_targets,
                    bound_names=bound_names,
                    module_name=module_name,
                )
                for stage in expr.stages
            ),
        )
    if isinstance(expr, AxonExprBind):
        value = _rewrite_expr(
            expr.value,
            unqualified_targets=unqualified_targets,
            qualified_targets=qualified_targets,
            bound_names=bound_names,
            module_name=module_name,
        )
        nested_bound = set(bound_names)
        nested_bound.add(expr.var)
        body = _rewrite_expr(
            expr.body,
            unqualified_targets=unqualified_targets,
            qualified_targets=qualified_targets,
            bound_names=nested_bound,
            module_name=module_name,
        )
        return AxonExprBind(value=value, var=expr.var, body=body)
    if isinstance(expr, AxonExprIf):
        return AxonExprIf(
            cond=_rewrite_expr(
                expr.cond,
                unqualified_targets=unqualified_targets,
                qualified_targets=qualified_targets,
                bound_names=bound_names,
                module_name=module_name,
            ),
            true_expr=_rewrite_expr(
                expr.true_expr,
                unqualified_targets=unqualified_targets,
                qualified_targets=qualified_targets,
                bound_names=bound_names,
                module_name=module_name,
            ),
            false_expr=_rewrite_expr(
                expr.false_expr,
                unqualified_targets=unqualified_targets,
                qualified_targets=qualified_targets,
                bound_names=bound_names,
                module_name=module_name,
            ),
        )
    if isinstance(expr, AxonExprTernary):
        return AxonExprTernary(
            cond=_rewrite_expr(
                expr.cond,
                unqualified_targets=unqualified_targets,
                qualified_targets=qualified_targets,
                bound_names=bound_names,
                module_name=module_name,
            ),
            true_expr=_rewrite_expr(
                expr.true_expr,
                unqualified_targets=unqualified_targets,
                qualified_targets=qualified_targets,
                bound_names=bound_names,
                module_name=module_name,
            ),
            false_expr=_rewrite_expr(
                expr.false_expr,
                unqualified_targets=unqualified_targets,
                qualified_targets=qualified_targets,
                bound_names=bound_names,
                module_name=module_name,
            ),
        )
    if isinstance(expr, AxonExprBinary):
        return AxonExprBinary(
            op=expr.op,
            left=_rewrite_expr(
                expr.left,
                unqualified_targets=unqualified_targets,
                qualified_targets=qualified_targets,
                bound_names=bound_names,
                module_name=module_name,
            ),
            right=_rewrite_expr(
                expr.right,
                unqualified_targets=unqualified_targets,
                qualified_targets=qualified_targets,
                bound_names=bound_names,
                module_name=module_name,
            ),
        )
    if isinstance(expr, AxonExprLambda):
        nested_bound = set(bound_names)
        nested_bound.add(expr.var)
        return AxonExprLambda(
            var=expr.var,
            body=_rewrite_expr(
                expr.body,
                unqualified_targets=unqualified_targets,
                qualified_targets=qualified_targets,
                bound_names=nested_bound,
                module_name=module_name,
            ),
        )
    if isinstance(expr, AxonExprParen):
        return AxonExprParen(
            inner=_rewrite_expr(
                expr.inner,
                unqualified_targets=unqualified_targets,
                qualified_targets=qualified_targets,
                bound_names=bound_names,
                module_name=module_name,
            )
        )
    if isinstance(expr, AxonExprAscribe):
        return AxonExprAscribe(
            expr=_rewrite_expr(
                expr.expr,
                unqualified_targets=unqualified_targets,
                qualified_targets=qualified_targets,
                bound_names=bound_names,
                module_name=module_name,
            ),
            type_expr=expr.type_expr,
        )
    if isinstance(expr, AxonExprList):
        return AxonExprList(
            items=tuple(
                _rewrite_expr(
                    item,
                    unqualified_targets=unqualified_targets,
                    qualified_targets=qualified_targets,
                    bound_names=bound_names,
                    module_name=module_name,
                )
                for item in expr.items
            )
        )
    if isinstance(expr, AxonExprTuple):
        return AxonExprTuple(
            items=tuple(
                _rewrite_expr(
                    item,
                    unqualified_targets=unqualified_targets,
                    qualified_targets=qualified_targets,
                    bound_names=bound_names,
                    module_name=module_name,
                )
                for item in expr.items
            )
        )
    if isinstance(expr, AxonExprDo):
        return AxonExprDo(
            body=_rewrite_statements(
                expr.body,
                unqualified_targets=unqualified_targets,
                qualified_targets=qualified_targets,
                bound_names=set(bound_names),
                module_name=module_name,
            ),
            inline=expr.inline,
        )
    return expr


def _rewrite_statements(
    statements: tuple[AxonStatement, ...],
    *,
    unqualified_targets: dict[str, tuple[str, ...]],
    qualified_targets: dict[str, str],
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
                        unqualified_targets=unqualified_targets,
                        qualified_targets=qualified_targets,
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
                        _rewrite_expr(
                            value,
                            unqualified_targets=unqualified_targets,
                            qualified_targets=qualified_targets,
                            bound_names=local_bound,
                            module_name=module_name,
                        )
                        for value in stmt.values
                    )
                )
            )
            continue
        if isinstance(stmt, AxonYield):
            rewritten.append(
                AxonYield(
                    values=tuple(
                        _rewrite_expr(
                            value,
                            unqualified_targets=unqualified_targets,
                            qualified_targets=qualified_targets,
                            bound_names=local_bound,
                            module_name=module_name,
                        )
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
                        stmt.to_expr,
                        unqualified_targets=unqualified_targets,
                        qualified_targets=qualified_targets,
                        bound_names=local_bound,
                        module_name=module_name,
                    ),
                    from_expr=_rewrite_expr(
                        stmt.from_expr,
                        unqualified_targets=unqualified_targets,
                        qualified_targets=qualified_targets,
                        bound_names=local_bound,
                        module_name=module_name,
                    ),
                    step_expr=_rewrite_expr(
                        stmt.step_expr,
                        unqualified_targets=unqualified_targets,
                        qualified_targets=qualified_targets,
                        bound_names=local_bound,
                        module_name=module_name,
                    ),
                    body=_rewrite_statements(
                        stmt.body,
                        unqualified_targets=unqualified_targets,
                        qualified_targets=qualified_targets,
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
                        unqualified_targets=unqualified_targets,
                        qualified_targets=qualified_targets,
                        bound_names=set(local_bound),
                        module_name=module_name,
                    ),
                    kwargs={
                        key: (
                            _rewrite_expr(
                                value,
                                unqualified_targets=unqualified_targets,
                                qualified_targets=qualified_targets,
                                bound_names=local_bound,
                                module_name=module_name,
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


def _rewrite_modules_for_loaded_file(
    *,
    loaded: _LoadedSyntaxFile,
    namespace_to_loaded: dict[str, _LoadedSyntaxFile],
) -> tuple[AxonDefinition, ...]:
    local_modules = _local_module_names(loaded.ast)
    unqualified_targets = _build_unqualified_target_map(
        loaded=loaded,
        local_modules=local_modules,
        namespace_to_loaded=namespace_to_loaded,
    )
    qualified_targets = _build_qualified_target_map(
        loaded=loaded,
        namespace_to_loaded=namespace_to_loaded,
    )
    modules = _build_surface_modules(
        loaded.ast,
        validate=False,
        extra_imported_members=loaded.effective_imported_members
        if loaded.effective_imported_members
        else None,
    )
    out: list[AxonDefinition] = []
    for raw_module, module in zip(loaded.ast.modules, modules, strict=True):
        canonical_name = f"{loaded.namespace}.{module.name}" if loaded.namespace else module.name
        bound_names = _module_bound_names(
            module,
            type_aliases=loaded.ast.type_aliases,
        )
        out.append(
            AxonDefinition(
                name=canonical_name,
                path_param=module.path_param,
                path_params=module.path_params,
                params=module.params,
                returns=module.returns,
                statements=_rewrite_statements(
                    module.statements,
                    unqualified_targets=unqualified_targets,
                    qualified_targets=qualified_targets,
                    bound_names=bound_names,
                    module_name=canonical_name,
                ),
                body_expr=(
                    None
                    if raw_module.body_expr is None
                    else _rewrite_expr(
                        raw_module.body_expr,
                        unqualified_targets=unqualified_targets,
                        qualified_targets=qualified_targets,
                        bound_names=bound_names,
                        module_name=canonical_name,
                    )
                ),
                imports=(),
                imported_members=None,
                exports=(),
                symbols=None,
                pragmas=None,
                type_aliases=None,
                return_type_expr=module.return_type_expr,
            )
        )
    return tuple(out)


def _collect_expr_refs(
    expr: AxonExpr,
    *,
    bound_names: set[str],
    module_names: set[str],
    value_names: set[str],
) -> tuple[set[str], set[str]]:
    module_refs: set[str] = set()
    value_refs: set[str] = set()
    if isinstance(expr, AxonExprName):
        base, surface = _split_callable_surface_name(expr.name)
        if base not in bound_names:
            if base in module_names or "." in base:
                module_refs.add(base)
            elif base in value_names:
                value_refs.add(base)
        value_refs.update(
            _collect_surface_placeholder_refs(
                surface,
                bound_names=bound_names,
                value_names=value_names,
            )
        )
        return module_refs, value_refs
    if isinstance(expr, AxonExprPath):
        for name in _collect_path_placeholders(expr):
            if name not in bound_names and name in value_names:
                value_refs.add(name)
        return module_refs, value_refs
    if isinstance(expr, AxonExprParen):
        return _collect_expr_refs(
            expr.inner,
            bound_names=bound_names,
            module_names=module_names,
            value_names=value_names,
        )
    if isinstance(expr, AxonExprAscribe):
        return _collect_expr_refs(
            expr.expr,
            bound_names=bound_names,
            module_names=module_names,
            value_names=value_names,
        )
    if isinstance(expr, AxonExprList):
        for item in expr.items:
            m, c = _collect_expr_refs(
                item,
                bound_names=bound_names,
                module_names=module_names,
                value_names=value_names,
            )
            module_refs.update(m)
            value_refs.update(c)
        return module_refs, value_refs
    if isinstance(expr, AxonExprTuple):
        for item in expr.items:
            m, c = _collect_expr_refs(
                item,
                bound_names=bound_names,
                module_names=module_names,
                value_names=value_names,
            )
            module_refs.update(m)
            value_refs.update(c)
        return module_refs, value_refs
    if isinstance(expr, AxonExprPipe):
        m, c = _collect_expr_refs(
            expr.value,
            bound_names=bound_names,
            module_names=module_names,
            value_names=value_names,
        )
        module_refs.update(m)
        value_refs.update(c)
        for stage in expr.stages:
            m, c = _collect_expr_refs(
                stage,
                bound_names=bound_names,
                module_names=module_names,
                value_names=value_names,
            )
            module_refs.update(m)
            value_refs.update(c)
        return module_refs, value_refs
    if isinstance(expr, AxonExprBind):
        m, c = _collect_expr_refs(
            expr.value,
            bound_names=bound_names,
            module_names=module_names,
            value_names=value_names,
        )
        module_refs.update(m)
        value_refs.update(c)
        nested_bound = set(bound_names)
        nested_bound.add(expr.var)
        m, c = _collect_expr_refs(
            expr.body,
            bound_names=nested_bound,
            module_names=module_names,
            value_names=value_names,
        )
        module_refs.update(m)
        value_refs.update(c)
        return module_refs, value_refs
    if isinstance(expr, AxonExprIf | AxonExprTernary):
        for subexpr in (expr.cond, expr.true_expr, expr.false_expr):
            m, c = _collect_expr_refs(
                subexpr,
                bound_names=bound_names,
                module_names=module_names,
                value_names=value_names,
            )
            module_refs.update(m)
            value_refs.update(c)
        return module_refs, value_refs
    if isinstance(expr, AxonExprBinary):
        for subexpr in (expr.left, expr.right):
            m, c = _collect_expr_refs(
                subexpr,
                bound_names=bound_names,
                module_names=module_names,
                value_names=value_names,
            )
            module_refs.update(m)
            value_refs.update(c)
        return module_refs, value_refs
    if isinstance(expr, AxonExprCall):
        base, surface = _split_callable_surface_name(expr.callee)
        if not base.startswith("_") and base not in bound_names:
            if base in module_names or "." in base:
                module_refs.add(base)
            elif base in value_names:
                value_refs.add(base)
        value_refs.update(
            _collect_surface_placeholder_refs(
                surface,
                bound_names=bound_names,
                value_names=value_names,
            )
        )
        for arg in expr.args:
            m, c = _collect_expr_refs(
                arg,
                bound_names=bound_names,
                module_names=module_names,
                value_names=value_names,
            )
            module_refs.update(m)
            value_refs.update(c)
        for value in expr.kwargs.values():
            if isinstance(value, AxonExpr):
                m, c = _collect_expr_refs(
                    value,
                    bound_names=bound_names,
                    module_names=module_names,
                    value_names=value_names,
                )
                module_refs.update(m)
                value_refs.update(c)
        return module_refs, value_refs
    if isinstance(expr, AxonExprLambda):
        nested_bound = set(bound_names)
        nested_bound.add(expr.var)
        return _collect_expr_refs(
            expr.body,
            bound_names=nested_bound,
            module_names=module_names,
            value_names=value_names,
        )
    if isinstance(expr, AxonExprDo):
        return _collect_statement_refs(
            expr.body,
            bound_names=set(bound_names),
            module_names=module_names,
            value_names=value_names,
        )
    return module_refs, value_refs


def _collect_statement_refs(
    statements: tuple[AxonStatement, ...],
    *,
    bound_names: set[str],
    module_names: set[str],
    value_names: set[str],
) -> tuple[set[str], set[str]]:
    module_refs: set[str] = set()
    value_refs: set[str] = set()
    local_bound = set(bound_names)
    for stmt in statements:
        if isinstance(stmt, AxonBind):
            m, c = _collect_expr_refs(
                stmt.expr,
                bound_names=local_bound,
                module_names=module_names,
                value_names=value_names,
            )
            module_refs.update(m)
            value_refs.update(c)
            for target in stmt.targets:
                if target != "_":
                    local_bound.add(target)
            continue
        if isinstance(stmt, AxonReturn | AxonYield):
            for value in stmt.values:
                m, c = _collect_expr_refs(
                    value,
                    bound_names=local_bound,
                    module_names=module_names,
                    value_names=value_names,
                )
                module_refs.update(m)
                value_refs.update(c)
            continue
        if isinstance(stmt, AxonRepeat):
            for subexpr in (stmt.from_expr, stmt.to_expr, stmt.step_expr):
                m, c = _collect_expr_refs(
                    subexpr,
                    bound_names=local_bound,
                    module_names=module_names,
                    value_names=value_names,
                )
                module_refs.update(m)
                value_refs.update(c)
            loop_bound = set(local_bound)
            loop_bound.add(stmt.var)
            if stmt.carry:
                for name in stmt.carry:
                    if name != "_":
                        loop_bound.add(name)
            m, c = _collect_statement_refs(
                stmt.body,
                bound_names=loop_bound,
                module_names=module_names,
                value_names=value_names,
            )
            module_refs.update(m)
            value_refs.update(c)
            if stmt.targets:
                for target in stmt.targets:
                    if target != "_":
                        local_bound.add(target)
            continue
        if isinstance(stmt, AxonScopeBind):
            for part in stmt.prefix.parts:
                for match in _PATH_PLACEHOLDER_RE.finditer(part):
                    name = match.group(1)
                    if name not in local_bound and name in value_names:
                        value_refs.add(name)
            for kwarg_value in stmt.kwargs.values():
                if isinstance(kwarg_value, AxonExpr):
                    m, c = _collect_expr_refs(
                        kwarg_value,
                        bound_names=local_bound,
                        module_names=module_names,
                        value_names=value_names,
                    )
                    module_refs.update(m)
                    value_refs.update(c)
            m, c = _collect_statement_refs(
                stmt.body,
                bound_names=set(local_bound),
                module_names=module_names,
                value_names=value_names,
            )
            module_refs.update(m)
            value_refs.update(c)
            for target in stmt.targets:
                if target != "_":
                    local_bound.add(target)
    return module_refs, value_refs


def _build_module_dependency_graph(
    modules: tuple[AxonDefinition, ...],
    *,
    type_aliases: dict[str, TypeAliasDef],
) -> dict[str, set[str]]:
    module_deps: dict[str, set[str]] = {}
    module_names = {module.name for module in modules}
    for module in modules:
        bound_names = _module_bound_names(
            module,
            type_aliases=type_aliases,
            include_type_dims=False,
        )
        mod_refs, value_refs = _collect_statement_refs(
            module.statements,
            bound_names=bound_names,
            module_names=module_names,
            value_names=module_names,
        )
        value_refs.update(
            _module_type_definition_refs(
                module,
                definition_names=module_names,
                type_aliases=type_aliases,
            )
        )
        if module.body_expr is not None:
            body_mod_refs, body_value_refs = _collect_expr_refs(
                module.body_expr,
                bound_names=bound_names,
                module_names=module_names,
                value_names=module_names,
            )
            mod_refs.update(body_mod_refs)
            value_refs.update(body_value_refs)
        module_deps[module.name] = mod_refs | value_refs
    return module_deps


def _reachable_definitions(
    *,
    entrypoint: str | None,
    module_graph: dict[str, set[str]],
) -> set[str]:
    reachable_modules: set[str] = set()
    module_stack: list[str] = [entrypoint] if entrypoint is not None else []
    while module_stack:
        current = module_stack.pop()
        if current in reachable_modules:
            continue
        reachable_modules.add(current)
        module_stack.extend(
            dep for dep in module_graph.get(current, set()) if dep not in reachable_modules
        )
    return reachable_modules


def reachable_definitions(program: AxonFile, *, entrypoint: str | None = None) -> set[str]:
    """Return definition names reachable from an entrypoint in a closed program."""

    root_entrypoint = entrypoint
    if root_entrypoint is None:
        pragma_main = program.pragmas.get("main")
        if isinstance(pragma_main, str) and pragma_main:
            root_entrypoint = pragma_main
    if root_entrypoint is None and program.modules:
        root_entrypoint = program.modules[-1].name
    module_graph = _build_module_dependency_graph(
        program.modules,
        type_aliases=program.type_aliases,
    )
    return _reachable_definitions(
        entrypoint=root_entrypoint,
        module_graph=module_graph,
    )


def prune_unreachable_definitions(program: AxonFile, *, entrypoint: str | None = None) -> AxonFile:
    """Drop closed-program definitions unreachable from the selected entrypoint."""

    reachable_modules = reachable_definitions(program, entrypoint=entrypoint)
    return AxonFile(
        modules=tuple(module for module in program.modules if module.name in reachable_modules),
        imports=program.imports,
        imported_members=program.imported_members,
        exports=program.exports,
        pragmas=program.pragmas,
        type_aliases=program.type_aliases,
        origin_path=program.origin_path,
    )


def _resolved_unused_definition_diagnostics(
    program: AxonFile,
    *,
    loaded_files: tuple[_LoadedSyntaxFile, ...],
    root: Path,
    builtins_dir: Path,
) -> tuple[ValidationDiagnostic, ...]:
    module_sources: dict[str, Path] = {}
    root_module_names: tuple[str, ...] = ()
    for loaded in loaded_files:
        if loaded.path == root:
            root_module_names = tuple(module.name for module in loaded.ast.modules)
        for module in loaded.ast.modules:
            canonical = f"{loaded.namespace}.{module.name}" if loaded.namespace else module.name
            module_sources.setdefault(canonical, loaded.path)

    root_entrypoint = program.pragmas.get("main")
    if not isinstance(root_entrypoint, str) or not root_entrypoint:
        root_entrypoint = root_module_names[-1] if root_module_names else None
    reachable_modules = reachable_definitions(program, entrypoint=root_entrypoint)
    return tuple(
        warn_unused_definitions(
            all_module_names=tuple(module.name for module in program.modules),
            root_entrypoint=root_entrypoint,
            reachable_modules=reachable_modules,
            all_value_names=set(),
            reachable_values=set(),
            module_sources=module_sources,
            value_sources={},
            builtins_dir=builtins_dir,
        )
    )


def _collect_path_placeholders(expr: AxonExpr) -> set[str]:
    placeholders: set[str] = set()
    if isinstance(expr, AxonExprPath):
        for part in expr.parts:
            for match in _PATH_PLACEHOLDER_RE.finditer(part):
                placeholders.add(match.group(1))
    return placeholders


def _validate_surface_placeholders(
    surface: str,
    *,
    bound_names: set[str],
    value_names: set[str],
    current_module: str,
) -> None:
    unresolved = _path_placeholder_names(surface).difference(bound_names | value_names)
    if unresolved:
        missing = ", ".join(sorted(unresolved))
        raise ValueError(
            f"resolver produced unresolved path placeholders [{missing}] in module {current_module!r}"
        )


def _validate_module_expr(
    expr: AxonExpr,
    *,
    bound_names: set[str],
    module_names: set[str],
    value_names: set[str],
    current_module: str,
) -> None:
    if isinstance(expr, AxonExprName):
        base, surface = _split_callable_surface_name(expr.name)
        _validate_surface_placeholders(
            surface,
            bound_names=bound_names,
            value_names=value_names,
            current_module=current_module,
        )
        if base.startswith("_"):
            return
        if base in bound_names:
            return
        if base in value_names or base in module_names:
            return
        raise ValueError(
            f"resolver produced unresolved name {expr.name!r} in module {current_module!r}"
        )
    if isinstance(expr, AxonExprCall):
        base, surface = _split_callable_surface_name(expr.callee)
        _validate_surface_placeholders(
            surface,
            bound_names=bound_names,
            value_names=value_names,
            current_module=current_module,
        )
        if base not in bound_names and not base.startswith("_") and base not in module_names:
            if base in value_names:
                raise ValueError(
                    f"resolver produced invalid callable value definition {base!r} in module {current_module!r}"
                )
            raise ValueError(
                f"resolver produced unresolved callee {base!r} in module {current_module!r}"
            )
        for arg in expr.args:
            _validate_module_expr(
                arg,
                bound_names=bound_names,
                module_names=module_names,
                value_names=value_names,
                current_module=current_module,
            )
        for value in expr.kwargs.values():
            if isinstance(value, AxonExpr):
                _validate_module_expr(
                    value,
                    bound_names=bound_names,
                    module_names=module_names,
                    value_names=value_names,
                    current_module=current_module,
                )
        return
    if isinstance(expr, AxonExprPipe):
        _validate_module_expr(
            expr.value,
            bound_names=bound_names,
            module_names=module_names,
            value_names=value_names,
            current_module=current_module,
        )
        for stage in expr.stages:
            _validate_module_expr(
                stage,
                bound_names=bound_names,
                module_names=module_names,
                value_names=value_names,
                current_module=current_module,
            )
        return
    if isinstance(expr, AxonExprBind):
        _validate_module_expr(
            expr.value,
            bound_names=bound_names,
            module_names=module_names,
            value_names=value_names,
            current_module=current_module,
        )
        nested_bound = set(bound_names)
        nested_bound.add(expr.var)
        _validate_module_expr(
            expr.body,
            bound_names=nested_bound,
            module_names=module_names,
            value_names=value_names,
            current_module=current_module,
        )
        return
    if isinstance(expr, AxonExprIf | AxonExprTernary):
        for subexpr in (expr.cond, expr.true_expr, expr.false_expr):
            _validate_module_expr(
                subexpr,
                bound_names=bound_names,
                module_names=module_names,
                value_names=value_names,
                current_module=current_module,
            )
        return
    if isinstance(expr, AxonExprBinary):
        for subexpr in (expr.left, expr.right):
            _validate_module_expr(
                subexpr,
                bound_names=bound_names,
                module_names=module_names,
                value_names=value_names,
                current_module=current_module,
            )
        return
    if isinstance(expr, AxonExprLambda):
        nested_bound = set(bound_names)
        nested_bound.add(expr.var)
        _validate_module_expr(
            expr.body,
            bound_names=nested_bound,
            module_names=module_names,
            value_names=value_names,
            current_module=current_module,
        )
        return
    if isinstance(expr, AxonExprParen):
        _validate_module_expr(
            expr.inner,
            bound_names=bound_names,
            module_names=module_names,
            value_names=value_names,
            current_module=current_module,
        )
        return
    if isinstance(expr, AxonExprList):
        for item in expr.items:
            _validate_module_expr(
                item,
                bound_names=bound_names,
                module_names=module_names,
                value_names=value_names,
                current_module=current_module,
            )
        return
    if isinstance(expr, AxonExprTuple):
        for item in expr.items:
            _validate_module_expr(
                item,
                bound_names=bound_names,
                module_names=module_names,
                value_names=value_names,
                current_module=current_module,
            )
        return
    if isinstance(expr, AxonExprDo):
        _validate_module_statements(
            expr.body,
            bound_names=set(bound_names),
            module_names=module_names,
            value_names=value_names,
            current_module=current_module,
        )
        return
    placeholders = _collect_path_placeholders(expr)
    unresolved = placeholders.difference(bound_names | value_names)
    if unresolved:
        missing = ", ".join(sorted(unresolved))
        raise ValueError(
            f"resolver produced unresolved path placeholders [{missing}] in module {current_module!r}"
        )


def _validate_module_statements(
    statements: tuple[AxonStatement, ...],
    *,
    bound_names: set[str],
    module_names: set[str],
    value_names: set[str],
    current_module: str,
) -> None:
    local_bound = set(bound_names)
    for stmt in statements:
        if isinstance(stmt, AxonBind):
            _validate_module_expr(
                stmt.expr,
                bound_names=local_bound,
                module_names=module_names,
                value_names=value_names,
                current_module=current_module,
            )
            for target in stmt.targets:
                if target != "_":
                    local_bound.add(target)
            continue
        if isinstance(stmt, AxonReturn | AxonYield):
            for value in stmt.values:
                _validate_module_expr(
                    value,
                    bound_names=local_bound,
                    module_names=module_names,
                    value_names=value_names,
                    current_module=current_module,
                )
            continue
        if isinstance(stmt, AxonRepeat):
            for subexpr in (stmt.from_expr, stmt.to_expr, stmt.step_expr):
                _validate_module_expr(
                    subexpr,
                    bound_names=local_bound,
                    module_names=module_names,
                    value_names=value_names,
                    current_module=current_module,
                )
            loop_bound = set(local_bound)
            loop_bound.add(stmt.var)
            if stmt.carry:
                for name in stmt.carry:
                    if name != "_":
                        loop_bound.add(name)
            _validate_module_statements(
                stmt.body,
                bound_names=loop_bound,
                module_names=module_names,
                value_names=value_names,
                current_module=current_module,
            )
            if stmt.targets:
                for target in stmt.targets:
                    if target != "_":
                        local_bound.add(target)
            continue
        if isinstance(stmt, AxonScopeBind):
            placeholders = {
                match.group(1)
                for part in stmt.prefix.parts
                for match in _PATH_PLACEHOLDER_RE.finditer(part)
            }
            unresolved = placeholders.difference(local_bound | value_names)
            if unresolved:
                missing = ", ".join(sorted(unresolved))
                raise ValueError(
                    f"resolver produced unresolved scope placeholders [{missing}] in module {current_module!r}"
                )
            for kwarg_value in stmt.kwargs.values():
                if isinstance(kwarg_value, AxonExpr):
                    _validate_module_expr(
                        kwarg_value,
                        bound_names=local_bound,
                        module_names=module_names,
                        value_names=value_names,
                        current_module=current_module,
                    )
            _validate_module_statements(
                stmt.body,
                bound_names=set(local_bound),
                module_names=module_names,
                value_names=value_names,
                current_module=current_module,
            )
            for target in stmt.targets:
                if target != "_":
                    local_bound.add(target)


def _validate_resolved_program(
    *,
    modules: tuple[AxonDefinition, ...],
    type_aliases: dict[str, TypeAliasDef],
) -> None:
    names = [module.name for module in modules]
    if len(set(names)) != len(names):
        raise ValueError("resolver produced duplicate module names after canonicalization")
    value_names = set(names)
    for module in modules:
        if module.imports or module.imported_members:
            raise ValueError(f"resolver produced unresolved imports in module {module.name!r}")
    module_names = set(names)
    for module in modules:
        bound_names = _module_bound_names(module, type_aliases=type_aliases)
        _validate_module_statements(
            module.statements,
            bound_names=bound_names,
            module_names=module_names,
            value_names=value_names,
            current_module=module.name,
        )
    validate_axon_program(modules)


def resolve_loaded_axon_files(
    loaded_program: LoadedAxonProgram,
) -> tuple[AxonFile, tuple[ValidationDiagnostic, ...]]:
    for loaded_file in loaded_program.files:
        validate_axon_program(loaded_file.ast.modules)

    root = loaded_program.root_path
    by_namespace: dict[str, LoadedAxonFile] = {
        loaded.namespace: loaded for loaded in loaded_program.files if loaded.namespace is not None
    }
    ordered_files = tuple(
        _LoadedSyntaxFile(
            path=loaded.path,
            namespace=loaded.namespace,
            ast=loaded.ast,
            effective_imported_members=_effective_imported_members(loaded, by_namespace),
        )
        for loaded in loaded_program.files
    )
    namespace_to_loaded: dict[str, _LoadedSyntaxFile] = {
        loaded.namespace: loaded for loaded in ordered_files if loaded.namespace is not None
    }
    merged_type_aliases: dict[str, TypeAliasDef] = {}
    root_pragmas: dict[str, object] = {}

    for loaded in ordered_files:
        if loaded.path == root:
            root_pragmas = dict(loaded.ast.pragmas)
        _merge_type_aliases(
            merged_type_aliases,
            loaded.ast.type_aliases,
            file_path=loaded.path,
        )

    ordered_modules: list[AxonDefinition] = []
    for loaded in ordered_files:
        ordered_modules.extend(
            _rewrite_modules_for_loaded_file(
                loaded=loaded,
                namespace_to_loaded=namespace_to_loaded,
            )
        )
    ordered_modules_tuple = tuple(ordered_modules)

    _validate_resolved_program(
        modules=ordered_modules_tuple,
        type_aliases=merged_type_aliases,
    )
    canonical_modules = tuple(
        module
        if module.body_expr is None
        else AxonDefinition(
            name=module.name,
            path_param=module.path_param,
            path_params=module.path_params,
            params=module.params,
            returns=module.returns,
            statements=(),
            body_expr=module.body_expr,
            imports=(),
            imported_members=None,
            exports=(),
            symbols=None,
            pragmas=None,
            type_aliases=None,
            return_type_expr=module.return_type_expr,
        )
        for module in ordered_modules_tuple
    )
    out = AxonFile(
        modules=canonical_modules,
        imports=(),
        imported_members={},
        exports=(),
        pragmas=root_pragmas,
        type_aliases=dict(merged_type_aliases),
        origin_path=root,
    )
    validate_closed_axon_file(out)
    diagnostics = _resolved_unused_definition_diagnostics(
        out,
        loaded_files=ordered_files,
        root=root,
        builtins_dir=loaded_program.builtins_dir,
    )
    pruned = prune_unreachable_definitions(out)
    validate_closed_axon_file(pruned)
    return (pruned, diagnostics)


ResolveDiagnostic = ValidationDiagnostic


__all__ = [
    "ResolveDiagnostic",
    "prune_unreachable_definitions",
    "reachable_definitions",
    "resolve_loaded_axon_files",
]
