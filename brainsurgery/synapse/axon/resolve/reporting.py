from __future__ import annotations

from pathlib import Path

from ..ast import AxonExpr, AxonExprDo, AxonFile, AxonDefinition, AxonReturn
from ..load import LoadedAxonFile, LoadedAxonProgram
from ..validate import (
    ValidationDiagnostic,
    warn_unused_definitions,
    warn_unused_import_diagnostics,
)
from .core import reachable_definitions
from .usage import collect_import_usage


def _local_module_names(ast: AxonFile) -> tuple[str, ...]:
    return tuple(module.name for module in ast.modules)


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


def _surface_modules(
    ast: AxonFile,
    *,
    imported_members: dict[str, tuple[str, ...]],
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
                imported_members=imported_members or dict(ast.imported_members) or None,
                exports=ast.exports,
                symbols=None,
                pragmas=None,
                type_aliases=dict(ast.type_aliases) or None,
                return_type_expr=module.return_type_expr,
            )
        )
    return tuple(modules)


def resolve_validation_diagnostics(
    loaded: LoadedAxonProgram,
    resolved: AxonFile,
) -> tuple[ValidationDiagnostic, ...]:
    """Report non-fatal validation diagnostics for resolved Axon programs."""

    by_namespace = {
        loaded_file.namespace: loaded_file
        for loaded_file in loaded.files
        if loaded_file.namespace is not None
    }
    diagnostics: list[ValidationDiagnostic] = []
    for loaded_file in loaded.files:
        imported_members = _effective_imported_members(loaded_file, by_namespace)
        usage = collect_import_usage(
            _surface_modules(loaded_file.ast, imported_members=imported_members)
        )
        diagnostics.extend(
            warn_unused_import_diagnostics(
                file_path=loaded_file.path,
                ast=loaded_file.ast,
                usage=usage,
                enabled=loaded.builtins_dir not in loaded_file.path.parents,
            )
        )

    module_sources: dict[str, Path] = {}
    root_module_names: tuple[str, ...] = ()
    for loaded_file in loaded.files:
        namespace = loaded_file.namespace
        if loaded_file.path == loaded.root_path:
            root_module_names = _local_module_names(loaded_file.ast)
        for module_name in _local_module_names(loaded_file.ast):
            canonical_module = f"{namespace}.{module_name}" if namespace else module_name
            module_sources.setdefault(canonical_module, loaded_file.path)

    root_entrypoint = root_module_names[-1] if root_module_names else None
    reachable_modules = reachable_definitions(resolved, entrypoint=root_entrypoint)
    diagnostics.extend(
        warn_unused_definitions(
            all_module_names=tuple(module.name for module in resolved.modules),
            root_entrypoint=root_entrypoint,
            reachable_modules=reachable_modules,
            all_value_names=set(),
            reachable_values=set(),
            module_sources=module_sources,
            value_sources={},
            builtins_dir=loaded.builtins_dir,
        )
    )
    return tuple(diagnostics)


__all__ = ["resolve_validation_diagnostics"]
