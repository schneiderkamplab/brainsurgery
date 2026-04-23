from __future__ import annotations

from dataclasses import dataclass

from .ast import AxonFile, AxonModule, TypeExpr, TypeTuple


@dataclass(frozen=True)
class ModuleSignature:
    param_names: tuple[str, ...]
    params: tuple[TypeExpr | None, ...]
    returns: tuple[TypeExpr, ...]


def _surface_modules_from_file(ast: AxonFile) -> tuple[AxonModule, ...]:
    return ast.modules


def _module_return_types(module: AxonModule) -> tuple[TypeExpr, ...]:
    if module.return_type_expr is None:
        return ()
    if isinstance(module.return_type_expr, TypeTuple):
        return tuple(module.return_type_expr.items)
    return (module.return_type_expr,)


def _build_module_signatures_for_closed_program(
    modules: AxonFile | tuple[AxonModule, ...], *, main_module: str | None = None
) -> dict[str, ModuleSignature]:
    if isinstance(modules, AxonFile):
        modules = _surface_modules_from_file(modules)
    if not modules:
        return {}
    by_name = {module.name: module for module in modules}
    if len(by_name) != len(modules):
        raise ValueError("Axon signature build failed: duplicate module names")
    selected_main = modules[-1].name if main_module is None else main_module
    if selected_main not in by_name:
        raise ValueError(f"Axon signature build failed: unknown main module {selected_main!r}")
    return {
        module.name: ModuleSignature(
            param_names=tuple(param.name for param in module.params),
            params=tuple(param.type_expr for param in module.params),
            returns=_module_return_types(module),
        )
        for module in modules
    }


__all__ = ["ModuleSignature", "_build_module_signatures_for_closed_program"]
