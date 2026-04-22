import ast
from collections import defaultdict
from pathlib import Path

PACKAGE_ROOT = Path("brainsurgery")
TARGET_PACKAGES = (
    "core",
    "io",
    "engine",
    "algorithms",
    "expressions",
    "transforms",
    "cli",
    "web",
)

INTENTIONAL_UNUSED_PRIVATE_SYMBOLS = {
    "brainsurgery.cli.synapse:_resolve_axon_to_text",
}


def _package_exports(package: str) -> set[str]:
    init_path = PACKAGE_ROOT / package / "__init__.py"
    tree = ast.parse(init_path.read_text(encoding="utf-8"), filename=str(init_path))
    exports: set[str] = set()

    explicit_all: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == "__all__"
                    and isinstance(node.value, (ast.List, ast.Tuple))
                ):
                    for element in node.value.elts:
                        if isinstance(element, ast.Constant) and isinstance(element.value, str):
                            explicit_all.add(element.value)

    if explicit_all:
        return explicit_all

    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.level > 0:
            for alias in node.names:
                if alias.name != "*":
                    exports.add(alias.asname or alias.name)

    return exports


def _iter_package_modules(package: str) -> list[tuple[str, Path]]:
    pkg_dir = PACKAGE_ROOT / package
    modules: list[tuple[str, Path]] = []
    for path in sorted(pkg_dir.glob("*.py")):
        if path.name == "__init__.py":
            module = f"brainsurgery.{package}"
        else:
            module = f"brainsurgery.{package}.{path.stem}"
        modules.append((module, path))
    return modules


def _module_defined_symbols(tree: ast.Module) -> set[str]:
    symbols: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    symbols.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            symbols.add(node.target.id)
    return symbols


def test_symbol_visibility_contract_for_core_io_engine() -> None:
    export_violations: list[str] = []
    internal_name_violations: list[str] = []
    dead_symbol_violations: list[str] = []

    for package in TARGET_PACKAGES:
        exports = _package_exports(package)
        for symbol in sorted(exports):
            if symbol.startswith("_"):
                export_violations.append(f"{package}:{symbol}")

        sibling_imports: dict[str, set[str]] = defaultdict(set)
        module_loads: dict[str, set[str]] = defaultdict(set)
        module_decorated_defs: dict[str, set[str]] = defaultdict(set)
        module_defs: dict[str, set[str]] = {}

        for module, path in _iter_package_modules(package):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            module_defs[module] = _module_defined_symbols(tree)

            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    module_loads[module].add(node.id)
                if (
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                    and node.decorator_list
                ):
                    module_decorated_defs[module].add(node.name)

                if isinstance(node, ast.ImportFrom):
                    level = node.level or 0
                    imported_module = node.module or ""
                    same_package = level == 1 or (
                        level == 0 and imported_module.startswith(f"brainsurgery.{package}")
                    )
                    if not same_package:
                        continue

                    for alias in node.names:
                        if alias.name != "*":
                            sibling_imports[alias.name].add(module)

        for module, symbols in module_defs.items():
            for symbol in sorted(symbols):
                if symbol.startswith("__") and symbol.endswith("__"):
                    continue
                if symbol in exports:
                    continue

                importers = {m for m in sibling_imports.get(symbol, set()) if m != module}

                # Internal cross-module symbols must be clearly internal.
                if importers and not symbol.startswith("_"):
                    internal_name_violations.append(
                        f"{module}:{symbol} (imported by {', '.join(sorted(importers))})"
                    )

                # Non-exported symbols must be used either locally or by siblings.
                locally_used = symbol in module_loads.get(
                    module, set()
                ) or symbol in module_decorated_defs.get(module, set())
                if not importers and not locally_used:
                    violation = f"{module}:{symbol}"
                    if violation in INTENTIONAL_UNUSED_PRIVATE_SYMBOLS:
                        continue
                    dead_symbol_violations.append(violation)

    assert not export_violations, (
        "Exported symbols must not start with underscore. Violations: "
        + ", ".join(export_violations)
    )
    assert not internal_name_violations, (
        "Non-exported cross-module symbols must start with underscore. Violations: "
        + ", ".join(internal_name_violations)
    )
    assert not dead_symbol_violations, (
        "Remove symbols that are neither exported nor internally used. Violations: "
        + ", ".join(dead_symbol_violations)
    )
