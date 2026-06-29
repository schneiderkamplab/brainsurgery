import ast
from pathlib import Path

PACKAGE_ROOT = Path("brainsurgery")

CONFIG = {
    "algorithms": {"allowed": {"algorithms", "core"}},
    "expressions": {"allowed": {"expressions", "core"}},
    "transforms": {"allowed": {"transforms", "algorithms", "core", "engine", "expressions"}},
    "cli": {"allowed": {"cli", "core", "engine", "synapse"}},
    "web": {"allowed": {"web", "core", "engine", "http", "cli", "ui"}},
}


def _iter_python_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _package_exports(pkg: str) -> set[str]:
    init_path = PACKAGE_ROOT / pkg / "__init__.py"
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


def _assert_import_policy(pkg: str, allowed_internal: set[str]) -> None:
    pkg_dir = PACKAGE_ROOT / pkg

    offenders_reexport: list[str] = []
    offenders_allowed: list[str] = []
    offenders_direct_submodule: list[str] = []

    # 1/2: package module internal import policy
    for path in _iter_python_files(pkg_dir):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module == f"brainsurgery.{pkg}":
                    offenders_reexport.append(f"{path}:{node.lineno}")
                if node.level == 1 and node.module is None:
                    offenders_reexport.append(f"{path}:{node.lineno}")

                module = node.module or ""
                if node.level == 0 and module.startswith("brainsurgery."):
                    dep = module.split(".")[1]
                    if dep not in allowed_internal:
                        offenders_allowed.append(f"{path}:{node.lineno}")
                if node.level >= 2:
                    dep = module.split(".", 1)[0] if module else ""
                    if dep not in allowed_internal:
                        offenders_allowed.append(f"{path}:{node.lineno}")

            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("brainsurgery."):
                        dep = alias.name.split(".")[1]
                        if dep not in allowed_internal:
                            offenders_allowed.append(f"{path}:{node.lineno}")

    # 3: non-package modules should not import package submodules directly
    for path in _iter_python_files(PACKAGE_ROOT):
        if pkg_dir in path.parents:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level == 0 and module.startswith(f"brainsurgery.{pkg}."):
                    offenders_direct_submodule.append(f"{path}:{node.lineno}")
                if node.level == 2 and module.startswith(f"{pkg}."):
                    offenders_direct_submodule.append(f"{path}:{node.lineno}")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(f"brainsurgery.{pkg}."):
                        offenders_direct_submodule.append(f"{path}:{node.lineno}")

    # 4: package re-exports should be used by at least one non-package module
    exports = _package_exports(pkg)
    used: set[str] = set()
    for path in _iter_python_files(PACKAGE_ROOT):
        if pkg_dir in path.parents:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level == 0 and node.module == f"brainsurgery.{pkg}":
                for alias in node.names:
                    if alias.name != "*":
                        used.add(alias.name)
            if node.level == 1 and node.module == pkg:
                for alias in node.names:
                    if alias.name != "*":
                        used.add(alias.name)
            if node.level == 2 and node.module == pkg:
                for alias in node.names:
                    if alias.name != "*":
                        used.add(alias.name)
    unused_exports = sorted(exports - used)

    assert not offenders_reexport, (
        f"{pkg}: modules must import concrete sibling modules, not package re-exports. "
        f"Offenders: {', '.join(offenders_reexport)}"
    )
    assert not offenders_allowed, (
        f"{pkg}: modules imported disallowed internal packages. "
        f"Offenders: {', '.join(offenders_allowed)}"
    )
    assert not offenders_direct_submodule, (
        f"{pkg}: non-package modules imported {pkg} submodules directly. "
        f"Offenders: {', '.join(offenders_direct_submodule)}"
    )
    assert not unused_exports, (
        f"{pkg}: package re-exports should be minimal and used by non-{pkg} modules. "
        f"Unused exports: {', '.join(unused_exports)}"
    )


def test_algorithms_import_policy() -> None:
    _assert_import_policy("algorithms", CONFIG["algorithms"]["allowed"])


def test_expressions_import_policy() -> None:
    _assert_import_policy("expressions", CONFIG["expressions"]["allowed"])


def test_transforms_import_policy() -> None:
    _assert_import_policy("transforms", CONFIG["transforms"]["allowed"])


def test_cli_import_policy() -> None:
    _assert_import_policy("cli", CONFIG["cli"]["allowed"])


def test_web_import_policy() -> None:
    _assert_import_policy("web", CONFIG["web"]["allowed"])
