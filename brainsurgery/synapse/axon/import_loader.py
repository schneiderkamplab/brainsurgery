from __future__ import annotations

import os
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
)


@dataclass(frozen=True)
class _LoadedSyntaxFile:
    path: Path
    namespace: str | None
    parsed_source: ParsedProgramSource


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


def _apply_namespace(
    modules: tuple[AxonModule, ...], namespace: str | None
) -> tuple[AxonModule, ...]:
    if not namespace:
        return modules
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
                statements=module.statements,
                imports=module.imports,
                imported_members=module.imported_members,
                symbols=module.symbols,
                pragmas=module.pragmas,
                return_type_expr=module.return_type_expr,
                return_shape=module.return_shape,
            )
        )
    return tuple(namespaced)


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
    prelude_file = (builtins_dir / "Prelude.axon").resolve()
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

    if prelude_file.exists() and prelude_file != root:
        _load_file_syntax(prelude_file, namespace="Prelude")
    _load_file_syntax(root)

    # Local import avoids a parser<->import_loader import cycle at module import time.
    from .parser import build_axon_modules_from_parsed_source

    ordered_modules: list[AxonModule] = []
    loaded_by_namespace: dict[str, _LoadedSyntaxFile] = {
        loaded.namespace: loaded for loaded in ordered_files if loaded.namespace is not None
    }
    for loaded in ordered_files:
        imported_constants: dict[str, AxonExpr] = {}
        imported_constant_imports: list[str] = []
        for namespace, members in loaded.parsed_source.imported_members.items():
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
            extra_imports=tuple(imported_constant_imports) if imported_constant_imports else None,
        )
        ordered_modules.extend(_apply_namespace(modules, loaded.namespace))

    out = tuple(ordered_modules)
    validate_axon_program(out)
    return out


__all__ = ["load_axon_program_from_path"]
