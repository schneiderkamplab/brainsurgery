from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .ast_validation import validate_axon_program
from .grammar import ParsedProgramSource, parse_program_source
from .syntax_validation import validate_parsed_program_source
from .types import AxonModule


@dataclass(frozen=True)
class _LoadedSyntaxFile:
    path: Path
    namespace: str | None
    parsed_source: ParsedProgramSource


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


def _resolve_import_path(base_file: Path, import_name: str, builtins_dir: Path) -> Path:
    rel = Path(*import_name.split(".")).with_suffix(".axon")
    local_candidate = (base_file.parent / rel).resolve()
    if local_candidate.exists():
        return local_candidate
    builtin_candidate = (builtins_dir / rel).resolve()
    if builtin_candidate.exists():
        return builtin_candidate
    raise FileNotFoundError(
        f"Axon import {import_name!r} not found from {base_file}: "
        f"tried {local_candidate} and {builtin_candidate}"
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
            dep = _resolve_import_path(resolved, import_name, builtins_dir)
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
    for loaded in ordered_files:
        modules = build_axon_modules_from_parsed_source(loaded.parsed_source, validate=False)
        ordered_modules.extend(_apply_namespace(modules, loaded.namespace))

    out = tuple(ordered_modules)
    validate_axon_program(out)
    return out


__all__ = ["load_axon_program_from_path"]
