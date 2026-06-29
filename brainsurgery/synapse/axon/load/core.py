from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ..ast.source import AxonFile
from ..parse import parse_axon_program_from_path


@dataclass(frozen=True)
class LoadedAxonFile:
    path: Path
    namespace: str | None
    ast: AxonFile


@dataclass(frozen=True)
class LoadedAxonProgram:
    root_path: Path
    files: tuple[LoadedAxonFile, ...]
    builtins_dir: Path


def axon_search_paths() -> tuple[Path, ...]:
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


def resolve_import_path(
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


def load_axon_files_from_path(path: Path) -> LoadedAxonProgram:
    root = path.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Axon file not found: {root}")
    if not root.is_file():
        raise ValueError(f"Axon import root must be a file: {root}")

    seen_paths: set[Path] = set()
    visiting: list[Path] = []
    ordered_files: list[LoadedAxonFile] = []
    builtins_dir = (Path(__file__).resolve().parents[2] / "builtins").resolve()
    search_paths = axon_search_paths()

    def _load_file(file_path: Path, *, namespace: str | None = None) -> None:
        resolved = file_path.resolve()
        if resolved in seen_paths:
            return
        if resolved in visiting:
            cycle = " -> ".join(str(p) for p in [*visiting, resolved])
            raise ValueError(f"Cyclic Axon imports detected: {cycle}")
        visiting.append(resolved)
        ast = parse_axon_program_from_path(resolved)
        for import_name in sorted(ast.imports):
            dep = resolve_import_path(resolved, import_name, builtins_dir, search_paths)
            _load_file(dep, namespace=import_name)
        ordered_files.append(LoadedAxonFile(path=resolved, namespace=namespace, ast=ast))
        seen_paths.add(resolved)
        visiting.pop()

    _load_file(root)
    return LoadedAxonProgram(root_path=root, files=tuple(ordered_files), builtins_dir=builtins_dir)
