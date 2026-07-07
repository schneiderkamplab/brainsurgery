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
    builtins_overlays: tuple[str, ...] = ()


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
    base_file: Path,
    import_name: str,
    builtins_dir: Path,
    search_paths: tuple[Path, ...],
    builtins_overlays: tuple[str, ...] = (),
) -> Path:
    rel = Path(*import_name.split(".")).with_suffix(".axon")
    base_resolved = base_file.resolve()
    in_builtins = False
    try:
        base_resolved.relative_to(builtins_dir)
        in_builtins = True
    except ValueError:
        pass
    local_candidate = (base_file.parent / rel).resolve()
    if not in_builtins and local_candidate.exists():
        return local_candidate
    search_candidates: list[Path] = []
    for search_root in search_paths:
        candidate = (search_root / rel).resolve()
        search_candidates.append(candidate)
        if candidate.exists():
            return candidate
    overlay_candidates: list[Path] = []
    for overlay in builtins_overlays:
        candidate = (builtins_dir / overlay / rel).resolve()
        overlay_candidates.append(candidate)
        if candidate.exists():
            return candidate
    builtin_candidate = (builtins_dir / rel).resolve()
    if builtin_candidate.exists():
        return builtin_candidate
    tried = [
        str(local_candidate),
        *(str(path) for path in search_candidates),
        *(str(path) for path in overlay_candidates),
        str(builtin_candidate),
    ]
    raise FileNotFoundError(
        f"Axon import {import_name!r} not found from {base_file}: tried {', '.join(tried)}"
    )


def _resolve_import_paths(
    base_file: Path,
    import_name: str,
    builtins_dir: Path,
    search_paths: tuple[Path, ...],
    builtins_overlays: tuple[str, ...],
) -> tuple[Path, ...]:
    rel = Path(*import_name.split(".")).with_suffix(".axon")
    base_resolved = base_file.resolve()
    in_builtins = False
    try:
        base_resolved.relative_to(builtins_dir)
        in_builtins = True
    except ValueError:
        pass
    local_candidate = (base_file.parent / rel).resolve()
    if not in_builtins and local_candidate.exists():
        return (local_candidate,)
    search_candidates: list[Path] = []
    for search_root in search_paths:
        candidate = (search_root / rel).resolve()
        search_candidates.append(candidate)
        if candidate.exists():
            return (candidate,)
    overlay_candidates = tuple(
        candidate
        for overlay in builtins_overlays
        if (candidate := (builtins_dir / overlay / rel).resolve()).exists()
    )
    builtin_candidate = (builtins_dir / rel).resolve()
    if builtin_candidate.exists():
        return (builtin_candidate, *overlay_candidates)
    if overlay_candidates:
        return overlay_candidates
    tried = [
        str(local_candidate),
        *(str(path) for path in search_candidates),
        *(str((builtins_dir / overlay / rel).resolve()) for overlay in builtins_overlays),
        str(builtin_candidate),
    ]
    raise FileNotFoundError(
        f"Axon import {import_name!r} not found from {base_file}: tried {', '.join(tried)}"
    )


def _unique_concat(*items: tuple[str, ...]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for group in items:
        for item in group:
            if item in seen:
                continue
            seen.add(item)
            out.append(item)
    return tuple(out)


def _merge_overlay_axon_files(paths: tuple[Path, ...]) -> AxonFile:
    if len(paths) == 1:
        return parse_axon_program_from_path(paths[0])
    parsed = tuple(parse_axon_program_from_path(path) for path in paths)
    imports = _unique_concat(*(ast.imports for ast in parsed))
    imported_members: dict[str, tuple[str, ...]] = {}
    for ast in parsed:
        imported_members.update(ast.imported_members)
    exports = _unique_concat(*(ast.exports for ast in parsed))
    pragmas: dict[str, object] = {}
    for ast in parsed:
        pragmas.update(ast.pragmas)
    type_aliases = {}
    for ast in parsed:
        type_aliases.update(ast.type_aliases)
    modules_by_name = {}
    module_order: list[str] = []
    for ast in parsed:
        for module in ast.modules:
            if module.name not in modules_by_name:
                module_order.append(module.name)
            modules_by_name[module.name] = module
    return AxonFile(
        modules=tuple(modules_by_name[name] for name in module_order),
        imports=imports,
        imported_members=imported_members,
        exports=exports,
        pragmas=pragmas,
        type_aliases=type_aliases,
        origin_path=paths[-1],
    )


def _normalize_builtins_overlays(
    builtins_dir: Path,
    builtins_overlays: tuple[str, ...] | list[str] | None,
) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in builtins_overlays or ():
        overlay = str(raw).strip().strip("/")
        if not overlay:
            continue
        if Path(overlay).is_absolute() or ".." in Path(overlay).parts:
            raise ValueError(
                f"Builtins overlay {raw!r} must be a name below {builtins_dir}, not a path"
            )
        if overlay in seen:
            continue
        overlay_dir = (builtins_dir / overlay).resolve()
        try:
            overlay_dir.relative_to(builtins_dir)
        except ValueError as exc:
            raise ValueError(
                f"Builtins overlay {raw!r} must resolve below {builtins_dir}"
            ) from exc
        if not overlay_dir.is_dir():
            raise FileNotFoundError(f"Builtins overlay not found: {overlay_dir}")
        seen.add(overlay)
        normalized.append(overlay)
    return tuple(normalized)


def load_axon_files_from_path(
    path: Path,
    *,
    builtins_overlays: tuple[str, ...] | list[str] | None = None,
) -> LoadedAxonProgram:
    root = path.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Axon file not found: {root}")
    if not root.is_file():
        raise ValueError(f"Axon import root must be a file: {root}")

    seen_paths: set[tuple[Path, ...]] = set()
    visiting: list[tuple[Path, ...]] = []
    ordered_files: list[LoadedAxonFile] = []
    builtins_dir = (Path(__file__).resolve().parents[2] / "builtins").resolve()
    normalized_overlays = _normalize_builtins_overlays(builtins_dir, builtins_overlays)
    search_paths = axon_search_paths()

    def _load_file(file_paths: Path | tuple[Path, ...], *, namespace: str | None = None) -> None:
        resolved_paths = (
            (file_paths.resolve(),) if isinstance(file_paths, Path) else tuple(path.resolve() for path in file_paths)
        )
        if resolved_paths in seen_paths:
            return
        if resolved_paths in visiting:
            cycle = " -> ".join(" + ".join(str(p) for p in group) for group in [*visiting, resolved_paths])
            raise ValueError(f"Cyclic Axon imports detected: {cycle}")
        visiting.append(resolved_paths)
        ast = _merge_overlay_axon_files(resolved_paths)
        for import_name in sorted(ast.imports):
            dep = _resolve_import_paths(
                resolved_paths[-1],
                import_name,
                builtins_dir,
                search_paths,
                normalized_overlays,
            )
            _load_file(dep, namespace=import_name)
        ordered_files.append(LoadedAxonFile(path=resolved_paths[-1], namespace=namespace, ast=ast))
        seen_paths.add(resolved_paths)
        visiting.pop()

    _load_file(root)
    return LoadedAxonProgram(
        root_path=root,
        files=tuple(ordered_files),
        builtins_dir=builtins_dir,
        builtins_overlays=normalized_overlays,
    )
