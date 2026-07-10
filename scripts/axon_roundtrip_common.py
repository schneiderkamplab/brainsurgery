from __future__ import annotations

import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


_BARE_CACHE_RE = re.compile(r"\??CacheLayer(?!\[)|\??Cache(?!\[|Layer)\b")


@dataclass(frozen=True)
class RoundtripResult:
    total: int
    checked: int
    skipped_stale_cache: int
    failed: tuple[tuple[Path, str], ...]
    unstable: tuple[Path, ...]
    output_dir: Path

    @property
    def ok(self) -> bool:
        return not self.failed and not self.unstable


def default_axon_paths() -> list[Path]:
    roots = [Path("brainsurgery/synapse/builtins"), Path("brainsurgery/synapse/models")]
    return [path for root in roots for path in sorted(root.rglob("*.axon"))]


def selected_axon_paths(inputs: list[Path]) -> list[Path]:
    if not inputs:
        return default_axon_paths()
    out: list[Path] = []
    for item in inputs:
        if item.is_dir():
            out.extend(sorted(item.rglob("*.axon")))
        elif item.suffix == ".axon":
            out.append(item)
    return out


def has_stale_cache_signature(path: Path) -> bool:
    for line in path.read_text(encoding="utf-8").splitlines():
        if "::" in line and _BARE_CACHE_RE.search(line):
            return True
    return False


def iter_progress(paths: Iterable[Path], label: str):
    paths = list(paths)
    try:
        from tqdm import tqdm
    except Exception:
        total = len(paths)
        for idx, path in enumerate(paths, start=1):
            if idx == 1 or idx % 10 == 0 or idx == total:
                print(f"{label}: {idx}/{total} {path}", file=sys.stderr, flush=True)
            yield path
        return
    yield from tqdm(paths, desc=label, unit="file")


def prepare_output_dir(output_dir: Path, *, keep_existing: bool) -> None:
    if output_dir.exists() and not keep_existing:
        shutil.rmtree(output_dir)


def write_generation(output_dir: Path, generation: str, source_path: Path, rendered: str) -> Path:
    output_path = output_dir / generation / source_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    return output_path


def run_path_roundtrips(
    *,
    paths: list[Path],
    output_dir: Path,
    keep_existing: bool,
    include_stale_cache: bool,
    label: str,
    roundtrip_path: Callable[[Path, Path], bool],
) -> RoundtripResult:
    all_paths = selected_axon_paths(paths)
    checked_paths = [
        path for path in all_paths if include_stale_cache or not has_stale_cache_signature(path)
    ]
    prepare_output_dir(output_dir, keep_existing=keep_existing)

    failed: list[tuple[Path, str]] = []
    unstable: list[Path] = []
    for path in iter_progress(checked_paths, label):
        try:
            if not roundtrip_path(path, output_dir):
                unstable.append(path)
        except Exception as exc:
            failed.append((path, f"{type(exc).__name__}: {exc}"))

    return RoundtripResult(
        total=len(all_paths),
        checked=len(checked_paths),
        skipped_stale_cache=len(all_paths) - len(checked_paths),
        failed=tuple(failed),
        unstable=tuple(unstable),
        output_dir=output_dir,
    )


def print_result(result: RoundtripResult) -> None:
    print(
        f"total={result.total} checked={result.checked} "
        f"skipped_stale_cache={result.skipped_stale_cache} "
        f"failed={len(result.failed)} unstable={len(result.unstable)} "
        f"output={result.output_dir}"
    )
    for path, error in result.failed[:40]:
        print(f"FAIL {path}: {error}")
    for path in result.unstable[:40]:
        print(f"UNSTABLE {path}")
