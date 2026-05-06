#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

from brainsurgery.synapse.axon import (
    flatten_closed_axon_file,
    normalize_closed_axon_file,
    render_axon_file,
    resolve_axon_program_from_path,
    typecheck2_flat_axon_file,
    typecheck_flat_axon_file,
)
from brainsurgery.synapse.axon.validate import validate_typed_axon_file


_BARE_CACHE_RE = re.compile(r"\??CacheLayer(?!\[)|\??Cache(?!\[|Layer)\b")


def _default_paths() -> list[Path]:
    roots = [Path("brainsurgery/synapse/builtins"), Path("brainsurgery/synapse/models")]
    return [path for root in roots for path in sorted(root.rglob("*.axon"))]


def _selected_paths(inputs: list[Path]) -> list[Path]:
    if not inputs:
        return _default_paths()
    out: list[Path] = []
    for item in inputs:
        if item.is_dir():
            out.extend(sorted(item.rglob("*.axon")))
        elif item.suffix == ".axon":
            out.append(item)
    return out


def _has_stale_cache_signature(path: Path) -> bool:
    for line in path.read_text(encoding="utf-8").splitlines():
        if "::" in line and _BARE_CACHE_RE.search(line):
            return True
    return False


def _resolve_normalize_flatten_typecheck_render(
    path: Path,
    *,
    main_module: str | None,
    validate: bool,
    show_types: bool,
    show_inferred_expression_types: bool,
    typechecker: str,
) -> str:
    resolved = resolve_axon_program_from_path(path).ast
    normalized = normalize_closed_axon_file(resolved, main_module=main_module)
    flattened = flatten_closed_axon_file(normalized, main_module=main_module)
    if typechecker == "typecheck2":
        typed = typecheck2_flat_axon_file(flattened, main_module=main_module)
    else:
        typed = typecheck_flat_axon_file(flattened, main_module=main_module)
    if validate:
        validate_typed_axon_file(typed, main_module=main_module)
    return render_axon_file(
        typed,
        show_types=show_types,
        show_inferred_expr_types=show_inferred_expression_types,
    )


def _iter_progress(paths: list[Path], label: str):
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify resolve+normalize+flatten+typecheck render stability across three generations."
    )
    parser.add_argument("paths", nargs="*", type=Path, help="Axon files or directories.")
    parser.add_argument(
        "--main-module",
        help="Main definition used for pruning and validation. Use for focused model roundtrips.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tmp/axon-stage-roundtrip-typecheck"),
        help="Directory for render1/render2/render3 files.",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not remove the output directory before writing.",
    )
    parser.add_argument(
        "--include-stale-cache",
        action="store_true",
        help="Include files with known stale bare Cache/CacheLayer signatures.",
    )
    parser.add_argument(
        "--no-validate-typed",
        action="store_true",
        help="Skip typed-stage validation for each generation.",
    )
    parser.add_argument(
        "--show-types",
        action="store_true",
        help="Render stable typed definitions/signatures and include them in stability checks.",
    )
    parser.add_argument(
        "--show-inferred-expression-types",
        action="store_true",
        help="Also render inferred expression-level type ascriptions. This is debug output and may not be stable.",
    )
    parser.add_argument(
        "--typechecker",
        choices=("typecheck", "typecheck2"),
        default="typecheck",
        help="Typechecker implementation to use. Defaults to the existing typecheck.",
    )
    args = parser.parse_args()

    all_paths = _selected_paths(args.paths)
    paths = [
        path
        for path in all_paths
        if args.include_stale_cache or not _has_stale_cache_signature(path)
    ]
    skipped_stale_cache = len(all_paths) - len(paths)
    if args.output_dir.exists() and not args.keep_existing:
        shutil.rmtree(args.output_dir)

    should_validate = not args.no_validate_typed
    failed: list[tuple[Path, str]] = []
    unstable: list[Path] = []
    for path in _iter_progress(paths, "typecheck"):
        try:
            first = _resolve_normalize_flatten_typecheck_render(
                path,
                main_module=args.main_module,
                validate=should_validate,
                show_types=args.show_types,
                show_inferred_expression_types=args.show_inferred_expression_types,
                typechecker=args.typechecker,
            )

            first_path = args.output_dir / "render1" / path
            first_path.parent.mkdir(parents=True, exist_ok=True)
            first_path.write_text(first, encoding="utf-8")

            second = _resolve_normalize_flatten_typecheck_render(
                first_path,
                main_module=args.main_module,
                validate=should_validate,
                show_types=args.show_types,
                show_inferred_expression_types=args.show_inferred_expression_types,
                typechecker=args.typechecker,
            )

            second_path = args.output_dir / "render2" / path
            second_path.parent.mkdir(parents=True, exist_ok=True)
            second_path.write_text(second, encoding="utf-8")

            third = _resolve_normalize_flatten_typecheck_render(
                second_path,
                main_module=args.main_module,
                validate=should_validate,
                show_types=args.show_types,
                show_inferred_expression_types=args.show_inferred_expression_types,
                typechecker=args.typechecker,
            )

            third_path = args.output_dir / "render3" / path
            third_path.parent.mkdir(parents=True, exist_ok=True)
            third_path.write_text(third, encoding="utf-8")

            if first != second or second != third:
                unstable.append(path)
        except Exception as exc:
            failed.append((path, f"{type(exc).__name__}: {exc}"))

    print(
        f"total={len(all_paths)} checked={len(paths)} skipped_stale_cache={skipped_stale_cache} "
        f"failed={len(failed)} unstable={len(unstable)} output={args.output_dir}"
    )
    for path, error in failed[:40]:
        print(f"FAIL {path}: {error}")
    for path in unstable[:40]:
        print(f"UNSTABLE {path}")
    return 1 if failed or unstable else 0


if __name__ == "__main__":
    raise SystemExit(main())
