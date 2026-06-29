#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from brainsurgery.synapse.axon import normalize_closed_axon_file, render_axon_file
from brainsurgery.synapse.axon import resolve_axon_program_from_path
from brainsurgery.synapse.axon.validate import validate_normalized_axon_file

from scripts.axon_roundtrip_common import (
    RoundtripResult,
    print_result,
    run_path_roundtrips,
    write_generation,
)


def _resolve_normalize_render(path: Path, *, validate: bool) -> str:
    normalized = normalize_closed_axon_file(resolve_axon_program_from_path(path).ast)
    if validate:
        validate_normalized_axon_file(normalized)
    return render_axon_file(normalized)


def normalize_strong_roundtrip_path(
    path: Path,
    output_dir: Path,
    *,
    validate_normalized: bool = True,
) -> bool:
    first = _resolve_normalize_render(path, validate=validate_normalized)
    first_path = write_generation(output_dir, "render1", path, first)

    second = _resolve_normalize_render(first_path, validate=validate_normalized)
    second_path = write_generation(output_dir, "render2", path, second)

    third = _resolve_normalize_render(second_path, validate=validate_normalized)
    write_generation(output_dir, "render3", path, third)

    return first == second == third


def run_normalize_strong_roundtrip(
    paths: list[Path],
    *,
    output_dir: Path = Path("tmp/axon-stage-roundtrip-normalize-strong"),
    keep_existing: bool = False,
    include_stale_cache: bool = False,
    validate_normalized: bool = True,
) -> RoundtripResult:
    return run_path_roundtrips(
        paths=paths,
        output_dir=output_dir,
        keep_existing=keep_existing,
        include_stale_cache=include_stale_cache,
        label="normalize-strong",
        roundtrip_path=lambda path, out: normalize_strong_roundtrip_path(
            path,
            out,
            validate_normalized=validate_normalized,
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Strong normalize roundtrip: render, reparse, reresolve, renormalize."
    )
    parser.add_argument("paths", nargs="*", type=Path, help="Axon files or directories.")
    parser.add_argument("--output-dir", type=Path, default=Path("tmp/axon-stage-roundtrip-normalize-strong"))
    parser.add_argument("--keep-existing", action="store_true")
    parser.add_argument("--include-stale-cache", action="store_true")
    parser.add_argument("--no-validate-normalized", action="store_true")
    args = parser.parse_args()

    result = run_normalize_strong_roundtrip(
        args.paths,
        output_dir=args.output_dir,
        keep_existing=args.keep_existing,
        include_stale_cache=args.include_stale_cache,
        validate_normalized=not args.no_validate_normalized,
    )
    print_result(result)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
