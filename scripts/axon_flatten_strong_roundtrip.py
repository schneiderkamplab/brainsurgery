#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from brainsurgery.synapse.axon import (
    flatten_closed_axon_file,
    elaborate_closed_axon_file,
    normalize_closed_axon_file,
    render_axon_file,
    resolve_axon_program_from_path,
)
from brainsurgery.synapse.axon.validate import validate_flat_axon_file

from scripts.axon_roundtrip_common import (
    RoundtripResult,
    print_result,
    run_path_roundtrips,
    write_generation,
)


def _resolve_normalize_flatten_render(path: Path, *, validate: bool) -> str:
    normalized = normalize_closed_axon_file(resolve_axon_program_from_path(path).ast)
    flattened = flatten_closed_axon_file(elaborate_closed_axon_file(normalized))
    if validate:
        validate_flat_axon_file(flattened)
    return render_axon_file(flattened)


def flatten_strong_roundtrip_path(
    path: Path,
    output_dir: Path,
    *,
    validate_flat: bool = True,
) -> bool:
    first = _resolve_normalize_flatten_render(path, validate=validate_flat)
    first_path = write_generation(output_dir, "render1", path, first)

    second = _resolve_normalize_flatten_render(first_path, validate=validate_flat)
    second_path = write_generation(output_dir, "render2", path, second)

    third = _resolve_normalize_flatten_render(second_path, validate=validate_flat)
    write_generation(output_dir, "render3", path, third)

    return first == second == third


def run_flatten_strong_roundtrip(
    paths: list[Path],
    *,
    output_dir: Path = Path("tmp/axon-stage-roundtrip-flatten-strong"),
    keep_existing: bool = False,
    include_stale_cache: bool = False,
    validate_flat: bool = True,
) -> RoundtripResult:
    return run_path_roundtrips(
        paths=paths,
        output_dir=output_dir,
        keep_existing=keep_existing,
        include_stale_cache=include_stale_cache,
        label="flatten-strong",
        roundtrip_path=lambda path, out: flatten_strong_roundtrip_path(
            path,
            out,
            validate_flat=validate_flat,
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Strong flatten roundtrip: render, reparse, reresolve, renormalize, reflatten."
    )
    parser.add_argument("paths", nargs="*", type=Path, help="Axon files or directories.")
    parser.add_argument("--output-dir", type=Path, default=Path("tmp/axon-stage-roundtrip-flatten-strong"))
    parser.add_argument("--keep-existing", action="store_true")
    parser.add_argument("--include-stale-cache", action="store_true")
    parser.add_argument("--no-validate-flat", action="store_true")
    args = parser.parse_args()

    result = run_flatten_strong_roundtrip(
        args.paths,
        output_dir=args.output_dir,
        keep_existing=args.keep_existing,
        include_stale_cache=args.include_stale_cache,
        validate_flat=not args.no_validate_flat,
    )
    print_result(result)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
