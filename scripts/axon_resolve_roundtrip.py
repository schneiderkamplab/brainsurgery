#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from brainsurgery.synapse.axon import render_axon_file, resolve_axon_program_from_path
from brainsurgery.synapse.axon.validate import validate_closed_axon_file

from scripts.axon_roundtrip_common import (
    RoundtripResult,
    print_result,
    run_path_roundtrips,
    write_generation,
)


def resolve_roundtrip_path(
    path: Path,
    output_dir: Path,
    *,
    validate_closed: bool = True,
) -> bool:
    first_ast = resolve_axon_program_from_path(path).ast
    if validate_closed:
        validate_closed_axon_file(first_ast)
    first = render_axon_file(first_ast)
    first_path = write_generation(output_dir, "render1", path, first)

    second_ast = resolve_axon_program_from_path(first_path).ast
    if validate_closed:
        validate_closed_axon_file(second_ast)
    second = render_axon_file(second_ast)
    second_path = write_generation(output_dir, "render2", path, second)

    third_ast = resolve_axon_program_from_path(second_path).ast
    if validate_closed:
        validate_closed_axon_file(third_ast)
    third = render_axon_file(third_ast)
    write_generation(output_dir, "render3", path, third)

    return first == second == third


def run_resolve_roundtrip(
    paths: list[Path],
    *,
    output_dir: Path = Path("tmp/axon-stage-roundtrip-resolve"),
    keep_existing: bool = False,
    include_stale_cache: bool = False,
    validate_closed: bool = True,
) -> RoundtripResult:
    return run_path_roundtrips(
        paths=paths,
        output_dir=output_dir,
        keep_existing=keep_existing,
        include_stale_cache=include_stale_cache,
        label="resolve",
        roundtrip_path=lambda path, out: resolve_roundtrip_path(
            path,
            out,
            validate_closed=validate_closed,
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify load+resolve render stability across three rendered generations."
    )
    parser.add_argument("paths", nargs="*", type=Path, help="Axon files or directories.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tmp/axon-stage-roundtrip-resolve"),
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
        "--no-validate-closed",
        action="store_true",
        help="Skip closed validation of resolved generations.",
    )
    args = parser.parse_args()

    result = run_resolve_roundtrip(
        args.paths,
        output_dir=args.output_dir,
        keep_existing=args.keep_existing,
        include_stale_cache=args.include_stale_cache,
        validate_closed=not args.no_validate_closed,
    )
    print_result(result)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
