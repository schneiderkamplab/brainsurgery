#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from brainsurgery.synapse.axon import ast_equal, parse_axon_program, parse_axon_program_from_path
from brainsurgery.synapse.axon import render_axon_file

from scripts.axon_roundtrip_common import (
    RoundtripResult,
    iter_progress,
    prepare_output_dir,
    print_result,
    selected_axon_paths,
)


def parse_roundtrip_path(path: Path, output_dir: Path) -> bool:
    parsed = parse_axon_program_from_path(path)
    rendered = render_axon_file(parsed)
    reparsed = parse_axon_program(rendered)
    rerendered = render_axon_file(reparsed)

    output_path = output_dir / path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    return rerendered == rendered and ast_equal(parsed, reparsed)


def run_parse_roundtrip(
    paths: list[Path],
    *,
    output_dir: Path = Path("tmp/axon-stage-roundtrip-parse"),
    keep_existing: bool = False,
) -> RoundtripResult:
    selected = selected_axon_paths(paths)
    prepare_output_dir(output_dir, keep_existing=keep_existing)
    failed: list[tuple[Path, str]] = []
    unstable: list[Path] = []
    for path in iter_progress(selected, "parse"):
        try:
            if not parse_roundtrip_path(path, output_dir):
                unstable.append(path)
        except Exception as exc:
            failed.append((path, f"{type(exc).__name__}: {exc}"))
    return RoundtripResult(
        total=len(selected),
        checked=len(selected),
        skipped_stale_cache=0,
        failed=tuple(failed),
        unstable=tuple(unstable),
        output_dir=output_dir,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render parse-stage Axon ASTs and verify parse/render/parse stability."
    )
    parser.add_argument("paths", nargs="*", type=Path, help="Axon files or directories.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tmp/axon-stage-roundtrip-parse"),
        help="Directory for rendered parse-stage files.",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not remove the output directory before writing.",
    )
    args = parser.parse_args()

    result = run_parse_roundtrip(
        args.paths,
        output_dir=args.output_dir,
        keep_existing=args.keep_existing,
    )
    print_result(result)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
