#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from brainsurgery.synapse.axon import ast_equal, parse_axon_program, parse_axon_program_from_path
from brainsurgery.synapse.axon import render_axon_file


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

    paths = _selected_paths(args.paths)
    if args.output_dir.exists() and not args.keep_existing:
        shutil.rmtree(args.output_dir)

    failed: list[tuple[Path, str]] = []
    unstable: list[Path] = []
    for path in _iter_progress(paths, "parse"):
        try:
            parsed = parse_axon_program_from_path(path)
            rendered = render_axon_file(parsed)
            reparsed = parse_axon_program(rendered)
            if not ast_equal(parsed, reparsed):
                unstable.append(path)
            output_path = args.output_dir / path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(rendered, encoding="utf-8")
        except Exception as exc:
            failed.append((path, f"{type(exc).__name__}: {exc}"))

    print(
        f"checked={len(paths)} failed={len(failed)} unstable={len(unstable)} "
        f"output={args.output_dir}"
    )
    for path, error in failed[:40]:
        print(f"FAIL {path}: {error}")
    for path in unstable[:40]:
        print(f"UNSTABLE {path}")
    return 1 if failed or unstable else 0


if __name__ == "__main__":
    raise SystemExit(main())
