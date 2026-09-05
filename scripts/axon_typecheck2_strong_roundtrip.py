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
    typecheck2_flat_axon_file,
)
from brainsurgery.synapse.axon.validate import validate_typed_axon_file

from scripts.axon_roundtrip_common import (
    RoundtripResult,
    print_result,
    run_path_roundtrips,
    write_generation,
)


def _resolve_normalize_flatten_typecheck_render(
    path: Path,
    *,
    main_module: str | None,
    validate: bool,
    show_types: bool,
    show_inferred_expression_types: bool,
) -> str:
    normalized = normalize_closed_axon_file(
        resolve_axon_program_from_path(path).ast, main_module=main_module
    )
    flat = flatten_closed_axon_file(
        elaborate_closed_axon_file(normalized, main_module=main_module),
        main_module=main_module,
    )
    typed = typecheck2_flat_axon_file(flat, main_module=main_module)
    if validate:
        validate_typed_axon_file(typed, main_module=main_module)
    return render_axon_file(
        typed,
        show_types=show_types,
        show_inferred_expr_types=show_inferred_expression_types,
    )


def typecheck2_strong_roundtrip_path(
    path: Path,
    output_dir: Path,
    *,
    main_module: str | None = None,
    validate_typed: bool = True,
    show_types: bool = True,
    show_inferred_expression_types: bool = False,
) -> bool:
    first = _resolve_normalize_flatten_typecheck_render(
        path,
        main_module=main_module,
        validate=validate_typed,
        show_types=show_types,
        show_inferred_expression_types=show_inferred_expression_types,
    )
    first_path = write_generation(output_dir, "render1", path, first)

    second = _resolve_normalize_flatten_typecheck_render(
        first_path,
        main_module=main_module,
        validate=validate_typed,
        show_types=show_types,
        show_inferred_expression_types=show_inferred_expression_types,
    )
    second_path = write_generation(output_dir, "render2", path, second)

    third = _resolve_normalize_flatten_typecheck_render(
        second_path,
        main_module=main_module,
        validate=validate_typed,
        show_types=show_types,
        show_inferred_expression_types=show_inferred_expression_types,
    )
    write_generation(output_dir, "render3", path, third)

    return first == second == third


def run_typecheck2_strong_roundtrip(
    paths: list[Path],
    *,
    main_module: str | None = None,
    output_dir: Path = Path("tmp/axon-stage-roundtrip-typecheck2-strong"),
    keep_existing: bool = False,
    include_stale_cache: bool = False,
    validate_typed: bool = True,
    show_types: bool = True,
    show_inferred_expression_types: bool = False,
) -> RoundtripResult:
    return run_path_roundtrips(
        paths=paths,
        output_dir=output_dir,
        keep_existing=keep_existing,
        include_stale_cache=include_stale_cache,
        label="typecheck2-strong",
        roundtrip_path=lambda path, out: typecheck2_strong_roundtrip_path(
            path,
            out,
            main_module=main_module,
            validate_typed=validate_typed,
            show_types=show_types,
            show_inferred_expression_types=show_inferred_expression_types,
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Strong typecheck2 roundtrip: render, reparse, reresolve, renormalize, reflatten, retypecheck."
    )
    parser.add_argument("paths", nargs="*", type=Path, help="Axon files or directories.")
    parser.add_argument("--main-module")
    parser.add_argument("--output-dir", type=Path, default=Path("tmp/axon-stage-roundtrip-typecheck2-strong"))
    parser.add_argument("--keep-existing", action="store_true")
    parser.add_argument("--include-stale-cache", action="store_true")
    parser.add_argument("--no-validate-typed", action="store_true")
    parser.set_defaults(show_types=True)
    parser.add_argument("--show-types", dest="show_types", action="store_true")
    parser.add_argument("--no-show-types", dest="show_types", action="store_false")
    parser.add_argument("--show-inferred-expression-types", action="store_true")
    args = parser.parse_args()

    result = run_typecheck2_strong_roundtrip(
        args.paths,
        main_module=args.main_module,
        output_dir=args.output_dir,
        keep_existing=args.keep_existing,
        include_stale_cache=args.include_stale_cache,
        validate_typed=not args.no_validate_typed,
        show_types=args.show_types,
        show_inferred_expression_types=args.show_inferred_expression_types,
    )
    print_result(result)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
