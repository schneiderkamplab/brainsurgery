#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from brainsurgery.synapse.axon import (
    elaborate_closed_axon_file,
    flatten_closed_axon_file,
    graph_program_to_axon_file,
    lower_axon_program_to_graph_ir,
    normalize_closed_axon_file,
    parse_axon_program,
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


def _graph_render(
    typed,
    *,
    main_module: str | None,
    validate: bool,
    show_types: bool,
    show_inferred_expression_types: bool,
) -> str:
    graph = lower_axon_program_to_graph_ir(typed, main_module=main_module)
    axon = graph_program_to_axon_file(graph)
    axon = typecheck2_flat_axon_file(axon, main_module=graph.main_module)
    if validate:
        validate_typed_axon_file(axon, main_module=graph.main_module)
    return render_axon_file(
        axon,
        show_types=show_types,
        show_inferred_expr_types=show_inferred_expression_types,
    )


def _full_graph_render(
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
    elaborated = elaborate_closed_axon_file(normalized, main_module=main_module)
    flattened = flatten_closed_axon_file(elaborated, main_module=main_module)
    typed = typecheck2_flat_axon_file(flattened, main_module=main_module)
    return _graph_render(
        typed,
        main_module=main_module,
        validate=validate,
        show_types=show_types,
        show_inferred_expression_types=show_inferred_expression_types,
    )


def _weak_graph_render(
    source: str,
    *,
    main_module: str | None,
    validate: bool,
    show_types: bool,
    show_inferred_expression_types: bool,
) -> str:
    normalized = normalize_closed_axon_file(parse_axon_program(source), main_module=main_module)
    elaborated = elaborate_closed_axon_file(normalized, main_module=main_module)
    typed = typecheck2_flat_axon_file(elaborated, main_module=main_module)
    return _graph_render(
        typed,
        main_module=main_module,
        validate=validate,
        show_types=show_types,
        show_inferred_expression_types=show_inferred_expression_types,
    )


def graph_ir_weak_roundtrip_path(
    path: Path,
    output_dir: Path,
    *,
    main_module: str | None = None,
    validate_typed: bool = True,
    show_types: bool = True,
    show_inferred_expression_types: bool = False,
) -> bool:
    first = _full_graph_render(
        path,
        main_module=main_module,
        validate=validate_typed,
        show_types=show_types,
        show_inferred_expression_types=show_inferred_expression_types,
    )
    write_generation(output_dir, "render1", path, first)

    second = _weak_graph_render(
        first,
        main_module=main_module,
        validate=validate_typed,
        show_types=show_types,
        show_inferred_expression_types=show_inferred_expression_types,
    )
    write_generation(output_dir, "render2", path, second)

    third = _weak_graph_render(
        second,
        main_module=main_module,
        validate=validate_typed,
        show_types=show_types,
        show_inferred_expression_types=show_inferred_expression_types,
    )
    write_generation(output_dir, "render3", path, third)

    return first == second == third


def run_graph_ir_weak_roundtrip(
    paths: list[Path],
    *,
    main_module: str | None = None,
    output_dir: Path = Path("tmp/axon-stage-roundtrip-graph-ir-weak"),
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
        label="graph-ir-weak",
        roundtrip_path=lambda path, out: graph_ir_weak_roundtrip_path(
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
        description="Weak Graph IR roundtrip: full pipeline to graph-rendered Axon, then reparse/retypecheck/regraph without reresolve/reflatten."
    )
    parser.add_argument("paths", nargs="*", type=Path, help="Axon files or directories.")
    parser.add_argument("--main-module")
    parser.add_argument("--output-dir", type=Path, default=Path("tmp/axon-stage-roundtrip-graph-ir-weak"))
    parser.add_argument("--keep-existing", action="store_true")
    parser.add_argument("--include-stale-cache", action="store_true")
    parser.add_argument("--no-validate-typed", action="store_true")
    parser.set_defaults(show_types=True)
    parser.add_argument("--show-types", dest="show_types", action="store_true")
    parser.add_argument("--no-show-types", dest="show_types", action="store_false")
    parser.add_argument("--show-inferred-expression-types", action="store_true")
    args = parser.parse_args()

    result = run_graph_ir_weak_roundtrip(
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
