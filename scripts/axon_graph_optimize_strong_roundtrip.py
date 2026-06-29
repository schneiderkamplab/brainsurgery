#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from scripts.axon_graph_ir_strong_roundtrip import run_graph_ir_strong_roundtrip
from scripts.axon_roundtrip_common import print_result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Strong optimized Graph IR roundtrip: rerun resolve+normalize+elaborate+"
            "flatten+typecheck before each graph render, then optimize Graph IR."
        )
    )
    parser.add_argument("paths", nargs="*", type=Path, help="Axon files or directories.")
    parser.add_argument("--main-module")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tmp/axon-stage-roundtrip-graph-optimize-strong"),
    )
    parser.add_argument("--keep-existing", action="store_true")
    parser.add_argument("--include-stale-cache", action="store_true")
    parser.add_argument("--no-validate-typed", action="store_true")
    parser.set_defaults(show_types=True)
    parser.add_argument("--show-types", dest="show_types", action="store_true")
    parser.add_argument("--no-show-types", dest="show_types", action="store_false")
    parser.add_argument("--show-inferred-expression-types", action="store_true")
    args = parser.parse_args()

    result = run_graph_ir_strong_roundtrip(
        args.paths,
        main_module=args.main_module,
        output_dir=args.output_dir,
        optimize_ast=True,
        optimize_graph=True,
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
