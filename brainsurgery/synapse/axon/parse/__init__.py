from __future__ import annotations

from typing import Any

__all__ = [
    "parse_axon_module",
    "parse_axon_program",
    "parse_axon_program_from_path",
    "parse_expression_source",
    "parse_surface_program_source",
]


def __getattr__(name: str) -> Any:
    if name == "parse_expression_source":
        from .grammar import parse_expression_source

        return parse_expression_source
    if name == "parse_surface_program_source":
        from .grammar import parse_surface_program_source

        return parse_surface_program_source
    if name in {"parse_axon_module", "parse_axon_program", "parse_axon_program_from_path"}:
        from .parser import (
            parse_axon_module,
            parse_axon_program,
            parse_axon_program_from_path,
        )

        return {
            "parse_axon_module": parse_axon_module,
            "parse_axon_program": parse_axon_program,
            "parse_axon_program_from_path": parse_axon_program_from_path,
        }[name]
    raise AttributeError(name)
