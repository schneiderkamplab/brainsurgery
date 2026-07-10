from __future__ import annotations

from pathlib import Path

import pytest

from brainsurgery.synapse.axon.ast import ast_equal, render_axon_file
from brainsurgery.synapse.axon.parse import parse_axon_program
from scripts.axon_parse_roundtrip import parse_roundtrip_path


def _axon_paths() -> list[Path]:
    roots = (Path("brainsurgery/synapse/models"), Path("brainsurgery/synapse/builtins"))
    paths: list[Path] = []
    for root in roots:
        paths.extend(sorted(root.glob("**/*.axon")))
    return paths


@pytest.mark.parametrize("axon_path", _axon_paths(), ids=lambda path: path.as_posix())
def test_parser_render_roundtrip_is_canonical(axon_path: Path, tmp_path: Path) -> None:
    assert parse_roundtrip_path(axon_path, tmp_path)


def test_parser_render_roundtrip_preserves_duplicate_pragmas() -> None:
    source = """
{-# CHECKPOINT "a" #-}
{-# CHECKPOINT "b" #-}

main x = x
"""
    parsed = parse_axon_program(source)
    rendered = render_axon_file(parsed)

    assert "{-# CHECKPOINT \"a\" #-}" in rendered
    assert "{-# CHECKPOINT \"b\" #-}" in rendered
    assert ast_equal(parsed, parse_axon_program(rendered))
