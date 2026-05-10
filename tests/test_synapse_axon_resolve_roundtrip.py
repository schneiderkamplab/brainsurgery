from __future__ import annotations

from pathlib import Path

import pytest

from brainsurgery.synapse.axon.ast import render_axon_file
from brainsurgery.synapse.axon.parse import parse_axon_program
from brainsurgery.synapse.axon.resolve import resolve_axon_program_from_path
from brainsurgery.synapse.axon.validate import validate_closed_axon_file
from scripts.axon_resolve_roundtrip import resolve_roundtrip_path


def _model_axon_paths() -> list[Path]:
    return sorted(Path("brainsurgery/synapse/models").glob("**/*.axon"))


@pytest.mark.parametrize("axon_path", _model_axon_paths(), ids=lambda path: path.as_posix())
def test_resolve_render_roundtrip_is_closed_and_canonical(
    axon_path: Path,
    tmp_path: Path,
) -> None:
    assert resolve_roundtrip_path(axon_path, tmp_path)


def test_resolve_keeps_path_template_lexical_dependencies(tmp_path: Path) -> None:
    (tmp_path / "Config.axon").write_text(
        """
has_key :: Path -> Bool
has_key path = true

int :: Path -> ?Int -> Int
int path ?default=0 = default
""".strip()
        + "\n",
        encoding="utf-8",
    )
    root = tmp_path / "main.axon"
    root.write_text(
        """
{-# MAIN "main" #-}
import Config

CFG = Config.has_key @@text_config ? "text_config." : ""
EXPERTS = Config.int @@'{CFG}num_experts' default=8

main :: Int -> Int
main x = EXPERTS
""".strip()
        + "\n",
        encoding="utf-8",
    )

    resolved = resolve_axon_program_from_path(root).ast
    names = {module.name for module in resolved.modules}
    rendered = render_axon_file(resolved)

    assert "CFG" in names
    assert "EXPERTS" in names
    assert "@@'{CFG}num_experts'" in rendered
    validate_closed_axon_file(parse_axon_program(rendered))
