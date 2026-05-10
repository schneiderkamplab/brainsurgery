from __future__ import annotations

from pathlib import Path

import pytest

from scripts.axon_graph_ir_strong_roundtrip import graph_ir_strong_roundtrip_path
from scripts.axon_graph_ir_weak_roundtrip import graph_ir_weak_roundtrip_path


def _graph_ir_roundtrip_paths() -> list[Path]:
    return sorted(Path("brainsurgery/synapse/models").glob("**/*.axon"))


@pytest.mark.parametrize("axon_path", _graph_ir_roundtrip_paths(), ids=lambda path: path.as_posix())
def test_graph_ir_weak_roundtrip_is_canonical(axon_path: Path, tmp_path: Path) -> None:
    assert graph_ir_weak_roundtrip_path(axon_path, tmp_path / "weak"), (
        f"Graph IR weak roundtrip changed rendered output for {axon_path}; "
        f"artifacts: {tmp_path}"
    )


@pytest.mark.parametrize("axon_path", _graph_ir_roundtrip_paths(), ids=lambda path: path.as_posix())
def test_graph_ir_strong_roundtrip_is_canonical(axon_path: Path, tmp_path: Path) -> None:
    assert graph_ir_strong_roundtrip_path(axon_path, tmp_path / "strong"), (
        f"Graph IR strong roundtrip changed rendered output for {axon_path}; "
        f"artifacts: {tmp_path}"
    )
