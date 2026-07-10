from __future__ import annotations

from pathlib import Path

import pytest

from scripts.axon_flatten_weak_roundtrip import flatten_weak_roundtrip_path


def _model_axon_paths() -> list[Path]:
    return sorted(Path("brainsurgery/synapse/models").glob("**/*.axon"))


@pytest.mark.parametrize("axon_path", _model_axon_paths(), ids=lambda path: path.as_posix())
def test_flatten_weak_render_roundtrip_is_flat_and_canonical(
    axon_path: Path,
    tmp_path: Path,
) -> None:
    assert flatten_weak_roundtrip_path(axon_path, tmp_path)
