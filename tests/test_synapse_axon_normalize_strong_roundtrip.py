from __future__ import annotations

from pathlib import Path

import pytest

from scripts.axon_normalize_strong_roundtrip import normalize_strong_roundtrip_path


def _model_axon_paths() -> list[Path]:
    return sorted(Path("brainsurgery/synapse/models").glob("**/*.axon"))


@pytest.mark.parametrize("axon_path", _model_axon_paths(), ids=lambda path: path.as_posix())
def test_normalize_strong_render_roundtrip_is_canonical(
    axon_path: Path,
    tmp_path: Path,
) -> None:
    assert normalize_strong_roundtrip_path(axon_path, tmp_path)
