from __future__ import annotations

from pathlib import Path

import pytest

from scripts.axon_typecheck2_strong_roundtrip import typecheck2_strong_roundtrip_path


def _typecheck2_roundtrip_paths() -> list[Path]:
    return sorted(Path("brainsurgery/synapse/models").glob("**/*.axon"))


@pytest.mark.parametrize("axon_path", _typecheck2_roundtrip_paths(), ids=lambda path: path.as_posix())
def test_typecheck2_strong_render_roundtrip_is_canonical(
    axon_path: Path,
    tmp_path: Path,
) -> None:
    assert typecheck2_strong_roundtrip_path(axon_path, tmp_path, show_types=True), (
        f"typecheck2 strong roundtrip changed rendered output for {axon_path}; "
        f"artifacts: {tmp_path}"
    )
