from __future__ import annotations

from pathlib import Path


def test_apertus_8b_fixture_downloads_model_dir(apertus_8b_local_path: Path) -> None:
    assert apertus_8b_local_path.name == "apertus_8b"
    assert (apertus_8b_local_path / "config.json").exists()
    assert (apertus_8b_local_path / "model.safetensors").exists() or (
        apertus_8b_local_path / "model.safetensors.index.json"
    ).exists()
