from __future__ import annotations

from pathlib import Path


def test_modernbert_fixture_downloads_model_dir(modernbert_local_path: Path) -> None:
    assert modernbert_local_path.name == "modernbert"
    assert (modernbert_local_path / "config.json").exists()
    assert (modernbert_local_path / "model.safetensors").exists() or (
        modernbert_local_path / "model.safetensors.index.json"
    ).exists()
