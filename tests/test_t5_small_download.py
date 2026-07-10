from __future__ import annotations

from pathlib import Path


def test_t5_small_fixture_downloads_model_dir(t5_small_local_path: Path) -> None:
    assert t5_small_local_path.name == "t5_small"
    assert (t5_small_local_path / "config.json").exists()
    assert (t5_small_local_path / "model.safetensors").exists() or (
        t5_small_local_path / "model.safetensors.index.json"
    ).exists()
