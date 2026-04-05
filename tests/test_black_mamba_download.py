from __future__ import annotations

from pathlib import Path


def test_black_mamba_fixture_downloads_model_dir(black_mamba_local_path: Path) -> None:
    assert black_mamba_local_path.name in {"black_mamba", "black_mamba_2_8b"}
    assert (black_mamba_local_path / "config.json").exists()
    assert (
        (black_mamba_local_path / "model.safetensors").exists()
        or (black_mamba_local_path / "model.safetensors.index.json").exists()
        or (black_mamba_local_path / "pytorch_model.bin").exists()
    )
