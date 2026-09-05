from __future__ import annotations

from pathlib import Path


def test_smollm_135m_fixture_downloads_model_dir(smollm_135m_local_path: Path) -> None:
    assert smollm_135m_local_path.name == "smollm_135m"
    assert (smollm_135m_local_path / "config.json").exists()
    assert (smollm_135m_local_path / "model.safetensors").exists() or (
        smollm_135m_local_path / "model.safetensors.index.json"
    ).exists()
