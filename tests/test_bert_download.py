from __future__ import annotations

from pathlib import Path


def test_bert_fixture_downloads_model_dir(bert_local_path: Path) -> None:
    assert bert_local_path.name == "bert"
    assert (bert_local_path / "config.json").exists()
    assert (bert_local_path / "model.safetensors").exists() or (
        bert_local_path / "model.safetensors.index.json"
    ).exists()
