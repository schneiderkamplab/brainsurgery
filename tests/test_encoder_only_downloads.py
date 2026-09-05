from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("fixture_name", "model_dir_name"),
    [
        ("roberta_local_path", "roberta"),
        ("camembert_local_path", "camembert"),
        ("xlm_roberta_local_path", "xlm_roberta"),
        ("distilbert_local_path", "distilbert"),
        ("electra_local_path", "electra"),
        ("albert_local_path", "albert"),
        ("deberta_v2_local_path", "deberta_v2"),
        ("longformer_local_path", "longformer"),
    ],
)
def test_encoder_only_fixtures_download_model_dirs(
    request: pytest.FixtureRequest,
    fixture_name: str,
    model_dir_name: str,
) -> None:
    local_path = request.getfixturevalue(fixture_name)
    assert isinstance(local_path, Path)
    assert local_path.name == model_dir_name
    assert (local_path / "config.json").exists()
    assert (
        (local_path / "model.safetensors").exists()
        or (local_path / "model.safetensors.index.json").exists()
        or (local_path / "pytorch_model.bin").exists()
    )
