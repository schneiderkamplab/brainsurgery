from __future__ import annotations

from pathlib import Path

import pytest

import tests.conftest as test_fixtures
from brainsurgery.synapse.axon_test_matrix import run_axon_test_matrix
from tests.model_downloads import MATRIX_AXON_MODEL_DIR_PAIRS, MODEL_SPECS


def test_modernbert_fixture_is_declared() -> None:
    assert hasattr(test_fixtures, "modernbert_local_path")


def test_modernbert_is_registered_in_download_specs_and_matrix() -> None:
    assert MODEL_SPECS["modernbert"].repo_id == "answerdotai/ModernBERT-base"
    assert MODEL_SPECS["modernbert"].local_dir == "modernbert"
    assert ("modernbert", "modernbert") in MATRIX_AXON_MODEL_DIR_PAIRS


def test_matrix_resolves_modernbert_to_modernbert_axon(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    examples_dir = tmp_path / "examples"
    models_dir = tmp_path / "models"
    examples_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    (examples_dir / "modernbert.axon").write_text(
        "modernbert :: Tensor -> Tensor\nmodernbert x = x\n",
        encoding="utf-8",
    )
    (models_dir / "modernbert").mkdir(parents=True, exist_ok=True)

    exit_code = run_axon_test_matrix(
        examples_dir=examples_dir,
        models_dir=models_dir,
        dry_run=True,
        include=["modernbert"],
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "modernbert.axon" in out
    assert "/models/modernbert" in out
