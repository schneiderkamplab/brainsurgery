from __future__ import annotations

from pathlib import Path

import pytest

import tests.conftest as test_fixtures
from brainsurgery.synapse.axon_test_matrix import run_axon_test_matrix
from tests.model_downloads import MATRIX_AXON_MODEL_DIR_PAIRS, MODEL_SPECS


def test_bert_fixture_is_declared() -> None:
    assert hasattr(test_fixtures, "bert_local_path")


def test_bert_is_registered_in_download_specs_and_matrix() -> None:
    assert MODEL_SPECS["bert"].repo_id == "google-bert/bert-base-uncased"
    assert MODEL_SPECS["bert"].local_dir == "bert"
    assert ("bert", "bert") in MATRIX_AXON_MODEL_DIR_PAIRS


def test_matrix_resolves_bert_to_bert_axon(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    examples_dir = tmp_path / "examples"
    models_dir = tmp_path / "models"
    examples_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    (examples_dir / "bert.axon").write_text(
        "bert :: Tensor -> Tensor\nbert x = x\n",
        encoding="utf-8",
    )
    (models_dir / "bert").mkdir(parents=True, exist_ok=True)

    exit_code = run_axon_test_matrix(
        examples_dir=examples_dir,
        models_dir=models_dir,
        dry_run=True,
        include=["bert"],
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "bert.axon" in out
    assert "/models/bert" in out
