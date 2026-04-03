from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import tests.conftest as test_fixtures
from brainsurgery.synapse import lower_axon_program_to_synapse_spec, parse_axon_program_from_path
from brainsurgery.synapse.axon_test_matrix import (
    _Pair,
    _resolve_model_task_for_pair,
    run_axon_test_matrix,
)
from tests.model_downloads import MATRIX_AXON_MODEL_DIR_PAIRS, MODEL_SPECS


def _load_axon_spec(path: Path) -> dict[str, Any]:
    modules = parse_axon_program_from_path(path)
    return lower_axon_program_to_synapse_spec(modules)


def test_t5_small_fixture_is_declared() -> None:
    assert hasattr(test_fixtures, "t5_small_local_path")
    assert hasattr(test_fixtures, "mt5_small_local_path")
    assert hasattr(test_fixtures, "bart_base_local_path")
    assert hasattr(test_fixtures, "mbart_large_50_m2m_local_path")
    assert hasattr(test_fixtures, "marian_en_de_local_path")
    assert hasattr(test_fixtures, "t5gemma_s_s_ul2_local_path")
    assert hasattr(test_fixtures, "t5gemma2_270m_local_path")


def test_t5_small_is_registered_in_download_specs_and_matrix() -> None:
    assert MODEL_SPECS["t5_small"].repo_id == "google-t5/t5-small"
    assert MODEL_SPECS["t5_small"].local_dir == "t5_small"
    assert ("t5_small", "t5_small") in MATRIX_AXON_MODEL_DIR_PAIRS
    assert MODEL_SPECS["mt5_small"].repo_id == "google/mt5-small"
    assert MODEL_SPECS["bart_base"].repo_id == "facebook/bart-base"
    assert MODEL_SPECS["mbart_large_50_m2m"].repo_id == "facebook/mbart-large-50-many-to-many-mmt"
    assert MODEL_SPECS["marian_en_de"].repo_id == "Helsinki-NLP/opus-mt-en-de"
    assert MODEL_SPECS["t5gemma_s_s_ul2"].repo_id == "google/t5gemma-s-s-ul2"
    assert MODEL_SPECS["t5gemma2_270m"].repo_id == "google/t5gemma-2-270m-270m"
    assert ("mt5", "mt5_small") in MATRIX_AXON_MODEL_DIR_PAIRS
    assert ("bart", "bart_base") in MATRIX_AXON_MODEL_DIR_PAIRS
    assert ("mbart", "mbart_large_50_m2m") in MATRIX_AXON_MODEL_DIR_PAIRS
    assert ("marian", "marian_en_de") in MATRIX_AXON_MODEL_DIR_PAIRS
    assert ("t5gemma", "t5gemma_s_s_ul2") in MATRIX_AXON_MODEL_DIR_PAIRS
    assert ("t5gemma2", "t5gemma2_270m") in MATRIX_AXON_MODEL_DIR_PAIRS


def test_matrix_resolves_t5_small_to_t5_small_axon(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    examples_dir = tmp_path / "examples"
    models_dir = tmp_path / "models"
    examples_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    (examples_dir / "t5_small.axon").write_text(
        "t5_small :: Tensor -> Tensor\nt5_small x = do\n  return x\n", encoding="utf-8"
    )
    (models_dir / "t5_small").mkdir(parents=True, exist_ok=True)

    exit_code = run_axon_test_matrix(
        examples_dir=examples_dir,
        models_dir=models_dir,
        dry_run=True,
        include=["t5_small"],
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "t5_small.axon" in out
    assert "/models/t5_small" in out


def test_matrix_auto_task_resolves_t5_small_to_seq2seq_lm() -> None:
    pair = _Pair(axon_path=Path("examples/t5_small.axon"), model_dir=Path("models/t5_small"))
    assert _resolve_model_task_for_pair(pair) == "seq2seq_lm"


@pytest.mark.parametrize(
    ("axon_name", "model_dir_name"),
    [
        ("mt5", "mt5_small"),
        ("bart", "bart_base"),
        ("mbart", "mbart_large_50_m2m"),
        ("marian", "marian_en_de"),
        ("t5gemma", "t5gemma_s_s_ul2"),
        ("t5gemma2", "t5gemma2_270m"),
    ],
)
def test_matrix_auto_task_resolves_seq2seq_families(
    axon_name: str,
    model_dir_name: str,
) -> None:
    pair = _Pair(
        axon_path=Path(f"examples/{axon_name}.axon"),
        model_dir=Path(f"models/{model_dir_name}"),
    )
    assert _resolve_model_task_for_pair(pair) == "seq2seq_lm"


def test_t5_axon_lowers_with_expected_symbols(repo_root: Path) -> None:
    spec = _load_axon_spec(repo_root / "examples" / "t5.axon")

    assert spec.get("synapse") == 1
    model = spec.get("model", {})
    assert model.get("outputs") == {"logits": "logits"}

    symbols = model.get("symbols", {})
    assert symbols.get("D") == 512
    assert symbols.get("V") is None
    assert symbols.get("L_ENC") == 6
    assert symbols.get("L_DEC") == 6
    assert symbols.get("H") == 8
    assert symbols.get("FFN") == 2048
    assert symbols.get("EPS") == 1.0e-06
    assert symbols.get("NUM_BUCKETS") == 32
    assert symbols.get("MAX_DISTANCE") == 128
