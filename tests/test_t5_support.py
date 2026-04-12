from __future__ import annotations

from pathlib import Path

import pytest

import tests.conftest as test_fixtures
from brainsurgery.synapse.axon_test import _infer_model_task
from brainsurgery.synapse.axon_test_matrix import (
    _Pair,
    _resolve_model_task_for_pair,
    run_axon_test_matrix,
)
from tests.model_downloads import MATRIX_AXON_MODEL_DIR_PAIRS, MODEL_SPECS


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


def test_t5gemma_family_files_are_split_by_checkpoint_family(repo_root: Path) -> None:
    t5gemma = (
        repo_root / "brainsurgery/synapse/models/t5gemma/generic-t5gemma-ul2.axon"
    ).read_text(encoding="utf-8")
    t5gemma_prefixlm = (
        repo_root / "brainsurgery/synapse/models/t5gemma/generic-t5gemma-prefixlm.axon"
    ).read_text(encoding="utf-8")
    t5gemma2 = (repo_root / "brainsurgery/synapse/models/t5gemma/generic-t5gemma-2.axon").read_text(
        encoding="utf-8"
    )

    assert "prefixlm" not in t5gemma
    assert "ul2" not in t5gemma_prefixlm
    assert "google/t5gemma-2-270m-270m" in t5gemma2
    assert "google/t5gemma-2-1b-1b" in t5gemma2
    assert "google/t5gemma-2-4b-4b" in t5gemma2


def test_generic_t5gemma_ul2_uses_mask_independent_position_ids(repo_root: Path) -> None:
    t5gemma = (
        repo_root / "brainsurgery/synapse/models/t5gemma/generic-t5gemma-ul2.axon"
    ).read_text(encoding="utf-8")

    assert "enc_pos_ids <- position_ids input_ids" in t5gemma
    assert "dec_pos_ids <- position_ids decoder_input_ids" in t5gemma


def test_axon_test_auto_infers_seq2seq_for_generic_t5gemma(repo_root: Path) -> None:
    axon = repo_root / "brainsurgery/synapse/models/t5gemma/generic-t5gemma-ul2.axon"
    weights = repo_root / "models/google/t5gemma-s-s-ul2"
    assert _infer_model_task(axon_file=axon, weights=weights) == "seq2seq_lm"


def test_axon_test_task_pragma_overrides_heuristic(tmp_path: Path) -> None:
    axon = tmp_path / "weird_name.axon"
    axon.write_text(
        "\n".join(
            [
                '{-# CHECKPOINTS "google/t5gemma-s-s-ul2" #-}',
                '{-# TASK "masked_lm" #-}',
                "",
                "main :: Tensor[B,S,D] -> Tensor[B,S,D]",
                "main x = do",
                "  return x",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    weights = Path("/work/training/brainsurgery/models/google/t5gemma-s-s-ul2")
    assert _infer_model_task(axon_file=axon, weights=weights) == "masked_lm"


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
