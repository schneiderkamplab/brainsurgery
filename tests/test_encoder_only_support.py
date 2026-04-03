from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import tests.conftest as test_fixtures
from brainsurgery.synapse import lower_axon_program_to_synapse_spec, parse_axon_program_from_path
from brainsurgery.synapse.axon_test_matrix import run_axon_test_matrix
from tests.model_downloads import MATRIX_AXON_MODEL_DIR_PAIRS, MODEL_SPECS

_MODEL_CASES: list[tuple[str, str, str]] = [
    ("roberta", "roberta", "roberta_local_path"),
    ("roberta", "camembert", "camembert_local_path"),
    ("roberta", "xlm_roberta", "xlm_roberta_local_path"),
    ("distilbert", "distilbert", "distilbert_local_path"),
    ("electra", "electra", "electra_local_path"),
    ("albert", "albert", "albert_local_path"),
    ("longformer", "longformer", "longformer_local_path"),
    ("deberta_v2", "deberta_v2", "deberta_v2_local_path"),
]


def _load_axon_spec(path: Path) -> dict[str, Any]:
    modules = parse_axon_program_from_path(path)
    return lower_axon_program_to_synapse_spec(modules)


@pytest.mark.parametrize(("axon_stem", "model_dir", "fixture_name"), _MODEL_CASES)
def test_encoder_only_fixture_and_registration(
    axon_stem: str,
    model_dir: str,
    fixture_name: str,
) -> None:
    assert hasattr(test_fixtures, fixture_name)
    assert model_dir in MODEL_SPECS
    assert (axon_stem, model_dir) in MATRIX_AXON_MODEL_DIR_PAIRS


@pytest.mark.parametrize(("axon_stem", "model_dir", "_"), _MODEL_CASES)
def test_matrix_resolves_encoder_only_pairs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    axon_stem: str,
    model_dir: str,
    _: str,
) -> None:
    examples_dir = tmp_path / "examples"
    models_dir = tmp_path / "models"
    examples_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    (examples_dir / f"{axon_stem}.axon").write_text(
        f"{axon_stem} :: Tensor -> Tensor\n{axon_stem} x = x\n",
        encoding="utf-8",
    )
    (models_dir / model_dir).mkdir(parents=True, exist_ok=True)

    exit_code = run_axon_test_matrix(
        examples_dir=examples_dir,
        models_dir=models_dir,
        dry_run=True,
        include=[model_dir],
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert f"{axon_stem}.axon" in out
    assert f"/models/{model_dir}" in out


@pytest.mark.parametrize(
    "axon_stem",
    ["roberta", "distilbert", "electra", "albert", "longformer", "deberta_v2"],
)
def test_encoder_only_axon_lowers_with_logits_output(repo_root: Path, axon_stem: str) -> None:
    spec = _load_axon_spec(repo_root / "examples" / f"{axon_stem}.axon")
    assert spec.get("synapse") == 1
    model = spec.get("model", {})
    assert model.get("outputs") == {"logits": "logits"}
