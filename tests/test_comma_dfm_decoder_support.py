from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import tests.conftest as test_fixtures
from brainsurgery.synapse import lower_axon_program_to_synapse_spec, parse_axon_program_from_path
from brainsurgery.synapse.axon_test import (
    _load_auto_config_with_compat_fallback,
    _normalize_rope_numeric_fields,
)
from brainsurgery.synapse.axon_test_matrix import run_axon_test_matrix
from tests.model_downloads import MATRIX_AXON_MODEL_DIR_PAIRS, MODEL_SPECS


def _load_axon_spec(path: Path) -> dict[str, Any]:
    modules = parse_axon_program_from_path(path)
    return lower_axon_program_to_synapse_spec(modules)


def test_comma_and_dfm_decoder_fixtures_are_declared() -> None:
    assert hasattr(test_fixtures, "comma_local_path")
    assert hasattr(test_fixtures, "dfm_decoder_local_path")


def test_comma_and_dfm_decoder_are_registered_in_download_specs_and_matrix() -> None:
    assert MODEL_SPECS["comma"].repo_id == "common-pile/comma-v0.1-1t"
    assert MODEL_SPECS["comma"].local_dir == "comma"

    assert (
        MODEL_SPECS["dfm_decoder"].repo_id == "danish-foundation-models/dfm-decoder-open-v0-7b-pt"
    )
    assert MODEL_SPECS["dfm_decoder"].local_dir == "dfm_decoder"
    assert ("dfm_decoder", "dfm_decoder") in MATRIX_AXON_MODEL_DIR_PAIRS
    assert ("dfm_decoder", "comma") in MATRIX_AXON_MODEL_DIR_PAIRS


def test_matrix_resolves_comma_to_shared_dfm_decoder_axon(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    examples_dir = tmp_path / "examples"
    models_dir = tmp_path / "models"
    examples_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    (examples_dir / "dfm_decoder.axon").write_text(
        "dfm_decoder :: Tensor -> Tensor\ndfm_decoder x = x\n",
        encoding="utf-8",
    )
    (models_dir / "comma").mkdir(parents=True, exist_ok=True)

    exit_code = run_axon_test_matrix(
        examples_dir=examples_dir,
        models_dir=models_dir,
        dry_run=True,
        include=["comma"],
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "dfm_decoder.axon" in out
    assert "/models/comma" in out


def test_dfm_decoder_axon_lowers_with_expected_symbols(repo_root: Path) -> None:
    spec = _load_axon_spec(repo_root / "brainsurgery" / "synapse" / "models" / "comma" / "comma-v0.1-1t.axon")
    assert spec.get("synapse") == 1

    model = spec.get("model", {})
    assert model.get("outputs") == {"logits": "logits", "new_kv": "new_kv"}

    symbols = model.get("symbols", {})
    assert symbols.get("D") == 4096
    assert symbols.get("V") == 64256
    assert symbols.get("L") == 32
    assert symbols.get("H") == 32
    assert symbols.get("KVH") == 32
    assert symbols.get("KVD") == 4096
    assert symbols.get("FFN") == 11008
    assert symbols.get("EPS") == 1.0e-05
    assert symbols.get("THETA") == 100000.0
    assert symbols.get("C") == 16384

    blocks = model.get("blocks", {})
    assert "comma_block" in blocks


def test_dfm_decoder_rope_normalization_preserves_nondefault_theta(repo_root: Path) -> None:
    cfg = _load_auto_config_with_compat_fallback(
        repo_root / "models" / "dfm_decoder", trust_remote_code=False
    )
    assert getattr(cfg, "rope_scaling") == {"rope_theta": 100000.0, "rope_type": "default"}
    assert getattr(cfg, "rope_parameters") == {"rope_theta": 100000.0, "rope_type": "default"}

    cfg = _normalize_rope_numeric_fields(cfg)

    assert getattr(cfg, "rope_scaling") == {
        "rope_theta": 100000.0,
        "rope_type": "default",
        "type": "default",
    }
    assert getattr(cfg, "rope_parameters") == {
        "rope_theta": 100000.0,
        "rope_type": "default",
        "type": "default",
    }
