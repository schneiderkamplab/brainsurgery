from __future__ import annotations

from pathlib import Path

import pytest

from brainsurgery.synapse.axon_test_matrix import run_axon_test_matrix


def _write_fixture_pair(root: Path, name: str, *, model_dir_name: str | None = None) -> None:
    examples = root / "examples"
    models = root / "models"
    examples.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    (examples / f"{name}.axon").write_text(
        f"{name} :: Tensor -> Tensor\n{name} x = x\n", encoding="utf-8"
    )
    model_dir = model_dir_name if model_dir_name is not None else name
    (models / model_dir).mkdir(parents=True, exist_ok=True)


def test_axon_test_matrix_include_runs_only_selected_pairs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_fixture_pair(tmp_path, "gpt2")
    _write_fixture_pair(tmp_path, "gemma3")

    exit_code = run_axon_test_matrix(
        examples_dir=tmp_path / "examples",
        models_dir=tmp_path / "models",
        dry_run=True,
        include=["gpt2"],
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "gpt2.axon" in out
    assert "gemma3.axon" not in out


def test_axon_test_matrix_exclude_skips_selected_pairs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_fixture_pair(tmp_path, "gpt2")
    _write_fixture_pair(tmp_path, "gemma3")

    exit_code = run_axon_test_matrix(
        examples_dir=tmp_path / "examples",
        models_dir=tmp_path / "models",
        dry_run=True,
        exclude=["gpt2"],
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "gpt2.axon" not in out
    assert "gemma3.axon" in out


def test_axon_test_matrix_include_axon_filename_is_exact(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_fixture_pair(tmp_path, "gpt2", model_dir_name="gpt2")
    _write_fixture_pair(tmp_path, "gpt2_kv", model_dir_name="gpt2")

    exit_code = run_axon_test_matrix(
        examples_dir=tmp_path / "examples",
        models_dir=tmp_path / "models",
        dry_run=True,
        include=["gpt2.axon"],
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "gpt2.axon" in out
    assert "gpt2_kv.axon" not in out


def test_axon_test_matrix_rejects_include_and_exclude_together(tmp_path: Path) -> None:
    _write_fixture_pair(tmp_path, "gpt2")
    _write_fixture_pair(tmp_path, "gemma3")

    with pytest.raises(ValueError, match="either include or exclude"):
        run_axon_test_matrix(
            examples_dir=tmp_path / "examples",
            models_dir=tmp_path / "models",
            dry_run=True,
            include=["gpt2"],
            exclude=["gemma3"],
        )
