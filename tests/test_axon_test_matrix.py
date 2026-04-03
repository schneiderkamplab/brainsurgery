from __future__ import annotations

from pathlib import Path

import pytest

import brainsurgery.synapse.axon_test_matrix as matrix_mod
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


def test_resolve_pairs_skips_config_axons(tmp_path: Path) -> None:
    _write_fixture_pair(tmp_path, "olmo2", model_dir_name="olmo_2_1b")
    _write_fixture_pair(tmp_path, "olmo2_config", model_dir_name="olmo_2_1b")

    pairs = matrix_mod._resolve_pairs(tmp_path / "examples", tmp_path / "models")

    assert {(pair.axon_path.stem, pair.model_dir.name) for pair in pairs} == {
        ("olmo2", "olmo_2_1b")
    }


def test_resolve_pairs_excludes_flexolmo_model_dir_but_keeps_flexmath(tmp_path: Path) -> None:
    examples = tmp_path / "examples"
    models = tmp_path / "models"
    examples.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)

    (examples / "flexolmo.axon").write_text(
        "flexolmo :: Tensor -> Tensor\nflexolmo x = x\n", encoding="utf-8"
    )
    (models / "flexmath").mkdir()
    (models / "flexolmo").mkdir()

    pairs = matrix_mod._resolve_pairs(examples, models)

    assert {(pair.axon_path.stem, pair.model_dir.name) for pair in pairs} == {
        ("flexolmo", "flexmath")
    }


def test_resolve_pairs_covers_non_config_examples_and_model_aliases(tmp_path: Path) -> None:
    examples = tmp_path / "examples"
    models = tmp_path / "models"
    examples.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)

    example_names = [
        "dfm_decoder",
        "olmo2",
        "olmo3",
        "olmo3_1025_7b",
        "phi3",
        "phi3_mini_4k_instruct",
        "roberta",
        "t5",
        "t5_small",
    ]
    for name in example_names:
        (examples / f"{name}.axon").write_text(f"{name} :: Tensor -> Tensor\n{name} x = x\n")
    (examples / "olmo3_config.axon").write_text(
        "olmo3_config :: Tensor -> Tensor\nolmo3_config x = x\n"
    )

    model_dir_names = [
        ".hf_modules_cache",
        "camembert",
        "comma",
        "glm_4_5_air",
        "nemotron3",
        "olmo_2_1b",
        "olmo_2_7b",
        "olmo_2_13b",
        "olmo3_1025_7b",
        "olmo3_7b_instruct",
        "olmo3_7b_think",
        "phi3_mini_4k_instruct",
        "phi3_small_8k_instruct microsoft",
        "t5_base",
        "t5_small",
        "test",
        "xlm_roberta",
    ]
    for name in model_dir_names:
        (models / name).mkdir(parents=True, exist_ok=True)

    pairs = matrix_mod._resolve_pairs(examples, models)
    covered_axons = {pair.axon_path.stem for pair in pairs}
    covered_models = {pair.model_dir.name for pair in pairs}

    assert covered_axons == set(example_names)
    assert {
        "camembert",
        "comma",
        "olmo_2_1b",
        "olmo_2_7b",
        "olmo_2_13b",
        "olmo3_1025_7b",
        "olmo3_7b_instruct",
        "olmo3_7b_think",
        "phi3_mini_4k_instruct",
        "phi3_small_8k_instruct microsoft",
        "t5_base",
        "t5_small",
        "xlm_roberta",
    } <= covered_models
    assert ".hf_modules_cache" not in covered_models
    assert "glm_4_5_air" not in covered_models
    assert "nemotron3" not in covered_models
    assert "test" not in covered_models


def test_axon_test_matrix_auto_task_uses_masked_lm_for_bert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_fixture_pair(tmp_path, "bert")
    captured_model_tasks: list[str] = []

    def _fake_run_pair(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args
        captured_model_tasks.append(str(kwargs["model_task"]))
        return {
            "hf_time": 1.0,
            "axon_time": 1.0,
            "speed_ratio_axon_over_hf": 1.0,
            "max_diff": 0.0,
            "max_rel_diff": 0.0,
            "mean_rel_diff": 0.0,
            "top1_eq": True,
            "masked_max_diff": None,
            "masked_last_max_diff": None,
            "masked_mean_rel_diff": None,
            "masked_max_rel_diff": None,
            "masked_top1_eq": None,
        }

    monkeypatch.setattr(matrix_mod, "_run_pair", _fake_run_pair)

    exit_code = run_axon_test_matrix(
        examples_dir=tmp_path / "examples",
        models_dir=tmp_path / "models",
        table_format="plain",
    )
    assert exit_code == 0
    assert captured_model_tasks == ["masked_lm"]


def test_axon_test_matrix_auto_task_uses_masked_lm_for_modernbert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_fixture_pair(tmp_path, "modernbert")
    captured_model_tasks: list[str] = []

    def _fake_run_pair(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args
        captured_model_tasks.append(str(kwargs["model_task"]))
        return {
            "hf_time": 1.0,
            "axon_time": 1.0,
            "speed_ratio_axon_over_hf": 1.0,
            "max_diff": 0.0,
            "max_rel_diff": 0.0,
            "mean_rel_diff": 0.0,
            "top1_eq": True,
            "masked_max_diff": None,
            "masked_last_max_diff": None,
            "masked_mean_rel_diff": None,
            "masked_max_rel_diff": None,
            "masked_top1_eq": None,
        }

    monkeypatch.setattr(matrix_mod, "_run_pair", _fake_run_pair)

    exit_code = run_axon_test_matrix(
        examples_dir=tmp_path / "examples",
        models_dir=tmp_path / "models",
        table_format="plain",
    )
    assert exit_code == 0
    assert captured_model_tasks == ["masked_lm"]


@pytest.mark.parametrize(
    ("axon_name", "model_dir_name"),
    [
        ("roberta", None),
        ("roberta", "camembert"),
        ("roberta", "xlm_roberta"),
        ("distilbert", None),
        ("electra", None),
        ("albert", None),
        ("longformer", None),
    ],
)
def test_axon_test_matrix_auto_task_uses_masked_lm_for_encoder_only_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    axon_name: str,
    model_dir_name: str | None,
) -> None:
    _write_fixture_pair(tmp_path, axon_name, model_dir_name=model_dir_name)
    captured_model_tasks: list[str] = []

    def _fake_run_pair(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args
        captured_model_tasks.append(str(kwargs["model_task"]))
        return {
            "hf_time": 1.0,
            "axon_time": 1.0,
            "speed_ratio_axon_over_hf": 1.0,
            "max_diff": 0.0,
            "max_rel_diff": 0.0,
            "mean_rel_diff": 0.0,
            "top1_eq": True,
            "masked_max_diff": None,
            "masked_last_max_diff": None,
            "masked_mean_rel_diff": None,
            "masked_max_rel_diff": None,
            "masked_top1_eq": None,
        }

    monkeypatch.setattr(matrix_mod, "_run_pair", _fake_run_pair)

    exit_code = run_axon_test_matrix(
        examples_dir=tmp_path / "examples",
        models_dir=tmp_path / "models",
        table_format="plain",
    )
    assert exit_code == 0
    assert captured_model_tasks == ["masked_lm"]


def test_axon_test_matrix_task_override_is_applied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_fixture_pair(tmp_path, "bert")
    captured_model_tasks: list[str] = []

    def _fake_run_pair(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args
        captured_model_tasks.append(str(kwargs["model_task"]))
        return {
            "hf_time": 1.0,
            "axon_time": 1.0,
            "speed_ratio_axon_over_hf": 1.0,
            "max_diff": 0.0,
            "max_rel_diff": 0.0,
            "mean_rel_diff": 0.0,
            "top1_eq": True,
            "masked_max_diff": None,
            "masked_last_max_diff": None,
            "masked_mean_rel_diff": None,
            "masked_max_rel_diff": None,
            "masked_top1_eq": None,
        }

    monkeypatch.setattr(matrix_mod, "_run_pair", _fake_run_pair)

    exit_code = run_axon_test_matrix(
        examples_dir=tmp_path / "examples",
        models_dir=tmp_path / "models",
        table_format="plain",
        model_task_override="causal_lm",
    )
    assert exit_code == 0
    assert captured_model_tasks == ["causal_lm"]
