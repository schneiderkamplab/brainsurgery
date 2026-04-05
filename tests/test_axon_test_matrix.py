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
        ("olmo2", "olmo_2_1b"),
        ("olmo2", "olmo_2_7b"),
        ("olmo2", "olmo_2_13b"),
    }


def test_resolve_pairs_keeps_flexolmo_and_flexmath_model_dirs(tmp_path: Path) -> None:
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
        ("flexolmo", "flexmath"),
        ("flexolmo", "flexolmo"),
        ("flexolmo", "flexolmo_7x7b_1t"),
    }


def test_resolve_pairs_maps_gpt_oss_120b_to_gpt_oss_20b_axon(tmp_path: Path) -> None:
    examples = tmp_path / "examples"
    models = tmp_path / "models"
    examples.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)

    (examples / "gpt_oss_20b.axon").write_text(
        "gpt_oss_20b :: Tensor -> Tensor\ngpt_oss_20b x = x\n",
        encoding="utf-8",
    )
    (models / "gpt_oss_20b").mkdir()
    (models / "gpt_oss_120b").mkdir()

    pairs = matrix_mod._resolve_pairs(examples, models)

    assert {(pair.axon_path.stem, pair.model_dir.name) for pair in pairs} == {
        ("gpt_oss_20b", "gpt_oss_20b"),
        ("gpt_oss_20b", "gpt_oss_120b"),
    }


def test_resolve_pairs_maps_bart_base_to_bart_axon(tmp_path: Path) -> None:
    examples = tmp_path / "examples"
    models = tmp_path / "models"
    examples.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)

    (examples / "bart.axon").write_text(
        "bart :: Tensor -> Tensor\nbart x = x\n",
        encoding="utf-8",
    )
    (models / "bart_base").mkdir()

    pairs = matrix_mod._resolve_pairs(examples, models)

    assert {(pair.axon_path.stem, pair.model_dir.name) for pair in pairs} == {
        ("bart", "bart_base"),
    }


def test_resolve_pairs_maps_deepseek_v2_lite_to_matching_axon(tmp_path: Path) -> None:
    examples = tmp_path / "examples"
    models = tmp_path / "models"
    examples.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)

    (examples / "deepseek_v2_lite.axon").write_text(
        "deepseek_v2_lite :: Tensor -> Tensor\ndeepseek_v2_lite x = x\n",
        encoding="utf-8",
    )
    (models / "deepseek_v2_lite").mkdir()

    pairs = matrix_mod._resolve_pairs(examples, models)

    assert {(pair.axon_path.stem, pair.model_dir.name) for pair in pairs} == {
        ("deepseek_v2_lite", "deepseek_v2_lite"),
    }


def test_resolve_pairs_includes_registered_models_without_local_dirs(tmp_path: Path) -> None:
    examples = tmp_path / "examples"
    models = tmp_path / "models"
    examples.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)

    (examples / "marian.axon").write_text(
        "marian :: Tensor -> Tensor\nmarian x = x\n",
        encoding="utf-8",
    )

    pairs = matrix_mod._resolve_pairs(examples, models)

    assert [(pair.axon_path.stem, pair.model_dir.name) for pair in pairs] == [
        ("marian", "marian_en_de")
    ]


def test_resolve_pairs_covers_non_config_examples_and_model_aliases(tmp_path: Path) -> None:
    examples = tmp_path / "examples"
    models = tmp_path / "models"
    examples.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)

    example_names = [
        "dfm_decoder",
        "olmo2",
        "olmo3",
        "phi3_mini_4k_instruct",
        "phi3minimedium",
        "phi3small",
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
        "phi3_mini_128k_instruct",
        "phi3_medium_4k_instruct",
        "phi3_medium_128k_instruct",
        "phi3_small_8k_instruct",
        "phi3_small_128k_instruct",
        "t5_base",
        "t5_small",
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
        "phi3_mini_128k_instruct",
        "phi3_medium_4k_instruct",
        "phi3_medium_128k_instruct",
        "phi3_small_8k_instruct",
        "phi3_small_128k_instruct",
        "t5_base",
        "t5_small",
        "xlm_roberta",
    } <= covered_models
    assert ".hf_modules_cache" not in covered_models


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
    assert captured_model_tasks
    assert all(task == "masked_lm" for task in captured_model_tasks)


def test_apply_billions_params_filter_uses_remote_lower_bound_for_max(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair = matrix_mod._Pair(
        axon_path=tmp_path / "examples" / "marian.axon",
        model_dir=tmp_path / "models" / "marian_en_de",
    )

    monkeypatch.setattr(matrix_mod, "_estimate_model_param_count", lambda model_dir: None)
    monkeypatch.setattr(
        matrix_mod,
        "estimate_remote_param_count_lower_bound",
        lambda *, repo_root, spec: 3_000_000_000,
    )

    kept, skipped = matrix_mod._apply_billions_params_filter(
        [pair],
        min_billions_params=None,
        max_billions_params=2,
    )

    assert kept == []
    assert len(skipped) == 1
    assert skipped[0].pair == pair
    assert skipped[0].param_count == 3_000_000_000
    assert skipped[0].is_exact is False


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
    assert captured_model_tasks
    assert all(task == "masked_lm" for task in captured_model_tasks)


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
    assert captured_model_tasks
    assert all(task == "masked_lm" for task in captured_model_tasks)


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


def test_apply_billions_params_filter_supports_min_and_max(tmp_path: Path) -> None:
    pairs = [
        matrix_mod._Pair(axon_path=tmp_path / "small.axon", model_dir=tmp_path / "small"),
        matrix_mod._Pair(axon_path=tmp_path / "mid.axon", model_dir=tmp_path / "mid"),
        matrix_mod._Pair(axon_path=tmp_path / "large.axon", model_dir=tmp_path / "large"),
        matrix_mod._Pair(
            axon_path=tmp_path / "unknown.axon",
            model_dir=tmp_path / "unknown",
        ),
    ]
    param_counts = {
        tmp_path / "small": 500_000_000,
        tmp_path / "mid": 12_000_000_000,
        tmp_path / "large": 30_000_000_000,
        tmp_path / "unknown": None,
    }

    def _fake_estimate(model_dir: Path) -> int | None:
        return param_counts[model_dir]

    original = matrix_mod._estimate_model_param_count
    matrix_mod._estimate_model_param_count = _fake_estimate
    try:
        kept, skipped = matrix_mod._apply_billions_params_filter(
            pairs,
            min_billions_params=10,
            max_billions_params=20,
        )
    finally:
        matrix_mod._estimate_model_param_count = original

    assert [pair.model_dir.name for pair in kept] == ["mid", "unknown"]
    assert [(item.pair.model_dir.name, item.param_count) for item in skipped] == [
        ("small", 500_000_000),
        ("large", 30_000_000_000),
    ]


def test_apply_billions_params_filter_rejects_invalid_bounds(tmp_path: Path) -> None:
    pair = matrix_mod._Pair(axon_path=tmp_path / "x.axon", model_dir=tmp_path / "x")

    with pytest.raises(ValueError, match="min_billions_params must be > 0"):
        matrix_mod._apply_billions_params_filter(
            [pair],
            min_billions_params=0,
            max_billions_params=None,
        )

    with pytest.raises(ValueError, match="max_billions_params must be > 0"):
        matrix_mod._apply_billions_params_filter(
            [pair],
            min_billions_params=None,
            max_billions_params=0,
        )

    with pytest.raises(ValueError, match="min_billions_params must be <= max_billions_params"):
        matrix_mod._apply_billions_params_filter(
            [pair],
            min_billions_params=20,
            max_billions_params=10,
        )


def test_resolve_worker_devices_repeats_cpu_for_multiple_processes() -> None:
    assert matrix_mod._resolve_worker_devices("cpu", 3) == ["cpu", "cpu", "cpu"]


def test_run_axon_test_matrix_uses_parallel_helper_when_processes_gt_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_fixture_pair(tmp_path, "gpt2")
    captured: dict[str, object] = {}

    def _fake_parallel(*args, **kwargs):  # type: ignore[no-untyped-def]
        captured["runnable_pairs"] = args[0]
        captured.update(kwargs)
        pair = args[0][0]
        return (
            [
                matrix_mod._SummaryRow(
                    axon_file=pair.axon_path.name,
                    model_dir=str(pair.model_dir),
                    hf_runtime_s="1",
                    axon_runtime_s="1",
                    runtime_ratio="1",
                    eval_max_abs_diff="0",
                    eval_max_rel_diff="0",
                    debug_max_logit_diff="0",
                    debug_max_rel_diff="0",
                    mean_rel_diff="0",
                    masked_max_diff="N/A",
                    masked_last_max_diff="N/A",
                    masked_mean_rel_diff="N/A",
                    masked_max_rel_diff="N/A",
                    eval_top1_eq="True",
                    debug_top1_eq="True",
                    masked_top1_eq="N/A",
                )
            ],
            1,
            0,
        )

    monkeypatch.setattr(matrix_mod, "_run_runnable_pairs_parallel", _fake_parallel)

    exit_code = run_axon_test_matrix(
        examples_dir=tmp_path / "examples",
        models_dir=tmp_path / "models",
        table_format="plain",
        processes=2,
        log_dir=tmp_path / "logs",
    )

    assert exit_code == 0
    assert captured["processes"] == 2
    assert captured["device"] == "cpu"
    assert captured["prompts"] == ["The future of AI is"]
    assert captured["log_dir"] == tmp_path / "logs"
    runnable_pairs = captured["runnable_pairs"]
    assert isinstance(runnable_pairs, list)
    assert len(runnable_pairs) == 1


def test_worker_log_path_includes_pid_axon_and_model(tmp_path: Path) -> None:
    pair = matrix_mod._Pair(tmp_path / "examples" / "gpt2.axon", tmp_path / "models" / "gpt2")

    path = matrix_mod._worker_log_path(tmp_path / "logs", pair, 1234)

    assert path == tmp_path / "logs" / "log-1234-gpt2-gpt2.txt"


def test_run_worker_loop_writes_log_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pair = matrix_mod._Pair(tmp_path / "examples" / "gpt2.axon", tmp_path / "models" / "gpt2")
    seen: list[tuple] = []

    class _Queue:
        def put(self, item):  # type: ignore[no-untyped-def]
            seen.append(item)

    monkeypatch.setattr(matrix_mod, "_maybe_ensure_pair_model_ready", lambda pair: None)

    def _fake_run_pair(*args, **kwargs):  # type: ignore[no-untyped-def]
        print("worker hello")
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
        }, None

    monkeypatch.setattr(matrix_mod, "_run_pair_with_fallback", _fake_run_pair)

    matrix_mod._run_worker_loop(
        0,
        pair,
        "cpu",
        _Queue(),
        {
            "dtype": "float32",
            "max_len": 16,
            "text": ["The future of AI is"],
            "verbose": False,
            "no_capture_output": False,
            "compile_hf": False,
            "compile_axon": False,
            "compile_backend": None,
            "compile_mode": None,
            "compile_fullgraph": False,
            "compile_dynamic": False,
            "model_task_override": None,
        },
        str(tmp_path / "logs"),
    )

    assert len(seen) == 1
    _, _, _, captured_output, log_path = seen[0]
    assert "worker hello" in captured_output
    assert log_path is not None
    log_file = Path(log_path)
    assert log_file.exists()
    assert log_file.parent == tmp_path / "logs"
    assert f"-{pair.axon_path.stem}-{pair.model_dir.name}.txt" in log_file.name
    assert "worker hello" in log_file.read_text(encoding="utf-8")


def test_run_axon_test_matrix_downloads_only_filtered_pairs_in_serial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_fixture_pair(tmp_path, "small")
    _write_fixture_pair(tmp_path, "large")
    (tmp_path / "models" / "small" / "model.safetensors").write_text("stub", encoding="utf-8")
    (tmp_path / "models" / "large" / "model.safetensors").write_text("stub", encoding="utf-8")

    monkeypatch.setattr(
        matrix_mod,
        "_estimate_model_param_count",
        lambda model_dir: 500_000_000 if model_dir.name == "small" else 3_000_000_000,
    )
    monkeypatch.setattr(
        matrix_mod,
        "MODEL_SPECS",
        {
            "small": matrix_mod.MODEL_SPECS.get("gpt2").__class__("small", "repo/small"),
            "large": matrix_mod.MODEL_SPECS.get("gpt2").__class__("large", "repo/large"),
        },
    )
    downloaded: list[str] = []

    def _fake_ensure_model_downloaded(*, repo_root: Path, spec, status_cb=None):  # type: ignore[no-untyped-def]
        del repo_root, status_cb
        downloaded.append(str(spec.local_dir))
        return tmp_path / "models" / str(spec.local_dir)

    def _fake_run_pair(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
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

    monkeypatch.setattr(matrix_mod, "ensure_model_downloaded", _fake_ensure_model_downloaded)
    monkeypatch.setattr(matrix_mod, "_run_pair", _fake_run_pair)
    monkeypatch.setattr(
        matrix_mod,
        "_resolve_pairs",
        lambda examples_dir, models_dir: [
            matrix_mod._Pair(tmp_path / "examples" / "small.axon", tmp_path / "models" / "small"),
            matrix_mod._Pair(tmp_path / "examples" / "large.axon", tmp_path / "models" / "large"),
        ],
    )

    exit_code = run_axon_test_matrix(
        examples_dir=tmp_path / "examples",
        models_dir=tmp_path / "models",
        table_format="plain",
        max_billions_params=2,
    )

    assert exit_code == 0
    assert downloaded == ["small"]


def test_apply_billions_params_filter_uses_local_bin_lower_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pair = matrix_mod._Pair(tmp_path / "examples" / "big.axon", tmp_path / "models" / "big")
    pair.model_dir.mkdir(parents=True)
    pt_bin = pair.model_dir / "pytorch_model.bin"
    with pt_bin.open("wb") as fh:
        fh.truncate(12_000_000_000)

    monkeypatch.setattr(
        matrix_mod,
        "estimate_remote_param_count_lower_bound",
        lambda repo_root, spec: (_ for _ in ()).throw(
            AssertionError("remote estimate should not run")
        ),
    )
    monkeypatch.setattr(
        matrix_mod,
        "MODEL_SPECS",
        {"big": matrix_mod.MODEL_SPECS.get("gpt2").__class__("big", "repo/big")},
    )

    kept, skipped = matrix_mod._apply_billions_params_filter(
        [pair],
        min_billions_params=None,
        max_billions_params=2,
    )

    assert kept == []
    assert len(skipped) == 1
    assert skipped[0].pair == pair
    assert skipped[0].is_exact is False


def test_run_pair_with_fallback_retries_cuda_oom_on_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pair = matrix_mod._Pair(tmp_path / "examples" / "bert.axon", tmp_path / "models" / "bert")
    calls: list[str] = []

    def _fake_run_pair(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args
        calls.append(str(kwargs["device"]))
        if kwargs["device"] == "cuda:0":
            raise RuntimeError("CUDA out of memory while allocating tensor")
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

    result, retry_message = matrix_mod._run_pair_with_fallback(
        pair,
        model_task="masked_lm",
        device="cuda:0",
        dtype="float32",
        max_len=16,
        text=["The future of AI is"],
        verbose=False,
        no_capture_output=False,
        compile_hf=False,
        compile_axon=False,
        compile_backend=None,
        compile_mode=None,
        compile_fullgraph=False,
        compile_dynamic=False,
    )

    assert calls == ["cuda:0", "cpu"]
    assert retry_message is not None
    assert "retrying on cpu" in retry_message.lower()
    assert result["top1_eq"] is True


def test_run_runnable_pairs_serial_retries_cuda_oom_on_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pair = matrix_mod._Pair(tmp_path / "examples" / "bert.axon", tmp_path / "models" / "bert")
    calls: list[str] = []

    monkeypatch.setattr(matrix_mod, "_maybe_ensure_pair_model_ready", lambda pair: None)
    monkeypatch.setattr(matrix_mod, "_resolve_model_task_for_pair", lambda pair: "masked_lm")

    def _fake_run_pair(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args
        calls.append(str(kwargs["device"]))
        if kwargs["device"] == "cuda:0":
            raise RuntimeError("CUDA out of memory while allocating tensor")
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

    rows, passed, failed = matrix_mod._run_runnable_pairs_serial(
        [pair],
        model_task_override=None,
        device="cuda:0",
        dtype="float32",
        max_len=16,
        prompts=["The future of AI is"],
        verbose=False,
        no_capture_output=False,
        compile_hf=False,
        compile_axon=False,
        compile_backend=None,
        compile_mode=None,
        compile_fullgraph=False,
        compile_dynamic=False,
    )

    assert calls == ["cuda:0", "cpu"]
    assert passed == 1
    assert failed == 0
    assert len(rows) == 1
    out = capsys.readouterr().out
    assert "retrying on cpu" in out.lower()
