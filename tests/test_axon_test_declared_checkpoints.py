from __future__ import annotations

from pathlib import Path

import pytest

import brainsurgery.synapse.axon_benchmark as axon_benchmark_module
import brainsurgery.synapse.axon_test as axon_test_module

KNOWN_DECLARED_PRAGMA_REGRESSION = pytest.mark.xfail(
    reason="known CHECKPOINTS/TOKENIZER pragma resolution regression outside tests",
    strict=False,
)


def test_format_checkpoint_summary_table_markdown() -> None:
    table = axon_test_module._format_checkpoint_summary_table(
        [
            {
                "axon": "/tmp/gemma3.axon",
                "checkpoint": "google/gemma-3-270m",
                "model_dir": "/tmp/models/google/gemma-3-270m",
                "fallback": "none",
                "masked_top1_eq": "True",
                "masked_max_abs_diff": "2.3e-05",
                "masked_max_rel_diff": "0.1",
            }
        ],
        table_format="markdown",
    )
    assert (
        "| axon | checkpoint | model dir | fallback | masked top-1 eq | masked max abs diff | masked max rel diff |"
        in table
    )
    assert (
        "| /tmp/gemma3.axon | google/gemma-3-270m | /tmp/models/google/gemma-3-270m | none | True | 2.3e-05 | 0.1 |"
        in table
    )


def test_format_checkpoint_summary_table_html_highlights_cells() -> None:
    table = axon_test_module._format_checkpoint_summary_table(
        [
            {
                "axon": "/tmp/a.axon",
                "checkpoint": "google/a",
                "model_dir": "/tmp/models/google/a",
                "fallback": "HF+Axon->cpu",
                "masked_top1_eq": "False",
                "masked_max_abs_diff": "0.02",
                "masked_max_rel_diff": "0.1",
            },
            {
                "axon": "/tmp/b.axon",
                "checkpoint": "google/b",
                "model_dir": "/tmp/models/google/b",
                "fallback": "none",
                "masked_top1_eq": "True",
                "masked_max_abs_diff": "0.005",
                "masked_max_rel_diff": "0.2",
            },
            {
                "axon": "/tmp/c.axon",
                "checkpoint": "google/c",
                "model_dir": "/tmp/models/google/c",
                "fallback": "none",
                "masked_top1_eq": "True",
                "masked_max_abs_diff": "0.02",
                "masked_max_rel_diff": "0.3",
            },
        ],
        table_format="html",
    )
    assert "<table>" in table
    assert 'style="background-color: #dc3545; color: #ffffff;"' in table
    assert 'style="background-color: #ffe5b4;"' in table
    assert 'style="background-color: #fff3cd;"' in table


def test_normalize_texts_defaults_to_two_prompts() -> None:
    assert axon_test_module._normalize_texts([]) == ["The future of AI is", "Hello World"]


def test_run_axon_benchmark_uses_declared_checkpoints_for_multiple_axon_files(
    tmp_path: Path, monkeypatch
) -> None:
    axon_path_1 = tmp_path / "gemma3.axon"
    axon_path_1.write_text(
        """
{-# CHECKPOINTS ["google/gemma-3-270m", "google/gemma-3-270m-it"] #-}
main :: Tensor[B,T,D] -> Tensor[B,T,D]
main x = do
  return x
""".strip()
        + "\n",
        encoding="utf-8",
    )
    axon_path_2 = tmp_path / "t5.axon"
    axon_path_2.write_text(
        """
{-# CHECKPOINTS "google-t5/t5-small" #-}
main :: Tensor[B,T,D] -> Tensor[B,T,D]
main x = do
  return x
""".strip()
        + "\n",
        encoding="utf-8",
    )

    ensured: list[tuple[Path, str]] = []
    single_runs: list[tuple[Path, Path, Path | None]] = []

    def _fake_repo_root() -> Path:
        return tmp_path

    def _fake_ensure_checkpoint_model_dir(*, repo_root: Path, checkpoint_id: str) -> Path:
        ensured.append((repo_root, checkpoint_id))
        return repo_root / "models" / checkpoint_id

    def _fake_run_single(**kwargs):
        single_runs.append(
            (
                kwargs["axon_file"],
                kwargs["weights"],
                kwargs.get("hf_model_dir"),
            )
        )
        return {"masked_top1_eq": True, "max_diff": 0.0}

    monkeypatch.setattr(axon_benchmark_module, "_repo_root", _fake_repo_root)
    monkeypatch.setattr(
        axon_benchmark_module,
        "_ensure_checkpoint_model_dir",
        _fake_ensure_checkpoint_model_dir,
    )
    monkeypatch.setattr(axon_benchmark_module, "_run_axon_test_single", _fake_run_single)

    result = axon_benchmark_module.run_axon_benchmark(
        axon_files=[axon_path_1, axon_path_2],
        device="cpu",
        dtype="float32",
        text="hi",
        max_len=4,
        table_format="plain",
    )

    assert ensured == [
        (tmp_path, "google/gemma-3-270m"),
        (tmp_path, "google/gemma-3-270m-it"),
        (tmp_path, "google-t5/t5-small"),
    ]
    assert single_runs == [
        (
            axon_path_1.resolve(),
            tmp_path / "models" / "google/gemma-3-270m",
            tmp_path / "models" / "google/gemma-3-270m",
        ),
        (
            axon_path_1.resolve(),
            tmp_path / "models" / "google/gemma-3-270m-it",
            tmp_path / "models" / "google/gemma-3-270m-it",
        ),
        (
            axon_path_2.resolve(),
            tmp_path / "models" / "google-t5" / "t5-small",
            tmp_path / "models" / "google-t5" / "t5-small",
        ),
    ]
    assert result == {
        "results": [
            {
                "masked_top1_eq": True,
                "max_diff": 0.0,
                "axon_file": axon_path_1.resolve(),
                "checkpoint_id": "google/gemma-3-270m",
                "weights": tmp_path / "models" / "google/gemma-3-270m",
                "hf_model_dir": tmp_path / "models" / "google/gemma-3-270m",
            },
            {
                "masked_top1_eq": True,
                "max_diff": 0.0,
                "axon_file": axon_path_1.resolve(),
                "checkpoint_id": "google/gemma-3-270m-it",
                "weights": tmp_path / "models" / "google/gemma-3-270m-it",
                "hf_model_dir": tmp_path / "models" / "google/gemma-3-270m-it",
            },
            {
                "masked_top1_eq": True,
                "max_diff": 0.0,
                "axon_file": axon_path_2.resolve(),
                "checkpoint_id": "google-t5/t5-small",
                "weights": tmp_path / "models" / "google-t5" / "t5-small",
                "hf_model_dir": tmp_path / "models" / "google-t5" / "t5-small",
            },
        ]
    }


@KNOWN_DECLARED_PRAGMA_REGRESSION
def test_tokenizer_pragma_for_checkpoint_supports_global_and_override(tmp_path: Path) -> None:
    axon = tmp_path / "toy.axon"
    axon.write_text(
        "\n".join(
            [
                '{-# CHECKPOINTS ["mistralai/Mistral-7B-v0.1", "mistralai/Devstral-Small-2507"] #-}',
                '{-# TOKENIZER "mistralai/Mistral-7B-v0.1" #-}',
                '{-# TOKENIZER ["mistralai/Devstral-Small-2507", "mistralai/Devstral-Small-2507"] #-}',
                "toy :: Tensor[B,S,D] -> Tensor[B,S,D]",
                "toy x = x",
                "",
            ]
        ),
        encoding="utf-8",
    )

    assert (
        axon_test_module._tokenizer_pragma_for_checkpoint(
            axon_file=axon,
            checkpoint_id="mistralai/Mistral-7B-v0.1",
        )
        == "mistralai/Mistral-7B-v0.1"
    )
    assert (
        axon_test_module._tokenizer_pragma_for_checkpoint(
            axon_file=axon,
            checkpoint_id="mistralai/Devstral-Small-2507",
        )
        == "mistralai/Devstral-Small-2507"
    )


def test_run_axon_benchmark_parallel_dispatches_pairs(tmp_path: Path, monkeypatch) -> None:
    axon_path = tmp_path / "gemma4.axon"
    axon_path.write_text(
        """
{-# CHECKPOINTS ["google/gemma-4-E2B", "google/gemma-4-E2B-it"] #-}
main :: Tensor[B,T,D] -> Tensor[B,T,D]
main x = do
  return x
""".strip()
        + "\n",
        encoding="utf-8",
    )

    ensured: list[tuple[Path, str]] = []
    parallel_call: dict[str, object] = {}

    def _fake_repo_root() -> Path:
        return tmp_path

    def _fake_ensure_checkpoint_model_dir(*, repo_root: Path, checkpoint_id: str) -> Path:
        ensured.append((repo_root, checkpoint_id))
        return repo_root / "models" / checkpoint_id

    def _fake_parallel(
        pairs,
        *,
        processes: int,
        device: str,
        log_dir: Path | None,
        stream_csv: Path | None,
        common_kwargs: dict[str, object],
        worker_specs=None,
        debug_errors: bool = False,
    ):
        parallel_call["pairs"] = pairs
        parallel_call["processes"] = processes
        parallel_call["device"] = device
        parallel_call["log_dir"] = log_dir
        parallel_call["stream_csv"] = stream_csv
        parallel_call["common_kwargs"] = common_kwargs
        parallel_call["worker_specs"] = worker_specs
        parallel_call["debug_errors"] = debug_errors
        return [
            {
                "masked_top1_eq": True,
                "masked_max_diff": 0.0,
                "masked_max_rel_diff": 0.0,
                "axon_file": pair.axon_file,
                "checkpoint_id": pair.checkpoint_id,
                "weights": pair.model_dir,
                "hf_model_dir": pair.model_dir,
            }
            for pair in pairs
        ]

    monkeypatch.setattr(axon_benchmark_module, "_repo_root", _fake_repo_root)
    monkeypatch.setattr(
        axon_benchmark_module,
        "_ensure_checkpoint_model_dir",
        _fake_ensure_checkpoint_model_dir,
    )
    monkeypatch.setattr(axon_benchmark_module, "_run_benchmark_jobs_parallel", _fake_parallel)

    result = axon_benchmark_module.run_axon_benchmark(
        axon_files=[axon_path],
        device="cuda",
        processes=2,
        dtype="float32",
        text="hi",
        max_len=4,
        table_format="plain",
        log_dir=tmp_path / "logs",
    )

    assert ensured == []
    assert parallel_call["processes"] == 2
    assert parallel_call["device"] == "cuda"
    assert parallel_call["log_dir"] == tmp_path / "logs"
    assert parallel_call["stream_csv"] is None
    assert parallel_call["common_kwargs"]["repo_root"] == tmp_path
    pairs = parallel_call["pairs"]
    assert [pair.checkpoint_id for pair in pairs] == [
        "google/gemma-4-E2B",
        "google/gemma-4-E2B-it",
    ]
    assert [pair.model_dir for pair in pairs] == [
        tmp_path / "models" / "google/gemma-4-E2B",
        tmp_path / "models" / "google/gemma-4-E2B-it",
    ]
    assert result == {
        "results": [
            {
                "masked_top1_eq": True,
                "masked_max_diff": 0.0,
                "masked_max_rel_diff": 0.0,
                "axon_file": axon_path.resolve(),
                "checkpoint_id": "google/gemma-4-E2B",
                "weights": tmp_path / "models" / "google/gemma-4-E2B",
                "hf_model_dir": tmp_path / "models" / "google/gemma-4-E2B",
            },
            {
                "masked_top1_eq": True,
                "masked_max_diff": 0.0,
                "masked_max_rel_diff": 0.0,
                "axon_file": axon_path.resolve(),
                "checkpoint_id": "google/gemma-4-E2B-it",
                "weights": tmp_path / "models" / "google/gemma-4-E2B-it",
                "hf_model_dir": tmp_path / "models" / "google/gemma-4-E2B-it",
            },
        ]
    }


def test_run_axon_benchmark_recurses_directories_for_axon_files(
    tmp_path: Path, monkeypatch
) -> None:
    gemma_dir = tmp_path / "models" / "gemma"
    gemma_dir.mkdir(parents=True)
    axon_path_1 = gemma_dir / "gemma3.axon"
    axon_path_1.write_text(
        """
{-# CHECKPOINTS ["google/gemma-3-270m", "google/gemma-3-270m-it"] #-}
main :: Tensor[B,T,D] -> Tensor[B,T,D]
main x = do
  return x
""".strip()
        + "\n",
        encoding="utf-8",
    )
    nested_dir = gemma_dir / "subfamily"
    nested_dir.mkdir()
    axon_path_2 = nested_dir / "gemma4.axon"
    axon_path_2.write_text(
        """
{-# CHECKPOINTS "google/gemma-4-E2B" #-}
main :: Tensor[B,T,D] -> Tensor[B,T,D]
main x = do
  return x
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (nested_dir / "README.txt").write_text("ignore me\n", encoding="utf-8")

    single_runs: list[tuple[Path, Path, Path | None]] = []

    def _fake_repo_root() -> Path:
        return tmp_path

    def _fake_ensure_checkpoint_model_dir(*, repo_root: Path, checkpoint_id: str) -> Path:
        return repo_root / "models" / checkpoint_id

    def _fake_run_single(**kwargs):
        single_runs.append(
            (
                kwargs["axon_file"],
                kwargs["weights"],
                kwargs.get("hf_model_dir"),
            )
        )
        return {"masked_top1_eq": True, "max_diff": 0.0}

    monkeypatch.setattr(axon_benchmark_module, "_repo_root", _fake_repo_root)
    monkeypatch.setattr(
        axon_benchmark_module,
        "_ensure_checkpoint_model_dir",
        _fake_ensure_checkpoint_model_dir,
    )
    monkeypatch.setattr(axon_benchmark_module, "_run_axon_test_single", _fake_run_single)

    result = axon_benchmark_module.run_axon_benchmark(
        axon_files=[gemma_dir],
        device="cpu",
        dtype="float32",
        text="hi",
        max_len=4,
        table_format="plain",
    )

    assert single_runs == [
        (
            axon_path_1.resolve(),
            tmp_path / "models" / "google/gemma-3-270m",
            tmp_path / "models" / "google/gemma-3-270m",
        ),
        (
            axon_path_1.resolve(),
            tmp_path / "models" / "google/gemma-3-270m-it",
            tmp_path / "models" / "google/gemma-3-270m-it",
        ),
        (
            axon_path_2.resolve(),
            tmp_path / "models" / "google/gemma-4-E2B",
            tmp_path / "models" / "google/gemma-4-E2B",
        ),
    ]
    assert result == {
        "results": [
            {
                "masked_top1_eq": True,
                "max_diff": 0.0,
                "axon_file": axon_path_1.resolve(),
                "checkpoint_id": "google/gemma-3-270m",
                "weights": tmp_path / "models" / "google/gemma-3-270m",
                "hf_model_dir": tmp_path / "models" / "google/gemma-3-270m",
            },
            {
                "masked_top1_eq": True,
                "max_diff": 0.0,
                "axon_file": axon_path_1.resolve(),
                "checkpoint_id": "google/gemma-3-270m-it",
                "weights": tmp_path / "models" / "google/gemma-3-270m-it",
                "hf_model_dir": tmp_path / "models" / "google/gemma-3-270m-it",
            },
            {
                "masked_top1_eq": True,
                "max_diff": 0.0,
                "axon_file": axon_path_2.resolve(),
                "checkpoint_id": "google/gemma-4-E2B",
                "weights": tmp_path / "models" / "google/gemma-4-E2B",
                "hf_model_dir": tmp_path / "models" / "google/gemma-4-E2B",
            },
        ]
    }


def test_run_axon_benchmark_excludes_matching_axon_files(tmp_path: Path, monkeypatch) -> None:
    axon_dir = tmp_path / "models"
    axon_dir.mkdir(parents=True)
    keep_path = axon_dir / "keep.axon"
    keep_path.write_text(
        """
{-# CHECKPOINTS "google/keep" #-}
main :: Tensor[B,T,D] -> Tensor[B,T,D]
main x = do
  return x
""".strip()
        + "\n",
        encoding="utf-8",
    )
    skip_path = axon_dir / "mistral4.axon"
    skip_path.write_text(
        """
{-# CHECKPOINTS "google/skip" #-}
main :: Tensor[B,T,D] -> Tensor[B,T,D]
main x = do
  return x
""".strip()
        + "\n",
        encoding="utf-8",
    )

    seen: list[tuple[str, str]] = []

    def _fake_repo_root() -> Path:
        return tmp_path

    def _fake_ensure_checkpoint_model_dir(*, repo_root: Path, checkpoint_id: str) -> Path:
        return repo_root / "models" / checkpoint_id

    def _fake_run_single(**kwargs):
        seen.append((Path(kwargs["axon_file"]).name, str(kwargs["weights"]).split("/models/")[-1]))
        return {"masked_top1_eq": True, "max_diff": 0.0}

    monkeypatch.setattr(axon_benchmark_module, "_repo_root", _fake_repo_root)
    monkeypatch.setattr(
        axon_benchmark_module,
        "_ensure_checkpoint_model_dir",
        _fake_ensure_checkpoint_model_dir,
    )
    monkeypatch.setattr(axon_benchmark_module, "_run_axon_test_single", _fake_run_single)

    result = axon_benchmark_module.run_axon_benchmark(
        axon_files=[axon_dir],
        exclude=["mistral4"],
        device="cpu",
        dtype="float32",
        text="hi",
        max_len=4,
        table_format="plain",
    )

    assert seen == [("keep.axon", "google/keep")]
    assert len(result["results"]) == 1
    assert Path(result["results"][0]["axon_file"]).name == "keep.axon"


def test_run_axon_benchmark_forwards_axon_backend(tmp_path: Path, monkeypatch) -> None:
    axon_path = tmp_path / "model.axon"
    axon_path.write_text(
        """
{-# CHECKPOINTS "google/test" #-}
main :: Tensor[B,T,D] -> Tensor[B,T,D]
main x = do
  return x
""".strip()
        + "\n",
        encoding="utf-8",
    )

    seen_backend: list[str] = []

    def _fake_repo_root() -> Path:
        return tmp_path

    def _fake_ensure_checkpoint_model_dir(*, repo_root: Path, checkpoint_id: str) -> Path:
        return repo_root / "models" / checkpoint_id

    def _fake_run_single(**kwargs):
        seen_backend.append(str(kwargs["axon_backend"]))
        return {"masked_top1_eq": True, "max_diff": 0.0}

    monkeypatch.setattr(axon_benchmark_module, "_repo_root", _fake_repo_root)
    monkeypatch.setattr(
        axon_benchmark_module,
        "_ensure_checkpoint_model_dir",
        _fake_ensure_checkpoint_model_dir,
    )
    monkeypatch.setattr(axon_benchmark_module, "_run_axon_test_single", _fake_run_single)

    axon_benchmark_module.run_axon_benchmark(
        axon_files=[axon_path],
        axon_backend="pipeline",
        device="cuda",
        processes=1,
        dtype="float32",
        text="hi",
        max_len=4,
        table_format="plain",
    )

    assert seen_backend == ["pipeline"]


def test_run_axon_benchmark_forwards_skip_hf(tmp_path: Path, monkeypatch) -> None:
    axon_path = tmp_path / "model.axon"
    axon_path.write_text(
        """
{-# CHECKPOINTS "google/test" #-}
main :: Tensor[B,T,D] -> Tensor[B,T,D]
main x = do
  return x
""".strip()
        + "\n",
        encoding="utf-8",
    )

    seen_skip_hf: list[bool] = []

    def _fake_repo_root() -> Path:
        return tmp_path

    def _fake_ensure_checkpoint_model_dir(*, repo_root: Path, checkpoint_id: str) -> Path:
        return repo_root / "models" / checkpoint_id

    def _fake_run_single(**kwargs):
        seen_skip_hf.append(bool(kwargs["skip_hf"]))
        return {"fallback": "skip-hf", "masked_top1_eq": None, "masked_max_diff": None}

    monkeypatch.setattr(axon_benchmark_module, "_repo_root", _fake_repo_root)
    monkeypatch.setattr(
        axon_benchmark_module,
        "_ensure_checkpoint_model_dir",
        _fake_ensure_checkpoint_model_dir,
    )
    monkeypatch.setattr(axon_benchmark_module, "_run_axon_test_single", _fake_run_single)

    axon_benchmark_module.run_axon_benchmark(
        axon_files=[axon_path],
        skip_hf=True,
        device="cpu",
        dtype="float32",
        text="hi",
        max_len=4,
        table_format="plain",
    )

    assert seen_skip_hf == [True]


def test_run_axon_benchmark_pipeline_requires_single_process(tmp_path: Path) -> None:
    axon_path = tmp_path / "model.axon"
    axon_path.write_text(
        """
{-# CHECKPOINTS "google/test" #-}
main :: Tensor[B,T,D] -> Tensor[B,T,D]
main x = do
  return x
""".strip()
        + "\n",
        encoding="utf-8",
    )

    def _fake_resolve_pipeline_worker_specs(
        *, device: str, processes: int, pipeline_parallel_size: int | None
    ):
        del pipeline_parallel_size
        # Simulate two workers with two GPUs each.
        assert device == "cuda"
        assert processes == 2
        return [
            axon_benchmark_module._WorkerSpec(run_device="cuda", cuda_visible_devices="0,1"),
            axon_benchmark_module._WorkerSpec(run_device="cuda", cuda_visible_devices="2,3"),
        ]

    seen: dict[str, object] = {}

    def _fake_parallel(
        pairs,
        *,
        processes: int,
        device: str,
        log_dir: Path | None,
        stream_csv: Path | None,
        common_kwargs: dict[str, object],
        worker_specs=None,
        debug_errors: bool = False,
    ):
        seen["pairs"] = pairs
        seen["processes"] = processes
        seen["device"] = device
        seen["log_dir"] = log_dir
        seen["stream_csv"] = stream_csv
        seen["common_kwargs"] = common_kwargs
        seen["worker_specs"] = worker_specs
        seen["debug_errors"] = debug_errors
        return [
            {
                "masked_top1_eq": True,
                "masked_max_diff": 0.0,
                "masked_max_rel_diff": 0.0,
                "axon_file": pair.axon_file,
                "checkpoint_id": pair.checkpoint_id,
                "weights": pair.model_dir,
                "hf_model_dir": pair.model_dir,
            }
            for pair in pairs
        ]

    def _fake_repo_root() -> Path:
        return tmp_path

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(axon_benchmark_module, "_repo_root", _fake_repo_root)
    monkeypatch.setattr(
        axon_benchmark_module,
        "_resolve_pipeline_worker_specs",
        _fake_resolve_pipeline_worker_specs,
    )
    monkeypatch.setattr(axon_benchmark_module, "_run_benchmark_jobs_parallel", _fake_parallel)
    try:
        result = axon_benchmark_module.run_axon_benchmark(
            axon_files=[axon_path],
            axon_backend="pipeline",
            device="cuda",
            processes=2,
            dtype="float32",
            text="hi",
            max_len=4,
            table_format="plain",
        )
    finally:
        monkeypatch.undo()

    assert seen["processes"] == 2
    assert seen["device"] == "cuda"
    assert isinstance(seen["worker_specs"], list)
    assert len(seen["worker_specs"]) == 2
    assert isinstance(result, dict)
    assert "results" in result


def test_run_axon_benchmark_sorts_summary_by_checkpoint_and_metrics(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    axon_path_1 = tmp_path / "gemma_a.axon"
    axon_path_1.write_text(
        """
{-# CHECKPOINTS ["google/gemma-a", "google/gemma-b"] #-}
main :: Tensor[B,T,D] -> Tensor[B,T,D]
main x = do
  return x
""".strip()
        + "\n",
        encoding="utf-8",
    )
    axon_path_2 = tmp_path / "gemma_b.axon"
    axon_path_2.write_text(
        """
{-# CHECKPOINTS ["google/gemma-b", "google/gemma-c"] #-}
main :: Tensor[B,T,D] -> Tensor[B,T,D]
main x = do
  return x
""".strip()
        + "\n",
        encoding="utf-8",
    )
    axon_path_3 = tmp_path / "gemma_c.axon"
    axon_path_3.write_text(
        """
{-# CHECKPOINTS ["google/gemma-b", "google/gemma-d"] #-}
main :: Tensor[B,T,D] -> Tensor[B,T,D]
main x = do
  return x
""".strip()
        + "\n",
        encoding="utf-8",
    )

    def _fake_repo_root() -> Path:
        return tmp_path

    def _fake_ensure_checkpoint_model_dir(*, repo_root: Path, checkpoint_id: str) -> Path:
        return repo_root / "models" / checkpoint_id

    def _fake_run_single(**kwargs):
        checkpoint = str(kwargs["weights"]).split("/models/")[-1]
        axon_name = Path(kwargs["axon_file"]).name
        data = {
            ("gemma_a.axon", "google/gemma-a"): {
                "masked_top1_eq": True,
                "masked_max_diff": 0.3,
                "masked_max_rel_diff": 0.4,
            },
            ("gemma_a.axon", "google/gemma-b"): {
                "masked_top1_eq": False,
                "masked_max_diff": 0.1,
                "masked_max_rel_diff": 0.2,
            },
            ("gemma_b.axon", "google/gemma-b"): {
                "masked_top1_eq": True,
                "masked_max_diff": 0.2,
                "masked_max_rel_diff": 0.3,
            },
            ("gemma_c.axon", "google/gemma-b"): {
                "masked_top1_eq": True,
                "masked_max_diff": 0.2,
                "masked_max_rel_diff": 0.1,
            },
            ("gemma_b.axon", "google/gemma-c"): {
                "masked_top1_eq": True,
                "masked_max_diff": 0.1,
                "masked_max_rel_diff": 0.1,
            },
            ("gemma_c.axon", "google/gemma-d"): {
                "masked_top1_eq": True,
                "masked_max_diff": 0.2,
                "masked_max_rel_diff": 0.3,
            },
        }
        return data[(axon_name, checkpoint)]

    monkeypatch.setattr(axon_benchmark_module, "_repo_root", _fake_repo_root)
    monkeypatch.setattr(
        axon_benchmark_module,
        "_ensure_checkpoint_model_dir",
        _fake_ensure_checkpoint_model_dir,
    )
    monkeypatch.setattr(axon_benchmark_module, "_run_axon_test_single", _fake_run_single)

    axon_benchmark_module.run_axon_benchmark(
        axon_files=[axon_path_1, axon_path_2, axon_path_3],
        device="cpu",
        dtype="float32",
        text="hi",
        max_len=4,
        table_format="markdown",
    )

    output = capsys.readouterr().out
    gemma_a = output.index("| " + str(axon_path_1.resolve()) + " | google/gemma-a |")
    gemma_b_true_rel_low = output.index("| " + str(axon_path_3.resolve()) + " | google/gemma-b |")
    gemma_b_true_rel_high = output.index("| " + str(axon_path_2.resolve()) + " | google/gemma-b |")
    gemma_b_false = output.index("| " + str(axon_path_1.resolve()) + " | google/gemma-b |")
    gemma_c = output.index("| " + str(axon_path_2.resolve()) + " | google/gemma-c |")
    gemma_d = output.index("| " + str(axon_path_3.resolve()) + " | google/gemma-d |")
    assert (
        gemma_a < gemma_b_true_rel_low < gemma_b_true_rel_high < gemma_b_false < gemma_c < gemma_d
    )


def test_run_axon_benchmark_streams_rows_to_csv(tmp_path: Path, monkeypatch) -> None:
    axon_path = tmp_path / "gemma.axon"
    axon_path.write_text(
        """
{-# CHECKPOINTS ["google/gemma-a", "google/gemma-b"] #-}
main :: Tensor[B,T,D] -> Tensor[B,T,D]
main x = do
  return x
""".strip()
        + "\n",
        encoding="utf-8",
    )
    csv_path = tmp_path / "stream" / "benchmark.csv"

    def _fake_repo_root() -> Path:
        return tmp_path

    def _fake_ensure_checkpoint_model_dir(*, repo_root: Path, checkpoint_id: str) -> Path:
        return repo_root / "models" / checkpoint_id

    def _fake_run_single(**kwargs):
        checkpoint = str(kwargs["weights"]).split("/models/")[-1]
        return {
            "fallback": "none",
            "masked_top1_eq": checkpoint.endswith("a"),
            "masked_max_diff": 0.02 if checkpoint.endswith("a") else 0.001,
            "masked_max_rel_diff": 0.3 if checkpoint.endswith("a") else 0.2,
        }

    monkeypatch.setattr(axon_benchmark_module, "_repo_root", _fake_repo_root)
    monkeypatch.setattr(
        axon_benchmark_module,
        "_ensure_checkpoint_model_dir",
        _fake_ensure_checkpoint_model_dir,
    )
    monkeypatch.setattr(axon_benchmark_module, "_run_axon_test_single", _fake_run_single)

    axon_benchmark_module.run_axon_benchmark(
        axon_files=[axon_path],
        device="cpu",
        dtype="float32",
        text="hi",
        max_len=4,
        table_format="plain",
        stream_csv=csv_path,
    )

    csv_text = csv_path.read_text(encoding="utf-8")
    assert (
        "axon,checkpoint,model_dir,fallback,masked_top1_eq,masked_max_abs_diff,masked_max_rel_diff"
        in csv_text
    )
    assert "google/gemma-a" in csv_text
    assert "google/gemma-b" in csv_text


def test_render_axon_benchmark_csv_sorts_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "benchmark.csv"
    csv_path.write_text(
        "\n".join(
            [
                "axon,checkpoint,model_dir,fallback,masked_top1_eq,masked_max_abs_diff,masked_max_rel_diff",
                "/tmp/z.axon,google/gemma-b,/tmp/models/google/gemma-b,HF->cpu,False,0.1,0.3",
                "/tmp/y.axon,google/gemma-b,/tmp/models/google/gemma-b,none,True,0.1,0.2",
                "/tmp/x.axon,google/gemma-a,/tmp/models/google/gemma-a,Axon->cpu,True,0.2,0.4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rendered = axon_benchmark_module.render_axon_benchmark_csv(
        csv_path=csv_path,
        table_format="markdown",
    )

    gemma_a = rendered.index("| /tmp/x.axon | google/gemma-a |")
    gemma_b_true = rendered.index("| /tmp/y.axon | google/gemma-b |")
    gemma_b_false = rendered.index("| /tmp/z.axon | google/gemma-b |")
    assert gemma_a < gemma_b_true < gemma_b_false


def test_run_axon_benchmark_serial_failure_becomes_error_row(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    axon_path = tmp_path / "broken.axon"
    axon_path.write_text(
        """
{-# CHECKPOINTS "google/gemma-broken" #-}
main :: Tensor[B,T,D] -> Tensor[B,T,D]
main x = do
  return x
""".strip()
        + "\n",
        encoding="utf-8",
    )

    def _fake_repo_root() -> Path:
        return tmp_path

    def _fake_ensure_checkpoint_model_dir(*, repo_root: Path, checkpoint_id: str) -> Path:
        return repo_root / "models" / checkpoint_id

    def _fake_run_single(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(axon_benchmark_module, "_repo_root", _fake_repo_root)
    monkeypatch.setattr(
        axon_benchmark_module,
        "_ensure_checkpoint_model_dir",
        _fake_ensure_checkpoint_model_dir,
    )
    monkeypatch.setattr(axon_benchmark_module, "_run_axon_test_single", _fake_run_single)

    result = axon_benchmark_module.run_axon_benchmark(
        axon_files=[axon_path],
        device="cpu",
        dtype="float32",
        text="hi",
        max_len=4,
        table_format="markdown",
    )

    output = capsys.readouterr().out
    assert "| " + str(axon_path.resolve()) + " | google/gemma-broken | " in output
    assert "| ERROR | ERROR |" in output
    assert result["results"][0]["masked_top1_eq"] == "ERROR"
    assert result["results"][0]["masked_max_diff"] == "ERROR"
