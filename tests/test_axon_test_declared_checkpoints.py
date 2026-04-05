from __future__ import annotations

from pathlib import Path

import brainsurgery.synapse.axon_test as axon_test_module


def test_format_checkpoint_summary_table_markdown() -> None:
    table = axon_test_module._format_checkpoint_summary_table(
        [
            {
                "axon": "/tmp/gemma3.axon",
                "checkpoint": "google/gemma-3-270m",
                "model_dir": "/tmp/models/google/gemma-3-270m",
                "masked_top1_eq": "True",
                "masked_max_abs_diff": "2.3e-05",
                "masked_max_rel_diff": "0.1",
            }
        ],
        table_format="markdown",
    )
    assert (
        "| axon | checkpoint | model dir | masked top-1 eq | masked max abs diff | masked max rel diff |"
        in table
    )
    assert (
        "| /tmp/gemma3.axon | google/gemma-3-270m | /tmp/models/google/gemma-3-270m | True | 2.3e-05 | 0.1 |"
        in table
    )


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

    monkeypatch.setattr(axon_test_module, "_repo_root", _fake_repo_root)
    monkeypatch.setattr(
        axon_test_module,
        "_ensure_checkpoint_model_dir",
        _fake_ensure_checkpoint_model_dir,
    )
    monkeypatch.setattr(axon_test_module, "_run_axon_test_single", _fake_run_single)

    result = axon_test_module.run_axon_benchmark(
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
