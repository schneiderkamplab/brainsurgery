from __future__ import annotations

import contextlib
import io
from pathlib import Path

import pytest

from brainsurgery.synapse import run_axon_test
from tests.model_downloads import MATRIX_AXON_MODEL_DIR_PAIRS
from tests.test_flags import LONG_TEST_ENV, run_long_tests_enabled


def _matrix_pairs(repo_root: Path) -> list[tuple[Path, Path]]:
    examples_dir = repo_root / "examples"
    models_dir = repo_root / "models"
    return [
        (examples_dir / f"{axon_stem}.axon", models_dir / model_dir_name)
        for axon_stem, model_dir_name in MATRIX_AXON_MODEL_DIR_PAIRS
    ]


_REPO_ROOT = Path(__file__).resolve().parents[1]
_PAIRS = _matrix_pairs(_REPO_ROOT)
_RUN_MATRIX = run_long_tests_enabled()


@pytest.mark.parametrize(
    ("axon_path", "model_dir"),
    _PAIRS,
    ids=[f"{axon.name}__{model.name}" for axon, model in _PAIRS],
)
@pytest.mark.skipif(
    not _RUN_MATRIX,
    reason=f"set {LONG_TEST_ENV}=1 to enable matrix perf-style tests",
)
def test_axon_matrix_quality(
    axon_path: Path,
    model_dir: Path,
    ensure_matrix_test_models: None,
    pytestconfig: pytest.Config,
) -> None:
    pytest.importorskip("transformers")
    _ = ensure_matrix_test_models

    with contextlib.redirect_stdout(io.StringIO()):
        result = run_axon_test(
            axon_file=axon_path,
            weights=model_dir,
            hf_model_dir=model_dir,
            device="cpu",
            dtype="float32",
            text="The future of AI is",
            max_len=16,
        )

    hf_time = float(result["hf_time"])
    axon_time = float(result["axon_time"])
    ratio = float(result["speed_ratio_axon_over_hf"])
    abs_delta = abs(axon_time - hf_time)

    reporter = pytestconfig.pluginmanager.get_plugin("terminalreporter")
    line = (
        f"[matrix-time] {axon_path.name} | hf={hf_time:.4f}s "
        f"axon={axon_time:.4f}s ratio={ratio:.3f}x delta={abs_delta:.4f}s"
    )
    if reporter is not None:
        reporter.write_line(line)
    else:
        print(line, flush=True)

    assert not (ratio >= 2.0 and abs_delta > 1.0), (
        f"performance gate failed for {axon_path.name}: "
        f"ratio={ratio:.3f}x, delta={abs_delta:.4f}s "
        f"(hf={hf_time:.4f}s, axon={axon_time:.4f}s)"
    )
    assert result["max_diff"] < 1.0e-3
    assert result["top1_eq"] is True
