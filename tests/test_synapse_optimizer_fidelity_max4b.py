from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pytest
import torch

from brainsurgery.synapse.axon_benchmark import run_axon_benchmark
from tests.test_flags import LONG_TEST_ENV, run_long_tests_enabled


def _generic_axon_paths(repo_root: Path) -> list[Path]:
    return sorted((repo_root / "brainsurgery" / "synapse" / "models").glob("**/generic-*.axon"))


def _result_top1_ok(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _result_abs_diff(value: Any) -> float:
    if value is None:
        return float("inf")
    return float(value)


@pytest.mark.skipif(
    not run_long_tests_enabled(),
    reason=f"set {LONG_TEST_ENV}=1 to enable max-4B optimizer fidelity tests",
)
def test_safe_optimized_codegen2_fidelity_for_generic_models_max4b(repo_root: Path) -> None:
    axon_paths = _generic_axon_paths(repo_root)
    assert axon_paths, "expected generic Axon model files"

    device = os.environ.get("BS_FIDELITY_DEVICE")
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    processes = int(os.environ.get("BS_FIDELITY_PROCESSES", "1"))
    run_id = time.strftime("pytest-max4b-opt-fidelity-%Y%m%d-%H%M%S")
    log_dir = repo_root / "log" / run_id

    result = run_axon_benchmark(
        axon_files=axon_paths,
        device=device,
        processes=processes,
        dtype="float32",
        text="Hello world",
        max_len=16,
        axon_backend="codegen2-torch",
        axon_typechecker="typecheck2",
        optimize_ast=True,
        optimize_graph=True,
        max_billion_parameters=4,
        table_format="plain",
        log_dir=log_dir,
        stream_csv=log_dir / "stream.csv",
        debug_errors=True,
    )

    rows = result["results"]
    assert rows, "expected at least one max-4B checkpoint row"
    bad_rows: list[str] = []
    for row in rows:
        top1 = row.get("masked_top1_eq")
        diff = row.get("masked_max_abs_diff", row.get("max_diff"))
        if not _result_top1_ok(top1) or _result_abs_diff(diff) >= 1.0e-2:
            bad_rows.append(
                f"{row.get('axon_file')} :: {row.get('checkpoint_id')} "
                f"top1={top1!r} max_abs={diff!r} fallback={row.get('fallback')!r}"
            )

    assert not bad_rows, "max-4B optimized fidelity failures:\n" + "\n".join(bad_rows)
