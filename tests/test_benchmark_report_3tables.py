from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


def test_benchmark_report_uses_vllm_top_logprobs_for_quality(tmp_path: Path) -> None:
    stream = tmp_path / "stream.csv"
    with stream.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "backend",
                "axon",
                "checkpoint",
                "fallback",
                "masked_top1_eq",
                "masked_max_abs_diff",
                "vllm_top_logprobs_top1_eq",
                "vllm_top_logprobs_hf_topk_covered",
                "vllm_top_logprobs_max_abs_diff",
                "hf_time",
                "axon_time",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "backend": "codegen2-vllm",
                "axon": "brainsurgery/synapse/models/foo/generic-foo.axon",
                "checkpoint": "test/Foo",
                "fallback": "none",
                "masked_top1_eq": "N/A",
                "masked_max_abs_diff": "N/A",
                "vllm_top_logprobs_top1_eq": "True",
                "vllm_top_logprobs_hf_topk_covered": "True",
                "vllm_top_logprobs_max_abs_diff": "9.5e-07",
                "hf_time": "1.0",
                "axon_time": "0.9",
            }
        )

    result = subprocess.run(
        [sys.executable, "scripts/benchmark_report_3tables.py", str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (
        "| Healthy | Skipped | Error rows | masked_top1_eq != True | "
        "masked_max_abs_diff >= 1e-3 |"
    ) in result.stdout
    assert "| 1 | 0 | 0 | 0 | 0 |" in result.stdout
    assert "| (none) |  |  |  |  |  |  |  |  |" in result.stdout
