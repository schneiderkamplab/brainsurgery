#!/usr/bin/env python3
"""Run run_axon_benchmark on Gemma4-Dense-Test (Axon vs HF transformers)."""
from __future__ import annotations

from pathlib import Path

from brainsurgery.synapse import run_axon_benchmark

LOG_DIR = Path("log/gemma4-31b-test-axon-vs-hf")


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    run_axon_benchmark(
        axon_files=[
            Path("brainsurgery/synapse/models/gemma4/Gemma4-Dense-Test.axon"),
        ],
        device="cuda",
        processes=1,
        dtype="float32",
        benchmark_mode="auto",
        forward_warmup=3,
        forward_repeat=10,
        generate_warmup=1,
        generate_repeat=3,
        max_len=64,
        debug_errors=True,
        log_dir=LOG_DIR,
        stream_csv=LOG_DIR / "stream.csv",
    )
    print(f"CSV: {LOG_DIR / 'stream.csv'}")


if __name__ == "__main__":
    main()
