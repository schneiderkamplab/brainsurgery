#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from brainsurgery.synapse.axon_benchmark import run_axon_benchmark


def main() -> None:
    axons = [
        Path("brainsurgery/synapse/models/apertus/Apertus-70B-2509.axon"),
        Path("brainsurgery/synapse/models/apertus/Apertus-8B-2509.axon"),
        Path("brainsurgery/synapse/models/apertus/generic-apertus.axon"),
        Path("brainsurgery/synapse/models/exaone4/EXAONE-4.0-1.2B.axon"),
        Path("brainsurgery/synapse/models/exaone4/EXAONE-4.0-32B.axon"),
        Path("brainsurgery/synapse/models/exaone4/generic-exaone4.axon"),
        Path("brainsurgery/synapse/models/llama3/Llama-3.1-70B-Instruct.axon"),
        Path("brainsurgery/synapse/models/llama3/Llama-3.1-70B.axon"),
        Path("brainsurgery/synapse/models/llama3/Llama-3.1-8B.axon"),
        Path("brainsurgery/synapse/models/llama3/Llama-3.2-1B.axon"),
        Path("brainsurgery/synapse/models/llama3/Llama-3.2-3B.axon"),
        Path("brainsurgery/synapse/models/llama3/Llama-3.3-70B-Instruct.axon"),
        Path("brainsurgery/synapse/models/llama3/Meta-Llama-3-70B-Instruct.axon"),
        Path("brainsurgery/synapse/models/llama3/Meta-Llama-3-70B.axon"),
        Path("brainsurgery/synapse/models/llama3/Meta-Llama-3-8B.axon"),
        Path("brainsurgery/synapse/models/llama3/generic-llama3-basic.axon"),
        Path("brainsurgery/synapse/models/llama3/generic-llama3.axon"),
    ]

    log_dir = Path("log-rope-freqscale-max10b-g0-5-20260412-r2")
    log_dir.mkdir(parents=True, exist_ok=True)
    result = run_axon_benchmark(
        axon_files=axons,
        device="cuda",
        processes=6,
        dtype="float32",
        axon_backend="codegen",
        oom_cpu_fallback=False,
        debug_errors=True,
        log_dir=log_dir,
        stream_csv=log_dir / "stream.csv",
        max_billion_parameters=10.0,
    )
    print("rows", len(result["results"]))
    print("csv", log_dir / "stream.csv")


if __name__ == "__main__":
    main()
