from __future__ import annotations

from pathlib import Path

from brainsurgery.synapse import run_axon_benchmark


def main() -> None:
    log_dir = Path("log-gemma4-moe-derived-experts-g45")
    run_axon_benchmark(
        axon_files=[
            Path("brainsurgery/synapse/models/gemma4/generic-gemma-4-moe.axon"),
            Path("brainsurgery/synapse/models/gemma4/gemma-4-26B-A4B.axon"),
        ],
        device="cuda",
        processes=2,
        dtype="float32",
        debug_errors=True,
        log_dir=log_dir,
        stream_csv=log_dir / "stream.csv",
    )
    print(f"CSV: {log_dir / 'stream.csv'}")


if __name__ == "__main__":
    main()
