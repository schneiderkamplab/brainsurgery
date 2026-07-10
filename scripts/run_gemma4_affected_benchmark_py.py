from __future__ import annotations

import sys
from pathlib import Path

from brainsurgery.synapse import run_axon_benchmark


def main() -> None:
    log_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "log-gemma4-inline-remove-builtins-p6")
    stream_csv = log_dir / "stream.csv"
    axon_files = [
        Path("brainsurgery/synapse/models/gemma4/generic-gemma-4-e.axon"),
        Path("brainsurgery/synapse/models/gemma4/gemma-4-E2B.axon"),
        Path("brainsurgery/synapse/models/gemma4/gemma-4-E4B.axon"),
    ]
    print(f"starting benchmark: log_dir={log_dir}")
    run_axon_benchmark(
        axon_files=axon_files,
        device="cuda",
        processes=6,
        dtype="float32",
        debug_errors=True,
        log_dir=log_dir,
        stream_csv=stream_csv,
    )
    print(f"CSV: {stream_csv}")


if __name__ == "__main__":
    main()
