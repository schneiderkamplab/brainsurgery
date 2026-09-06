#!/usr/bin/env python3
"""Evaluation-only fault injector for BrainSurgery sharded publication."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


class InjectedSaveFailure(RuntimeError):
    """Deterministic evaluation exception raised after one complete shard."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("exception_after_first", "pause_after_first"),
        required=True,
    )
    parser.add_argument("--marker", type=Path)
    parser.add_argument("plan", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "pause_after_first" and args.marker is None:
        raise SystemExit("--marker is required for pause_after_first")

    # Running a source file sets sys.path[0] to this directory, whereas the
    # installed console script resolves the editable repository package. Add
    # that same repository root explicitly for this evaluation subprocess.
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))

    # The patch exists only for the lifetime of this evaluation subprocess.
    import brainsurgery
    from brainsurgery.engine import checkpoint_io

    original = checkpoint_io._save_safetensors_shard
    completed = 0

    def injected(path, shard):
        nonlocal completed
        if completed == 0:
            original(path, shard)
            completed = 1
            if args.mode == "pause_after_first":
                assert args.marker is not None
                args.marker.write_text(
                    f"pid={os.getpid()}\nfirst_shard={path.name}\n", encoding="utf-8"
                )
                while True:
                    time.sleep(1)
            return
        raise InjectedSaveFailure("InjectedSaveFailure: stopped after one complete shard")

    checkpoint_io._save_safetensors_shard = injected
    brainsurgery.main(
        [
            str(args.plan),
            "--provider",
            "inmemory",
            "--num-workers",
            "1",
            "--no-summarize",
        ]
    )


if __name__ == "__main__":
    main()
