#!/usr/bin/env python
"""Recompute the execution-related fields of harness.json from transcript.jsonl.

    .venv/bin/python usability_tests/resummarise.py [--root DIR]

Use after changing how run_claude.py counts executions; other fields
(timestamps, wall clock, cap, ids) are kept. Runs without a transcript are left alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_claude import summarise  # noqa: E402

HERE = Path(__file__).resolve().parent
FIELDS = ("tool_calls", "executions", "failed_executions", "first_execution_success", "executions_until_first_success")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=HERE)
    args = parser.parse_args()
    n = 0
    for transcript in sorted(args.root.glob("*/*/*/*/transcript.jsonl")):
        harness_path = transcript.parent / "harness.json"
        if not harness_path.exists():
            continue
        events = [json.loads(line) for line in transcript.read_text().splitlines() if line.strip()]
        fresh = summarise(events)
        harness = json.loads(harness_path.read_text())
        old_classes = {f["n"]: f.get("error_class", "") for f in harness.get("failed_executions", [])}
        for f in fresh["failed_executions"]:
            f["error_class"] = old_classes.get(f["n"], "")
        before = harness.get("executions")
        harness.update({k: fresh[k] for k in FIELDS})
        harness_path.write_text(json.dumps(harness, indent=2) + "\n")
        print(f"{transcript.parent.relative_to(args.root)}: executions {before} -> {fresh['executions']}")
        n += 1
    print(f"resummarised {n} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
