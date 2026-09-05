#!/usr/bin/env python
"""Aggregate run records into the study tables.

    .venv/bin/python usability_tests/analyze.py [--root DIR] [--json]

Walks <root>/<agent>/<target>/<effort>/<test>-<condition>-<repeat>/ and reads
run.json, harness.json, grade.json and review.json. Prints, per
(agent, target, effort, condition), per (agent, effort, condition) pooled over
targets, and pooled over everything per (effort, condition) and per condition:

    runs, success rate (final grade PASS), first-execution success rate,
    median retries (executions - 1), failed executions per run,
    median tokens in (uncached + cache reads + cache writes) and out, cost, median time to solution (wall clock of
    the solve phase for passing runs), bug-detection rate and false-alarm rate.

Missing files are counted, not fatal, so partial studies can be inspected.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def median(values: list[float]) -> str:
    values = [v for v in values if v is not None]
    return f"{statistics.median(values):.4g}" if values else "-"


def rate(num: int, den: int) -> str:
    return f"{num}/{den} ({100 * num / den:.0f}%)" if den else "-"


def collect(root: Path) -> list[dict]:
    runs = []
    for run_json in sorted(root.glob("*/*/*/*/run.json")):
        sandbox = run_json.parent
        run = read(run_json) or {}
        harness = read(sandbox / "harness.json") or {}
        grade = read(sandbox / "grade.json") or {}
        review = read(sandbox / "review.json") or {}
        detected = review.get("detected")
        if detected is None:
            detected = review.get("auto_says_defective")
        runs.append({
            "agent": run.get("agent", sandbox.parts[-4]),
            "target": run.get("target", sandbox.parts[-3]),
            "effort": run.get("effort", sandbox.parts[-2]),
            "test": run.get("test"),
            "condition": run.get("condition"),
            "repeat": run.get("repeat"),
            "passed": grade.get("passed"),
            "graded": "passed" in grade,
            "first_ok": harness.get("first_execution_success"),
            "executions": harness.get("executions"),
            "failed_execs": len(harness.get("failed_executions", []) or []),
            "tokens_in": harness.get("tokens_in_total", harness.get("tokens_in")),
            "tokens_out": harness.get("tokens_out"),
            "cost_usd": harness.get("cost_usd"),
            "wall_s": harness.get("wall_clock_s"),
            "cap_hit": harness.get("cap_hit"),
            "review_kind": review.get("artifact_kind"),
            "review_detected": detected,
        })
    return runs


def summarize(rows: list[dict]) -> dict:
    graded = [r for r in rows if r["graded"]]
    passed = [r for r in graded if r["passed"]]
    with_h = [r for r in rows if r["executions"] is not None]
    reviewed = [r for r in rows if r["review_kind"] in ("defective", "correct") and r["review_detected"] is not None]
    defective = [r for r in reviewed if r["review_kind"] == "defective"]
    correct = [r for r in reviewed if r["review_kind"] == "correct"]
    return {
        "runs": len(rows),
        "success": rate(len(passed), len(graded)),
        "first_exec_ok": rate(sum(1 for r in with_h if r["first_ok"]), len(with_h)),
        "median_retries": median([r["executions"] - 1 for r in with_h]),
        "failed_execs_per_run": median([r["failed_execs"] for r in with_h]),
        "cap_hit": sum(1 for r in rows if r["cap_hit"] not in (None, "none", False)),
        "median_tokens_in": median([r["tokens_in"] for r in rows]),
        "median_tokens_out": median([r["tokens_out"] for r in rows]),
        "median_cost_usd": median([r["cost_usd"] for r in rows]),
        "median_time_to_solution_s": median([r["wall_s"] for r in passed]),
        "bug_detected": rate(sum(1 for r in defective if r["review_detected"]), len(defective)),
        "false_alarms": rate(sum(1 for r in correct if r["review_detected"]), len(correct)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=HERE)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    runs = collect(args.root)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in runs:
        groups[(r["agent"], r["target"], r["effort"], r["condition"])].append(r)
        groups[(r["agent"], "all", r["effort"], r["condition"])].append(r)
        groups[("all", "all", r["effort"], r["condition"])].append(r)
        groups[("all", "all", "all", r["condition"])].append(r)
    tables = {"|".join(k): summarize(v) for k, v in sorted(groups.items())}
    if args.json:
        print(json.dumps({"runs": runs, "tables": tables}, indent=2))
        return 0
    if not runs:
        print("no runs found under", args.root)
        return 0
    cols = list(next(iter(tables.values())).keys())
    print("| agent | target | effort | cond | " + " | ".join(cols) + " |")
    print("|" + "---|" * (4 + len(cols)))
    for key, row in tables.items():
        agent, target, effort, cond = key.split("|")
        print(f"| {agent} | {target} | {effort} | {cond} | " + " | ".join(str(row[c]) for c in cols) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
