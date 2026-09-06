#!/usr/bin/env python
"""Audit Codex records against transcripts and report unfinished bookkeeping.

This command is read-only. It recomputes transcript-derived harness fields and
flags failed executions without an error class and reviews whose ``detected``
value has not been confirmed by the experimenter.

    .venv/bin/python usability_tests/audit_codex.py --agent astra
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_codex import summarise_codex  # noqa: E402

HERE = Path(__file__).resolve().parent
SUMMARY_FIELDS = (
    "turns",
    "tool_calls",
    "tokens_in",
    "tokens_out",
    "cache_read_tokens",
    "tokens_in_total",
    "reasoning_tokens",
    "executions",
    "first_execution_success",
    "executions_until_first_success",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def audit_run(run_dir: Path) -> list[str]:
    issues: list[str] = []
    try:
        harness = read_json(run_dir / "harness.json")
        events = [json.loads(line) for line in (run_dir / "transcript.jsonl").read_text().splitlines() if line]
    except (OSError, json.JSONDecodeError) as exc:
        return [f"records/transcript unreadable: {exc}"]

    fresh = summarise_codex(events, harness.get("final_text", ""))
    for field in SUMMARY_FIELDS:
        if harness.get(field) != fresh.get(field):
            issues.append(f"transcript mismatch {field}: record={harness.get(field)!r}, transcript={fresh.get(field)!r}")

    recorded_failures = harness.get("failed_executions") or []
    fresh_failures = fresh.get("failed_executions") or []
    if [item.get("n") for item in recorded_failures] != [item.get("n") for item in fresh_failures]:
        issues.append("transcript mismatch failed-execution numbers")
    for failure in recorded_failures:
        if not failure.get("error_class"):
            issues.append(f"bookkeeping: execution {failure.get('n')} needs error_class")

    try:
        review = read_json(run_dir / "review.json")
    except (OSError, json.JSONDecodeError) as exc:
        issues.append(f"review unreadable: {exc}")
    else:
        if review.get("detected") is None:
            issues.append("bookkeeping: review.detected needs experimenter confirmation")
    if not (run_dir / "grade.json").is_file():
        issues.append("grade.json missing")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--root", type=Path, default=HERE)
    args = parser.parse_args()
    agent_root = args.root / args.agent
    harnesses = sorted(agent_root.glob("*/*/*/harness.json"))
    if not harnesses:
        print(f"no Codex harnesses found under {agent_root}", file=sys.stderr)
        return 2

    issue_count = 0
    for harness_path in harnesses:
        run_dir = harness_path.parent
        issues = audit_run(run_dir)
        rel = run_dir.relative_to(args.root)
        if issues:
            issue_count += len(issues)
            print(f"PENDING {rel}")
            for issue in issues:
                print(f"  {issue}")
        else:
            print(f"OK      {rel}")
    print(f"audited {len(harnesses)} runs; {issue_count} issue(s)")
    return 1 if issue_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
