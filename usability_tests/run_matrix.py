#!/usr/bin/env python
"""Run many cells of the study with Claude Code, resumably and in parallel.

    .venv/bin/python usability_tests/run_matrix.py --agent sonnet5 --model claude-sonnet-5 --effort medium \
        [--repeat 1] [--targets gpt-2 olmo-1b pythia-1b] [--tests T1 ... T5] [--conditions P F B] \
        [--parallel 3] [--venv] [--log-dir log/usability-<agent>-<effort>]

Each cell is one `run_claude.py` invocation. Cells whose run directory already
holds grade.json are skipped, so an interrupted matrix can be resumed with the
same command. Per-cell stdout/stderr goes to <log-dir>/<target>-<test>-<cond>-<repeat>.log
and a one-line status per cell to <log-dir>/matrix.log.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from targets import TARGETS, TESTS  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def infra_failure(run_dir: Path) -> bool:
    """True when the solve phase ended with an API/CLI error rather than a model outcome
    (rate limit or session limit: is_error with an api_error_status; CLI crash: unknown subtype)."""
    try:
        harness = json.loads((run_dir / "harness.json").read_text())
    except (OSError, json.JSONDecodeError):
        return True
    if harness.get("is_error") or harness.get("api_error_status"):
        return True
    return harness.get("result_subtype") not in ("success", "error_max_turns") and harness.get("cap_hit") != "time"


def rate_limited(run_dir: Path) -> str | None:
    """The limit message when the cell died on a 429 (rate or session limit), else None."""
    try:
        harness = json.loads((run_dir / "harness.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if harness.get("api_error_status") == 429 or (harness.get("is_error") and "limit" in (harness.get("final_text") or "")):
        return (harness.get("final_text") or "429")[:120]
    return None


def run_cell(args, target: str, test: str, cond: str, log_dir: Path) -> str:
    run_dir = HERE / args.agent / target / args.effort / f"{test}-{cond}-{args.repeat}"
    tag = f"{target} {test} {cond} r{args.repeat}"
    if (run_dir / "grade.json").exists():
        if infra_failure(run_dir):
            shutil.rmtree(run_dir)
            note = " (previous attempt failed before the model could work: infrastructure error; rerunning)"
        else:
            return f"skip   {tag} (already graded)"
    elif run_dir.exists():
        return f"stale  {tag} (run dir exists without grade.json; remove it to rerun)"
    else:
        note = ""
    cmd = [sys.executable, str(HERE / "run_claude.py"), test, cond, "--agent", args.agent, "--model", args.model,
           "--target", target, "--effort", args.effort, "--repeat", str(args.repeat),
           "--max-turns", str(args.max_turns), "--timeout", str(args.timeout)]
    if args.venv:
        cmd.append("--venv")
    if args.keep_artifacts:
        cmd.append("--keep-artifacts")
    log = log_dir / f"{target}-{test}-{cond}-{args.repeat}.log"
    start = dt.datetime.now()
    for attempt in range(1, args.max_attempts + 1):
        with log.open("a") as fh:
            fh.write(f"# attempt {attempt} {dt.datetime.now().isoformat(timespec='seconds')}\n")
            fh.flush()
            rc = subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT).returncode
        limit = rate_limited(run_dir)
        if limit is None:
            break
        # Rate or session limit: this attempt never reached the model. Discard it,
        # wait, and try again so the cell is measured, not lost.
        shutil.rmtree(run_dir, ignore_errors=True)
        note = f" (attempt {attempt} hit a limit: {limit!r}; waited {args.limit_wait_s}s)"
        print(f"limit  {tag}: {limit!r}; waiting {args.limit_wait_s}s before attempt {attempt + 1}", flush=True)
        time.sleep(args.limit_wait_s)
    secs = (dt.datetime.now() - start).total_seconds()
    return f"{'PASS' if rc == 0 else 'FAIL'}   {tag} {secs:.0f}s (log: {log.name}){note}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--effort", required=True, choices=("low", "medium", "high"))
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--targets", nargs="+", default=sorted(TARGETS), choices=sorted(TARGETS))
    parser.add_argument("--tests", nargs="+", default=sorted(TESTS), choices=sorted(TESTS))
    parser.add_argument("--conditions", nargs="+", default=["P", "F", "B"], choices=["P", "F", "B"])
    parser.add_argument("--parallel", type=int, default=3)
    parser.add_argument("--venv", action="store_true")
    parser.add_argument("--keep-artifacts", action="store_true")
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--max-attempts", type=int, default=6,
                        help="attempts per cell when the API returns a rate/session limit")
    parser.add_argument("--limit-wait-s", type=int, default=900, help="wait between such attempts")
    args = parser.parse_args()
    log_dir = args.log_dir or REPO / "log" / f"usability-{args.agent}-{args.effort}"
    log_dir.mkdir(parents=True, exist_ok=True)
    cells = [(t, s, c) for t in args.targets for s in args.tests for c in args.conditions]
    matrix_log = log_dir / "matrix.log"
    with matrix_log.open("a") as fh:
        fh.write(f"# {dt.datetime.now().isoformat(timespec='seconds')} {len(cells)} cells, parallel={args.parallel}\n")
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(run_cell, args, t, s, c, log_dir): (t, s, c) for t, s, c in cells}
        for fut in as_completed(futures):
            line = fut.result()
            print(line, flush=True)
            with matrix_log.open("a") as fh:
                fh.write(line + "\n")
    print(f"done; see {matrix_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
