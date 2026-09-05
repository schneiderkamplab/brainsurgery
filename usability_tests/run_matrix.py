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
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from targets import TARGETS, TESTS  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def run_cell(args, target: str, test: str, cond: str, log_dir: Path) -> str:
    run_dir = HERE / args.agent / target / args.effort / f"{test}-{cond}-{args.repeat}"
    tag = f"{target} {test} {cond} r{args.repeat}"
    if (run_dir / "grade.json").exists():
        return f"skip   {tag} (already graded)"
    if run_dir.exists():
        return f"stale  {tag} (run dir exists without grade.json; remove it to rerun)"
    cmd = [sys.executable, str(HERE / "run_claude.py"), test, cond, "--agent", args.agent, "--model", args.model,
           "--target", target, "--effort", args.effort, "--repeat", str(args.repeat),
           "--max-turns", str(args.max_turns), "--timeout", str(args.timeout)]
    if args.venv:
        cmd.append("--venv")
    log = log_dir / f"{target}-{test}-{cond}-{args.repeat}.log"
    start = dt.datetime.now()
    with log.open("w") as fh:
        rc = subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT).returncode
    secs = (dt.datetime.now() - start).total_seconds()
    return f"{'PASS' if rc == 0 else 'FAIL'}   {tag} {secs:.0f}s (log: {log.name})"


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
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--log-dir", type=Path, default=None)
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
