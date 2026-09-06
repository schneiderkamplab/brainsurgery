#!/usr/bin/env python
"""Run one repeat of the Codex study matrix, resumably and in parallel.

Example (from the repository root):

    .venv/bin/python usability_tests/run_matrix_codex.py \
        --agent astra --model gpt-6-astra --effort light --repeat 1 \
        --parallel 1 --venv --timeout 1800 --price-in 10 --price-out 50 \
        --log-dir log/usability-astra-light-r1

Each cell is one ``run_codex.py`` invocation. A cell is complete when its run
directory contains harness.json, grade.json and review.json. Incomplete or
infrastructure-failed directories are reported and preserved unless
``--rerun-infrastructure-failures`` is explicitly passed.

After the matrix, run ``audit_codex.py``. Transcript consistency is automatic,
but an experimenter must still classify failed executions and confirm each
review's ``detected`` value.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from targets import TARGETS, TESTS  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
REQUIRED_RECORDS = ("harness.json", "grade.json", "review.json")


def complete(run_dir: Path) -> bool:
    return all((run_dir / name).is_file() for name in REQUIRED_RECORDS)


def infrastructure_failure(run_dir: Path) -> bool:
    """True when Codex itself failed before producing a participant outcome."""
    try:
        harness = json.loads((run_dir / "harness.json").read_text())
    except (OSError, json.JSONDecodeError):
        return True
    return harness.get("exit_code") not in (0, None) and harness.get("cap_hit") != "time"


def run_cell(args, target: str, test: str, condition: str, log_dir: Path) -> str:
    run_dir = HERE / args.agent / target / args.effort / f"{test}-{condition}-{args.repeat}"
    tag = f"{target} {test} {condition} r{args.repeat}"
    if complete(run_dir) and not infrastructure_failure(run_dir):
        return f"skip   {tag} (already complete)"
    if run_dir.exists():
        if infrastructure_failure(run_dir) and args.rerun_infrastructure_failures:
            if args.dry_run:
                prior = " (would remove an infrastructure-failed attempt)"
            else:
                shutil.rmtree(run_dir)
                prior = " (removed an infrastructure-failed attempt)"
        else:
            return f"stale  {tag} (preserved incomplete/infrastructure-failed run directory)"
    else:
        prior = ""

    reasoning = args.reasoning_effort or ("low" if args.effort == "light" else args.effort)
    cmd = [
        sys.executable,
        str(HERE / "run_codex.py"),
        test,
        condition,
        "--agent",
        args.agent,
        "--model",
        args.model,
        "--target",
        target,
        "--effort",
        args.effort,
        "--reasoning-effort",
        reasoning,
        "--repeat",
        str(args.repeat),
        "--timeout",
        str(args.timeout),
        "--price-in",
        str(args.price_in),
        "--price-out",
        str(args.price_out),
        "--price-cache-read",
        str(args.price_cache_read),
        "--price-cache-write",
        str(args.price_cache_write),
    ]
    if args.venv:
        cmd.append("--venv")
    if args.keep_artifacts:
        cmd.append("--keep-artifacts")

    log = log_dir / f"{target}-{test}-{condition}-{args.repeat}.log"
    if args.dry_run:
        return f"would  {tag} -> {log.name}{prior}"
    started = dt.datetime.now()
    with log.open("w") as fh:
        rc = subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT).returncode
    seconds = (dt.datetime.now() - started).total_seconds()
    state = "PASS" if rc == 0 else "FAIL"
    return f"{state:<6} {tag} {seconds:.0f}s (log: {log.name}){prior}"


def check_prior_repeat(args, cells: list[tuple[str, str, str]]) -> list[Path]:
    if args.repeat <= 1:
        return []
    prior = args.repeat - 1
    return [
        HERE / args.agent / target / args.effort / f"{test}-{condition}-{prior}"
        for target, test, condition in cells
        if not complete(HERE / args.agent / target / args.effort / f"{test}-{condition}-{prior}")
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--effort", required=True, choices=("light", "medium", "high"))
    parser.add_argument("--reasoning-effort", choices=("minimal", "low", "medium", "high"))
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--targets", nargs="+", default=sorted(TARGETS), choices=sorted(TARGETS))
    parser.add_argument("--tests", nargs="+", default=sorted(TESTS), choices=sorted(TESTS))
    parser.add_argument("--conditions", nargs="+", default=["P", "F", "B"], choices=["P", "F", "B"])
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--venv", action="store_true")
    parser.add_argument("--keep-artifacts", action="store_true")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--price-in", type=float, required=True, help="USD per million input tokens")
    parser.add_argument("--price-out", type=float, required=True, help="USD per million output tokens")
    parser.add_argument("--price-cache-read", type=float,
                        help="USD per million cache-read tokens (default: --price-in)")
    parser.add_argument("--price-cache-write", type=float,
                        help="USD per million cache-write tokens (default: --price-in)")
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rerun-infrastructure-failures", action="store_true")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    if args.parallel < 1:
        parser.error("--parallel must be at least 1")
    if args.price_cache_read is None:
        args.price_cache_read = args.price_in
    if args.price_cache_write is None:
        args.price_cache_write = args.price_in

    cells = [(target, test, condition) for target in args.targets for test in args.tests
             for condition in args.conditions]
    missing_prior = check_prior_repeat(args, cells)
    if missing_prior:
        print(f"refusing repeat {args.repeat}: {len(missing_prior)} corresponding repeat "
              f"{args.repeat - 1} cells are incomplete", file=sys.stderr)
        for path in missing_prior[:10]:
            print(f"  {path.relative_to(HERE)}", file=sys.stderr)
        return 2

    log_dir = args.log_dir or REPO / "log" / f"usability-{args.agent}-{args.effort}-r{args.repeat}"
    log_dir.mkdir(parents=True, exist_ok=True)
    matrix_log = log_dir / "matrix.log"
    with matrix_log.open("a") as fh:
        fh.write(f"# {dt.datetime.now().isoformat(timespec='seconds')} {len(cells)} cells, "
                 f"parallel={args.parallel}\n")

    states: list[str] = []
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(run_cell, args, target, test, condition, log_dir): (target, test, condition)
                   for target, test, condition in cells}
        for future in as_completed(futures):
            line = future.result()
            states.append(line)
            print(line, flush=True)
            with matrix_log.open("a") as fh:
                fh.write(line + "\n")
    stale = sum(line.startswith("stale") for line in states)
    print(f"done; see {matrix_log}; run audit_codex.py before analysis")
    return 1 if stale else 0


if __name__ == "__main__":
    raise SystemExit(main())
