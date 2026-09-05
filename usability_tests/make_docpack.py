#!/usr/bin/env python
"""Assemble the documentation pack handed to participants in condition B.

    .venv/bin/python usability_tests/make_docpack.py [--out DIR] [--example PLAN ...]

Writes DIR (default usability_tests/docpack/):
    README.md                  the BrainSurgery README
    interfaces-reference.md    docs/interfaces-reference.md
    help.txt                   built-in help for every transform and assert expression
    examples/                  worked example plans; by default the two frozen ones:
                               flexmore_examples/olmo_1b_0724_hf_dense_to_expert_moe.yaml
                               validation/validation.yaml

The pack for condition B0 is only help.txt. Condition P gets no pack.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ANSI = re.compile(r"\x1b\[[0-9;]*m")
ASSERT_EXPRS = [
    "all", "any", "count", "dimensions", "dtype", "equal", "exists", "iszero", "not", "shape",
]


def brainsurgery_bin() -> str:
    candidate = REPO / ".venv" / "bin" / "brainsurgery"
    return str(candidate) if candidate.exists() else "brainsurgery"


def list_commands(tmp: Path) -> list[str]:
    plan = tmp / "list.yaml"
    plan.write_text("transforms:\n  - help: {}\n")
    text = run_plan(plan)
    names = re.findall(r"^\s*│?\s{2,}([a-z_]+)\s*│?$", text, flags=re.MULTILINE)
    return sorted(set(names))


def run_plan(plan: Path) -> str:
    proc = subprocess.run(
        [brainsurgery_bin(), str(plan), "--no-summarize", "--log-level", "error"],
        cwd=REPO, capture_output=True, text=True, env={"COLUMNS": "100", "PATH": "/usr/bin:/bin"},
    )
    if proc.returncode != 0:
        print(proc.stdout[-2000:], proc.stderr[-2000:], file=sys.stderr)
        raise SystemExit(f"help dump failed for {plan}")
    return ANSI.sub("", proc.stdout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=HERE / "docpack")
    parser.add_argument("--example", type=Path, action="append", default=None,
                        help="example plan to include (default: the two frozen examples below)")
    args = parser.parse_args()

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / "README.md", out / "README.md")
    shutil.copy(REPO / "docs" / "interfaces-reference.md", out / "interfaces-reference.md")

    tmp = out / ".tmp"
    tmp.mkdir(exist_ok=True)
    commands = list_commands(tmp)
    if not commands:
        raise SystemExit("could not parse the command list from `help: {}` output")
    entries = ["  - help: {}"] + [f"  - help: {name}" for name in commands]
    entries += [f"  - help: {{ assert: {expr} }}" for expr in ASSERT_EXPRS]
    plan = tmp / "help.yaml"
    plan.write_text("transforms:\n" + "\n".join(entries) + "\n")
    (out / "help.txt").write_text(run_plan(plan))
    shutil.rmtree(tmp)

    if args.example is None:
        args.example = [REPO / "flexmore_examples" / "olmo_1b_0724_hf_dense_to_expert_moe.yaml",
                        REPO / "validation" / "validation.yaml"]
    examples = out / "examples"
    if args.example:
        examples.mkdir(exist_ok=True)
        for path in args.example:
            shutil.copy(path, examples / path.name)

    print(f"[docpack] {out}: README.md, interfaces-reference.md, help.txt ({len(commands)} commands), "
          f"{len(args.example)} example plan(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
