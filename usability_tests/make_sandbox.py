#!/usr/bin/env python
"""Create an isolated sandbox for one run: (agent, target, effort, test, condition, repeat).

    .venv/bin/python usability_tests/make_sandbox.py T2 B --agent sonnet5 --target gpt-2 --effort medium [--repeat 1] [--venv]

Results are organised by coding-agent model, surgery-target model and effort tier:

    usability_tests/<agent>/<target>/<effort>/<test>-<condition>-<repeat>/

for example usability_tests/sonnet5/gpt-2/medium/T2-B-1/. Effort tiers use the
vendor's own names: low, medium, high for Claude Code; light, medium, high for
OpenAI models. Each run directory holds
exactly what the participant may see, and, with --venv, its own Python
environment built with uv from the condition's pinned requirements:

    <agent>/<target>/<effort>/<test>-<condition>-<repeat>/
      PROMPT.md            condition preamble + task specification
      CLAUDE.md, AGENTS.md copies of PROMPT.md so agents that read CLAUDE.md or AGENTS.md load it automatically
      TASK.md              the task specification alone
      record-template.md   self-report fields
      requirements-*.txt   what the environment contains (constraints-B.txt for B)
      F-allowed.md         condition F only
      docpack/             condition B only
      .claude/settings.json  permission denies: no web, no installs, no reads outside the sandbox
      inputs/              symlink to the target's read-only inputs
      out/<test>/          empty; the participant writes here
      .venv/               with --venv: private environment
      run.json             metadata (ids, timestamps, repository commit, env fingerprint)

Nothing from references/, solutions/, review/ or grade.py is copied.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from targets import TARGETS, TESTS  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CONDITIONS = ("P", "F", "B")
EFFORTS = ("low", "light", "medium", "high")


def sh(cmd: list[str], cwd: Path) -> None:
    print("[sandbox] $", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def repo_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def make_venv(sandbox: Path, condition: str) -> str:
    venv = sandbox / ".venv"
    sh(["uv", "venv", "--python", "3.13", str(venv)], sandbox)
    python = venv / "bin" / "python"
    if condition == "B":
        # Non-editable install: the package is built into a wheel and copied into the
        # sandbox environment, so the repository checkout (its docs, tests, wiki and
        # this kit) is not reachable through the installed package. An editable
        # install exposes the checkout path, and pilot participants followed it.
        sh(["uv", "pip", "install", "--python", str(python), "-c", str(sandbox / "constraints-B.txt"),
            "--no-cache-dir", str(REPO)], sandbox)
        # The wheel metadata records where it was built from; drop that pointer too.
        for direct_url in venv.glob("lib/python*/site-packages/brainsurgery-*.dist-info/direct_url.json"):
            direct_url.unlink()
    else:
        sh(["uv", "pip", "install", "--python", str(python), "-r",
            str(sandbox / f"requirements-{condition}.txt")], sandbox)
    freeze = subprocess.run(["uv", "pip", "freeze", "--python", str(python)],
                            capture_output=True, text=True, check=True).stdout
    (sandbox / "env-freeze.txt").write_text(freeze)
    return freeze


def create_sandbox(test: str, condition: str, *, agent: str, target: str, effort: str, repeat: int = 1,
                   venv: bool = False, root: Path = HERE) -> Path:
    if test not in TESTS or condition not in CONDITIONS or target not in TARGETS or effort not in EFFORTS:
        raise SystemExit(f"[sandbox] unknown test/condition/target/effort: {test} {condition} {target} {effort}")
    inputs = (HERE / "inputs" / target).resolve()
    if not inputs.exists():
        raise SystemExit(f"[sandbox] inputs for {target} missing; run setup.py first")
    run_id = f"{test}-{condition}-{repeat}"
    sandbox = (root / agent / target / effort / run_id).resolve()
    if sandbox.exists():
        raise SystemExit(f"[sandbox] {sandbox} already exists; use a fresh --repeat")
    sandbox.mkdir(parents=True)

    task_md = (HERE / "tasks" / f"{test}-{TESTS[test]}" / f"TASK-{target}.md").read_text()
    cond_md = (HERE / "conditions" / f"{condition}.md").read_text()
    prompt = cond_md.rstrip() + "\n\n---\n\n" + task_md
    (sandbox / "TASK.md").write_text(task_md)
    for name in ("PROMPT.md", "CLAUDE.md", "AGENTS.md"):
        (sandbox / name).write_text(prompt)
    shutil.copy(HERE / "record-template.md", sandbox / "record-template.md")
    shutil.copy(HERE / "conditions" / f"requirements-{condition}.txt", sandbox / f"requirements-{condition}.txt")
    if condition == "B":
        shutil.copy(HERE / "conditions" / "constraints-B.txt", sandbox / "constraints-B.txt")
        docpack = HERE / "docpack"
        if not docpack.exists():
            raise SystemExit("[sandbox] docpack missing; run make_docpack.py first")
        shutil.copytree(docpack, sandbox / "docpack")
    if condition == "F":
        shutil.copy(HERE / "conditions" / "F-allowed.md", sandbox / "F-allowed.md")
    (sandbox / ".claude").mkdir()
    shutil.copy(HERE / "conditions" / "sandbox-settings.json", sandbox / ".claude" / "settings.json")
    (sandbox / "inputs").symlink_to(inputs)
    (sandbox / "out" / test).mkdir(parents=True)

    freeze = make_venv(sandbox, condition) if venv else ""
    (sandbox / "run.json").write_text(json.dumps({
        "run_id": run_id,
        "agent": agent,
        "target": target,
        "effort": effort,
        "test": test,
        "condition": condition,
        "repeat": repeat,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "repo_commit": repo_commit(),
        "venv": venv,
        "env_freeze_lines": len(freeze.splitlines()),
    }, indent=2) + "\n")
    return sandbox


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("test", choices=sorted(TESTS))
    parser.add_argument("condition", choices=CONDITIONS)
    parser.add_argument("--agent", required=True, help="coding-agent model directory name, e.g. sonnet5")
    parser.add_argument("--target", required=True, choices=sorted(TARGETS), help="surgery-target model")
    parser.add_argument("--effort", required=True, choices=EFFORTS,
                        help="effort tier: low/medium/high (Claude Code) or light/medium/high (OpenAI)")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--root", type=Path, default=HERE, help="parent of <agent>/<target>/<effort>/ (default: this directory)")
    parser.add_argument("--venv", action="store_true", help="build a private environment with uv")
    args = parser.parse_args()
    sandbox = create_sandbox(args.test, args.condition, agent=args.agent, target=args.target, effort=args.effort,
                             repeat=args.repeat, venv=args.venv, root=args.root)
    print(f"[sandbox] ready: {sandbox}")
    print(f"[sandbox] prompt: {sandbox / 'PROMPT.md'}")
    if args.venv:
        print(f"[sandbox] activate: source {sandbox / '.venv' / 'bin' / 'activate'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
