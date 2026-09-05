#!/usr/bin/env python
"""Run one participant session with Claude Code and record everything.

    .venv/bin/python usability-tests/run_claude.py T2 B --agent sonnet5 --model claude-sonnet-5 \
        --target gpt-2 --effort medium --repeat 1 [--venv] [--max-turns 40] [--timeout 900] [--skip-review]

Phases:

1. sandbox: make_sandbox.create_sandbox(...) builds
   <agent>/<target>/<effort>/<test>-<condition>-<repeat>/ (with --venv, a private environment).
2. solve: `claude -p --effort <effort>` runs in the sandbox with PROMPT.md as the prompt, the
   sandbox's .venv/bin first on PATH, permissions bypassed inside the sandbox
   (the sandbox's .claude/settings.json still denies web access, installs and
   reads outside the sandbox). The stream-json transcript is saved as
   transcript.jsonl and summarised into harness.json: turns, tool calls,
   tokens, cost, wall clock, executions of the participant's script/plan and
   which of them failed, cap hit.
3. grade: grade.py writes grade.json.
4. review (bug detection): a fresh single-turn `claude -p` session receives the
   task specification plus one artifact for the same task and condition
   language (P or F: Python; B: plan), either the defective version from
   review/<target>/ or the correct reference, alternating by repeat parity
   (odd repeat: defective, even: correct), and is asked whether it meets the
   specification and what is wrong. review.json stores the verdict text, the
   artifact kind, a heuristic reading of the verdict, and `detected: null`
   for the experimenter to confirm against review/<target>/answers.json.

Other agents: reproduce the same phases and write the same JSON files; analyze.py
only reads run.json, harness.json, grade.json and review.json.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_sandbox import create_sandbox  # noqa: E402
from targets import TARGETS, TESTS  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
# An "execution" is a Bash command that runs the participant's artifact: the script
# (python ... solution.py), the plan (brainsurgery ... plan.yaml, as a program, not just
# the word), a run.sh, or a merge tool invoked on the participant's own config.
EXEC_RE = re.compile(
    r"(?:^|[;&|(]\s*|\n\s*)(?:\S*/)?(?:python[0-9.]*\s+\S*solution\.py|brainsurgery\s+\S*plan\.yaml"
    r"|(?:bash\s+|sh\s+|\./)?\S*run\.sh|mergekit-\w+\s)"
)
DEFECT_WORDS = re.compile(
    r"\b(does not|doesn't|do not|don't|not (?:meet|satisfy|match|implement)|incorrect|wrong|bug|defect|"
    r"mistake|off[- ]by[- ]one|missing|violat)", re.IGNORECASE)


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def run_claude(prompt: str, *, cwd: Path, model: str, effort: str, max_turns: int, timeout: int,
               env: dict, extra: list[str]) -> tuple[list[dict], int, float, bool]:
    cmd = ["claude", "-p", prompt, "--model", model, "--effort", effort, "--max-turns", str(max_turns),
           "--output-format", "stream-json", "--verbose", *extra]
    start = time.monotonic()
    events: list[dict] = []
    timed_out = False
    with subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, err = proc.communicate()
            timed_out = True
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        if err.strip():
            (cwd / "claude-stderr.txt").write_text(err)
    return events, proc.returncode, time.monotonic() - start, timed_out


def summarise(events: list[dict]) -> dict:
    tool_calls = 0
    executions: list[dict] = []
    pending: dict[str, dict] = {}
    for ev in events:
        msg = ev.get("message") or {}
        content = msg.get("content") or []
        if ev.get("type") == "assistant":
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tool_calls += 1
                    cmd = (block.get("input") or {}).get("command", "")
                    if block.get("name") == "Bash" and EXEC_RE.search(cmd or ""):
                        pending[block.get("id")] = {"n": len(executions) + 1, "command": cmd[:300]}
                        executions.append(pending[block["id"]])
        elif ev.get("type") == "user":
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result" and block.get("tool_use_id") in pending:
                    rec = pending.pop(block["tool_use_id"])
                    text = block.get("content")
                    if isinstance(text, list):
                        text = " ".join(t.get("text", "") for t in text if isinstance(t, dict))
                    text = str(text or "")
                    rec["is_error"] = bool(block.get("is_error"))
                    rec["message"] = text.strip().splitlines()[-1][:300] if text.strip() else ""
    result = next((ev for ev in reversed(events) if ev.get("type") == "result"), {})
    usage = result.get("usage") or {}
    failed = [e for e in executions if e.get("is_error")]
    first_ok_index = next((e["n"] for e in executions if e.get("is_error") is False), None)
    return {
        "turns": result.get("num_turns"),
        "tool_calls": tool_calls,
        "tokens_in": usage.get("input_tokens"),
        "tokens_out": usage.get("output_tokens"),
        "cache_read_tokens": usage.get("cache_read_input_tokens"),
        "cache_write_tokens": usage.get("cache_creation_input_tokens"),
        "tokens_in_total": sum(usage.get(k) or 0 for k in
                               ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")),
        "cost_usd": result.get("total_cost_usd"),
        "duration_api_ms": result.get("duration_api_ms"),
        "session_id": result.get("session_id"),
        "result_subtype": result.get("subtype"),
        "executions": len(executions),
        "failed_executions": [{k: v for k, v in e.items() if k != "is_error"} | {"error_class": ""} for e in failed],
        "first_execution_success": (executions[0].get("is_error") is False) if executions else False,
        "executions_until_first_success": first_ok_index if first_ok_index is not None else "never",
        "final_text": (result.get("result") or "")[:2000],
    }


def review_prompt(task_md: str, artifact: str, language: str) -> str:
    return (
        "You are reviewing a checkpoint-editing artifact against its specification. Read the "
        "specification, then the artifact, and answer two questions: (1) Does the artifact do exactly "
        "what the specification requires? Answer YES or NO on the first line. (2) If NO, state precisely "
        "what is wrong and which specification clause it violates, in at most five sentences. Do not run "
        "anything; judge from reading alone.\n\n# Specification\n\n" + task_md +
        f"\n\n# Artifact ({language})\n\n```\n{artifact}\n```\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("test", choices=sorted(TESTS))
    parser.add_argument("condition", choices=("P", "F", "B"))
    parser.add_argument("--agent", required=True, help="directory name for the coding-agent model, e.g. sonnet5")
    parser.add_argument("--model", required=True, help="Claude model id passed to `claude --model`")
    parser.add_argument("--target", required=True, choices=sorted(TARGETS))
    parser.add_argument("--effort", required=True, choices=("low", "medium", "high"),
                        help="Claude Code effort tier, passed as `claude --effort`")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--venv", action="store_true", help="build a private environment for the sandbox")
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--timeout", type=int, default=900, help="solve-phase cap in seconds")
    parser.add_argument("--max-budget-usd", type=float, default=None)
    parser.add_argument("--skip-review", action="store_true")
    parser.add_argument("--root", type=Path, default=HERE)
    args = parser.parse_args()

    sandbox = create_sandbox(args.test, args.condition, agent=args.agent, target=args.target, effort=args.effort,
                             repeat=args.repeat, venv=args.venv, root=args.root)
    print(f"[run] sandbox {sandbox}", flush=True)
    env = dict(os.environ)
    if args.venv:
        env["PATH"] = f"{sandbox / '.venv' / 'bin'}:{env.get('PATH', '')}"
        env["VIRTUAL_ENV"] = str(sandbox / ".venv")
    extra = ["--dangerously-skip-permissions"]
    if args.max_budget_usd is not None:
        extra += ["--max-budget-usd", str(args.max_budget_usd)]

    # ---- solve
    prompt = (sandbox / "PROMPT.md").read_text()
    started = now()
    events, rc, wall, timed_out = run_claude(prompt, cwd=sandbox, model=args.model, effort=args.effort,
                                             max_turns=args.max_turns, timeout=args.timeout, env=env, extra=extra)
    (sandbox / "transcript.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    summary = summarise(events)
    cap = "time" if timed_out else ("turns" if summary.get("result_subtype") == "error_max_turns" else "none")
    harness = {
        "phase": "solve", "driver": "run_claude.py", "model_id": args.model, "agent": args.agent,
        "target": args.target, "effort": args.effort, "test": args.test, "condition": args.condition,
        "repeat": args.repeat,
        "started_at": started, "finished_at": now(), "wall_clock_s": round(wall, 1),
        "exit_code": rc, "cap_hit": cap, "max_turns": args.max_turns, "timeout_s": args.timeout,
        **summary,
    }
    (sandbox / "harness.json").write_text(json.dumps(harness, indent=2) + "\n")
    print(f"[run] solve: {wall:.0f}s, turns={summary['turns']}, executions={summary['executions']}, "
          f"cost={summary['cost_usd']}", flush=True)

    # ---- grade
    subprocess.run([sys.executable, str(HERE / "grade.py"), args.test, "--target", args.target,
                    "--out", str(sandbox / "out" / args.test), "--json",
                    "--write", str(sandbox / "grade.json")], check=False, capture_output=True)
    grade = json.loads((sandbox / "grade.json").read_text()) if (sandbox / "grade.json").exists() else {}
    print(f"[run] grade: {'PASS' if grade.get('passed') else 'FAIL'} {grade.get('findings', [])[:2]}", flush=True)

    # ---- review (bug detection)
    if not args.skip_review:
        lang = "yaml" if args.condition == "B" else "py"
        kind = "defective" if args.repeat % 2 == 1 else "correct"
        if kind == "defective":
            art = HERE / "review" / args.target / ("B" if lang == "yaml" else "P") / f"{args.test}-defective.{lang}"
        else:
            art = HERE / "solutions" / args.target / ("B" if lang == "yaml" else "P") / f"{args.test}.{lang}"
        rprompt = review_prompt((sandbox / "TASK.md").read_text(), art.read_text(),
                                "BrainSurgery plan" if lang == "yaml" else "Python script")
        rstart = now()
        revents, rrc, rwall, _ = run_claude(rprompt, cwd=sandbox, model=args.model, effort=args.effort, max_turns=1,
                                            timeout=300, env=env, extra=["--tools", "", "--dangerously-skip-permissions"])
        rsum = summarise(revents)
        verdict = rsum["final_text"]
        first_line = verdict.strip().splitlines()[0] if verdict.strip() else ""
        says_defective = first_line.strip().upper().startswith("NO") or bool(DEFECT_WORDS.search(first_line))
        answers = json.loads((HERE / "review" / args.target / "answers.json").read_text())
        review = {
            "phase": "review", "artifact_kind": kind, "artifact": str(art.relative_to(HERE)),
            "started_at": rstart, "finished_at": now(), "wall_clock_s": round(rwall, 1),
            "tokens_in": rsum["tokens_in"], "tokens_out": rsum["tokens_out"], "cost_usd": rsum["cost_usd"],
            "verdict_text": verdict, "auto_says_defective": says_defective,
            "detected": None, "expected_defect": answers[args.test] if kind == "defective" else None,
            "note": "experimenter: set `detected` (true if the stated problem matches expected_defect; for a "
                    "correct artifact, true means a false alarm).",
        }
        (sandbox / "review.json").write_text(json.dumps(review, indent=2) + "\n")
        print(f"[run] review ({kind}): says_defective={says_defective}", flush=True)
    return 0 if grade.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
