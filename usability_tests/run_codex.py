#!/usr/bin/env python
"""Run one participant session with OpenAI Codex CLI and record everything.

    .venv/bin/python usability_tests/run_codex.py T2 B --agent <agent-name> --model <model-id> \
        --target gpt-2 --effort light --repeat 1 [--reasoning-effort low] [--venv] [--timeout 1800] \
        --price-in <usd/M> --price-out <usd/M> [--price-cache-read <usd/M>] [--price-cache-write <usd/M>]

Same phases and the same record files as run_claude.py, so analyze.py treats
both vendors alike:

1. sandbox: make_sandbox.create_sandbox(...) builds
   <agent>/<target>/<effort>/<test>-<condition>-<repeat>/ (with --venv, a private environment).
2. solve: `codex exec` runs non-interactively in the sandbox with PROMPT.md as
   the prompt, the sandbox's .venv/bin first on PATH, `--sandbox workspace-write`
   (Codex's own sandbox: writes only inside the working directory, no network)
   and `--json`, which streams JSONL events. The stream is saved as
   transcript.jsonl and summarised into harness.json: turns, tool calls,
   tokens, executions of the participant's artifact and which failed, cap hit.
   Codex does not report cost; cost_usd is computed from the input, output,
   cache-read and cache-write price arguments (USD per million tokens) when
   given, otherwise left null for the analysis to fill from the vendor rate
   card.
3. grade: grade.py writes grade.json.
4. review (bug detection): a fresh single-turn `codex exec --sandbox read-only`
   session judges one artifact for the same task (defective on odd repeats,
   correct on even), exactly as in run_claude.py, into review.json.

Effort: --effort names the tier directory (light/medium/high for OpenAI models,
as agreed for the study); --reasoning-effort is the value passed to Codex as
`-c model_reasoning_effort=...` (Codex accepts minimal, low, medium, high; it
defaults to the tier name, so pass --reasoning-effort low for the light tier).

Codex CLI is a moving target. This driver is validated against `codex exec
--json` as of Codex CLI 0.153.0 (events: thread.started, turn.started,
item.started/item.completed with item.type in {agent_message, reasoning,
command_execution, ...}, turn.completed with usage). On each new CLI version,
run one cell, then compare
harness.json with transcript.jsonl (or run audit_codex.py) and adjust
summarise_codex() if the event names differ. Everything downstream only needs
the harness.json fields.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_sandbox import create_sandbox  # noqa: E402
from run_claude import DEFECT_WORDS, EXEC_RE, review_prompt  # noqa: E402
from targets import TARGETS, TESTS  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SHELL_LC_RE = re.compile(r"^(?:\S*/)?(?:zsh|bash|sh)\s+-lc\s+(['\"])(.*)\1$", re.DOTALL)


def is_execution_command(command: str) -> bool:
    """Recognize artifact runs, including commands wrapped by `codex exec` in a shell."""
    command = command.strip()
    shell_match = SHELL_LC_RE.match(command)
    if shell_match:
        command = shell_match.group(2)
    return bool(EXEC_RE.search(command))


def is_tool_item(item_type: str) -> bool:
    """Return whether a Codex stream item represents one tool invocation."""
    return item_type in {"command_execution", "file_change", "mcp_tool_call", "web_search"}


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def run_codex(prompt: str, *, cwd: Path, model: str, reasoning_effort: str, timeout: int, env: dict,
              sandbox_mode: str, extra: list[str]) -> tuple[list[dict], int, float, bool, str]:
    last_message = cwd / ".codex-last-message.txt"
    cmd = ["codex", "exec", "--json", "--skip-git-repo-check", "--sandbox", sandbox_mode,
           "--model", model, "-c", f"model_reasoning_effort={json.dumps(reasoning_effort)}",
           "--output-last-message", str(last_message), *extra, prompt]
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
            (cwd / "codex-stderr.txt").write_text(err)
    final = last_message.read_text() if last_message.exists() else ""
    if last_message.exists():
        last_message.unlink()
    return events, proc.returncode, time.monotonic() - start, timed_out, final


def _walk(obj, key):
    """Yield every value of `key` anywhere inside a nested JSON object."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                yield v
            else:
                yield from _walk(v, key)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v, key)


def summarise_codex(events: list[dict], final_text: str) -> dict:
    turns = 0
    tool_calls = 0
    execution_order: list[str] = []
    execution_results: dict[str, dict] = {}
    unkeyed_executions: list[dict] = []
    usage: dict = {}
    for ev in events:
        etype = ev.get("type", "")
        if etype in ("turn.started", "turn.completed"):
            if etype == "turn.completed":
                turns += 1
                for u in _walk(ev, "usage"):
                    if isinstance(u, dict):
                        for k, v in u.items():
                            if isinstance(v, (int, float)):
                                usage[k] = usage.get(k, 0) + v
            continue
        item = ev.get("item") if isinstance(ev.get("item"), dict) else ev
        itype = item.get("type", "") or item.get("item_type", "")
        if etype.endswith("completed") and is_tool_item(itype):
            tool_calls += 1
        if "command" in itype or "command" in item:
            cmd = item.get("command")
            if isinstance(cmd, list):
                cmd = " ".join(map(str, cmd))
            cmd = str(cmd or "")
            item_id = str(item.get("id") or "")
            if etype.endswith("started") and is_execution_command(cmd) and item_id:
                execution_order.append(item_id)
            if etype.endswith("completed"):
                if is_execution_command(cmd):
                    exit_code = item.get("exit_code")
                    output = str(item.get("aggregated_output") or item.get("output") or "")
                    result = {
                        "command": cmd[:300],
                        "is_error": (exit_code not in (0, None)) or item.get("status") == "failed",
                        "message": output.strip().splitlines()[-1][:300] if output.strip() else "",
                    }
                    if item_id:
                        execution_results[item_id] = result
                        if item_id not in execution_order:
                            execution_order.append(item_id)
                    else:
                        unkeyed_executions.append(result)
    executions = [execution_results[item_id] for item_id in execution_order if item_id in execution_results]
    executions.extend(unkeyed_executions)
    for n, execution in enumerate(executions, 1):
        execution["n"] = n
    failed = [e for e in executions if e["is_error"]]
    first_ok = next((e["n"] for e in executions if not e["is_error"]), None)
    tokens_in = usage.get("input_tokens")
    cached = usage.get("cached_input_tokens") or usage.get("cache_read_input_tokens") or 0
    cache_written = usage.get("cache_write_input_tokens") or usage.get("cache_creation_input_tokens") or 0
    return {
        "turns": turns or None,
        "tool_calls": tool_calls,
        "tokens_in": max(0, tokens_in - cached - cache_written) if tokens_in is not None else None,
        "tokens_out": usage.get("output_tokens"),
        "cache_read_tokens": cached or None,
        "cache_write_tokens": cache_written or None,
        "tokens_in_total": tokens_in,
        "reasoning_tokens": usage.get("reasoning_output_tokens"),
        "usage_raw": usage,
        "executions": len(executions),
        "failed_executions": [{k: v for k, v in e.items() if k != "is_error"} | {"error_class": ""} for e in failed],
        "first_execution_success": (not executions[0]["is_error"]) if executions else False,
        "executions_until_first_success": first_ok if first_ok is not None else "never",
        "final_text": final_text[:2000],
    }


def cost(summary: dict, price_in: float | None, price_out: float | None,
         price_cache_read: float | None = None, price_cache_write: float | None = None) -> float | None:
    if price_in is None or price_out is None or summary["tokens_in_total"] is None:
        return None
    cache_read_rate = price_in if price_cache_read is None else price_cache_read
    cache_write_rate = price_in if price_cache_write is None else price_cache_write
    return round(
        (summary["tokens_in"] or 0) / 1e6 * price_in
        + (summary["cache_read_tokens"] or 0) / 1e6 * cache_read_rate
        + (summary["cache_write_tokens"] or 0) / 1e6 * cache_write_rate
        + (summary["tokens_out"] or 0) / 1e6 * price_out,
        6,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("test", choices=sorted(TESTS))
    parser.add_argument("condition", choices=("P", "F", "B"))
    parser.add_argument("--agent", required=True, help="directory name for the coding-agent model")
    parser.add_argument("--model", required=True, help="model id passed to `codex exec --model`")
    parser.add_argument("--target", required=True, choices=sorted(TARGETS))
    parser.add_argument("--effort", required=True, choices=("light", "low", "medium", "high"),
                        help="tier directory name")
    parser.add_argument("--reasoning-effort", default=None,
                        help="value for model_reasoning_effort (default: the tier name; use low for light)")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--venv", action="store_true")
    parser.add_argument("--timeout", type=int, default=1800, help="solve-phase cap in seconds (the study uses 1800)")
    parser.add_argument("--price-in", type=float, default=None, help="USD per 1M input tokens, for cost_usd")
    parser.add_argument("--price-out", type=float, default=None, help="USD per 1M output tokens, for cost_usd")
    parser.add_argument("--price-cache-read", type=float, default=None,
                        help="USD per 1M cache-read tokens (default: --price-in)")
    parser.add_argument("--price-cache-write", type=float, default=None,
                        help="USD per 1M cache-write tokens (default: --price-in)")
    parser.add_argument("--skip-review", action="store_true")
    parser.add_argument("--keep-artifacts", action="store_true",
                        help="keep the sandbox environment and output checkpoints (default: delete after grading)")
    parser.add_argument("--root", type=Path, default=HERE)
    args = parser.parse_args()
    if shutil.which("codex") is None:
        print("[run] codex CLI not found on PATH", file=sys.stderr)
        return 2
    reasoning = args.reasoning_effort or ("low" if args.effort == "light" else args.effort)

    sandbox = create_sandbox(args.test, args.condition, agent=args.agent, target=args.target, effort=args.effort,
                             repeat=args.repeat, venv=args.venv, root=args.root)
    print(f"[run] sandbox {sandbox}", flush=True)
    env = dict(os.environ)
    if args.venv:
        env["PATH"] = f"{sandbox / '.venv' / 'bin'}:{env.get('PATH', '')}"
        env["VIRTUAL_ENV"] = str(sandbox / ".venv")

    # ---- solve
    prompt = (sandbox / "PROMPT.md").read_text()
    started = now()
    events, rc, wall, timed_out, final = run_codex(prompt, cwd=sandbox, model=args.model, reasoning_effort=reasoning,
                                                   timeout=args.timeout, env=env, sandbox_mode="workspace-write",
                                                   extra=[])
    (sandbox / "transcript.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    summary = summarise_codex(events, final)
    harness = {
        "phase": "solve", "driver": "run_codex.py", "model_id": args.model, "agent": args.agent,
        "target": args.target, "effort": args.effort, "reasoning_effort": reasoning, "test": args.test,
        "condition": args.condition, "repeat": args.repeat, "started_at": started, "finished_at": now(),
        "wall_clock_s": round(wall, 1), "exit_code": rc, "cap_hit": "time" if timed_out else "none",
        "timeout_s": args.timeout,
        "cost_usd": cost(summary, args.price_in, args.price_out,
                         args.price_cache_read, args.price_cache_write),
        "cost_source": "rate card via run_codex.py price arguments" if args.price_in is not None else None,
        **summary,
    }
    (sandbox / "harness.json").write_text(json.dumps(harness, indent=2) + "\n")
    print(f"[run] solve: {wall:.0f}s, turns={summary['turns']}, executions={summary['executions']}", flush=True)

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
        sub = "B" if lang == "yaml" else "P"
        art = (HERE / "review" / args.target / sub / f"{args.test}-defective.{lang}" if kind == "defective"
               else HERE / "solutions" / args.target / sub / f"{args.test}.{lang}")
        rprompt = review_prompt((sandbox / "TASK.md").read_text(), art.read_text(),
                                "BrainSurgery plan" if lang == "yaml" else "Python script")
        rstart = now()
        revents, rrc, rwall, _, rfinal = run_codex(rprompt, cwd=sandbox, model=args.model, reasoning_effort=reasoning,
                                                   timeout=300, env=env, sandbox_mode="read-only", extra=[])
        rsum = summarise_codex(revents, rfinal)
        verdict = rsum["final_text"]
        first_line = verdict.strip().splitlines()[0] if verdict.strip() else ""
        says_defective = first_line.strip().upper().startswith("NO") or bool(DEFECT_WORDS.search(first_line))
        answers = json.loads((HERE / "review" / args.target / "answers.json").read_text())
        review = {
            "phase": "review", "artifact_kind": kind, "artifact": str(art.relative_to(HERE)),
            "started_at": rstart, "finished_at": now(), "wall_clock_s": round(rwall, 1),
            "tokens_in": rsum["tokens_in_total"], "tokens_out": rsum["tokens_out"],
            "cost_usd": cost(rsum, args.price_in, args.price_out,
                             args.price_cache_read, args.price_cache_write),
            "verdict_text": verdict, "auto_says_defective": says_defective,
            "detected": None, "expected_defect": answers[args.test] if kind == "defective" else None,
            "note": "experimenter: set `detected` (true if the stated problem matches expected_defect; for a "
                    "correct artifact, true means a false alarm).",
        }
        (sandbox / "review.json").write_text(json.dumps(review, indent=2) + "\n")
        print(f"[run] review ({kind}): says_defective={says_defective}", flush=True)
    if not args.keep_artifacts:
        from run_claude import cleanup_sandbox
        cleanup_sandbox(sandbox)
    return 0 if grade.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
