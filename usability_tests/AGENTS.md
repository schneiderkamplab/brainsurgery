# usability_tests/AGENTS.md

Instructions for a coding agent working in this directory as the
**experimenter** (the agent that runs the study). Participants never see this
file: a participant session starts inside a sandbox directory that has its own
`AGENTS.md`/`CLAUDE.md` (the task prompt) and cannot see this directory.

## What this directory is

A self-contained usability-study kit for BrainSurgery: five checkpoint-editing
tests (T1..T5) on three surgery-target models (gpt-2, olmo-1b, pythia-1b),
each solved under three conditions (P: Python/torch, F: free choice from an
allowed package list, B: BrainSurgery plan) by several coding-agent models.
`README.md` explains the design; this file is the operating procedure.

## Roles, and the one hard rule

- Experimenter: creates sandboxes, launches participant sessions, grades,
  fills records, runs the analysis. That is you when you work here.
- Participant: a fresh agent session whose working directory is a sandbox.
  It gets `PROMPT.md` and nothing else.
- Hard rule: never act as a participant from this directory, and never copy
  anything from `solutions/`, `review/` or `references/` into a sandbox. The
  study is void if a participant sees a reference solution, the grader, or
  the review answers.

## Procedure for one run

```bash
# once per machine
.venv/bin/python -c "import torch, safetensors, brainsurgery"
.venv/bin/python usability_tests/setup.py            # needs models/gpt2, models/olmo-1b-0724-hf, models/pythia-1b
.venv/bin/python usability_tests/make_docpack.py

# one run with a Claude model (sandbox -> solve -> grade -> review, all recorded)
.venv/bin/python usability_tests/run_claude.py T2 B --agent sonnet5 --model claude-sonnet-5 \
    --target gpt-2 --effort medium --repeat 1 --venv

# one run with an OpenAI model through the Codex CLI (same phases and record files)
.venv/bin/python usability_tests/run_codex.py T2 B --agent <agent-name> --model <model-id> \
    --target gpt-2 --effort light --reasoning-effort low --repeat 1 --venv --price-in <usd/M> --price-out <usd/M>

# one run with any other agent: build the sandbox, drive the agent yourself,
# then grade and write harness.json / review.json with the same fields
.venv/bin/python usability_tests/make_sandbox.py T2 B --agent <agent-name> --target gpt-2 --effort <tier> --repeat 1 --venv
#   ... run the agent with <sandbox>/PROMPT.md as its prompt and <sandbox> as cwd ...
.venv/bin/python usability_tests/grade.py T2 --target gpt-2 --out <sandbox>/out/T2 --json --write <sandbox>/grade.json

# after any number of runs
.venv/bin/python usability_tests/analyze.py
```

Agent directory names are short and stable for the whole study: `fable51`,
`opus5`, `sonnet5`, then whatever the team uses for other vendors. Model ids
go in `harness.json` (`model_id`), not in directory names. Effort tiers are a
directory level and use the vendor's own names: `low`, `medium`, `high` for
Claude Code (passed as `claude --effort`), `light`, `medium`, `high` for
OpenAI models. Results live at
`<agent>/<target>/<effort>/<test>-<condition>-<repeat>/`.

## Full matrix on this machine

3 agents x 3 targets x 3 effort tiers x 5 tests x 3 conditions x k repeats
(405 runs per repeat). Run repeats with
odd numbers first (their review phase gets the defective artifact), then even
ones (correct artifact), so bug detection and false alarms are balanced.
Condition F is ready: `conditions/F-allowed.md` lists the allowed packages
(derived from the paper's related-systems table) and `requirements-F.txt` is
their pinned lock; the pilot ran all 15 F cells with it. Replace both files
before the full run only if the team wants a different list.

## What is recorded, where

| File in the run directory | Written by | Content |
|---|---|---|
| `run.json` | `make_sandbox.py` | ids, timestamps, repository commit, environment fingerprint |
| `harness.json` | driver | turns, tool calls, tokens in/out (+cache), cost, wall clock, executions, failed executions with error class, first-execution success, executions until first success, cap hit |
| `grade.json` | `grade.py` | PASS/FAIL, findings, metrics |
| `review.json` | driver | bug-detection phase: artifact kind (defective/correct), verdict text, heuristic reading, `detected` (experimenter-confirmed), tokens and cost |
| `out/<test>/REPORT.md` | participant | self-report (attempts, pitfalls, unclear points) |
| `transcript.jsonl` | driver | the full stream-json transcript |

After each run: open `harness.json`, classify every failed execution with
one of the error classes in `record-template.md`, and set `detected` in
`review.json` by comparing the verdict with `expected_defect`. Only then does
`analyze.py` report bug-detection rates from confirmed values (until then it
falls back to the heuristic).

## Editing the kit

- Task text, reference solutions, plans and review artifacts are generated.
  Edit `targets.py` (model facts) or `generate.py` (templates), run
  `generate.py`, then `setup.py --force` to rebuild references, then re-verify
  every plan with `grade.py`. Never hand-edit files under `tasks/`,
  `solutions/` or `review/`.
- Freeze the kit before the pilot: the commit hash recorded in `run.json` is
  the pre-registration. Do not change tasks, prompts, doc pack, requirements
  or grader once runs have started; if you must, start a new agent directory.
- Do not commit checkpoints, environments or copied inputs; `.gitignore`
  keeps only the small study data of each run.
- `grade.py` must stay independent of `brainsurgery` (it may import only
  torch, safetensors and `targets.py`).

## Isolation

Inputs are shared by every sandbox through symlinks and are made read-only by
`setup.py` (input files and the base checkpoints they point at). Keep them
that way: in the pilot a participant copied `inputs/base` with `cp -r`, got
the symlinks, and wrote a corrupted test checkpoint straight through one of
them into the shared GPT-2 base, invalidating a later cell. If a base file is
ever modified, re-download it and re-verify against `references/` before
running anything else.

Each run has its own directory and, with `--venv`, its own environment. The
sandbox `.claude/settings.json` denies web access, package installs and reads
outside the sandbox for Claude Code participants; other agents need the
equivalent configured in their own tooling. For strict isolation run the
participant in a container that mounts only the sandbox and the inputs.
