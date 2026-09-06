# BrainSurgery usability tests

A usability-study kit for BrainSurgery. Five checkpoint-editing tests, each
instantiated on three surgery-target models and solved under three
conditions by coding agents acting as proxy practitioners. It produces the
evidence the reviewers of the paper ("BrainSurgery: Reproducible and Reliable
Declarative Weight Manipulations for Model Editing and Upcycling") asked for:
success rate, retries, errors, tokens and cost, time to solution, and
bug-detection ability, per test, target, condition and agent. `AGENTS.md`
here is the operating procedure for the agent running the study.

The tests are deliberately not the ones shown in the paper. They are
realistic checkpoint-surgery jobs that exercise the same capability classes
the paper claims (bulk targeting, structural edits, precision casting,
arithmetic, low-rank adapters, validation, sharded export).

## The five tests

| Test | Objective | Capability class | Output |
|---|---|---|---|
| T1 layer-prune | Remove transformer blocks and renumber the rest contiguously | structural edit, bulk rename | single file |
| T2 head-prune | Remove one attention head (a middle one) from every layer, across all head-bearing projections | slicing, concat, layout knowledge | single file |
| T3 mixed-precision-export | Projection matrices to bfloat16, everything else float32, drop buffers, sharded with index | precision cast, bulk targeting, sharding | sharded |
| T4 task-vector-merge | `base + 0.4 (ft1 - base) + 0.4 (ft2 - base)` on the MLP tensors after verifying all other tensors match | multi-checkpoint arithmetic, validation | single file |
| T5 lora-merge | Fold a PEFT-style LoRA adapter (r=16, alpha=32) into the base attention weights | low-rank adapter, matmul, transpose, sharding | sharded |

Each test has a "Why it is meaningful" section in its specification and a
per-target instantiation (`tasks/<test>/TASK-<target>.md`) with concrete
names, shapes, dtypes, counts and layout facts. The specifications are
tool-neutral: they state what must be true of the output and which checks
the solution must perform, and nothing about how to do it.

## Surgery targets

| Target | Model | Layers | Dtype | What makes it different |
|---|---|---|---|---|
| `gpt-2` | GPT-2 (124M), `openai-community/gpt2` | 12 | fp32 | fused `[q k v]` projection, Conv1D `[in, out]` layout, causal-mask buffers, single file |
| `olmo-1b` | OLMo-1B-0724-hf, `allenai/OLMo-1B-0724-hf` | 16 | fp32 | separate q/k/v/o, no biases, no norm parameters, sharded input (two shards + index), 4.7 GB |
| `pythia-1b` | Pythia-1B, `EleutherAI/pythia-1b` | 16 | fp16 | fused QKV interleaved per head (GPT-NeoX), three buffer kinds incl. a uint8 mask, half-precision input that must be upcast for arithmetic |

All facts live in `targets.py`; `generate.py` renders the task texts,
reference solutions, plans and review artifacts from them.

## Conditions

| Code | Tool | Environment | Doc pack |
|---|---|---|---|
| P | Python script with torch and safetensors | `conditions/requirements-P.txt` (pinned) | none |
| F | Free choice from the allowed list: merge toolkits, adapter libraries, HF utilities, key-rewriting tools, or scripts on top of them | `conditions/requirements-F.txt` (pinned lock resolved from the list) | the tools' own docs |
| B | A BrainSurgery YAML plan run with the `brainsurgery` CLI, no Python | the repository installed editable at the recorded commit, `conditions/constraints-B.txt` | `docpack/` |

`conditions/F-allowed.md` is the paper's related-systems list restricted to
what is pip-installable and useful here; `requirements-F.txt` is its resolved
lock. The team may replace both before the pilot. Every task has at least one
plausible route through existing tools (merge toolkits slice layers and do
task arithmetic, transformers prunes heads and saves with a dtype, peft merges
adapters, torch-state-bridge rewrites keys), so F is a real alternative and
not a disguised copy of P.

All three conditions pin torch 2.14.0, safetensors 0.5.3 and numpy 2.5.2 so
runs are comparable across conditions and over the weeks of the study.

## Coding agents and effort tiers

Two drivers produce identical record files, so `analyze.py` treats all
vendors alike:

- `run_claude.py`: Claude models through Claude Code (`claude -p`). Used on
  this machine for Fable 5.1, Opus 5 and Sonnet 5.
- `run_codex.py`: OpenAI models through the Codex CLI (`codex exec --json`).
  The model ids are chosen by whoever runs it; nothing in the kit names one.
  Codex does not report cost, so pass the input, output, cache-read and
  cache-write rates (USD per million tokens), or leave `cost_usd` null for
  the analysis. Written without
  a Codex installation to test on: on first use run one cell and compare
  `harness.json` with `transcript.jsonl` (see the docstring).

For full Codex runs on Linux, `run_matrix_codex.py` runs one resumable effort
tier and repeat, while `run_full_codex.sh` covers all three effort tiers for
one repeat. The full launcher verifies the data manifest before starting and
defaults to sequential cells. After each batch, `audit_codex.py --agent NAME`
checks every transcript-derived harness field and lists the failed-execution
classes and review decisions that still require experimenter confirmation.

Any other agent can be driven from a copy of this repository with
`make_sandbox.py` plus a driver that writes the same record files (see
`AGENTS.md` and `record-template.md`).

Every agent runs every cell at three effort tiers, using the vendor's own
names: `low`, `medium`, `high` for Claude Code (`claude --effort`), and
`light`, `medium`, `high` for OpenAI models (their reasoning-effort setting).
The tier is a directory level in the results and a field in `run.json` and
`harness.json`, so the analysis reports every measure per tier.

## Where results go

    usability_tests/<agent>/<target>/<effort>/<test>-<condition>-<repeat>/
    usability_tests/sonnet5/gpt-2/medium/T2-B-1/          for example

Agent directory names are short and stable (`fable51`, `opus5`, `sonnet5`,
...). Each run directory is the participant's sandbox and, after the run,
holds that run's study data: `PROMPT.md`, the authored artifact under
`out/<test>/`, the participant's `REPORT.md`, `run.json`, `harness.json`,
`grade.json`, `review.json`, `env-freeze.txt`. `.gitignore` keeps those and
ignores checkpoints, environments, transcripts and copied inputs.

## What is measured

| Measure | Source |
|---|---|
| Success rate | `grade.json` PASS over runs, per cell and pooled |
| Retries, errors | `harness.json`: executions of the script/plan, failed executions with an error class each, first-execution success, executions until first success |
| Tokens and cost | `harness.json`: input/output/cache tokens and cost from the provider (Claude Code reports `total_cost_usd`) |
| Time to solution | `harness.json` wall clock of the solve phase, reported over passing runs |
| Bug-detection ability | `review.json`: after solving, the same model reviews one artifact for the same task (defective on odd repeats, correct on even) and must say whether it meets the specification; the experimenter confirms `detected` against `review/<target>/answers.json`. Reported as detection rate on defective artifacts and false-alarm rate on correct ones |
| Self-report | `out/<test>/REPORT.md`: attempts, pitfalls, unclear points, tools used (F) |

`analyze.py` aggregates all of it into one table per (agent, target,
effort, condition) plus rows pooled over targets, over agents, and overall.

## Environment isolation

Every run gets its own sandbox directory and, with `--venv`, its own Python
environment built by uv from the condition's pinned requirements. The
sandbox contains the prompt (also as `CLAUDE.md` and `AGENTS.md` so agents
load it automatically), the task, a read-only `inputs/` symlink, an empty
`out/<test>/`, the condition's doc pack or allowed list, and a
`.claude/settings.json` that denies web access, package installs and reads
outside the sandbox. Nothing from `references/`, `solutions/`, `review/` or
`grade.py` is inside a sandbox. Each task tells the participant this in its
"Environment" section.

## Layout

| Path | What |
|---|---|
| `targets.py` | Facts about the three surgery targets; the single source for everything generated |
| `generate.py` | Renders `tasks/*/TASK-<target>.md`, `solutions/<target>/{P,B}/`, `review/<target>/` |
| `tasks/<test>/TASK-<target>.md` | Specification given to the participant |
| `conditions/{P,F,B}.md` | Condition preamble prepended to the task |
| `conditions/requirements-*.txt`, `constraints-B.txt`, `F-allowed.md`, `sandbox-settings.json` | Environment contents and sandbox permissions |
| `record-template.md` | Every recorded field, error classes, review record, participant self-report |
| `setup.py` | Builds inputs and hidden references for every target |
| `make_docpack.py` | Assembles `docpack/` for condition B (README, interfaces reference, full `help` dump, two frozen example plans) |
| `make_sandbox.py` | Creates one run directory (and optionally its environment) |
| `run_claude.py`, `run_codex.py` | Drive one participant session with Claude Code or Codex CLI: sandbox, solve, grade, review |
| `run_matrix.py`, `run_matrix_codex.py` | Run a Claude or Codex matrix resumably and in parallel |
| `run_full_claude.sh`, `run_full_codex.sh` | Run a full vendor matrix; the Codex launcher handles one repeat at a time on Linux |
| `audit_codex.py` | Recompute Codex harness fields from transcripts and report manual bookkeeping still required |
| `resummarise.py` | Recomputes execution counts in `harness.json` from saved Claude transcripts |
| `grade.py` | Grades an output against `references/<target>/<test>`; independent of BrainSurgery |
| `analyze.py` | Aggregates run records into the study tables |
| `make_manifest.py`, `manifest.sha256` | Checksums of every input, reference and doc-pack file; `--verify` proves another machine runs the same study |
| `pack_data.sh` | Bundles the generated inputs and references (~35 GB) for transfer to another machine |
| `solutions/<target>/P/*.py`, `solutions/<target>/B/*.yaml` | Reference baselines and plans. Hidden from participants. The Python ones generate the references |
| `review/<target>/` | Defective variants of the references, one injected defect per test, plus `answers.json` |
| `AGENTS.md`, `CLAUDE.md` | Operating procedure for the experimenter agent |
| `inputs/<target>`, `references/<target>` | Symlinks into the data root (gitignored) |

## Setup

```bash
cd /path/to/brainsurgery
.venv/bin/python -c "import torch, safetensors, brainsurgery"
# base checkpoints (once): models/gpt2, models/olmo-1b-0724-hf, models/pythia-1b
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
# pinned revisions (also in targets.py); a different revision changes every derived file
snapshot_download("openai-community/gpt2", revision="607a30d783dfa663caf39e06633721c8d4cfcd7e",
                  local_dir="models/gpt2", allow_patterns=["*.json", "*.safetensors", "*.txt"])
snapshot_download("allenai/OLMo-1B-0724-hf", revision="d7cbab742d80589e714b1a2d7f838dcd21cbe143",
                  local_dir="models/olmo-1b-0724-hf", allow_patterns=["*.json", "*.safetensors", "*.txt"])
snapshot_download("EleutherAI/pythia-1b", revision="f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2",
                  local_dir="models/pythia-1b", allow_patterns=["*.json", "*.safetensors", "*.txt"])
PY
.venv/bin/python usability_tests/generate.py     # only after editing targets.py or generate.py
.venv/bin/python usability_tests/setup.py        # ~5 min, writes ~48 GB under models/usability_tests
.venv/bin/python usability_tests/make_docpack.py
```

`setup.py` writes `inputs/<target>/{base,ft1,ft2,lora}` and
`references/<target>/T1..T5` under `--data-root` (default
`models/usability_tests`). The fine-tunes are synthetic frozen-backbone
fine-tunes (seeded low-rank deltas on the MLP weights, small noise on MLP
biases, everything else bit-identical); the adapters are seeded PEFT-style
LoRA factors with an `adapter_config.json`. References come from the Python
baselines only, so grading is independent of the tool under test.

## Grading

`grade.py <test> --target <target> [--out PATH] [--json] [--write FILE]`
checks, in order: loadability (single `.safetensors`, torch `.pt`, or a
sharded directory with an index); the sharding rule for T3 and T5, where a
shard may exceed the budget only if it holds a single tensor; exact key set;
per-tensor shape and dtype; values. Values are bit-exact except for tensors
produced by floating-point arithmetic (the merged MLP tensors of T4 and the
merged attention weights of T5), which use a relative Frobenius tolerance of
1e-5 for float32 outputs and 1e-3 for half-precision outputs.

## Verification

Verified 2026-09-05 on this machine: every plan in `solutions/<target>/B`
passes `grade.py` against the Python-generated references under the default
provider, and every defective artifact in `review/<target>/` (Python and plan)
fails it. Wall clock is one BrainSurgery run on CPU; line counts are
non-blank, non-comment lines (`validation/count_lines.py`), the Python column
excluding the 45-line shared loader/writer `_ckpt.py`.

| Test | gpt-2 Python / plan / s | olmo-1b Python / plan / s | pythia-1b Python / plan / s |
|---|---|---|---|
| T1 | 27 / 16 / 4 | 27 / 19 / 5 | 27 / 19 / 4 |
| T2 | 32 / 60 / 4 | 34 / 83 / 6 | 32 / 64 / 5 |
| T3 | 20 / 14 / 4 | 20 / 13 / 4 | 20 / 14 / 5 |
| T4 | 28 / 20 / 4 | 28 / 20 / 10 | 28 / 24 / 7 |
| T5 | 31 / 20 / 4 | 31 / 19 / 4 | 31 / 21 / 4 |

T2 is much longer as a plan than as a script because `concat` takes single
tensor references only, so every per-layer concatenation is spelled out (16
layers times up to four tensors for OLMo); the renumbering moves in T1 are
spelled out for the same reason. Both are genuine usability findings about
the DSL and stay in the task set.

## Results, repeat 1

Sonnet 5, Opus 5 and Fable 5.1 through Claude Code; three effort tiers; 3
targets x 5 tests x 3 conditions; one repeat; 405 cells, all under commit
8b7c76a or later with the fixed doc pack. Records are under
`usability_tests/<agent>/<target>/<effort>/`; the full table is `analyze.py`.

| Condition | Success | First run OK | Median cost (USD) | Median time to solution | Median output tokens | Bug detected |
|---|---|---|---|---|---|---|
| P (Python) | 135/135 | 132/135 (98%) | 0.39 | 61 s | 3.8k | 135/135 |
| F (free choice) | 135/135 | 122/135 (90%) | 0.49 | 87 s | 5.3k | 134/135 |
| B (BrainSurgery) | 135/135 | 134/135 (99%) | 0.77 | 131 s | 7.0k | 135/135 |

By agent (pooled over targets and tiers):

| Agent | Cond | First run OK | Median cost | Median time |
|---|---|---|---|---|
| Sonnet 5 | P / F / B | 98% / 84% / 98% | 0.16 / 0.21 / 0.48 | 61 / 90 / 168 s |
| Opus 5 | P / F / B | 98% / 93% / 100% | 0.38 / 0.54 / 0.76 | 69 / 113 / 134 s |
| Fable 5.1 | P / F / B | 98% / 93% / 100% | 0.51 / 0.61 / 1.04 | 57 / 77 / 115 s |

By effort tier (pooled over agents and targets):

| Tier | Cond | First run OK | Median cost | Median time |
|---|---|---|---|---|
| low | P / F / B | 96% / 91% / 98% | 0.29 / 0.36 / 0.57 | 49 / 67 / 96 s |
| medium | P / F / B | 98% / 89% / 100% | 0.37 / 0.53 / 0.76 | 61 / 85 / 121 s |
| high | P / F / B | 100% / 91% / 100% | 0.53 / 0.69 / 1.13 | 86 / 130 / 183 s |

Reading, for repeat 1: every one of the 405 cells passed, in every
condition, on every target, for every agent and tier, with no cap hits, so
correctness does not separate the conditions for these agents. Effort does,
and the ordering is the same everywhere: a BrainSurgery plan costs about
twice a Python script and takes about twice as long, the free-choice condition
sits in between and has the lowest first-run success (tool invocations fail
before the model falls back to a script), and raising the effort tier scales
cost and time in all conditions without changing outcomes. The plan condition
has the highest first-run success (134 of 135), which is what the executable
checks in a plan are for. Bug detection is at or near 100 percent in all
conditions, but only defective artifacts were shown (odd repeat); the
false-alarm rate needs repeat 2. Repeat 1 cost 280.63 USD in total (233 solve,
47 review) and about 5 hours of wall clock at four cells in parallel.

## Pilot, and what it changed

A pilot (Sonnet 5, medium effort, one repeat, 45 cells) ran before the study
and is preserved in git history (commit 603b806 on the `usability-study`
branch); it is not part of the study data. It forced four changes, all made
before the study started:

- three documentation gaps that cost the plan condition most of its extra
  turns were fixed in the BrainSurgery README, the interfaces reference and
  the `assert equal` help text: shard sizes are binary units and count tensor
  data only (an oversized tensor goes alone in its shard); which alias a
  multi-input plan writes as output; and that `assert equal`'s `right` is a
  rewrite of each `left` match, so capture groups work across aliases;
- condition B installs BrainSurgery non-editable so the repository source is
  not reachable from the sandbox;
- inputs and base checkpoints are read-only (a participant wrote through a
  copied symlink into the shared GPT-2 base);
- execution counting matches real invocations only.

The doc pack is regenerated from the fixed documentation; every study cell
sees the same pack.

## Fixed on the way

The in-memory provider used to fail at save time on any output that still
held a non-contiguous tensor (a `permute` result or a `phlora` factor),
after all transforms had succeeded, leaving partial shards on disk. Fixed
2026-09-05 in `brainsurgery/io/safetensors.py` (tensors are packed at save
time), regression test in `tests/test_io.py`, recorded in `wiki/log.md`.
Output is still not atomic on other save-time failures.

## Open items

1. The condition-F allowed list and lock are the kit's proposal from the
   paper's related-systems table; the team may replace them before the pilot.
2. Repeats `k` and the per-run caps (default in `run_claude.py`: 40 turns,
   15 minutes). Pilot one agent with k=2 across all cells, then fix them.
3. The sandbox permission denies cover Claude Code participants; other
   agents need the equivalent in their own driver, or a container.
