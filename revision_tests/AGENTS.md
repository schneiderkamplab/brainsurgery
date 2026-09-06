# revision_tests/AGENTS.md

Global repository policy in `../AGENTS.md` applies here.

## Purpose

This directory contains the reproducible evaluation work added in response to
the EACL 2027 reviews. It is separate from `usability_tests/`, which owns the
coding-agent study and its participant-isolation protocol.

## Scope

- Keep committed plans, protocols, small fixtures, test code, analysis code,
  and compact machine-readable summaries here.
- Write raw execution artifacts to `../log/revision_tests/<run_id>/`.
- Keep checkpoints, model caches, transferred archives, virtual environments,
  credentials, and large generated outputs out of Git.
- Use underscore-separated file and directory names.

## Experimental discipline

- Every reported result must identify the Git commit, exact command, machine
  fingerprint, model revision, input manifest, and output location.
- Do not combine timing or memory measurements from different machines in one
  performance curve unless the hardware is explicitly an experimental factor.
- Prefer independent oracles and hand-verifiable fixtures. Do not use
  BrainSurgery itself to generate both the tested output and its reference.
- Separate lossless preservation claims from tolerance-based claims for lossy
  arithmetic or dtype conversions.
- Preserve failures and negative results. Never silently replace a failed run
  with a rerun in the same run directory.
- Any change to `brainsurgery/*` remains subject to the root change-discipline
  and model-special-casing policies.

## Plans and status

- `plans/revision_plan.md` is the reviewer-concern and prioritization map.
- `plans/execution_plan.md` separates work that runs locally on macOS from
  work that requires the Linux/CUDA backend.
- Update the applicable plan checkbox and its evidence link when a work item is
  completed.
- Plans must contain repository-relative paths and no credentials, private
  reviewer identities, or machine-local absolute paths.

## Test-area contracts

- `correctness/`: semantic correctness and unintended-change detection.
- `robustness/`: malformed input, assertion, interruption, and publication
  failure behavior.
- `scaling/`: time, memory, I/O, sharding, and checkpoint-size experiments.
- `downstream/`: task or behavioral quality after intentionally lossy edits.
- `competing_tools/`: comparisons on genuinely overlapping operations.
- `behavioral/`: prompt-suite provenance and behavioral regression tests.

Each area README defines its intended evidence and stop conditions. Add a local
`AGENTS.md` before an area develops stricter operational rules.
