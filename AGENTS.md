# AGENTS.md

This file defines global conventions for contributors and agents in this repository.
Detailed local rules live in subfolder `AGENTS.md` files.

Related local policies:

- `brainsurgery/AGENTS.md`: package-wide Python import/export boundaries.
- `wiki/AGENTS.md`: LLM wiki memory/process for this repo.
- `docs/AGENTS.md`: co-maintenance rules for user + agent docs work.
- `scripts/AGENTS.md`: benchmark/run execution conventions.
- `brainsurgery/synapse/*/AGENTS.md`: local boundaries for compiler/runtime/model work.
- `tests/AGENTS.md`: test-fix contract and escalation policy.

## Model Special-Casing Policy

- Model-specific special casing is allowed only in HF loading/config-adaptation paths.
- Do not add model-specific branches to parser, typechecker, lowering, codegen, runtime, or core builtins unless explicitly approved.
- If any such special casing is detected, report it and provide a concrete elimination plan (target module, refactor path, risk, validation).
- Allowed HF/model-quirk integration files are limited to loading/integration entrypoints:
  - `brainsurgery/synapse/axon_test.py`
  - `brainsurgery/synapse/axon/tokenization.py`
  - `brainsurgery/transforms/infer_runtime.py`
  - (and explicit future additions approved in review)
- Runtime/compiler layers (`brainsurgery/synapse/axon/*`, `runtime.py`, `pipeline_*`, `codegen.py`, `builtins/*.axon`) must not carry HF namespace quirks or model-family routing.
- Builtins must not use model-specific absolute default paths (`@@...`) in signatures.

## LLM Wiki Policy

- The `wiki/` directory is the persistent, compounding working memory for agents (adapted from the LLM-wiki pattern).
- Store operational overviews, runbooks, and script-maintenance notes there.
- Keep `scripts/` under agent control; document every maintained script in `wiki/` with purpose, CLI, inputs/outputs, and ownership.
- `wiki/AGENTS.md` contains the concrete ingest/query/lint workflow; treat this root file as the global policy layer.

## Benchmark and Reporting Delegation

- Benchmark execution policy is defined in `scripts/AGENTS.md`.
- Reporting format and durable run-memory conventions are defined in `wiki/AGENTS.md`.
- New benchmark/run artifacts must be written below repo-root `log/`
  (for example `log/<run-id>`), not as top-level `log-*` files or
  directories.
- Root policy requires consistency with those files but does not duplicate their details.

## Change Discipline

- If a change alters behavior in `brainsurgery/*`, get approval before landing substantial semantic changes.
- Keep edits minimal, conservative, and evidence-backed.
- Avoid compatibility shims unless explicitly requested.
- Prefer generic reusable implementations and avoid duplicated logic.
- Maintain policy guards in tests for special-casing boundaries (forbidden model-family/HF-quirk patterns in restricted layers).
