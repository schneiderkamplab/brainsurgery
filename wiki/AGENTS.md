# wiki/AGENTS.md

This file defines the "llmwiki" operating model for this repository.

Global repo policy lives in `../AGENTS.md` and applies here.

## Purpose

- `wiki/` is persistent agent memory and overview space.
- It is a maintained knowledge layer between chat history and source code.
- It stores durable context: architecture summaries, recurring failure classes, benchmark/reporting playbooks, and script operating notes.

## Core Model (LLM Wiki v2)

Four knowledge tiers:

1. Raw artifacts: source code, logs, test outputs, benchmark CSVs, and external references.
2. Episodic memory: dated operational observations in `wiki/log.md`.
3. Semantic/procedural memory: synthesized topic pages in `wiki/*.md`.
4. Schema/rules: workflow and quality constraints in `AGENTS.md` files.

The goal is accumulation: do not rediscover the same context in every thread.

## Memory Lifecycle

- Capture:
  - Add wiki notes when knowledge is likely to matter across sessions: recurring bugs, benchmark protocols, script contracts, stage invariants, model-family migration facts, and settled decisions.
  - Do not capture transient command output unless it explains a durable behavior or run result.
- Consolidate:
  - Prefer updating an existing topic page over creating a near-duplicate page.
  - Move repeated `wiki/log.md` observations into a topic page when they become a reusable rule or runbook.
- Validate:
  - Link claims to source files, commands, logs, tests, or external references.
  - Include dates for environment-dependent facts and benchmark outcomes.
  - Mark confidence explicitly when a claim is inferred rather than directly observed.
- Retire:
  - Do not silently delete contradicted knowledge. Mark it `superseded-by`, `fixed-by`, or `obsolete as of YYYY-MM-DD` unless it is pure duplication.
  - Remove stale operational steps only after the replacement path is documented.

## Wiki Page Metadata

Use this lightweight header when a page contains durable operational knowledge:

```md
---
status: active
last-confirmed: YYYY-MM-DD
owners: agents
confidence: high|medium|low
---
```

- `status`: `active`, `draft`, `superseded`, or `archived`.
- `last-confirmed`: date of last validation against code, tests, or logs.
- `owners`: usually `agents`; name a human/team only if explicitly established.
- `confidence`: `high` for directly validated facts, `medium` for strong inference, `low` for hypotheses or TODOs.

## Operating Workflows

- Ingest:
  - When new long-lived facts appear (new run class, recurring error class, script contract), update relevant wiki pages.
  - Keep entries short, factual, and link to concrete files/commands.
  - Record `validated-by`, `caused-by`, `fixed-by`, `depends-on`, or `supersedes` relationships where they clarify impact.
- Query:
  - For repeated tasks, check `wiki/` first, then execute.
  - Promote useful answers into wiki pages instead of leaving them only in chat.
  - Prefer exact paths, commands, and log IDs over memory from prior conversation.
- Lint:
  - Periodically remove stale notes, merge duplicates, and add missing cross-links.
  - Flag contradictions explicitly with date and resolution status.
  - Ensure every maintained script in `../scripts/` has a `wiki/scripts.md` entry.
  - Ensure `wiki/index.md` lists every non-AGENTS wiki page with a one-line purpose.

## Required Files

- `wiki/index.md`: content index of wiki pages (one-line summary each).
- `wiki/log.md`: append-only operational log (`## [YYYY-MM-DD] <event>` format).
- `wiki/scripts.md`: inventory of maintained scripts in `../scripts/`.
- `wiki/memory.md`: LLM Wiki v2 page conventions, lifecycle details, relationship vocabulary, and maintenance checklist.
- `wiki/axon-compiler-policy.md`: Axon compiler/runtime no-special-casing policy.
- `wiki/roundtrips.md`: staged Axon roundtrip contracts and test/script mapping.
- `wiki/benchmarks.md`: benchmark execution and reporting runbook.

## Scripts Ownership (Repo-Specific)

- `scripts/` is fully under agent control for automation.
- Every script that is created/changed must be documented in `wiki/scripts.md`:
  - path
  - purpose
  - invocation examples
  - required env vars
  - outputs/artifacts
  - known failure modes

## Quality Bar

- No speculative notes presented as facts.
- Prefer links to source files/commands over prose-only memory.
- Keep wiki pages composable: one topic per page, with cross-links.
- Keep pages concise enough to scan. Put raw output in `log/` or `tmp/`, not in wiki pages.
- Wiki edits should be behaviorally inert unless paired with explicit code/test changes.
- If a wiki claim changes how agents should work, update the nearest applicable `AGENTS.md` too.
- Do not store secrets, private tokens, credentials, or unnecessary PII.

## Axon Semantic Attribution

- Do not infer Axon semantics from definition/module names such as
  `Attention.*`, `Positions.*`, or `Cache.*`. Axon definitions are ordinary
  user definitions unless they lower to explicit primitives or validated graph
  intrinsics.
- Compiler/runtime optimizations must prove semantics from typed AST/Graph IR
  structure, primitive operations, provenance facts, constraints, and domain
  facts. Names may be used for display, diagnostics, and import resolution, but
  not as semantic evidence.
- Backend-specific graph intrinsics such as `__torch_*` must be introduced only
  by opt-in graph optimization for a compatible backend, and only after a
  provenance/structure proof over actual primitive computations. Backend-neutral
  graph optimization must not emit backend-specific intrinsics.

## Benchmark Reporting Standard

- Use the repository standard 3-table format for benchmark updates:
  1. Overview (total/completed/completion %, errors, >=1e-3 count, elapsed, ETA)
  2. Exceptions (`ERROR`, `masked_top1_eq != True`, or `masked_max_abs_diff >= 1e-3`)
  3. Unpaired (only when both generic/materialized exist and differ)
- Normalize paths in tables to concise repo-relative model paths.
- Do not include synthetic non-rows (for example `None | None | ...`).
- Treat `log/<run-id>` as the canonical location for new benchmark logs and
  reports. Do not create new top-level `log-*` artifacts.
- Record rerun outcomes as transient vs reproducible in `wiki/log.md` when relevant.
