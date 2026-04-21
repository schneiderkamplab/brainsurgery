# wiki/AGENTS.md

This file defines the "llmwiki" operating model for this repository.

Global repo policy lives in `../AGENTS.md` and applies here.

## Purpose

- `wiki/` is persistent agent memory and overview space.
- It is a maintained knowledge layer between chat history and source code.
- It stores durable context: architecture summaries, recurring failure classes, benchmark/reporting playbooks, and script operating notes.

## Core Model (Adapted from LLM Wiki)

Three layers:

1. Raw artifacts: source code, logs, test outputs, and external references (immutable facts).
2. Wiki pages (`wiki/*.md`): synthesized, cross-linked operational knowledge (maintained by agents).
3. Schema/rules (`AGENTS.md` files): workflow and quality constraints.

The goal is accumulation: do not rediscover the same context in every thread.

## Operating Workflows

- Ingest:
  - When new long-lived facts appear (new run class, recurring error class, script contract), update relevant wiki pages.
  - Keep entries short, factual, and link to concrete files/commands.
- Query:
  - For repeated tasks, check `wiki/` first, then execute.
  - Promote useful answers into wiki pages instead of leaving them only in chat.
- Lint:
  - Periodically remove stale notes, merge duplicates, and add missing cross-links.
  - Flag contradictions explicitly with date and resolution status.

## Required Files

- `wiki/index.md`: content index of wiki pages (one-line summary each).
- `wiki/log.md`: append-only operational log (`## [YYYY-MM-DD] <event>` format).
- `wiki/scripts.md`: inventory of maintained scripts in `../scripts/`.

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
