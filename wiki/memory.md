---
status: active
last-confirmed: 2026-05-20
owners: agents
confidence: high
---

# LLM Wiki v2 Memory Model

This page defines the repository-local memory conventions used with `wiki/AGENTS.md`.

Validated-by: LLM Wiki v2 gist, root `AGENTS.md`, `wiki/AGENTS.md`, and repo inspection on 2026-05-20.

## Knowledge Tiers

- Raw artifacts: source files, tests, benchmark CSVs, generated logs, command transcripts, and external references. These are evidence, not wiki memory.
- Working memory: current-turn observations that have not yet been consolidated. Usually stays in chat or `tmp/`.
- Episodic memory: dated events and outcomes in `wiki/log.md`.
- Semantic memory: stable facts, invariants, failure classes, and architecture notes in topic pages.
- Procedural memory: runbooks, script contracts, benchmark workflows, and maintenance checklists.
- Rules/schema: `AGENTS.md` files that govern agent behavior.

## Lifecycle

- Capture:
  - Record only knowledge likely to matter in future sessions.
  - Use dated log entries for one-time outcomes and topic pages for reusable knowledge.
- Consolidate:
  - Promote repeated log observations into semantic/procedural pages.
  - Merge duplicate claims instead of growing parallel notes.
- Validate:
  - Attach `validated-by` evidence: repo path, command, log run ID, test, or external source.
  - Use `last-confirmed` dates for environment-dependent or fast-moving facts.
- Retire:
  - Mark outdated content with `superseded-by`, `fixed-by`, or `obsolete as of YYYY-MM-DD`.
  - Do not silently remove contradicted knowledge unless it is pure duplication.
- Forget/deprioritize:
  - Keep transient bugs in `wiki/log.md`; only promote them if they recur or explain durable behavior.
  - Architecture decisions and workflow contracts decay slowly; one-off failures decay quickly.

## Page Metadata

Durable topic pages should begin with:

```md
---
status: active
last-confirmed: YYYY-MM-DD
owners: agents
confidence: high|medium|low
---
```

- `status`: `active`, `draft`, `superseded`, or `archived`.
- `last-confirmed`: date last checked against code, tests, logs, or an authoritative source.
- `owners`: usually `agents`; name humans only when explicitly established.
- `confidence`: `high` for validated claims, `medium` for strong inference, `low` for hypotheses/TODOs.

## Relationship Vocabulary

Use these labels in prose when they clarify impact:

- `uses`: direct invocation or consumption.
- `depends-on`: correctness requires another invariant or artifact.
- `validated-by`: evidence confirms the claim.
- `caused-by`: known root cause.
- `fixed-by`: change or process that resolved an issue.
- `supersedes`: newer guidance replaces older guidance.
- `superseded-by`: older guidance kept for history but not followed.
- `contradicts`: claim conflicts with another claim and needs resolution.

## Search And Navigation

- Keep `wiki/index.md` as the human-readable catalog.
- For small wiki size, `rg` over `wiki/` plus the index is sufficient.
- If the wiki grows beyond a few hundred pages, add generated search/index artifacts rather than expanding `index.md` into a huge document.
- Prefer entity/relationship wording in pages so future graph-style traversal is possible without rewriting all content.

## Automation Hooks

Current repository hooks are manual scripts, not daemonized automation.

- On script changes: update `wiki/scripts.md`.
- On benchmark/reporting changes: update `wiki/benchmarks.md` and append `wiki/log.md`.
- On staged pipeline contract changes: update `wiki/roundtrips.md` and relevant `AGENTS.md`.
- On model-family migration facts: add or update a model-family topic page before relying on chat history.
- On wiki writes: check `wiki/index.md` coverage and avoid storing secrets.

## Privacy And Governance

- Never write credentials, private tokens, API keys, or unnecessary PII into `wiki/`.
- Store raw logs under `log/` or `tmp/`; summarize only durable facts in wiki pages.
- Bulk wiki rewrites should leave an audit entry in `wiki/log.md`.
- Reversible changes are preferred: mark stale content before deleting it.

## Maintenance Checklist

- `wiki/index.md` lists every non-AGENTS wiki page.
- `wiki/scripts.md` documents every maintained script in `../scripts/`.
- `wiki/log.md` contains a dated entry for each nontrivial wiki policy rewrite.
- Benchmark summaries use the repository 3-table format and point to `log/<run-id>`.
- Operational claims that change agent behavior are mirrored in the nearest applicable `AGENTS.md`.
