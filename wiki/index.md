---
status: active
last-confirmed: 2026-07-23
owners: agents
confidence: high
---

# Wiki Index

Validated-by: repo inspection of `wiki/` on 2026-07-23.

## Core Pages

- [index.md](index.md): Human-readable catalog of wiki pages and policy entry points.
- [memory.md](memory.md): LLM Wiki v2 memory tiers, lifecycle, metadata, relationships, automation hooks, governance, and maintenance checklist.
- [axon-compiler-policy.md](axon-compiler-policy.md): No definition-name special-casing policy for Axon compiler/runtime stages.
- [scripts.md](scripts.md): Current inventory and operating notes for `../scripts/`.
- [roundtrips.md](roundtrips.md): Axon stage roundtrip definitions, weak/strong contracts, and pytest integration.
- [benchmarks.md](benchmarks.md): Benchmark execution/reporting conventions and canonical log layout.
- [model-family-gap-plan.md](model-family-gap-plan.md): Exhaustive implementation/evidence ledger for text and specialized-generation model-family mappings.
- [log.md](log.md): Append-only chronology of durable wiki updates and operational events.

## Policy Entry Points

- [../AGENTS.md](../AGENTS.md): Root contributor and agent conventions, including global LLM Wiki policy.
- [AGENTS.md](AGENTS.md): Wiki-local operating model and quality rules.
- [../scripts/AGENTS.md](../scripts/AGENTS.md): Script and benchmark execution conventions.

## Maintenance

- Every non-AGENTS wiki page must be listed here.
- Every maintained script in `../scripts/` must be documented in [scripts.md](scripts.md).
- Dated operational outcomes belong in [log.md](log.md); repeated patterns should be promoted into topic pages.
