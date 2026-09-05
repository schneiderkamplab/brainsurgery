---
paths:
  - "brainsurgery/synapse/**"
---

# Synapse / Axon rules

Full policy: `brainsurgery/synapse/AGENTS.md`, `brainsurgery/synapse/axon/AGENTS.md`,
`brainsurgery/synapse/models/AGENTS.md`, `brainsurgery/synapse/builtins/AGENTS.md`.
Read the one that governs the file you are editing before changing it.

@../../brainsurgery/synapse/AGENTS.md

## Non-negotiable

- No model-family names, HF namespace quirks, or checkpoint-path routing in
  `axon/*`, `ops/*`, `builtins/*.axon`, `runtime*`, `pipeline*`, or `codegen*`.
  Model quirks live only in HF loading entrypoints (`axon_test.py`,
  `axon/tokenization.py`, `transforms/infer_runtime.py`).
- Do not infer semantics from definition names. Optimizations use typed AST,
  Graph IR structure, primitive ops, provenance, or constraints.
- Tensor dimensions come from `TypeExpr` only. No parallel shape metadata on AST nodes.
- No parser-side semantic rewrites (eta expansion, constant folding).
- Backend-specific `__<backend>_*` intrinsics only from explicit opt-in graph
  optimization for that backend, never from backend-neutral paths.
- Builtins must not use model-specific absolute default paths (`@@...`) in signatures.

## Models

- `generic-<family>.axon` is the source of truth. Edit it, then rematerialize
  with `scripts/rematerialize_all_generic.sh`. Do not hand-edit materialized files.
- Model files use builtins, never raw primitive `_op` calls.
- Do not reintroduce deprecated Compat-style calls where a current builtin exists.

## Verify

- `pytest -q tests/test_agents_policy_guards.py` after any edit here.
- Roundtrip tests for compiler-stage changes; see `skills/roundtrip/SKILL.md`.
- For fidelity-affecting changes, state which backends you validated and how.
  Grammar, typing, lowering, or runtime semantic changes need explicit user approval.
