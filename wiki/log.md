---
status: active
last-confirmed: 2026-05-20
owners: agents
confidence: high
---

# Wiki Log

## [2026-05-20] llmwiki v2 | lifecycle-managed memory policy

- Updated root `AGENTS.md` and `wiki/AGENTS.md` to align with LLM Wiki v2 conventions: memory tiers, lifecycle, confidence/source metadata, supersession, typed relationships, quality checks, and privacy filtering.
- Added `wiki/memory.md` as the repository-local reference for memory page metadata, relationship vocabulary, ingest rules, and maintenance checklist.
- Updated `wiki/index.md` to include the new memory conventions page.

## [2026-05-20] llmwiki v2 | stale wiki refresh from repo inspection

- Treated existing wiki content as stale and regenerated the core pages from current repository state.
- Rewrote `wiki/scripts.md` to cover all current scripts in `../scripts/`, including roundtrip scripts, benchmark reporting, checkpoint generators, materialization, targeted helpers, and legacy launch/render helpers.
- Added `wiki/roundtrips.md` for weak/strong stage roundtrip contracts and `wiki/benchmarks.md` for canonical benchmark execution/reporting conventions.
- Updated `wiki/index.md` and `wiki/AGENTS.md` to include the new required pages.

## [2026-05-20] axon policy | no definition-name special-casing

- Added `wiki/axon-compiler-policy.md` to make explicit that typecheck, optimize, lowering, codegen, runtime, and related Axon compiler stages must not special-case ordinary Axon definitions by name.
- Clarified the allowed boundary: primitive operation semantics may be encoded on primitives, while normal definitions such as `Config.dim`, `Tensor.size`, `NN.embedding`, and `Cache.update` must be handled through generic Axon/Graph IR rules.
- Updated `wiki/index.md` and `wiki/AGENTS.md` to include the policy page.

## [2026-04-20] init | bootstrap llmwiki structure

- Added `wiki/AGENTS.md`, `wiki/index.md`, `wiki/log.md`, and `wiki/scripts.md`.
- Established cross-references with root `AGENTS.md`.

## [2026-05-09] test checkpoints | regenerated min4B family fixtures

- Regenerated all 31 `models/test/*-Test` checkpoints in float32 with `scripts/create_min4_family_test_models.py`.
- Regenerated `models/test/DeepSeek-V4-Test` with `scripts/create_deepseek_v4_random.py`.
- `DeepSeek-V2-Test` now preserves the real DeepSeek-V2-Lite `q_lora_rank=null` / `q_proj` attention weight layout.
- `Phi-MoE-Test` now materializes locally via native Transformers `phimoe` plus checkpoint-key/config normalization for Axon parity.

## [2026-05-10] max32B benchmark | Phi-3-small-128k HF module-cache repair

- `log/max32b-v1-removal-p8` had two transient `microsoft/Phi-3-small-128k-instruct` error rows caused by a missing `tokenization_phi3_small.py` in the local HF dynamic-module cache revision `models/microsoft/.hf_modules_cache/transformers_modules/Phi_hyphen_3_hyphen_small_hyphen_128k_hyphen_instruct/4b6b36edd054c256/`.
- Repaired the local cache from `models/microsoft/Phi-3-small-128k-instruct/*.py`.
- Reran the two affected rows in `log/max32b-phi3small-128k-cache-rerun`: both materialized and generic rows passed with `masked_top1_eq=True` and max abs below `3e-05`.

## [2026-05-24] benchmark | model-family fidelity fix list

- `log/all-models-fullopt-hf-max4b-20260524-011933` completed the full `<=4B` full-opt HF comparison: 273 unique rows, 0 runtime errors, 3 top-1-bad rows, median Axon/HF time ratio `0.606`.
- `log/all-models-fullopt-hf-4to16b-20260524-014611` completed the `4B..16B` full-opt HF comparison: 178 planned rows, 0 runtime errors, 8 top-1-bad rows, median Axon/HF time ratio `0.323`.
- Durable fix list recorded in `docs/TODO.md`: fix family fidelity for `deepseekv2`, `deepseekv4`, `mt5`, and `olmo3`; separately resolve the `phi3small` generic-vs-materialized mismatch.

## [2026-05-24] benchmark reporting | four-table report contract

- Updated `scripts/benchmark_report_3tables.py` and the `report` skill to emit four standard benchmark tables: progress, issue rows, generic-vs-materialized comparison, and `Axon/HF >= 1.0` timing outliers.
- The progress table now includes timed row count, Axon faster count, and Axon/HF >= 1.0 count.
