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

## [2026-07-06] codegen2-vllm | native-vs-generated throughput parity confirmed on longer generation

- Confirmed codegen2-vllm generated models match native vLLM throughput on longer
  generation lengths (max_tokens=256, seq_len=1024, TP=2, CUDA graphs, `ignore_eos=True`,
  dummy weights). Runner: `/tmp/opencode/bench_long_gen.py` (mirrors
  `scripts/benchmark_vllm_throughput.py:run_benchmark` but forces `ignore_eos` and
  auto-forces `TRITON_ATTN` for heterogeneous head dims).
- gemma-4-E2B (`brainsurgery/synapse/models/gemma4/gemma-4-E2B.axon`, global_head_dim=512,
  TRITON_ATTN): generated/native ratio 1.000 (bs=1), 1.000 (bs=4), 1.001 (bs=16).
  Results: `/tmp/opencode/e2b_long_gen.json`.
- gemma-4-31B (`brainsurgery/synapse/models/gemma4/gemma-4-31B.axon`): generated/native
  ratio 1.025 (bs=1), 1.048 (bs=4), 1.031 (bs=16) — generated matches/slightly exceeds native.
  Results: `/tmp/opencode/31b_long_gen.json`.
- Note: the 31B generated file (`/tmp/opencode/generated_vllm_gemma4_31b.py`, dated
  2026-07-05 20:32) was STALE — it predated the last `codegen2_vllm/core.py` edit
  (21:22) that produced the matching E2B file (21:23). Regenerated 31B with current
  codegen via `/tmp/opencode/gen_31b.py` before benchmarking. Future runs: regenerate
  model code after any codegen change before benchmarking.
- Caveat: the 31B config omits `global_head_dim` (matching the original 31B validation),
  so both native and generated run homogeneous head_dim=256; the perf comparison is
  apples-to-apples on identical configs. Real 31B has heterogeneous 256/512 head dims.

## [2026-07-06] run_axon_benchmark | Axon-vs-HF fidelity confirmed for gemma4 E2B + 31B

- Ran `run_axon_benchmark` (Axon `codegen2-torch` backend vs HF transformers, `dtype=float32`)
  to cross-check the vLLM throughput results with correctness/fidelity data.
- gemma-4-E2B (real weights, `google/gemma-4-E2B-it`, ~4B params, single A6000):
  `masked_top1_eq=True`, `masked_max_abs_diff=5.7e-05`.
  Generate: HF 23.69 tok/s, Axon 23.29 tok/s (Axon/HF 0.983).
  Log: `log/gemma4-e2b-axon-vs-hf/stream.csv`. Runner: `scripts/run_gemma4_e2b_benchmark.py`.
- Gemma4-Dense-Test (test checkpoint from `google/gemma-4-31B`, 37M params, 4 layers):
  `masked_top1_eq=True`, `masked_max_abs_diff=3.2e-06`.
  Generate: HF 158.39 tok/s, Axon 204.34 tok/s (Axon/HF 1.290).
  Log: `log/gemma4-31b-test-axon-vs-hf/stream.csv`. Runner: `scripts/run_gemma4_31b_test_benchmark.py`.
  Note: 31B real weights (62B params) do not fit on a single A6000 in float32; test checkpoint
  used for fidelity. Timing ratio not representative of full-scale 31B.
- Gemma4-E-Test (test checkpoint from `google/gemma-4-E2B`, 308M params, 4 layers):
  `masked_top1_eq=True`, `masked_max_abs_diff=1.7e-06`.
  Generate: HF 133.93 tok/s, Axon 167.81 tok/s (Axon/HF 1.253).
  Log: `log/gemma4-etest-axon-vs-hf/stream.csv`. Runner: `scripts/run_gemma4_etest_benchmark.py`.
- All three pass fidelity (`masked_top1_eq=True`, max abs diff well below 1e-3 threshold).
  The vLLM throughput parity (ratio 1.000–1.048) and the HF fidelity (`masked_top1_eq=True`)
  together confirm: codegen produces both correct and performance-equivalent model code.
- bfloat16 note: `run_axon_benchmark` with `dtype="bfloat16"` fails on E2B with
  `RuntimeError: expected scalar type Float but found BFloat16` (RMSNorm `cast_float=true`
  upcasts to float32, mixing with bfloat16 tensors). Using `dtype="float32"` (the default)
  works. This is a known codegen2-torch dtype-mixing issue, not a vLLM-path issue.
