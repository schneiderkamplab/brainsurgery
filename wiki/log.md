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

## [2026-07-04] axon-benchmark | multi-backend rows and Rich monitor

- Added native ordered multi-backend support to `brainsurgery synapse axon-benchmark` via `--axon-backends`, emitting one stream CSV row per backend with a new `backend` column.
- Added `scripts/monitor_axon_benchmark.py` as a Rich live monitor for benchmark run directories, including recursive stream CSVs, parent logs, paired-runner status files, GPU memory/utilization, active jobs, and recent failures.
- Validated by `brainsurgery synapse axon-benchmark ... --axon-backends codegen2-torch,codegen2-jax --dry-run --stream-csv ...` and `pytest -q tests/test_synapse_axon_import_loader.py tests/test_synapse_cli_optimize_flags.py`.

## [2026-07-01] mlx codegen | 5 fixes + mx.compile implemented on feat-mlx

- Cherry-picked all 5 MLX codegen fixes from `feat-serving` to `feat-mlx`:
  1. `params_has_root` → `self._flat_tensors` (not torch parent's `state_dict_tensors`)
  2. `list_length`/`list_append` → `0 if x is None else len(x)` / `x if x is not None else []`
  3. `_path_template_part` → instance method with dict cache (`_path_cache`)
  4. `_param`/`_optional_param` → `dict.get(key)` early-exit before `_materialize_expert_bank_for_path`
  5. `use_cache`/`past_kv` `_to_mlx` skip in `forward()` and `_forward()`
- Implemented `compile(max_kv_length)` method in MLX codegen: wraps `_forward` with `mx.compile`, pre-compiles KV shapes 0..max_kv_length via dummy decode warmup.
- Updated `scripts/bench_raw_throughput_compare.py` with `mlx+compile` backend config.
- Results on `feat-mlx` (Gemma-3-270M, 3 trials, 64 decode steps, warmup 1100 shapes):

  | Backend | p=16 | p=64 | p=256 | p=512 | p=1024 |
  |---------|------|------|-------|-------|--------|
  | torch (MPS) | 42 | 42 | 41 | 40 | 40 |
  | mlx | 600 | 595 | 587 | 577 | 461 |
  | **mlx+compile** | **2091** | **2177** | **1858** | **1868** | **1840** |

- MLX+compile is ~50x faster than torch MPS, ~3.5x faster than MLX baseline.
- Warmup cost: ~6.1s for 1100 shapes. Per-shape compile: ~5.5ms.
- Memory cost: ~61 KB/shape (small KV) to ~567 KB/shape (large KV). 1100 shapes ≈ 40 MB active + cache.
- `feat-mlx` has `_mlx_rope` graph intrinsic bug (`bool()` on mlx array) when MLX backend intrinsics are enabled. Benchmark does not enable MLX intrinsics (matching `feat-serving` behavior).
- Depends-on: the `_to_mlx` skip fix (without it, `mx.compile` recompiles every step because `np.asarray()` creates new Python objects that break tracing).
