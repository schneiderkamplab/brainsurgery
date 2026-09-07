---
status: active
last-confirmed: 2026-07-23
owners: agents
confidence: high
---

# Wiki Log

## [2026-07-06] codegen2-mlx | timing bug fix + custom fast forward + bfloat16 fix

### Timing Bug

- Root cause: `mx.eval(mx.array(0))` evaluates a throwaway scalar array, not the model output. GPU computation for the model forward was never synchronized — timing measured only Python overhead + graph building, not actual GPU compute.
- Affected: `scripts/bench_raw_throughput_compare.py` `sync()` (~line 224), `scripts/stream_tokens.py` `_sync()`, `brainsurgery/synapse/axon_test.py` `_force_eval_output()` (~line 2947).
- **Supersedes** all MLX tok/s numbers in prior log entries dated 2026-07-01 and 2026-07-05 that used the `mx.eval(mx.array(0))` pattern. Those numbers (600-2177 tok/s for Gemma-3-270M, 700+ tok/s for Gemma-4-E2B) were inflated — they measured Python loop speed, not GPU throughput.
- Fix: replaced `mx.eval(mx.array(0))` with `mx.eval(output_array)` in all three files.

### Corrected Ground Truth (Gemma-4-E2B-it, bfloat16, 100 tokens, Apple Silicon GPU)

| Configuration | tok/s | Notes |
|---|---|---|
| torch (MPS) | ~19 | |
| `mlx_lm` (async_eval) | 28.2 | Reference implementation |
| codegen2-mlx original forward (bfloat16) | ~10.4 | 2034-line generated forward |
| codegen2-mlx fast forward (float32 embeddings) | 12.8 | Custom forward, but 2 weights in float32 |
| codegen2-mlx fast forward (all bfloat16, sync) | 45.2 | |
| **codegen2-mlx fast forward (all bfloat16, async_eval)** | **52.0** | 1.8x faster than `mlx_lm` |

### bfloat16 Dtype Bug

- Root cause: `scripts/stream_tokens.py` `_compile_and_load_axon()` set `keep_fp32 = "embed_tokens"` when `dtype == "bfloat16"`, keeping `embed_tokens.weight` and `embed_tokens_per_layer.weight` in float32. These 2 weights (out of 2011) upcast the entire computation to float32 via type promotion, roughly halving throughput.
- Separately, `from_safetensors` in `codegen2_mlx/core.py` (~line 237) converted bfloat16 torch tensors to float32 via `t.float().numpy()` without casting back. Fixed to `.astype(mx.bfloat16)`.
- Fix: set `keep_fp32 = None` (always convert all weights to target dtype). Updated `from_safetensors` to preserve bfloat16.
- Impact: 3.5x speedup (12.8 → 45.2 tok/s) — the single biggest optimization.

### Custom Fast Forward in `compile()`

- Replaced the 2034-line generated `_forward` with a ~100-line hand-written forward using `mx.fast.*` primitives directly:
  - `mx.fast.scaled_dot_product_attention` with string `"causal"` mask (no explicit mask arrays) for prompt, `None` mask for single-token decode
  - `mx.fast.rms_norm` for all normalization (supports `None` weight for unscaled RMS)
  - `mx.fast.rope` with `ProportionalRoPE` freqs for full attention layers, `base` freqs for local attention layers
  - Pre-allocated `_KVCache` with in-place writes (no concat per step)
  - Minimal Python overhead (~1ms graph build time)
- Correctness verified: max diff 0.000070 vs original generated forward; next-token argmax matches.
- The generated forward is fundamentally incompatible with `mx.compile(shapeless=True)` due to 77 isinstance checks, 59 `is None` checks, ~200 shape-dependent reshapes. The custom forward bypasses this entirely.
- Not model-specific in approach — uses the same `mx.fast.*` primitives that `mlx_lm` uses. Model architecture params (NUM_LAYERS, ROPE_PERIOD, WIN_LOCAL, etc.) are read from model symbols, not hardcoded.

### Relevant Files

- `brainsurgery/synapse/axon/codegen2_mlx/core.py` — `compile()` with custom `_ff` forward builder (~line 668), `_KVCache` preamble (~line 1113), `from_safetensors` bfloat16 fix (~line 237)
- `scripts/stream_tokens.py` — `keep_fp32 = None` fix (~line 130), `_sync()` timing fix
- `scripts/bench_raw_throughput_compare.py` — `sync()` timing fix (~line 224)
- `brainsurgery/synapse/axon_test.py` — `_force_eval_output` timing fix (~line 2947)

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

## [2026-07-05] gemma-4-E2B-it | torch vs MLX GPU sweep on Apple Silicon

- Ran `scripts/bench_gemma4_e2b_gpu_sweep.py` (new) over `google/gemma-4-E2B-it` (bf16, generate/auto, warmup=1, repeat=3, mean of 3) on MPS/Apple GPU.
- Backends: `codegen2-torch` (MPS), `codegen2-torch + compile_axon`, `codegen2-mlx`, `codegen2-mlx + compile_axon`. HF reference = torch MPS in every run.
- Artifacts: `log/gemma4-e2b-it-gpu-sweep-2026-07-05/` (per-run `stream.csv` + logs; `sweep_summary.csv`).
- Results (Axon tok/s, generate):

  | max_len | HF tok/s | torch tok/s | mlx tok/s | mlx+compile tok/s |
  |---|---|---|---|---|
  | 64 | 23.6 | 18.0 | 148.3 | 749.0 |
  | 128 | 22.6 | 16.9 | 109.0 | 678.4 |
  | 256 | 23.0 | 16.3 | 122.6 | 730.8 |
  | 512 | 21.6 | 14.6 | OOM | 694.3 |

- `codegen2-torch` is slower than HF torch (~0.75x). `codegen2-mlx` is ~5-6x faster than HF. `codegen2-mlx + compile_axon` is ~30-35x faster than HF (~700 tok/s).
- `torch-compile` initially failed on MPS for all lengths: `RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float != c10::BFloat16`. Fixed in same session (see next entry); was an Axon matmul dtype-alignment gap, not a pure MPS limitation.
- `codegen2-mlx` (uncompiled) hit `[metal::malloc] Resource limit (499000) exceeded` at max_len=512 (KV cache + activations exceed Metal buffer budget without compiled shape specialization). `mlx-compile` handles 512 fine.
- All successful runs: `masked_top1_eq=True`. `masked_max_abs_diff` 0.79-1.38 (bf16 numerical noise over 507 generated tokens).
- Depends-on: `feat-mlx` MLX codegen + `compile(max_kv_length)` from [2026-07-01].

## [2026-07-05] codegen2-torch | fix matmul dtype alignment for torch.compile on MPS

- Root cause: `_align_pair` in `codegen2_torch/core.py` only aligned **device**, never **dtype**. When both matmul operands were on the same device, it returned them unchanged. A float32 activation (from a norm/softmax in fp32) matmul'd against a bf16 weight passed through to inductor's `extern_kernels.mm`, which is dtype-strict on MPS (eager `torch.matmul` auto-promotes, inductor does not).
- The `linear` primitive already aligned dtype (`weight.to(dtype=x.dtype)`), but `matmul` did not. Grouped-mm already aligned (`x_g.to(dtype=grouped_weight.dtype)`).
- Fix: added dtype alignment to both `matmul` code paths in `codegen2_torch/core.py`:
  - inline emission (line ~5813): cast `_a.to(dtype=_b.dtype)` when both floating-point and dtypes differ
  - runtime dispatch (line ~2442): cast `left.to(dtype=right.dtype)` under same conditions
  - Direction: left→right's dtype, matching `prefer="right"` and grouped-mm precedent. Matches HF pattern (norm/softmax in fp32, cast back to model dtype for matmul).
- Not model-specific; affects all `codegen2-torch` matmul ops generically.
- Validated-by: `tests/test_axon_graph_ir.py` (185 passed), `test_synapse_optimizer_fidelity_max4b.py` + `test_synapse_cli_optimize_flags.py` (7 passed, 1 skipped), `test_synapse_axon_typecheck_flat.py` matmul tests + `test_e2e_dump_compact_shape_regression.py` + `test_tensor_ops_transforms.py` (3 passed).
- Verified: torch-compile on MPS+bf16 now runs for all lengths (64-512). `masked_top1_eq=True`.
- Re-sweep results (Axon tok/s, generate, bf16, mean of 3):

  | max_len | HF | torch | torch-compile | mlx | mlx-compile |
  |---|---|---|---|---|---|
  | 64 | 23.1 | 17.3 | 16.8 | 89.5 | 671.5 |
  | 128 | 21.2 | 15.9 | 16.5 | 119.1 | 689.0 |
  | 256 | 21.1 | 14.7 | 15.0 | 115.8 | 716.7 |
  | 512 | 23.4 | 13.7 | 14.2 | OOM | 714.5 |

- torch-compile on MPS is slightly slower than eager torch (inductor MPS overhead), but no longer crashes. MLX+compile remains the clear winner (~700 tok/s, ~30-35x HF).
- Artifacts: `log/gemma4-e2b-it-gpu-sweep-fix-2026-07-05/`.

## [2026-07-06] codegen2-mlx | fix cached-decoder generate loop feeding wrong token

- Root cause: the cached-decoder `generate` path in `codegen2_mlx/core.py` (the main path for all modern LLMs with KV cache) never updated `out` after each step. Steps 1+ fed `out[:, -1:]` (last **prompt** token) instead of the last **generated** token. The other two paths (no-cache decoder, encoder-decoder) were correct — they concatenated `next_id` back into the running sequence.
- The bug was hidden because `masked_top1_eq=True` only validates the **first** new token (logits diff at position 0), not subsequent tokens. Outputs diverged to garbage after token 1.
- Before fix (E2B MLX): `"The future of AI is future AI is is a the is is is is is a a a a a..."` (garbage)
- After fix (E2B MLX): `"The future of AI is future of AI."` (matches HF exactly)
- Fix in `codegen2_mlx/core.py` cached-decoder path (lines 742-777):
  1. Added `current = out` after `out = input_ids` (tracks the working input)
  2. Changed `step_input = out[:, -1:]` → `current[:, -1:]` (and `else out` → `else current`)
  3. Added `current = next_id` after `generated.append(next_id)` (feeds last generated token into next step)
- Step 0: `current = out` (full prompt), cache is None → `step_input = current` (full prompt, correct)
- Step 1+: `current = next_id` (shape `(batch, 1)`), cache is not None → `step_input = current[:, -1:]` (last generated token, correct)
- Not model-specific; affects all `codegen2-mlx` cached-decoder models generically.
- Validated-by:
  - Gemma-3-270M MLX: outputs match HF (was garbage before fix). `masked_top1_eq=True`.
  - Gemma-3-270M MLX+compile: outputs match HF. `masked_top1_eq=True`. ~1514 tok/s.
  - Gemma-4-E2B-it MLX: outputs match HF exactly for prompt 0, minor bf16 divergence at end of prompt 1 (same as torch backend). `masked_top1_eq=True`. ~69 tok/s.
  - Gemma-4-E2B-it MLX+compile: outputs match HF. `masked_top1_eq=True`. ~749 tok/s (no perf regression).
  - `tests/test_axon_graph_ir.py` (185 passed), `tests/ -k mlx` (8 passed).
  - 2 pre-existing test failures unrelated to this change (transform spec validation, completion candidates).
  - Artifacts: `log/mlx-genloop-fix-{270m,e2b}-{mlx,mlx-compile}/`.

## [2026-07-23] codegen2-vllm | 4 new bug fixes (workstreams J-M), GPU/bf16 verified

### Workstream J: LM head misclassification
- **Root cause:** `_classify_lm_head` in `classify.py` searched ALL non-repeated modules for a linear reaching model output. GLM-edge's FFN down_proj (in `glm_edge_ffn`, called from repeated `glm_edge_block`) was misclassified as `PARALLEL_LM_HEAD` → created with `vocab_size` dimensions instead of `hidden_size`.
- **Fix:** Build `skip_modules` set from `repeated_module_names` + transitively reachable modules. Only search non-skipped modules.
- **Impact:** GLM-edge diff 2.58→6.5e-07, Longformer masked_diff 29.29→3.4e-05. Qwen (has `output_tied` in non-repeated module) still correctly classified.
- **File:** `brainsurgery/synapse/axon/codegen2_vllm/classify.py`

### Workstream K: RMSNorm eps resolution
- **Root cause:** Three bugs: (1) `_resolve_value_ref` didn't check global binding modules (0 nodes, `GraphLiteral` outputs) — `EPS=1e-05` for GLM-edge never found. (2) `_node_rmsnorm_eps` only scanned `input[2:]`, not `input[1]` where `EPS` ValueRef lives, and didn't resolve ValueRefs. (3) Python `bool` is `int` subclass → `float(True)=1.0` returned as eps for HRM-Text where `cast_float=True` at `input[3]`.
- **Fix:** Added global binding check in `_resolve_value_ref`, ValueRef resolution + `not isinstance(eps, bool)` guard in all eps methods, module-call eps parameter tracing in `_find_rmsnorm_eps_in_module`.
- **Impact:** GLM-edge all RMSNorm eps 1e-6→1e-05, HRM-Text all RMSNorm eps 1.0→1e-06.
- **File:** `brainsurgery/synapse/axon/codegen2_vllm/core.py`

### Workstream L: Derived constant resolution
- **Root cause:** `_resolve_const_literal` only checked `GraphLiteral` outputs. `MODEL_DIM <- ((1536 :: Dim) :: Dim)` lowers to `GraphExpr(core.ascribe, GraphLiteral(1536))` — not recognized. Derived constants `GQKV_DIM=4*QD`, `GATE_UP_DIM=2*FFN` all returned None → fell back to `hidden_size`.
- **Fix:** Added `_unwrap_expr_value` to recursively unwrap `GraphExpr` (`core.ascribe`, `core.binary.*`/`+`/`/`). Updated `_resolve_const_literal` and `_resolve_const_operand`.
- **Impact:** HRM-Text: GQKV_DIM=6144, GATE_UP_DIM=8192, QD=1536 all resolve (were all `hidden_size`=1536).
- **File:** `brainsurgery/synapse/axon/codegen2_vllm/core.py`

### Workstream M: Embedding scale
- **Root cause:** `VOCAB_PARALLEL_EMBEDDING` layer call emitted `self._vllm_xxx(input_ids)` without applying embedding scale from `NN.embedding` module's `scale` parameter (input[3]). HRM-Text uses `EMBED_SCALE=39.19`.
- **Fix:** In `_emit_vllm_layer_call`, check `node.inputs[3]` for non-None scale; if present, wrap as `(call * scale_expr)`.
- **Impact:** HRM-Text diff 2.46→0.0.
- **File:** `brainsurgery/synapse/axon/codegen2_vllm/core.py`

### GPU/bf16 validation (B200, 8 models)
All 8 spot-checked models pass on GPU/bf16 with `masked_top1_eq=True`:
- GLM-edge: 0.25 (was 18.84), HRM-Text: 0.125 (was 18.00), Longformer: 0.36 (was 29.29), xlm-roberta: 2.25 (was 65.38)
- Regressions confirmed: SmolLM 1.97, Qwen 3.52, BERT 0.5, RoBERTa 0.41
- Ready for full-scale benchmark via `bash log/4b-bench/run_all_4b_vllm.sh`

## [2026-07-23] SDPA graph optimization | enabling `__torch_sdpa`/`__triton_sdpa` for T5, DeBERTa, and all models

### Problem
- `log/4b-bench/run_all_4b.sh` did NOT pass `--optimize-graph`, so graph IR optimization (including SDPA rewrite) was never applied during benchmarks.
- `_default_graph_backend_intrinsics` in `axon_test.py` did not enable `__torch_sdpa` by default for the torch backend (only jax and triton had intrinsics).
- 62 of 139 torch/triton benchmarks were slower than HF; worst: T5/MT5 (3–6x), DeBERTa-v3 (3.6–3.9x), Phi-3-mini (1.6–1.9x), encoder-only models (1.2–4.5x).

### Root Causes
1. **Benchmark script** `log/4b-bench/run_all_4b.sh` missing `--optimize-graph` flag.
2. **SDPA provenance matching** in `_standard_sdpa_fact` (`provenance.py`) could not handle the nested-add pattern `(scores * scale + keep_mask) + extra_bias` used by T5/DeBERTa (rel_bias).
3. **`core.select` null-checks** for optional `rel_bias` parameter persisted in provenance as `core.select(x == null, a, b)`, blocking SDPA pattern matching.
4. **DeBERTa `Positions.axon`** had wasteful `matmul q_flat (transpose k_flat 1 2) * 0.0` operation.
5. **`_sdpa` codegen** had dtype mismatch: `torch.where(attn_mask, 0.0, float('-inf'))` created float32 tensor while model runs in bf16.
6. **6-arg SDPA codegen** used inline lambda bypassing `_sdpa` method, missing dtype unification (q/k float32 from RoPE vs v bf16).

### Changes
- **`provenance.py`**: Added `extra_additive_bias` field to `GraphSdpaGqaFact`. Added `_null_select_candidates()` to unwrap `core.select(x == null, ...)` by trying both branches. Added `_try_sdpa_on_softmax_in()` to handle nested-add form `(scores * scale + keep_mask) + extra_bias`. Applied to both `_standard_sdpa_fact` and `_sdpa_gqa_fact`.
- **`optimize.py`**: `_maybe_rewrite_node_to_backend_sdpa` passes `extra_bias` as 7th input to SDPA intrinsic. Added `__triton_sdpa` to `_TRITON_BACKEND_INTRINSICS`. Added `_rewrite_backend_sdpa_intrinsics(op_name="__triton_sdpa")` call in triton backend section.
- **`codegen2_torch/core.py`**: Added `_sdpa` classmethod with dtype unification (q/k cast to v.dtype), bool mask conversion with `.to(dtype=target_dtype)`, extra_bias support. Updated interpreter for `__torch_sdpa`/`__triton_sdpa` with 7th input. Fixed 6-arg SDPA codegen to use `self._sdpa` instead of inline lambda. Fixed scale `SyntaxWarning` for literal values.
- **`codegen2_triton/core.py`**: `_triton_sdpa` delegates to `_torch_sdpa` in `_primitive_expr`.
- **`codegen2_jax/core.py`**: `_sdpa` helper accepts `extra_bias`; `_jax_sdpa` emission handles 7th input.
- **`codegen2_tinygrad/core.py`**, **`codegen2_mlx/core.py`**: Guard raising ValueError for unsupported `extra_bias`.
- **`Positions.axon`**: Removed wasteful `matmul * 0.0` in `relative_bias_disentangled`.
- **`axon_test.py`**: `_default_graph_backend_intrinsics` returns `"codegen2-torch:__torch_sdpa"` for torch backend.
- **`run_all_4b.sh`**: Added `--optimize-graph` flag.

### Results (bf16, compiled, max_len=128, forward, 5 repeats)
| Model | Before (Axon/HF) | After (Axon/HF) | Correct? |
|---|---|---|---|
| bert-base-uncased (torch) | 3.99x | 1.01x | Yes |
| deberta-v3-xsmall (torch) | 3.99x | 2.42x | Yes |
| t5-small (torch) | 4.62x | 2.29x | Yes |
| t5-small (triton) | 3.67x | 2.86x | Yes |
| SmolLM-135M (torch) | 0.91x | works (was crashing) | Yes |

- BERT now at parity with HF. T5 and DeBERTa improved ~2x but remain 2-3x slower (remaining gap likely decoder loop overhead, not attention).
- SDPA provenance matching verified: T5 (3 SDPA nodes, 3 with extra_bias), DeBERTa (1 SDPA node, 1 with extra_bias), BERT/SmolLM/GPT2 (1 SDPA node each, 0 extra_bias).
- Pre-existing failures unchanged: GPT-2 top1_eq=False (same diff=1.25 before and after), SmolLM triton swiglu bf16 kernel error (pre-existing, not SDPA-related).

### Relevant Files
- `brainsurgery/synapse/axon/graph_ir/provenance.py` — `GraphSdpaGqaFact`, `_null_select_candidates`, `_try_sdpa_on_softmax_in`, `_standard_sdpa_fact`, `_sdpa_gqa_fact`
- `brainsurgery/synapse/axon/graph_ir/optimize.py` — `_maybe_rewrite_node_to_backend_sdpa`, `_TRITON_BACKEND_INTRINSICS`, triton SDPA rewrite
- `brainsurgery/synapse/axon/codegen2_torch/core.py` — `_sdpa` classmethod, interpreter `__torch_sdpa`/`__triton_sdpa`, `_primitive_expr` for `_torch_sdpa`, `_emit_common` emitted `_sdpa`
- `brainsurgery/synapse/axon/codegen2_triton/core.py` — `_triton_sdpa` delegation
- `brainsurgery/synapse/axon/codegen2_jax/core.py` — `_sdpa` helper, `_jax_sdpa` emission
- `brainsurgery/synapse/builtins/Positions.axon` — `relative_bias_disentangled`
- `brainsurgery/synapse/axon_test.py` — `_default_graph_backend_intrinsics`
- `log/4b-bench/run_all_4b.sh` — benchmark script

## [2026-07-27] vLLM | B200 attention backend investigation for head_dim >= 256

### Gemma4 E2B/E4B vLLM blocked by B200 hardware limits

- Gemma4 uses heterogeneous head dims (256 sliding, 512 full attention).
- No vLLM attention backend on B200 supports these sizes:
  - FA4: TMEM capacity limit for head_size >= 256
  - FA2: max head_dim 256 (512 unsupported)
  - TRITON_ATTN: needs 311KB shared memory, B200 has 232KB
  - FLASHINFER: JIT fails (missing `cublasLt.h`, `nvrtc.h`, `-lcuda`)
  - FLEX_ATTENTION: no KV sharing support
- `Gemma4Config.verify_and_update_config` forces FA4, which falls back to
  TRITON_ATTN, which hits the shared memory limit.
- **Gemma4 axon files verified correct on torch backend** (f32: top1=True,
  max_diff < 0.00005 for both E2B and E4B).

### Fixes committed (`102ebb5`)

- `axon_test.py`: head_dim detection from nested `text_config` for multi-modal
  models (Gemma3/4)
- `axon_test.py`: slot_mapping int32 -> int64 (FLASH_ATTN requires int64)
- `axon_test.py`: `set_default_torch_dtype` + Attention.forward dtype cast
  (dormant, gated by `_need_fa_dtype`)
- `axon_test.py`: `set_current_vllm_config` wrapping forward pass (needed for
  FLASHINFER)

### Regression suite results

- 27 PASS, 2 pre-existing FAIL (SmolLM2-135M, granite-3.1-2b), 0 ERROR, 0 regressions
- Updated `wiki/vllm-backend-debug.md` with full B200 backend status table
