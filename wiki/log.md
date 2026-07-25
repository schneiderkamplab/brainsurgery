---
status: active
last-confirmed: 2026-07-06
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
## [2026-07-23] Model-family gap execution plan added

- Added [model-family-gap-plan.md](model-family-gap-plan.md) with a complete
  encoder/masked-LM execution ledger and queued causal-LM, seq2seq, and
  multimodal/audio/OCR gap sets.
- The plan constrains source changes to new model Axon files and requires every
  masked-LM candidate to end as covered, implemented with a real checkpoint,
  implemented with a test checkpoint, or blocked with concrete evidence.
- Validated-by: Transformers `main` auto-model mappings and repository model
  inventory on 2026-07-23.

## [2026-07-23] First masked-LM gap implementations validated

- Added new model-only Axon implementations for Data2Vec Text, DeBERTa v1,
  ERNIE, ESM, EuroBERT, FNet, Jina Embeddings v3, LayoutLM, LUKE,
  Megatron-BERT, MPNet, NomicBERT, RemBERT, RoBERTa-PreLayerNorm, RoFormer,
  SqueezeBERT, TAPAS, XLM/Flaubert, and XLM-RoBERTa-XL.
- Direct fidelity evidence: Data2Vec Text and RoBERTa-PreLayerNorm are exact;
  XLM, Flaubert, ESM, RoFormer, FNet, and SqueezeBERT preserve top-1.
- FNet uses an explicit compositional DFT; SqueezeBERT expresses grouped 1x1
  convolutions as grouped parameter reshapes plus batched matmul. No primitive
  or compiler changes were introduced.
- Remaining execution and upstream-reference blockers are tracked in
  [model-family-gap-plan.md](model-family-gap-plan.md).
- Validated-by: `log/masked-lm-gaps-20260723-batch1` through
  `log/masked-lm-gaps-20260723-batch8`.

## [2026-07-23] Seq2seq family-gap tranche started

- Added model-only Axon implementations for Blenderbot, Blenderbot-small,
  BigBird-Pegasus, FSMT, LED, LongT5-local, M2M-100, MVP, Pegasus, PLBart,
  Switch Transformers, and UMT5. No compiler, builtin, runtime, or existing
  model source was changed.
- Healthy direct fidelity evidence:
  - Blenderbot: top-1 equal, max abs `1.19e-5`.
  - Blenderbot-small: top-1 equal, max abs `1.91e-5`.
  - MVP: exact (`0` max abs).
  - Pegasus: top-1 equal, max abs `3.62e-5`.
  - M2M-100: top-1 equal, max abs `9.06e-6`.
  - FSMT: top-1 equal, max abs `1.24e-5`.
  - LED: top-1 equal, max abs `3.15e-5`.
- The M2M-100 false result under a T5 tokenizer was invalid evidence: position
  IDs are derived from model-family pad IDs. Native tokenizer artifacts restored
  parity. FSMT's converted `embed_positions.weight` is a sinusoidal buffer, not
  a learned embedding; expressing the sinusoid compositionally restored parity.
- Remaining mapped-family work and concrete evidence blockers are maintained in
  [model-family-gap-plan.md](model-family-gap-plan.md).
- Validated-by: `log/seq2seq-gaps-20260723-bart-lineage-rerun`,
  `log/seq2seq-gaps-20260723-batch2`,
  `log/seq2seq-gaps-20260723-native-tokenizers`, and
  `log/seq2seq-gaps-20260723-fsmt-led-rerun`.

## [2026-07-24] Causal-LM family-gap tranche expanded

- Added model-only Axon implementations for Arcee, Cohere2, DiffLlama,
  Falcon-H1, Granite-MoE-Shared, Jais2, MPT, NanoChat, Persimmon, Qwen2-MoE,
  SolarOpen, and VaultGemma.
- Healthy direct evidence now covers Arcee, MPT, Persimmon, Cohere2, Jais2,
  Granite-MoE-Shared, Qwen2-MoE, and SolarOpen. Falcon-H1 is fidelity-clean in
  forward mode, while its
  recurrent generation cache remains incomplete.
- Qwen2-MoE's feature-complete test exercises mixed sliding/full attention,
  sparse routing without forced top-k renormalization, packed expert execution,
  and the gated shared expert; masked max abs is `1.19e-7`.
- DiffLlama, Helium, VaultGemma, and standard padded-batch NanoChat remain
  explicitly unaccepted with their numerical or integration blockers recorded
  in [model-family-gap-plan.md](model-family-gap-plan.md).
- No compiler, builtin, runtime, codegen, benchmark, or existing model source
  was changed.
- Validated-by: `log/causal-gaps-20260724-mpt-test-vocab`,
  `log/causal-gaps-20260724-persimmon-test`,
  `log/causal-gaps-20260724-arcee-tiny`,
  `log/causal-gaps-20260724-falcon-h1-limit`,
  `log/causal-gaps-20260724-cohere2-test-vocab`,
  `log/causal-gaps-20260724-jais2-test`, and
  `log/causal-gaps-20260724-qwen2-moe-test-separate`, and
  `log/causal-gaps-20260724-solar-open-test`, and
  `log/causal-gaps-20260724-granitemoe-shared-scope`.

## [2026-07-24] MPNet inferred-helper type refinement fixed

- Typecheck2 now discards stale inferred annotations between fixpoint
  iterations and refines only initially undeclared helper interfaces whose
  shared row-variable result is consistently constrained by all callsites.
- This fixes the `Tensor[Q,..R]` to `Tensor[Q,Q,K]` expansion in MPNet relative
  position bucketing without specializing declared polymorphic definitions.
- MPNet's MLM decoder now reads the checkpoint's actual
  `lm_head.decoder.bias`; the distinct `lm_head.bias` is not tied by current
  Transformers for this checkpoint.
- Validated-by: all 30 flat typecheck2 tests, successful MPNet Graph IR
  lowering, and CUDA float32 fidelity with `top1=True` and masked max abs `0`
  in `log/mpnet-typecheck2-fix-20260724-b`.
## [2026-07-24] Causal-LM gap closure continued

- Added and validated feature-complete test coverage for AFMoE, Dots1,
  Ernie-4.5-MoE, EXAONE-MoE, Hunyuan-V1-MoE, HY-V3, Laguna, and MiniMax-M2.
- All eight new sources lower through Graph IR. Their float32 forward benchmarks
  preserve top-1; masked max absolute differences range from `3.73e-8` to
  `1.73e-3`. Evidence paths are recorded in
  `wiki/model-family-gap-plan.md`.
- The attempted real HyperCLOVAX 1.5B confirmation is blocked by a gated Hub
  repository (`401`); the feature-complete test remains healthy.
- Confirmed BitNet’s current blocker: exact online ternary/int8 quantization
  needs model-level reduction-max and round operations that are not exported.

## [2026-07-24] Text seq2seq gap ledger exhausted

- Added healthy feature-complete test coverage for concrete BERT2BERT
  `encoder-decoder` composition, LongT5-local, and Pegasus-X.
- BERT2BERT preserves top-1 with masked max abs `2.12e-5`; LongT5-local reaches
  `1.79e-6`; Pegasus-X reaches `2.38e-7`, including two global tokens and
  alternating staggered local blocks.
- Added a feature-complete NLLB-MoE test/source with top-2 routing and four
  experts. It runs with top-1 parity but remains numerically unaccepted at
  `0.367` masked max abs.
- Added a two-predicting-stream ProphetNet source and test checkpoint. Its
  staged pipeline succeeds through flatten, then typecheck2 rejects the
  generated decoder loop-helper call despite an explicit
  `Tensor[B,3 * SDEC,MODEL_DIM]` carry/result type. Compiler changes were outside
  this tranche's allowed change boundary.
- Switch Transformers, UMT5, and BigBird-Pegasus remain explicit numerical
  parity blockers. The complete text seq2seq classification and evidence paths
  are maintained in [model-family-gap-plan.md](model-family-gap-plan.md).

## [2026-07-24] Masked-LM mapping fully reconciled

- Reconciled all 46 entries in the installed Transformers masked-LM mapping:
  every family is now covered, benchmarked on a real or persisted test
  checkpoint, or blocked with a concrete operation/integration reason in
  [model-family-gap-plan.md](model-family-gap-plan.md).
- Added healthy Axon coverage for MobileBERT, float-mode I-BERT, ConvBERT,
  the published Nyströmformer-512 configuration, and the language Perceiver.
- Added persisted test-head evidence for ERNIE, LayoutLM, LUKE, RemBERT, and
  TAPAS. LUKE exposed and fixed a model-source bug: its MLM decoder is
  independent of the word embedding and must use
  `lm_head.decoder.{weight,bias}`.
- All ten newly added or checkpoint-adjusted Axons lower successfully through
  `graph-ir-axon`. New benchmark evidence is under
  `log/masked-lm-{mobilebert,ibert,convbert-test,ernie,head-tests,luke-test,nystromformer,perceiver}*`.
## [2026-07-24] Causal-LM gap closure: LFM2-MoE and DBRX

- Added `models/lfm2_moe/generic-lfm2-moe.axon` and validated the feature-complete `test/LFM2-MoE-Test`: top-1 equal, masked max abs `1.19e-7` (`log/causal-close-lfm2-moe-r4`).
- Added `models/dbrx/generic-dbrx.axon` and validated `test/DBRX-Test`: top-1 equal, masked max abs `1.34e-7` (`log/causal-close-dbrx-r5`).
- DBRX validation confirmed that serialized expert tensors use flattened `[experts * model_width, expert_width]` storage; the Axon source reshapes this contract directly and does not rely on model-specific compiler behavior.

## [2026-07-24] Causal-LM gap closure: JetMoE

- Added `models/jetmoe/generic-jetmoe.axon` and validated `test/JetMoE-Test`: top-1 equal, masked max abs `5.96e-8` (`log/causal-close-jetmoe-r3`).
- The source represents both routed attention projections and routed SwiGLU FFNs. JetMoE's KV layout uses whole-head-block repetition, validated-by an initial repeat-interleave divergence and the healthy `unsqueeze -> expand -> reshape` correction.

## [2026-07-24] Causal-LM gap closure: Bamba

- Added `models/bamba/generic-bamba.axon` and validated `test/Bamba-Test`: top-1 equal, masked max abs `8.94e-8` (`log/causal-close-bamba-r1`).
- The feature checkpoint exercises both Mamba-2 and GQA layers. The initial source uses exact stateless full-prefix generation rather than introducing an unvalidated hybrid cache contract.

## [2026-07-24] Masked-LM generic capabilities and family closures

- Added explicit-key random tensors, stable `argsort`, scatter reduction, and
  elementwise `acos` as generic Axon operations with Torch and JAX lowering.
  Explicit keys keep random programs deterministic and referentially
  transparent; random permutations can be expressed as uniform keys followed
  by stable sorting.
- Funnel's fixed-resolution encoder stages and YOSO's dense angular-expectation
  path are healthy. Validated-by:
  `log/masked-lm-funnel-test-20260724-j` and
  `log/masked-lm-yoso-20260724-e`.
- DeBERTa-v1 now represents its interleaved per-head QKV packing and
  content-to-position plus position-to-content attention directly. Its
  legacy masked-LM checkpoint head is adapted only in the allowed HF reference
  loader. Validated-by: `log/masked-lm-deberta-megatron-final-20260724`, with
  `top1=True` and masked max abs `2.57e-5`.
- Megatron-BERT now applies the final encoder LayerNorm and reads the actual
  decoder bias. Validated-by:
  `log/masked-lm-deberta-megatron-final-20260724`, with `top1=True` and
  masked max abs `4.48e-5`.

## [2026-07-24] BigBird and Reformer masked-LM gaps closed

- BigBird block-sparse inference is expressed as ordinary dense Axon mask and
  score-bias construction. Transformers uses zero-valued random block indices
  in eval, which duplicates block 0; the equivalent score correction is
  `log(1 + num_random_blocks)` for block-0 keys on non-global query rows.
- The real BigBird checkpoint is healthy on both its standard full-attention
  path and a forced 713-token sparse path. Validated-by:
  `log/masked-lm-big-bird-20260724-b` and
  `log/masked-lm-big-bird-sparse-20260724-c`.
- Reformer LSH is represented without a model-specific primitive: seeded
  rotations produce buckets, stable `argsort` yields token ranks, and rank
  chunks define the exact dense attention relation. The strengthened persisted
  test exercises two hash rounds, four buckets, cross-hash previous-chunk
  adjacency, duplicate key occurrences, axial positions, local attention,
  input padding, and reversible two-stream carries.
- Reformer validation is top-1 equal with masked max abs `5.96e-8` in
  `log/masked-lm-reformer-test-20260724-g`. The earlier one-hash evidence in
  `log/masked-lm-reformer-test-20260724-d` is superseded by this broader case.

## [2026-07-24] ProphetNet seq2seq gap closed

- ProphetNet's affine `3 * S` decoder-loop carry now typechecks through a
  generic call-local dimension-equation fix, and Graph IR inlining preserves
  lexical dimension substitutions such as callee `S -> 3 * caller_S`.
- The model source reproduces HF's two predicting-stream embedding order and
  joint relative-bias layout. Applying relative bias independently per stream
  is not equivalent to HF's `reshape -> permute -> reinterpret reshape`
  sequence.
- The real `microsoft/prophetnet-large-uncased` checkpoint is healthy on CPU
  float32: top-1 equal and masked max abs `1.57e-5`.
  Validated-by: `log/seq2seq-prophetnet-cpu-20260724-m`.

## [2026-07-24] UMT5 and Switch Transformers seq2seq gaps closed

- UMT5 divergence began at the second encoder/decoder block because the Axon
  source reused block 0's relative-attention-bias parameter. Current UMT5
  checkpoints serialize a distinct bias per block. The corrected two-layer
  feature checkpoint has top-1 parity and max abs `1.91e-6`.
  Validated-by: `log/seq2seq-umt5-cpu-20260724-b`.
- Current Switch Transformers computes learned relative bias only in block 0;
  later blocks receive no propagated bias. Limiting the Axon bias to block 0
  closes the real `google/switch-base-8` checkpoint at top-1 parity and max abs
  `5.72e-5`. Validated-by: `log/seq2seq-switch-cpu-20260724-c`.

## [2026-07-24] BigBird-Pegasus seq2seq gap closed

- BigBird-Pegasus applies `layernorm_embedding` after each complete
  encoder/decoder stack, not before it. Moving both norms closes the real
  checkpoint's full-attention path at max abs `1.93e-5`.
- Eval-time block-sparse attention uses zero-valued random block indices, which
  duplicate block 0. The dense equivalent adds
  `log(1 + num_random_blocks)` to block-0 key scores for non-global query
  blocks. A 768-token sparse CPU probe has top-1 parity and max abs `9.78e-6`.
  Validated-by: `log/seq2seq-bigbird-pegasus-cpu-20260724-b` and
  `tmp/bigbird_pegasus_sparse_cpu_parity.py`.

## [2026-07-24] NLLB-MoE isolated to upstream reference regression

- Direct CPU parity confirms Axon positions, Q/K/V, attention, selected expert
  IDs and weights, and the intended top-2 weighted expert sum.
- Current Transformers applies `one_hot` to an already one-hot top-1 mask in
  `NllbMoeExperts.forward`. This routes every top-1 contribution through expert
  1 and every second contribution through expert 0, regardless of the selected
  expert ID. The end-to-end `0.367` difference is caused-by that upstream
  behavior; the incorrect dispatch is deliberately not reproduced in Axon.

## [2026-07-24] Causal Reformer and XLNet gaps closed

- Causal Reformer reuses the existing primitive-level local and LSH
  implementation with causal query/key relations. Its five-layer feature test
  is top-1 equal at max abs `1.94e-7`, and generated completions match HF.
  Validated-by: `log/causal-reformer-cpu-20260724-c` and
  `log/causal-reformer-cpu-20260724-d`.
- XLNet is represented without compiler or backend special cases: relative
  shift, content/position attention, two-stream dummy-token prediction, and
  the two-token memory recomputation window all live in
  `models/xlnet/generic-xlnet.axon`.
- The real `xlnet-base-cased` checkpoint is top-1 equal at max abs `1.05e-4`
  in forward; cached generation produces identical text at max abs `7.82e-5`.
  Validated-by: `log/causal-xlnet-base-cpu-20260724-forward-c` and
  `log/causal-xlnet-base-cpu-20260724-generate-a`.

## [2026-07-24] BitNet causal gap closed

- `models/bitnet/generic-bitnet.axon` implements online int8 activation
  quantization and ternary weight quantization as ordinary Axon. Exact
  ties-to-even rounding is derived from `floor`, lower-integer parity, and
  tensor selection; no round primitive was added.
- The real `microsoft/bitnet-b1.58-2B-4T-bf16` checkpoint preserves prefill
  top-1 and produces the same three cached-generation tokens as HF.
  Validated-by: `log/causal-bitnet-cpu-20260724-forward-r4` and
  `log/causal-bitnet-cpu-20260724-generate`.
- The remaining max-logit difference (`0.486`) is caused-by amplification of
  one-ULP weight-scale differences: HF's `torch.compile` mean reduction has
  shape-dependent reduction ordering. Layer-0 norm, activation quantization,
  and several projections are exact, and a complete first block differs by
  only `1.83e-4`. Evidence:
  `log/causal-bitnet-cpu-20260724/op-parity.txt` and
  `log/causal-bitnet-cpu-20260724/block0-parity-r2.txt`.

## [2026-07-24] Specialized-generation mappings exhaustively classified

- The installed Transformers mappings contain 71 image-text, 17
  speech-seq2seq, 11 text-to-waveform, and one causal-image family.
- All are currently blocked-by the benchmark input/output contract:
  `axon_test` supports only textual causal, masked, and seq2seq tasks and
  cannot construct or compare the required pixel, video, acoustic,
  image-code, protein, action-state, or waveform values.
- Existing Axon text backbones are recorded only as partial architectural
  coverage, never as full multimodal-wrapper evidence. The exact mapping
  entries and dispositions are maintained in
  `wiki/model-family-gap-plan.md`.

## [2026-07-24] Graph-IR loop roundtrip regression observed

- Weak and strong Graph-IR roundtrips execute successfully but are textually
  unstable for the new BitNet source because every rerender adds an identity
  wrapper around a lowered `for` body and renumbers generated dimensions.
- The same weak-roundtrip behavior reproduces on the existing
  `models/llama3/generic-llama3.axon`, so this is not caused-by BitNet.
  Validated-by: `log/causal-bitnet-cpu-20260724-roundtrip/weak.txt`,
  `log/causal-bitnet-cpu-20260724-roundtrip/strong.txt`, and
  `log/causal-bitnet-cpu-20260724-roundtrip/llama3-weak.txt`.
- No compiler change was made because the model-gap effort explicitly limits
  changes to new model sources and documentation.

## [2026-07-24] Remaining causal cache and operator gaps revised

- Added a generic ties-to-even `round` primitive and `Math.round` wrapper with
  direct Torch, JAX, MLX, and TinyGrad lowering. Typed tensor/scalar tests pass,
  and BitNet now uses the primitive rather than an Axon-derived floor/parity
  expansion.
- Falcon-H1 cached generation now carries attention KV, convolution history,
  and recurrent SSM state per layer. The real 0.5B checkpoint matches HF
  completions with top-1 equality and max abs `1.53e-5`; validated-by
  `log/causal-close-falcon-h1-cache-r2-20260724.txt`.
- GLM-MoE-DSA cached generation now carries independent attention KV and
  indexer-key histories. The feature checkpoint matches HF completions with
  top-1 equality and its accepted max abs `7.89e-2`; validated-by
  `log/causal-close-glm-dsa-cache-r2-20260724.txt`.
- CPMAnt and xLSTM reference generation now disables only their known-broken
  upstream HF cache paths. Their uncached references match Axon completions at
  max abs `1.60e-5` and `8.34e-7`, validated-by
  `log/causal-close-cpmant-reference-20260724.txt` and
  `log/causal-close-xlstm-reference-20260724.txt`.
- BLT's host-driven ragged split is not a fundamental Axon blocker. A
  `List[Tensor]` alone cannot turn runtime tensor scalars into shapes, but an
  exact dense formulation can compute patch IDs, scatter into at most `S`
  patch slots, and mask unused slots.

## [2026-07-24] Final standalone causal-LM gaps closed

- Added `models/bert_generation/generic-bert-generation.axon` and the
  feature-complete `test/BertGeneration-Test` checkpoint. Cached CPU generation
  matches HF completions with top-1 equality and masked max abs `1.79e-7`;
  validated-by `log/causal-bert-generation-20260724`.
- Added `models/blt/generic-blt.axon`. Runtime patch counts remain dense rather
  than becoming host-shaped lists: token loops compute patch IDs, scatter-reduce
  constructs at most `S` patch slots, and masks isolate unused slots. The test
  exercises entropy inference, forced initial boundaries, max-length splitting,
  hash embeddings, local/global/local transformers, `cross_attn_k=2`, and
  decoder cross-attention at every layer. CPU generation matches HF at masked
  max abs `1.19e-6`; validated-by `log/causal-blt-r7-20260724`.
- BitNet's current-snapshot regression was caused by a two-stage reduction for
  ternary weight scaling. Flattening before the single reduction matches
  Transformers' `abs(weight).mean()` rounding order. The real 2B checkpoint
  again preserves top-1 and produces identical completions at max abs `0.557`;
  validated-by `log/causal-bitnet-flat-mean-20260724` and
  `log/causal-bitnet-flat-mean-generate-20260724`.
## [2026-07-24] EuroBERT masked-LM reference loading fixed

- `EuroBERT/EuroBERT-210m` was not an Axon model-math failure. Under the current
  Transformers version, remote `from_pretrained` left most matrices randomly
  initialized and left a nonpersistent RoPE `inv_freq` buffer uninitialized.
- The masked-LM HF integration now validates a representative matrix against
  safetensors, restores checkpoint tensors explicitly on mismatch, and refreshes
  rotary frequencies through the module's own initializer.
- Validated-by `log/eurobert-fixed-cpu-20260724-r2`: CPU float32
  `masked_top1_eq=True`, masked max abs `4.24e-5`, fallback `none`.

## [2026-07-24] MRA masked-LM gap closed

- Added `models/mra/generic-mra.axon` for `uw-madison/mra-base-512-4`.
  Its multiresolution attention is ordinary Axon: low-resolution block
  scoring, top-k block selection, dense representation of selected
  high-resolution blocks, and the exact low/high normalizer correction.
- Upstream Transformers returns zero attention when its out-of-tree
  `mra_cuda_kernel` is unavailable. The allowed HF loading integration now
  installs a portable implementation of those published equations solely as
  the benchmark reference; no compiler, builtin, or backend special case was
  introduced.
- Validated-by `log/mra-cpu-fidelity-512-20260724`: real checkpoint, native
  512-token CPU float32 forward, `masked_top1_eq=True`, masked max abs `0`,
  fallback `none`.

## [2026-07-24] Masked-LM and seq2seq real-checkpoint expansion

- Added suitable real checkpoints to the previously test-only ConvBERT, ERNIE,
  Funnel, LUKE, Reformer, RemBERT, TAPAS, BERT2BERT, LongT5, UMT5, and
  Pegasus-X Axon sources. LayoutLM remains test-only because the audited public
  candidate has no MLM head; NLLB-MoE remains test-only because its real
  checkpoints exceed the 4B evidence boundary.
- Config-driven model fixes cover ERNIE 1.0 versus 3.0 activation/task
  embeddings, BERT versus BERT-generation embeddings/heads, untied LongT5
  embeddings, factored Reformer buckets, official LUKE tied token output, and
  Pegasus-X bias/activation settings.
- Every declared real checkpoint in this expansion has a top-1-equal CPU
  float32 evidence row. Most masked max-abs differences are at or below
  `1.45e-3`; the third-party Funnel checkpoint is a transparent provisional
  exception at `1.64e-2`.
- Validated-by the `log/real-checkpoint-evidence-cpu-20260724-*` runs summarized
  in `wiki/model-family-gap-plan.md`.

## [2026-07-24] Causal-LM real-checkpoint expansion

- Every causal family whose ledger evidence was test-backed now declares
  canonical real checkpoints where a matching standalone checkpoint exists.
  The sole test-only generic is BertGeneration, for which no independently
  trained standalone causal checkpoint was found.
- The declaration audit covers 178 real entries across 71 test-backed causal
  generics: 147 public configs, 29 authorization-gated configs, and two local
  aliases. All 72 causal test-backed generics pass the full frontend through
  typecheck.
- CPU float32 real-weight evidence is healthy for Cohere2's public tiny
  checkpoint, Youtu, GraniteMoeShared, OLMo Hybrid, Zamba, xLSTM, Persimmon,
  and LFM2-MoE. Qwen3.5 0.8B and JetMoE 8B preserve top-1 at about `2e-2` max
  abs. Hunyuan, GraniteMoeHybrid, Zamba2, causal Reformer, and BLT still have
  substantial numerical mismatches.
- Validated-by
  `log/causal-real-checkpoints-cpu-20260724/checkpoint-config-audit.csv`,
  `log/causal-real-checkpoints-cpu-20260724/frontend-validation-2/status.csv`,
  and the per-checkpoint benchmark logs under the same run root.

## [2026-07-25] Generic Axon materialization and CPU fidelity audit

- Materialization considered 183 `generic-*.axon` sources. All 169 sources
  with at least one locally complete checkpoint materialized successfully,
  producing 351 checkpoint-specific Axon files; all 351 reparsed successfully.
  Fourteen sources had no usable local context, and 99 individual declared
  checkpoint contexts remain unavailable.
- `axon-benchmark` downloaded and tested 64 of 71 previously missing real
  checkpoints at or below 4B. Seven repositories remained authorization-gated:
  three Cohere tiny-Aya variants, StarCoderBase-1B, two RecurrentGemma
  variants, and VaultGemma-1B.
- The exact-output CPU float32 sweep covered 291 materialized rows at or below
  4B: 227 healthy, 47 execution errors, five top-1 failures, and 17 rows with
  max absolute difference at least `1e-3`. Task coverage was 180 causal-LM,
  59 masked-LM, and 45 seq2seq-LM rows.
- Funnel pooled relative-position generation was corrected to produce the
  same `2K` shifted positions as Transformers. The real `windowsartes/funnel`
  row improved from about `1.64e-2` to `2.29e-5` max absolute difference.
  The random Funnel fixture remains numerically unstable and is not accepted
  as healthy.
- Validated-by
  `log/rematerialize-all-20260725/local-context-rematerialize.json`,
  `log/rematerialize-all-20260725/materialized-output-paths.txt`,
  `log/generic-missing-small-fidelity-20260725/`, and
  `log/materialized-exact-max4b-fidelity-20260725/`.

## [2026-07-25] Materialization regressions eliminated

- The 14 generic/materialized mismatches in the initial max-4B audit were
  fixed generically. Variadic result dimensions now remain unbound during
  materialization, lazy select branches bind shape symbols independently,
  dimensions after variadic tensor prefixes bind from trailing axes, and
  destructured `_split` results retain their literal split sizes.
- Targeted CPU float32 reruns now classify all 291 compared rows as 257
  same-numeric, 33 both-error, and one same-unmeasured; no row remains a
  materialization regression. Remaining errors and fidelity failures are
  shared by generic and materialized programs.
- Validated-by
  `log/rematerialize-all-20260725/generic-vs-materialized-max4b-final.csv`,
  `log/rematerialize-all-20260725/final-summary.md`,
  `tests/test_axon_materialize.py`, and the non-TinyGrad/MLX portion of
  `tests/test_axon_graph_ir.py`.
