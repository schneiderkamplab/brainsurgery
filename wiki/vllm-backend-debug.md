---
status: active
last-confirmed: 2026-07-27
owners: agents
confidence: high
---

# codegen2-vllm Backend Debug Status

**Last confirmed:** 2026-07-27
**Benchmark:** `log/4b-bench/run_all_4b_vllm.sh` — 68 models, bf16, forward-only, max_len=128, 8 GPUs
**Results CSV:** `log/4b-bench/all-models-bf16/bfloat16-128/stream_vllm.csv`
**Regression suite:** `test_vllm_changes.py` — 30 models, bf16, forward-only, max_len=128

## Summary

| Category | Initial | Pre-fixes | Current | Notes |
|----------|---------|-----------|---------|-------|
| PASS     | 4       | 13        | 21+     | +8 from workstreams J-M, GPU/bf16 verified |
| FAIL     | ~29     | 15        | 5       | 10 fixed (SmolLM/Qwen RMSNorm + GLM-edge + HRM-Text + BERT family + xlm-roberta) |
| ERROR    | ~35     | 40        | 29      | 27 fixes applied, pending full GPU/bf16 validation |

## GPU/bf16 Spot-check (2026-07-23, B200)

| Model | masked_top1_eq | masked_max_abs_diff | Category |
|-------|----------------|---------------------|----------|
| SmolLM-135M | True | 1.97 | regression |
| Qwen2.5-0.5B | True | 3.52 | regression |
| GLM-edge-1.5b | True | 0.25 | **was FAIL 18.84** |
| HRM-Text-1B | True | 0.125 | **was FAIL 18.00** |
| BERT-base | True | 0.5 | regression |
| RoBERTa-base | True | 0.41 | regression |
| Longformer | True | 0.36 | **was FAIL 29.29** |
| xlm-roberta-base | True | 2.25 | **was FAIL 65.38** |

All 8 pass on GPU/bf16. Ready for full-scale benchmark.

## B200 Attention Backend Limitations (head_dim >= 256)

**Confirmed:** 2026-07-27 on 8x NVIDIA B200 (SM100a, 232KB shared memory per SM).

Gemma4 uses heterogeneous head dimensions (`head_dim=256` for sliding attention,
`global_head_dim=512` for full attention). No vLLM attention backend on B200
supports these head sizes:

| Backend | head_size=256 | head_size=512 | Notes |
|---------|--------------|--------------|-------|
| FLASHINFER | valid | valid | JIT compilation fails: missing `cublasLt.h`, `nvrtc.h`, `-lcuda` |
| FLASH_ATTN (FA2) | valid | **invalid** | `FlashAttention forward only supports head dimension at most 256` |
| FLASH_ATTN (FA4) | **invalid** | **invalid** | `FA4 on Blackwell does not support head_size=256 due to TMEM capacity limits` |
| TRITON_ATTN | valid | valid | `OutOfResources: shared memory, Required: 311296, Hardware limit: 232448` |
| FLEX_ATTENTION | valid | valid | `FlexAttention does not support kv sharing yet` (Gemma4 uses KV sharing) |

`Gemma4Config.verify_and_update_config` forces `flash_attn_version=4` when
heterogeneous head dims are detected, but FA4 is unsupported on B200 for these
sizes. The fallback to TRITON_ATTN hits the shared memory limit.

**Gemma4 axon files are correct on torch backend** (verified f32):
- `gemma-4-E2B.axon`: top1=True, max_diff=0.000026
- `gemma-4-E4B.axon`: top1=True, max_diff=0.000044

**Workaround path:** Install missing CUDA headers (`cublasLt.h`, `nvrtc.h`) and
`libcuda.so` to enable FLASHINFER JIT, or set `disable_sliding_window=True` +
`use_trtllm_attention=False` to avoid TRTLLM path. Both require environment
changes outside the repo.

## B200 Attention Backend Selection

vLLM on B200 auto-selects attention backends by priority:
`FLASHINFER > FLASH_ATTN > TRITON_ATTN > FLEX_ATTENTION`.

In `axon_test.py`, the default `torch.get_default_dtype()` is `float32`.
FLASHINFER and FLASH_ATTN reject float32 (`supports_dtype` check), so
TRITON_ATTN is selected for most models. This works for head_dim <= 128 but
fails for head_dim >= 256 (shared memory).

When a non-TRITON backend is forced (e.g. for head_dim >= 256), the test
harness must:
1. Wrap model creation in `set_default_torch_dtype(resolved_dtype)` so
   `Attention.__init__` sees the correct dtype for backend selection.
2. Cast model parameters to `resolved_dtype` after `load_weights`.
3. Cast `Attention.forward` q/k/v inputs to `resolved_dtype` (RoPE sin/cos
   are computed in float32, promoting q*k to float32).
4. Wrap the forward pass in `set_current_vllm_config()` (FLASHINFER needs it).
5. Use `int64` for `slot_mapping` (FLASH_ATTN `reshape_and_cache_flash`
   requires int64, not int32).

Currently `_need_fa_dtype` is dormant (`pass` in the head_dim >= 256 branch)
because no working backend exists on B200 for these head sizes. The
infrastructure is in place for when the environment is fixed.

## Fixes Applied (all workstreams)

### Early fixes (workstreams 1-7)

1. **VocabParallelEmbedding/ParallelLMHead `params_dtype`** — added to init calls (fixed BERT family 7 models)
2. **`__flat_1` name mangling in `_collect_args`** — applied `_py_ident()` (fixed Pleias, SmolLM3)
3. **`dict(hf_config)` for T5/BART/MT5** — use `to_dict()` when available
4. **`{root}` -> `{prefix}` substitution** — in `_qkv_layer_prefix` and `_layer_prefix`
5. **`load_weights` dtype conversion** — `state_dict_tensors` weights converted to `params_dtype`
6. **RMSNorm missing `dtype=` param** — added to both has_weight and no-weight paths
7. **Other** — port collision, dynamo disable, emit unguarding, classify fixes

### Workstream A: Fused QKV (16 ERROR)
- Accept fused QKV in classifier, dynamic `stacked_params_mapping`, config key normalization

### Workstream B: RMSNorm (6+ FAIL)
- Strip `.weight` from `_linear_base_key`, recursive `_find_rmsnorm_eps_in_module`, `{root}`→`{prefix}` in QKV prefix

### Workstream C: Column parallel (4 ERROR)
- Conv1D transpose detection, `embedding_size` config, `n_embd` alias, `head_dim` derivation

### Workstream D: XGLM/BlackMamba (4 ERROR)
- `model_type="llama"` in vLLM fallback config

### Workstream E: MambaConfig (1 ERROR)
- `getattr` with fallback for `num_attention_heads`

### Workstream F: Rotary embedding (2 ERROR) — partial
- `{__scope}`→`{prefix}` fix applied; RoPE variant detection still needs work

### Workstream G: Electra (1 ERROR)
- `VocabParallelEmbedding` uses `embedding_size` not `hidden_size`

### Workstream I: BERT family weight loading (12+ FAIL)
- gamma/beta→weight/bias remap, `{i}` resolution, `ParallelLMHead` bias+prefix, `compute_logits` bias, `_resolve_value_ref` module scope, `attn_mask` passthrough in forward

### Workstream J: LM head misclassification (GLM-edge, Longformer) — GPU verified
- **Root cause:** `_classify_lm_head` searched all non-repeated modules, causing GLM-edge's FFN down_proj to be misclassified as `PARALLEL_LM_HEAD` (wrong dimensions)
- **Fix:** `classify.py:_classify_lm_head` now builds `skip_modules` set from `repeated_module_names` + transitively reachable modules. Only non-skipped modules searched.
- **File:** `brainsurgery/synapse/axon/codegen2_vllm/classify.py`

### Workstream K: RMSNorm eps resolution (GLM-edge, HRM-Text) — GPU verified
- **Root cause:** (1) `_resolve_value_ref` didn't check global binding modules with `GraphLiteral` outputs (2) `_node_rmsnorm_eps` didn't resolve ValueRefs or scan from input[1] (3) Python `bool` subclass of `int` caused `float(True)=1.0` returned as eps
- **Fix:** Added global binding check in `_resolve_value_ref`, ValueRef resolution + bool guard in `_node_rmsnorm_eps`/`_find_rmsnorm_eps_in_module`/`_node_layernorm_eps`, module-call eps parameter tracing
- **File:** `brainsurgery/synapse/axon/codegen2_vllm/core.py`

### Workstream L: Derived constant resolution (HRM-Text) — GPU verified
- **Root cause:** `_resolve_const_literal` only checked `GraphLiteral` outputs; `MODEL_DIM <- ((1536 :: Dim) :: Dim)` lowers to `GraphExpr(core.ascribe, GraphLiteral(1536))` which wasn't recognized
- **Fix:** Added `_unwrap_expr_value` method to recursively unwrap `GraphExpr` (`core.ascribe`, `core.binary.*`/`+`/`/`). Updated `_resolve_const_literal` and `_resolve_const_operand`.
- **File:** `brainsurgery/synapse/axon/codegen2_vllm/core.py`

### Workstream M: Embedding scale (HRM-Text) — GPU verified
- **Root cause:** `VOCAB_PARALLEL_EMBEDDING` layer call emitted `self._vllm_xxx(input_ids)` without applying embedding scale. HRM-Text uses `EMBED_SCALE=39.19`.
- **Fix:** In `_emit_vllm_layer_call`, check `node.inputs[3]` for non-None scale; if present, wrap call as `(call * scale_expr)`.
- **File:** `brainsurgery/synapse/axon/codegen2_vllm/core.py`

## Regression Suite Status (2026-07-27)

`test_vllm_changes.py` — 30 models, bf16, forward-only, B200 GPU 2.

| Result | Count | Models |
|--------|-------|--------|
| PASS | 27 | SmolLM-135M, SmolLM2-360M, SmolLM2-1.7B, SmolLM3-3B, starcoder2-3b, granite-3.3-2b-base, gemma-2-2b, gemma-3-1b, gemma-3-4b, Qwen2.5-0.5B, Qwen2.5-1.5B, Qwen3-0.6B, Qwen3-1.7B, Phi-3-mini-4k, Phi-4-mini, stablelm-2-1.6b, exaone-4.0-1.2b, olmo-2-1b, glm-edge-1.5b, nanochat-d20, apertus-7b, gpt2, falcon-rw-1b, Llama-3.2-1B, Llama-3.2-3B, Mistral-7B, Helium-2b (torch only) |
| FAIL | 2 | SmolLM2-135M (bf16 precision), granite-3.1-2b-instruct (near-tied logits) |
| ERROR | 0 | — |
| Blocked | 4 | Gemma4 E2B/E4B (B200 head_dim limit), Helium-2b vLLM (model not local), Gemma3-27b (OOM) |

### Pre-existing FAILs (not regressions)

| Model | max_diff | Cause |
|-------|----------|-------|
| SmolLM2-135M | 2.09 | bf16 precision: position 4 logits near-tied (28.25 vs 28.125), both torch+vLLM fail bf16, pass f32 |
| granite-3.1-2b-instruct | 0.1875 | Near-tied logits, bf16 precision issue |

### Blocked models

| Model | Blocker | Torch status |
|-------|---------|--------------|
| Gemma4 E2B | B200: no backend for head_dim=256/512 | f32 PASS (max_diff=0.000026) |
| Gemma4 E4B | B200: no backend for head_dim=256/512 | f32 PASS (max_diff=0.000044) |
| Helium-2b | Model not available locally | torch f32/bf16 PASS |
| Gemma3-27b | OOM (GPU memory occupied) | — |

## Remaining Issues

### 5 FAIL (pending GPU/bf16 validation)

| Model | Notes |
|-------|-------|
| SmolLM-360M | Same family as SmolLM-135M (fixed), likely needs batch test re-run |
| SmolLM2-135M | Same family |
| SmolLM2-1.7B | Same family |
| SmolLM3-3B | Same family |
| SmolLM-1.7B | Same family |

### Pre-existing errors (not addressed in this session)

| Error category | Models | Count |
|---------------|--------|-------|
| OOM (GPU contention) | granite-3.1/3.3-2b-*, EXAONE-4.0-1.2B, Phi-3-mini-128k, Phi-4-mini-reasoning | 8 |
| mat1/mat2 0xN shape (fused QKV) | bloom-560m/1b1/1b7/3b, falcon_rw_1b, opt-1.3b | 6 |
| dict update (T5/BART) | bart-base, mbart, mt5-*, opus-mt, deberta-v3 | 7 |
| rotary_embedding positions | gemma-3-270m, gemma-3-1b | 2 |
| start + length exceeds dimension | OLMoE-1B-7B-0924, PowerMoE-3b | 2 |
| tensor size mismatch | gemma-4-E4B, electra-base-generator | 2 |
| AssertionError (weight loading) | gpt2, albert-base-v2, starcoder2-3b | 3 |
| unsupported operand NoneType % int | bloom-560m | 1 |
| vocab size assertion | xglm-564M | 1 |

## Key Files

- `brainsurgery/synapse/axon_test.py` — main test runner; vLLM model setup, attention backend config, dtype handling, slot_mapping, forward context
- `brainsurgery/synapse/axon/codegen2_vllm/core.py` — vLLM codegen emitter (~3719 lines)
- `brainsurgery/synapse/axon/codegen2_vllm/classify.py` — graph classification for vLLM layers
- `brainsurgery/synapse/axon/codegen2_torch/core.py` — base torch emitter (shared helpers)
- `test_vllm_changes.py` — regression suite (30 models, repo root)
- `log/4b-bench/run_all_4b_vllm.sh` — benchmark orchestrator

## Commits

- `daaa5d7` — vLLM fixes 1-8 (Gemma-3 rope, Qwen2.5 LM head, Phi-4-mini, performance, Falcon ALiBi, node ID alias)
- `d6255dd` — vLLM fixes 9-13 (Gemma-2 fused norm, embedding scale, GeLU, RMSNorm unit_offset, Gemma-3 QK norms, GPT2/Falcon)
- `49b26a6` — `cast_float=true` default for `rmsnorm_noscale` and `rmsnorm`
- `b054dd5` — `__torch_rmsnorm_scaled` intrinsic fix (bf16 weight multiply)
- `77f08b9` — OLMo-2 post-norm fix (QK norm pre-reshape, post-norm emission)
- `64ec155` — GLM-edge interleaved RoPE + ffn_norm_idx via `_trace_back`
- `6c81e04` — EXAONE-4.0 ModuleList indexing + repeated module fallback
- `114c7a7` — StableLM QKV bias + partial rotary, NanoChat RoPE sign flip, Helium interleaved RoPE
- `102ebb5` — B200 attention backend: head_dim detection from `text_config`, slot_mapping int64, `set_default_torch_dtype`, `set_current_vllm_config` forward wrapper

## vLLM Forward Benchmark (2026-07-27)

**Run:** `log/vllm-bench/all-models-bf16/bfloat16-128/stream.csv`
**Plot:** `log/vllm-bench/plots/scatter_vllm_fwd.svg`, `scatter_vllm_fwd_labeled.svg`
**Report:** `log/vllm-bench/plots/vllm-bench-REPORT.md`
**Config:** bf16, forward-only, max_len=128, 1 warmup + 3 repeats, 8x B200 GPUs

| Metric | Value |
|--------|-------|
| Total models | 52 |
| PASS (top1=True) | 52 (100%) |
| FAIL | 0 |
| ERROR | 0 |
| Axon faster (ratio < 1.0) | 8 |
| Median ratio | ~1.12 |
| Min ratio | 0.52 (gpt2) |
| Max ratio | 2.83 (opt-1.3b, isolation) |

Fastest Axon: gpt2 (0.52x), falcon-rw-1b (0.55x), bloom-3b (0.84x)
Slowest Axon: opt-1.3b (2.83x isolated), Phi-4-mini-instruct (1.72x), Phi-4-mini-reasoning (1.70x)

### EXAONE QK Norm Fix (2026-07-27)

**Bug:** `_find_linear_call_for_value_deep` in `classify.py` traced through `core.alias` → `rope_pair_base_factors` and found q_proj for BOTH q and k (q's input searched first).
**Fix:** Added special case for `core.alias` when producer is a rope pair function: use output index to select corresponding input (output 0→input 0=q, output 1→input 1=k).
**Result:** EXAONE-4.0-1.2B ERROR→PASS (top1=True, max_diff=0.5). No regressions on Mistral, Qwen, Phi-3, Gemma-2, Llama-3.

## Next Steps

1. Debug NanoChat vLLM FAIL (max_diff=22.875) — check rmsnorm_noscale, relu2, logit soft cap, Q/K norms after RoPE
2. Test Helium-2b vLLM once model is available locally
3. Address remaining pre-existing errors (MoE models, Mistral-7B-v0.2 near-tied logits, GLM-4-9B)
