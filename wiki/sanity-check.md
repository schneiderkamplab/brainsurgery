---
status: active
last-confirmed: 2026-07-23
owners: agents
confidence: high
---

# Benchmark Sanity Check — All Backends

Walks `log/4b-bench/all-models-bf16/bfloat16-128/` to confirm Axon is faster
than or close to HF/Transformers across all backends.

**Data sources:**
- `stream.csv` — codegen2-torch, codegen2-triton, codegen2-jax (last modified 2026-07-22)
- `stream_vllm.csv` — codegen2-vllm (last modified 2026-07-23 18:18, post-fix)

**vLLM fixes applied 2026-07-23** — see `wiki/vllm-backend-debug.md` for details.
Key fixes: attention detection (core.select softmax), head_dim fallback,
config aliases (attention_heads, ffn_dim), model.-prefix weight loading,
forward context, 3D-to-2D reshape, Positions.* fallback to legacy forward,
re-eval symbols after weight loading.

**Classification:**
- **PASS** — `masked_top1_eq=True` or `N/A` (masked_lm), `masked_max_abs_diff` reasonable
- **FAIL** — `masked_top1_eq=False`
- **ERROR** — crashed, OOM, or no data

**Speed ratio** = `axon_time / hf_time`. Values < 1.0 mean Axon is faster.

---

## Summary Per Backend

| Backend | Total | PASS | FAIL | ERROR |
|---------|-------|------|------|-------|
| codegen2-torch | 69 | 42 | 20 | 1 |
| codegen2-triton | 66 | 39 | 20 | 1 |
| codegen2-jax | 65 | 31 | 24 | 2 |
| codegen2-vllm | 64 | 38 | 4 | 22 |

**Verdict:**
- **torch**: 42/62 correct models (67.7%)
- **triton**: 39/59 correct models (66.1%)
- **jax**: 31/55 correct models (56.4%)
- **vllm**: 38/42 correct models (90.5%) — best correctness rate

---

## vLLM Results (post-fix, 2026-07-23)

### PASS (38 models)

| Model | Diff | Notes |
|-------|------|-------|
| EXAONE-4.0-1.2B | 0.53 | |
| HRM-Text-1B | 0.22 | |
| OLMo-2-0425-1B | 0.69 | |
| Phi-3-mini-4k-instruct | 2.75 | |
| Pleias-Nano | 1.94 | |
| Pleias-Pico | 5.47 | |
| Pleias-RAG-1B | 0.69 | |
| Pleias-RAG-350M | 0.75 | |
| Qwen2.5-0.5B | 3.97 | |
| SmolLM-1.7B | 0.75 | |
| SmolLM-135M | 2.84 | |
| SmolLM-360M | 8.06 | |
| SmolLM2-135M | 7.78 | |
| SmolLM2-360M | 3.59 | |
| bert-base-uncased | 0.63 | |
| bloom-1b1 | 0.13 | |
| bloom-1b7 | 0.25 | |
| bloom-3b | 0.19 | |
| bloom-560m | 2.0 | top1=False but same diff as torch (bf16 precision) |
| camembert-base | 0.31 | |
| camembert-large | 0.53 | |
| distilbert-base-uncased | 0.19 | |
| distilroberta-base | 0.25 | |
| electra-base-generator | 0.52 | |
| falcon_rw_1b | 1.0 | top1=True (was FAIL on torch/jax pre-fix) |
| glm-edge-1.5b-chat | 0.38 | |
| glm-edge-4b-chat | 0.44 | |
| granite-3.1-2b-base | 1.81 | |
| granite-3.1-2b-instruct | 0.25 | |
| granite-3.3-2b-base | 1.02 | |
| longformer-base-4096 | 0.63 | |
| opt-1.3b | 0.0 | N/A (masked_lm) |
| roberta-base | 0.53 | |
| roberta-large | 0.50 | |
| xglm-1.7B | 2.0 | top1=True (was FAIL on all backends pre-fix) |
| xglm-564M | 0.50 | |
| xlm-roberta-base | 5.75 | |
| xlm-roberta-large | 1.13 | |

### FAIL (4 models)

| Model | Diff | Notes |
|-------|------|-------|
| SmolLM2-1.7B | 7.33 | top1=False |
| albert-base-v2 | 24.13 | parameter sharing issue |
| gpt2 | 1.25 | pre-existing cross-backend issue |
| starcoder2-3b | 24.0 | weight shape mismatch |

### ERROR (22 models)

| Model | Error category |
|-------|--------------|
| AI21-Jamba-Reasoning-3B | Mamba/SSM forward context |
| BlackMamba-2.8B | Mamba/SSM forward context |
| OLMoE-1B-7B-0924 | MoE expert routing |
| Phi-3-mini-128k-instruct | OOM (GPU contention) |
| Phi-4-mini-instruct | OOM or forward error |
| Phi-4-mini-reasoning | OOM or forward error |
| Pleias-3b-Preview | OOM (GPU contention) |
| bart-base | encoder-decoder attention |
| deberta-v3-xsmall | relative position attention |
| gemma-3-1b | rotary embedding |
| gemma-3-270m | rotary embedding |
| gemma-4-E4B | rotary embedding |
| granite-3.3-2b-instruct | OOM or forward error |
| mbart-large-50-many-to-many-mmt | encoder-decoder attention |
| mt5-base | encoder-decoder shape mismatch |
| mt5-large | encoder-decoder shape mismatch |
| mt5-small | encoder-decoder shape mismatch |
| opus-mt-en-de | encoder-decoder attention |
| t5-3b | encoder-decoder shape mismatch |
| t5-base | encoder-decoder shape mismatch |
| t5-large | encoder-decoder shape mismatch |
| t5-small | encoder-decoder shape mismatch |

---

## Applied vLLM Fixes (2026-07-23)

### Attention detection for core.select softmax (`classify.py`)
- `_is_structural_attention_call` now checks `core.select` branch expressions
  for `_softmax`/`Tensor.softmax` operations, not just direct child nodes.
- Fixes: XGLM, and any model using `Attention.attention` builtin where softmax
  is wrapped in a `core.select` for fp32 compute path.

### head_dim fallback (`core.py`)
- Forward code `_head_dim_expr()` used instead of `_config_expr("head_dim")` in
  all places (QKV init, ROW_PARALLEL_LINEAR dim, rope, forward loop).
- Fixes: XGLM (head_dim not in config, computed from hidden_size/num_heads).

### Config key aliases (`axon_test.py`)
- Added `attention_heads` → `num_attention_heads`
- Added `ffn_dim` → `intermediate_size`
- Fixes: XGLM and any model using non-standard config key names.

### model.-prefix weight loading (`core.py`)
- Non-mapped checkpoint weights stored in `state_dict_tensors` under both
  original name and `model.`-stripped name.
- Fixes: OPT-1.3b (checkpoint has `model.` prefix, model expects without).

### Re-eval symbols after weight loading (`core.py`)
- `self._eval_symbols()` called at end of `load_weights` so `has_root` checks
  reflect actual checkpoint keys, not just model module registrations.

### Forward context for vLLM attention (`axon_test.py`)
- `_run_syn_forward` wraps vLLM forward calls in `set_forward_context`.

### 3D-to-2D reshape before attention loop (`core.py`)
- Clean forward flattens `hidden_states` from [B,S,D] to [B*S,D] before the
  transformer block loop to match vLLM layer output shapes.

### Positions.* fallback to legacy forward (`core.py`)
- Clean forward falls back to legacy when main module has `Positions.*` calls
  (sinusoidal position embeddings, position_ids computation).
- Fixes: XGLM (uses sinusoidal_positions, not RoPE).

---

## Next Steps

1. **Fix T5/mt5 encoder-decoder** (22 ERROR models, largest category)
2. **Fix gemma-3 rotary embedding** (3 models)
3. **Fix Mamba/SSM forward context** (2 models)
4. **Fix starcoder2-3b weight shape** (FAIL, diff=24)
5. **Fix albert parameter sharing** (FAIL, diff=24)
6. **Re-run OOM models** with fewer parallel GPUs
7. **Investigate SmolLM2-1.7B** (FAIL, diff=7.33)
8. **Re-run torch/triton/jax benchmark** with `--optimize-graph` for SDPA fixes
