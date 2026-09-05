# Next Models

This document combines high-priority current gaps and classic baseline families that are good candidates for new `axon` coverage.

| Priority | Target family | Example checkpoint(s) | Why it's worth adding |
|---|---|---|---|
| 1 | Phi-4 | `microsoft/Phi-4-mini-instruct`, `microsoft/Phi-4-mini-reasoning` | Current Microsoft gap; strong small-model target. |
| 2 | Aya Expanse / Cohere | `CohereLabs/aya-expanse-8b` | Major multilingual family not covered locally. |
| 3 | Granite | `ibm-granite/granite-3.3-8b-instruct` | High-profile IBM family with strong enterprise relevance. |
| 4 | EXAONE 4 | `LGAI-EXAONE/EXAONE-4.0-1.2B` | Important Korean/English family supported in Transformers. |
| 5 | StarCoder2 | `bigcode/starcoder2-7b` | Clear missing code-model baseline. |
| 6 | BLOOM | `bigscience/bloom-560m`, `bigscience/bloom-1b7`, `bigscience/bloom-7b1` | Foundational multilingual causal LM family; still a useful historical and compatibility baseline. |
| 7 | OPT | `facebook/opt-1.3b`, `facebook/opt-6.7b`, `facebook/opt-13b` | Standard historical decoder-only baseline family, still widely referenced. |
| 8 | GPT-J / GPT-Neo | `EleutherAI/gpt-j-6b`, `EleutherAI/gpt-neo-1.3B`, `EleutherAI/gpt-neo-2.7B` | Older, but still useful for broad HF compatibility and regression coverage. |
| 9 | XGLM | `facebook/xglm-2.9B`, `facebook/xglm-7.5B` | Useful classic multilingual decoder baseline if broader language-family coverage is desired. |
| 10 | Phi-3.5-MoE / PhiMoE | `microsoft/Phi-3.5-MoE-instruct` | Extends existing Phi coverage into the MoE branch. |
| 11 | GraniteMoe | `ibm-granite/granitemoe-*`, `ibm-research/PowerMoE-3B` | Adds a modern sparse-MoE family outside the Mixtral/Qwen/Gemma cluster. |
| 12 | LFM2 | `LiquidAI/LFM2-1.2B`, `LiquidAI/LFM2-2.6B` | Good architecture-diversity target with on-device relevance. |
| 13 | BitNet | `microsoft/bitnet-b1.58-2B-4T` | High-profile architecture target, though less representative in plain Transformers benchmarking. |

## Suggested First Wave

If only a small number of families should be added next, start with:

1. `phi4`
2. `cohere` or `aya-expanse`
3. `granite`
4. `exaone4`
5. `starcoder2`
6. `bloom`

## Notes

- The current local model coverage already includes families such as Gemma, Llama, Qwen, Mistral, Mixtral, OLMo, Falcon, Phi-3, GPT-OSS, T5, RoBERTa, BERT, and related variants under `brainsurgery/synapse/models`.
- The first five additions above emphasize prominent current gaps in the modern Hugging Face / Transformers ecosystem.
- The classic baseline families improve historical benchmark coverage, compatibility testing, and regression tracking.

## Axon / Synapse TODO

### Groomed primitive ops

- `linear`
- `layernorm`
- `embedding`
- `repeat`
- `split`
- `chunk`
- `slice`
- `reshape`
- `permute`
- `transpose`
- `expand`
- `arange`
- `cast`
- `cumsum`
- `softmax`
- `topk`
- `sqrt`
- `eq`
- `le`
- `and`
- `where`
- `zeros_like`
- `min_like`

### Deprecated operations

- `bidirectional_mask`
- `causal_mask`
- `attention`
- `rope_pair`
- `reshape_heads`
- `merge_heads`
- `position_ids`
- `split_qkv_heads`
- `split_qkv_grouped`
- `linear_position_bias`
- `blocksparse_mask`
- `moe_select`
- `moe_scatter_add`
- `moe_grouped_ffn`
- `moe_grouped_swiglu_ffn`
- `softmax_topk_router`
- `sigmoid_topk_router`
- `nemotron_moe`
- `gemma4_router`
- `gemma4_moe_experts`
- `gemma4_per_layer_inputs`
- `gemma4_per_layer_input_at`
- `glm4_router`
- `mamba_scan`
- `mamba2_scan`
- `causal_conv1d`
- `t5_relative_position_bias`
- `disentangled_relative_bias`

### Other too-coarse/opaque ops to move into `Derived.axon`

- `repeat`
- `rmsnorm` (when a stable derived formulation is agreed)
- `param_scale`

### To be sorted (all current primitive ops)

- `_ir_alias`, `_ir_expr`, `activation`, `add`, `and`, `arange`, `attention`, `bidirectional_mask`, `blocksparse_mask`, `cache_seq_len`, `cache_update`, `cast`, `causal_conv1d`, `causal_mask`, `chunk`, `clamp`, `concat`, `config_float`, `config_has`, `config_int`, `config_str`, `config_value`, `cumsum`, `disentangled_relative_bias`, `div`, `embedding`, `eq`, `expand`, `floor`, `gemma4_moe_experts`, `gemma4_per_layer_input_at`, `gemma4_per_layer_inputs`, `gemma4_router`, `glm4_router`, `l2norm`, `layernorm`, `le`, `linear`, `linear_position_bias`, `list_append`, `list_index`, `list_init`, `log`, `mamba2_scan`, `mamba_scan`, `matmul`, `merge_heads`, `min_like`, `moe_grouped_ffn`, `moe_grouped_swiglu_ffn`, `moe_scatter_add`, `moe_select`, `mul`, `nemotron_moe`, `param_scale`, `params_has_root`, `params_root`, `permute`, `position_ids`, `repeat`, `reshape`, `reshape_heads`, `rmsnorm`, `rope_pair`, `select`, `sigmoid_topk_router`, `slice`, `softmax`, `softmax_topk_router`, `split`, `split_qkv_grouped`, `split_qkv_heads`, `sqrt`, `t5_relative_position_bias`, `topk`, `transpose`, `unsqueeze`, `where`, `zeros_like`

### Language/runtime follow-ups

- Full expression-level elementwise tensor comparisons such as `attn_mask == 0`.
- Once that larger task exists, tensor comparison expressions could replace some dedicated comparison primitives in Axon modules.
- Add `_to_idx` primitive (`Tensor -> IdxTensor`) as an explicit typed boundary, then tighten `embedding` to consume `IdxTensor` and avoid broad implicit `Tensor`/`IdxTensor` interchange.
- Add generic list-unpack syntax (Python-style starred target), e.g. `head, *tail <- xs` and `*init, last <- xs`; use it to simplify Cache/list handling (`List.index` head/last patterns).
