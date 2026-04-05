# Synapse Ops Overview

This document is a compact reference for the currently registered Synapse ops in
`brainsurgery/synapse/ops`.

- `Op` is the canonical runtime `OP_NAME`.
- `Surface Form` shows the usual Axon spelling when it differs from `OP_NAME`.
- `Inputs` is the accepted positional-argument count from lowering validation.
- `Outputs` is the runtime output arity at the op boundary.
- `Key kwargs` lists the main supported kwargs, with required ones marked `required`.

## Core Layers

| Op | Surface Form | Inputs | Outputs | Key kwargs | Description |
|---|---|---:|---:|---|---|
| `embedding` | `embedding` | 1 | 1 | `dim`, `scale` | Looks up embeddings from a parameter table and can apply a scalar scale. |
| `linear` | `linear` | 1 | 1 | `bias`, `bias_path`, `dim`, `expert`, `transpose`, `weight` | Applies an affine projection using inferred or explicit parameter paths. |
| `layernorm` | `layernorm` | 1 | 1 | `bias`, `dim`, `eps`, `weight` | Applies LayerNorm over the final dimension with optional explicit parameter paths. |
| `rmsnorm` | `rmsnorm` | 1 | 1 | `cast_float`, `dim`, `eps`, `unit_offset`, `with_scale` | Applies RMSNorm with optional float accumulation and scale handling. |
| `activation` | `act::kind(...)` | 1 | 1 | `fp32_accum`, `limit` | Applies a named activation such as `gelu`, `relu`, `silu`, or capped variants. |
| `param_scale` | `param_scale` | 1 | 1 | `scale` | Multiplies the input tensor by a resolved parameter tensor. |

## Attention, Masks, And Positional Bias

| Op | Surface Form | Inputs | Outputs | Key kwargs | Description |
|---|---|---:|---:|---|---|
| `attention` | `attention` | 3 | 1 | `causal`, `eager`, `float_mask_additive`, `float_mask_floor_keep`, `mask`, `padding_mask`, `scale`, `sink`, `sink_path` | Computes scaled dot-product attention with optional additive masks, padding masks, causal mode, and sink handling. |
| `causal_mask` | `causal_mask` | 2 | 1 | `early_exit`, `padding_mask`, `window` | Builds an additive causal attention mask, optionally windowed or padding-aware. |
| `bidirectional_mask` | `bidirectional_mask` | 2 | 1 | `padding_mask`, `window` | Builds a symmetric additive attention mask for full-context or local bidirectional attention. |
| `blocksparse_mask` | `blocksparse_mask` | 2 | 1 | `block_size` `required`, `homo_head`, `local_blocks` `required`, `padding_mask`, `vert_stride` `required` | Builds a block-sparse additive mask with causal, local, and vertical-stride structure. |
| `rope_pair` | `apply_rope_pair` | 2 | 2 | `position_ids` `required`, `theta`, `interleaved`, `scale_factor`, `rope_mode`, `truncate`, `partial_rotary_factor`, `attention_factor`, `short_factor`, `long_factor` | Applies rotary position embedding to a query/key tensor pair, including long-context variants. |
| `position_ids` | `arange_positions` | 2 | 1 | `pad_fill`, `past_length`, `use_attention_mask` | Derives position ids from token ids plus an attention mask or past-length offset. |
| `linear_position_bias` | `linear_position_bias` | 1 | 1 | `heads` `required`, `scale` | Builds a linear ALiBi-style additive bias tensor from an attention mask. |
| `t5_relative_position_bias` | `t5_relative_position_bias` | 2 | 1 | `bidirectional`, `max_distance`, `num_buckets` | Builds learned T5-style relative position bias from query/key lengths. |
| `disentangled_relative_bias` | `disentangled_relative_bias` | 2 | 1 | `rel_embeddings`, `position_buckets`, `max_relative_positions`, `c2p`, `p2c`, `share_att_key`, `apply_rel_layernorm` | Builds DeBERTa-style disentangled relative attention bias with optional projected content and position paths. |
| `cache_update` | `Cache.update` | 3 | 3 | none | Merges new key/value tensors into the KV cache and returns `k_ctx`, `v_ctx`, and `present`. |
| `cache_seq_len` | `Cache.seq_len` | 1 | 1 | none | Returns the current sequence length represented by a cache entry. |

## Tensor Shape And Data Movement

| Op | Surface Form | Inputs | Outputs | Key kwargs | Description |
|---|---|---:|---:|---|---|
| `reshape_heads` | `reshape_heads` | 1 | 1 | `head_dim`, `heads` | Reshapes hidden states into `[batch, heads, seq, head_dim]` form. |
| `merge_heads` | `merge_heads` | 1 | 1 | none | Folds attention heads back into the final hidden dimension. |
| `split` | `split` | 1 | dynamic | `dim`, `interleave`, `parts`, `sizes` | Splits one tensor into multiple outputs by part count or explicit sizes. |
| `split_qkv_heads` | `split_qkv_heads` | 1 | 3 | `heads` `required`, `layout` | Splits packed QKV activations into separate query, key, and value head tensors. |
| `split_qkv_grouped` | `split_qkv_grouped` | 1 | 3 | `head_dim`, `heads` `required`, `kv_heads` `required` | Splits grouped-query packed QKV activations into Q, K, and V tensors. |
| `repeat` | `repeat` / `repeat_kv` | 1-3 | 1 | `dim`, `repeats` | Repeats values along a chosen axis, commonly for KV-head expansion. |
| `concat` | `concat` | 2 | 1 | `dim` | Concatenates two tensors along a chosen dimension. |
| `topk` | `topk` | 1 | 2 | `dim`, `k` `required`, `largest`, `sorted` | Returns top-k values and indices along an axis. |
| `softmax` | `softmax` | 1 | 1 | `dim`, `dtype` | Applies softmax along a chosen dimension with optional dtype override. |
| `zeros_like` | `zeros_like` | 1 | 1 | none | Allocates a zero tensor matching the input tensor shape and dtype. |

## Math And Reductions

| Op | Surface Form | Inputs | Outputs | Key kwargs | Description |
|---|---|---:|---:|---|---|
| `add` | `add` or `x + y` | 2 | 1 | none | Performs elementwise or broadcast addition. |
| `mul` | `mul` or `x * y` | 2 | 1 | none | Performs elementwise or broadcast multiplication. |
| `div` | `div` or `x / y` | 2 | 1 | none | Performs elementwise or broadcast division. |
| `sum` | `sum` | 1 | 1 | `dim`, `keepdim` | Reduces by summation over all elements or along a specific axis. |
| `log` | `log` | 1 | 1 | none | Applies the natural logarithm elementwise. |
| `sqrt` | `sqrt` | 1 | 1 | none | Applies square root elementwise. |
| `clamp` | `clamp` | 1 | 1 | `max`, `min` | Clips tensor values to a configured minimum, maximum, or both. |

## State-Space And Sequence Ops

| Op | Surface Form | Inputs | Outputs | Key kwargs | Description |
|---|---|---:|---:|---|---|
| `causal_conv1d` | `causal_conv1d` | 1-2 | 1 or 2 | `activation` | Runs depthwise causal 1D convolution and can optionally return updated decode state. |
| `mamba_scan` | `mamba_scan` | 4-7 | 1 or 2 | `A`, `D`, `a_is_log` | Executes the selective state-space scan used by Mamba-style sequence blocks. |
| `mamba2_scan` | `mamba2_scan` | 5-6 | 1 or 2 | `A` `required`, `D` `required`, `dt_bias` `required`, `norm_weight` `required`, `n_groups` `required`, `head_dim` `required`, `time_step_min`, `time_step_max` | Executes the Mamba-2 scan with gated group RMSNorm and optional recurrent state output. |

## MoE And Routing

| Op | Surface Form | Inputs | Outputs | Key kwargs | Description |
|---|---|---:|---:|---|---|
| `moe_select` | `moe_select_tokens` | 3 | 4 | `expert` `required` | Selects the tokens and routing metadata assigned to one expert from top-k routing outputs. |
| `moe_scatter_add` | `moe_scatter_add` | 4 | 1 | `accum_dtype` | Accumulates expert updates back into token order using routing weights. |
| `moe_grouped_ffn` | `moe_grouped_ffn` | 3 | 1 | `alpha`, `down_bias`, `down_weight`, `gate_up_bias`, `gate_up_weight`, `has_bias`, `has_gate`, `limit`, `transpose` | Runs grouped expert FFN execution and weighted aggregation in one fused generic MoE op. |
| `gemma4_router` | `gemma4_router` | 1 | 2 | `top_k` `required`, `scalar_root_size`, `rms_eps` | Computes Gemma 4 router probabilities, top-k expert indices, and scaled routing weights. |
| `gemma4_moe_experts` | `gemma4_moe_experts` | 3 | 1 | none | Executes Gemma 4 expert projections for routed tokens and accumulates the weighted outputs. |
| `glm4_router` | `glm4_router` | 1 | 2 | `top_k` `required`, `n_group`, `topk_group`, `norm_topk_prob`, `routed_scaling_factor`, `weight`, `e_score_correction_bias` | Computes GLM-4 grouped router top-k indices and routing weights. |
| `nemotron_moe` | `nemotron_moe` | 1 | 1 | `top_k` `required`, `n_group` `required`, `topk_group` `required`, `routed_scaling_factor` `required`, `norm_topk_prob` `required` | Runs the Nemotron routed-expert block, including shared-expert fallback. |
| `nemotron_moe_expert_step` | `nemotron_moe_expert_step` | 3 | 1 | `expert` `required` | Executes one Nemotron expert update for the tokens routed to a specific expert id. |

## Containers And Control Flow

| Op | Surface Form | Inputs | Outputs | Key kwargs | Description |
|---|---|---:|---:|---|---|
| `list_init` | `init_list` | 0 | 1 | none | Creates an empty runtime list container. |
| `list_append` | `append` | 2 | 1 | none | Appends one value to a runtime list. |
| `list_index` | `index` | 2 | 1 | none | Reads one item from a list, tuple, or tensor by position. |
| `select` | conditional expression lowering | 0 | dynamic | `cond` `required` | Executes either the `_then` or `_else` branch graph and forwards its bound outputs. |

## Config And Parameter Introspection

| Op | Surface Form | Inputs | Outputs | Key kwargs | Description |
|---|---|---:|---:|---|---|
| `config_has` | `Config.has` | 1 | 1 | `root` | Tests whether a config key exists, optionally under a config root. |
| `config_int` | `Config.int` | 1 | 1 | `default`, `root` | Reads a config value and coerces it to an integer. |
| `config_float` | `Config.float` | 1 | 1 | `default`, `root` | Reads a config value and coerces it to a float. |
| `config_str` | `Config.str` | 1 | 1 | `default`, `root` | Reads a config value and coerces it to a string. |
| `config_value` | `Config.value` | 1 | 1 | `default`, `root` | Reads a config value without type coercion. |
| `params_has_root` | `Params.has_root` | 1 | 1 | none | Checks whether any parameter exists under a given root prefix. |
| `params_root` | `Params.root` | 1 | 1 | `default` | Resolves the first usable parameter root, with optional fallback default. |

## Model-Specific Utility Ops

| Op | Surface Form | Inputs | Outputs | Key kwargs | Description |
|---|---|---:|---:|---|---|
| `gemma4_per_layer_inputs` | `gemma4_per_layer_inputs` | 2 | 1 | `num_layers` `required`, `per_layer_dim` `required`, `embed_scale`, `projection_scale`, `combine_scale`, `rms_eps` | Builds Gemma 4 per-layer inputs by combining per-layer token embeddings with projected hidden states. |
| `gemma4_per_layer_input_at` | `gemma4_per_layer_input_at` | 2 | 1 | none | Selects one layer slice from a Gemma 4 per-layer input tensor. |

## IR-Only Internal Ops

| Op | Surface Form | Inputs | Outputs | Key kwargs | Description |
|---|---|---:|---:|---|---|
| `_ir_alias` | none | 1 | 1 | none | Internal IR node that aliases an existing value to a new binding. |
| `_ir_expr` | `_ir_const` compatibility path | 0 | 1 | none | Internal IR node that materializes an expression or constant into the graph. |

## Granularity Recommendations

The current op surface is no longer just a small set of generic tensor primitives. It
now mixes:

- stable reusable primitives such as `linear`, `attention`, `split`, `topk`, and `softmax`
- semantic model-domain primitives such as `cache_update`, `rope_pair`, and `mamba2_scan`
- model-family-specific helpers such as `gemma4_*`, `glm4_router`, and `nemotron_*`

That makes granularity the main design pressure. The goal should be to keep generic
ops small and reusable, keep model-specific ops explicitly scoped, and avoid adding
new "do everything" hybrids in the middle.

| Ops | Recommendation | Why |
|---|---|---|
| `attention` | Split policy extras from the core attention op. | `scale`, `mask`, `padding_mask`, and `causal` are core semantics. `eager`, `sink`, `sink_path`, and the float-mask behavior flags are execution-policy or compatibility knobs. Keeping them on one op makes attention mean both "compute attention" and "pick a backend/mask policy". |
| `rope_pair` | Split surface variants, keep one shared rotary backend. | This op has grown into a family: base RoPE, interleaved layout, partial rotary, multiple scaling schemes, truncation, and mode switching. The math core can stay shared, but the public surface is too overloaded for one entry point. |
| `mamba_scan` vs `mamba2_scan` | Keep separate and do not merge. | They are both scan-like, but they are not the same semantic primitive. `mamba2_scan` has a distinct contract around `dt_bias`, grouped RMSNorm, head grouping, and time-step clamping. Merging them would create an even more overloaded state-space op. |
| `causal_mask`, `bidirectional_mask`, `blocksparse_mask` | Keep distinct public ops, but share internal mask-building utilities. | These are different semantic bias families, and the names carry useful intent. However, they all build additive attention masks and should share shape validation and mask materialization internals where possible. |
| `linear_position_bias`, `t5_relative_position_bias`, `disentangled_relative_bias` | Keep separate; do not merge into one generic bias op. | These are genuinely different bias constructions with different parameterization and inductive bias. A catch-all relative-bias op would reduce clarity and encourage kwarg-driven branching. |
| `split_qkv_heads` and `split_qkv_grouped` | Keep both for now, but treat them as domain sugar over lower-level shape ops. | They express common packed-attention layouts cleanly. They should not become the start of a large family of packing-specific ops; if more variants appear, a lowering rewrite layer is preferable to many near-duplicate runtime primitives. |
| `repeat` | Keep one runtime op and prefer domain-specific surface aliases in Axon. | The runtime computation is generic axis repetition. The dominant semantic use is KV-head expansion, so aliases like `repeat_kv` are good at the language surface, but there is no need for another runtime op. |
| `linear` | Keep as the dense affine primitive, but do not let routing/expert behavior grow further inside it. | `bias_path`, explicit `weight`, and `expert` already push `linear` toward parameter-routing logic. If more expert-specific behavior appears, it should move into dedicated helper ops rather than turning `linear` into a generic parameter lookup escape hatch. |
| `gemma4_router`, `glm4_router`, `nemotron_moe` | Keep model-family-specific routing ops separate from generic MoE primitives. | These encode family-specific routing formulas and parameter conventions. They should stay explicit instead of being folded into `moe_select` or `moe_grouped_ffn`, which are more reusable sparse-execution primitives. |
| `gemma4_moe_experts`, `nemotron_moe_expert_step` | Consider lowering these to generic MoE building blocks if another family needs the same execution pattern. | Right now they are justified as family helpers. If similar expert-step ops start appearing for more families, the right move is probably a shared expert-execution primitive plus model-specific lowering, not a growing list of per-family expert kernels. |
| `gemma4_per_layer_inputs`, `gemma4_per_layer_input_at` | Keep isolated as model-specific utility ops and resist generalizing prematurely. | These are clear one-family helpers. They should not be expanded into a generic "per-layer tensor toolkit" unless multiple unrelated architectures need the same abstraction. |
| `config_has`, `config_int`, `config_float`, `config_str`, `config_value` | Keep the typed surface split, merge implementation internals. | The surface is readable and precise. The runtime work is mostly the same lookup/default/root-resolution machinery, so the right merge point is internal helpers, not the public API. |
| `params_has_root`, `params_root` | Keep distinct. | They answer different questions: existence versus resolution. Merging them would force sentinel-style behavior and make the API harder to reason about. |
| `add`, `mul`, `div`, `sum`, `log`, `sqrt`, `clamp`, `softmax`, `topk`, `zeros_like` | Keep as focused leaf ops; do not merge for surface reduction. | These are low-complexity, recognizable tensor primitives. Their granularity is already appropriate, and explicit ops help readability and backend targeting. |
| `list_init`, `list_append`, `list_index` | Keep minimal and avoid expanding the container family casually. | These ops are useful for cache/present plumbing, but every extra container primitive pushes Synapse closer to a general-purpose language. The current boundary is still reasonable. |
| `_ir_alias`, `_ir_expr` | Keep internal-only. | These are lowering artifacts and should not leak into the authored DSL surface. |

### Practical Priorities

If the immediate goal is to improve conceptual granularity without shrinking real
capability, the highest-value moves are:

1. Split `attention` into a smaller semantic core plus policy/backend wrappers or lowering rewrites.
2. Split `rope_pair` into clearer surface variants while preserving one shared implementation backbone.
3. Hold the line between generic MoE primitives and model-family routing/helper ops instead of adding more hybrid middle-ground ops.
4. Keep adding model-specific helpers only when the behavior is genuinely architecture-specific; otherwise lower to existing generic primitives.
