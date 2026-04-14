# Op Buckets Overview (Full Inventory)

This file inventories all builtins definitions and all primitives (`_xyz`) with an explicit `Type` classification.

Type values: `primitive`, `derived`, `alias`, `wrapper` (non-alias wrapper).

Rule update: direct primitive calls are now enforced as `_xyz` syntax and only from builtins (`*.axon` in builtin namespaces). Model code must call wrappers/aliases.

## Activations.axon (are there any activations that have paths?)

| Op | Type | Suggestion | Use in builtins | Use in models | Use in operators | Comment |
| --- | --- | --- | ---: | ---: | ---: | --- |
| gelu | alias | Keep as-is. | 0 | 67 | 0 |  |
| gelu_new | alias | Keep as-is. | 0 | 55 | 0 |  |
| gelu_pytorch_tanh | alias | Keep as-is. | 0 | 52 | 0 |  |
| gegelu | wrapper | Keep as-is. | 0 | 3 | 0 |  |
| relu | alias | Keep as-is. | 0 | 46 | 0 |  |
| relu2 | alias | Keep as-is. | 0 | 4 | 0 |  |
| sigmoid | alias | Keep as-is. | 0 | 10 | 0 |  |
| tanh | alias | Keep as-is. | 0 | 11 | 0 |  |
| silu | alias | Keep as-is. | 1 | 194 | 0 |  |
| swiglu | alias | Keep as-is. | 0 | 0 | 0 |  |
| xielu | alias | Keep as-is. | 0 | 3 | 0 |  |

## Cache.axon

| Op | Type | Suggestion | Use in builtins | Use in models | Use in operators | Comment |
| --- | --- | --- | ---: | ---: | ---: | --- |
| update | wrapper | Keep as-is. | 2 | 192 | 0 |  |
| init | wrapper | Keep as-is. | 5 | 196 | 0 |  |
| index | wrapper | Keep as-is. | 8 | 206 | 0 |  |
| append | wrapper | Keep as-is. | 4 | 206 | 0 |  |
| past_length | wrapper | Keep as-is. | 7 | 357 | 0 |  |

## Config.axon

| Op | Type | Suggestion | Use in builtins | Use in models | Use in operators | Comment |
| --- | --- | --- | ---: | ---: | ---: | --- |
| has_key | wrapper | Keep as-is. | 2 | 23 | 0 |  |
| has_value | wrapper | Keep as-is. | 2 | 0 | 0 |  |
| int | wrapper | Keep as-is. | 2 | 708 | 0 |  |
| float | wrapper | Keep as-is. | 2 | 232 | 0 |  |
| str | wrapper | Keep as-is. | 2 | 5 | 0 |  |
| bool | wrapper | Keep as-is. | 4 | 0 | 0 |  |
| list | wrapper | Keep as-is. | 2 | 0 | 0 |  |
| value | wrapper | Keep as-is. | 7 | 79 | 0 |  |

## Derived.axon

| Op | Type | Suggestion | Use in builtins | Use in models | Use in operators | Comment |
| --- | --- | --- | ---: | ---: | ---: | --- |
| reshape_head_basic | derived | Keep as-is. | 10 | 263 | 0 |  |
| merge_heads_basic | derived | Keep as-is. | 10 | 272 | 0 |  |
| position_ids_basic_masked | wrapper | Compatibility forward; canonical home is now `Positions.axon`. | 6 | 0 | 0 | Delegates to `Positions.position_ids_basic_masked`. |
| position_ids_basic_nomask | wrapper | Compatibility forward; canonical home is now `Positions.axon`. | 6 | 0 | 0 | Delegates to `Positions.position_ids_basic_nomask`. |
| position_ids_basic | wrapper | Compatibility forward; canonical home is now `Positions.axon`. | 5 | 0 | 0 | Delegates to `Positions.position_ids_basic`. |
| attention_basic | derived | Keep as-is. | 0 | 9 | 0 |  |
| attention_basic_hf | derived | Deprecated; remove. | 0 | 0 | 0 |  |
| mask_to_additive_basic | wrapper | Compatibility forward; eventually import/use `Masking.*` directly where practical. | 7 | 0 | 0 | Delegates to `Masking.mask_to_additive_basic`. |
| causal_mask_basic_nomask | wrapper | Compatibility forward; eventually import/use `Masking.*` directly where practical. | 7 | 0 | 0 | Delegates to `Masking.causal_mask_basic_nomask`. |
| causal_mask_basic_masked | wrapper | Compatibility forward; eventually import/use `Masking.*` directly where practical. | 6 | 0 | 0 | Delegates to `Masking.causal_mask_basic_masked`. |
| causal_mask_basic | wrapper | Compatibility forward; eventually import/use `Masking.*` directly where practical. | 5 | 7 | 0 | Delegates to `Masking.causal_mask_basic`. |
| bidirectional_mask_basic_nomask | wrapper | Compatibility forward; eventually import/use `Masking.*` directly where practical. | 7 | 0 | 0 | Delegates to `Masking.bidirectional_mask_basic_nomask`. |
| bidirectional_mask_basic_masked | wrapper | Compatibility forward; eventually import/use `Masking.*` directly where practical. | 6 | 0 | 0 | Delegates to `Masking.bidirectional_mask_basic_masked`. |
| bidirectional_mask_basic | wrapper | Compatibility forward; eventually import/use `Masking.*` directly where practical. | 5 | 7 | 0 | Delegates to `Masking.bidirectional_mask_basic`. |
| position_ids_scale_basic | wrapper | Compatibility forward; canonical home is now `Positions.axon`. | 6 | 0 | 0 |  |
| rope_pair_base_basic | wrapper | Compatibility forward; canonical home is now `Positions.axon`. | 6 | 0 | 0 |  |
| rope_pair_proportional_basic | wrapper | Compatibility forward; canonical home is now `Positions.axon`. | 6 | 0 | 0 |  |
| rope_pair_freq_scale_basic | wrapper | Compatibility forward; canonical home is now `Positions.axon`. | 6 | 2 | 0 |  |
| rope_pair_longrope_basic | wrapper | Compatibility forward; canonical home is now `Positions.axon`. | 6 | 0 | 0 |  |
| rope_pair_yarn_basic | wrapper | Compatibility forward; canonical home is now `Positions.axon`. | 6 | 0 | 0 |  |
| rope_pair_hf_yarn_basic | wrapper | Compatibility forward; canonical home is now `Positions.axon`. | 6 | 0 | 0 |  |
| relative_position_buckets_basic | wrapper | Compatibility forward; canonical home is now `Positions.axon`. | 6 | 0 | 0 |  |
| relative_position_bias_basic | wrapper | Compatibility forward; canonical home is now `Positions.axon`. | 5 | 5 | 0 |  |

## Masking.axon

| Op | Type | Suggestion | Use in builtins | Use in models | Use in operators | Comment |
| --- | --- | --- | ---: | ---: | ---: | --- |
| mask_to_additive_basic | derived | Keep as-is. | 7 | 0 | 0 |  |
| causal_mask_basic_nomask | derived | Keep as-is. | 7 | 0 | 0 |  |
| causal_mask_basic_masked | derived | Keep as-is. | 6 | 0 | 0 |  |
| causal_mask_basic | derived | Keep as-is. | 5 | 7 | 0 |  |
| bidirectional_mask_basic_nomask | derived | Keep as-is. | 7 | 0 | 0 |  |
| bidirectional_mask_basic_masked | derived | Keep as-is. | 6 | 0 | 0 |  |
| bidirectional_mask_basic | derived | Keep as-is. | 5 | 7 | 0 |  |

## List.axon

| Op | Type | Suggestion | Use in builtins | Use in models | Use in operators | Comment |
| --- | --- | --- | ---: | ---: | ---: | --- |
| init | alias | Keep as-is. | 2 | 269 | 0 |  |
| index | alias | Keep as-is. | 2 | 384 | 0 |  |
| append | alias | Keep as-is. | 1 | 192 | 0 |  |

## Math.axon

| Op | Type | Suggestion | Use in builtins | Use in models | Use in operators | Comment |
| --- | --- | --- | ---: | ---: | ---: | --- |
| add | alias | Keep as-is. | 4 | 12 | 1050 | `Use in operators` counts `+` binary expressions. |
| div | alias | Keep as-is. | 0 | 0 | 532 | `Use in operators` counts `/` binary expressions (including shape/symbol math). |
| mul | alias | Keep as-is. | 1 | 231 | 446 | `Use in operators` counts `*` binary expressions. |
| pow | alias | Keep as-is. | 0 | 0 | 0 | No Axon call sites; backend expression evaluators still handle `pow(...)` (runtime/codegen/materialization). |
| exp | alias | Keep as-is. | 5 | 0 | 0 |  |
| log | alias | Keep as-is. | 18 | 8 | 0 |  |
| floor | alias | Keep as-is. | 11 | 8 | 0 |  |
| sin | alias | Keep as-is. | 2 | 0 | 0 |  |
| cos | alias | Keep as-is. | 2 | 0 | 0 |  |
| sqrt | alias | Keep as-is. | 2 | 52 | 0 |  |
| clamp | alias | Keep as-is. | 0 | 0 | 0 |  |

## MoE.axon (GROOM ME!)

| Op | Type | Suggestion | Use in builtins | Use in models | Use in operators | Comment |
| --- | --- | --- | ---: | ---: | ---: | --- |
| select | wrapper | Alioas? | 1 | 62 | 0 |  |
| scatter_add | alias | Groom for primitive or not. | 1 | 62 | 0 |  |
| grouped_ffn | alias | Deprecated facade; remove after migration to grouped_swiglu_ffn_basic paths. | 0 | 3 | 0 |  |
| grouped_swiglu_ffn | alias | Deprecated facade; remove after migration to grouped_swiglu_ffn_basic paths. | 0 | 0 | 0 |  |
| grouped_swiglu_ffn_basic | wrapper | Keep as-is. | 1 | 42 | 0 |  |
| grouped_swiglu_ffn_basic_granite | wrapper | Keep as-is. | 0 | 2 | 0 |  |
| sparsemixer_router | wrapper | Keep as-is. | 0 | 1 | 0 |  |

## Params.axon (GROOM and maybe add param_scale?)

| Op | Type | Suggestion | Use in builtins | Use in models | Use in operators | Comment |
| --- | --- | --- | ---: | ---: | ---: | --- |
| has_root | wrapper | Keep as-is. | 0 | 48 | 0 |  |
| root | wrapper | Deprecated; remove after remaining callers (if any) migrate to `has_root`/explicit roots. | 0 | 0 | 0 |  |
| param_scale | wrapper | Keep as canonical home for parameter scaling. | 0 | 0 | 0 |  |

## Positions.axon

| Op | Type | Suggestion | Use in builtins | Use in models | Use in operators | Comment |
| --- | --- | --- | ---: | ---: | ---: | --- |
| position_ids | derived | Keep as-is. | 0 | 245 | 0 | Derived from basic ops (`cast/cumsum/where/slice/arange/reshape/expand`). |
| position_ids_basic_masked | derived | Keep as-is. | 6 | 0 | 0 |  |
| position_ids_basic_nomask | derived | Keep as-is. | 6 | 0 | 0 |  |
| position_ids_basic | derived | Keep as-is. | 5 | 0 | 0 |  |
| linear_position_bias | alias | Deprecated facade; migrate to Prelude/Derived replacements. | 0 | 14 | 0 |  |
| position_ids_scale_basic | derived | Keep as-is. | 6 | 0 | 0 |  |
| relative_position_buckets_basic | derived | Keep as-is. | 6 | 0 | 0 |  |
| relative_position_bias_basic | derived | Keep as-is. | 5 | 5 | 0 |  |
| rope_pair_base_basic | derived | Keep as-is. | 6 | 0 | 0 |  |
| rope_pair_proportional_basic | derived | Keep as-is. | 6 | 0 | 0 |  |
| rope_pair_freq_scale_basic | derived | Keep as-is. | 6 | 2 | 0 |  |
| rope_pair_longrope_basic | derived | Keep as-is. | 6 | 0 | 0 |  |
| rope_pair_yarn_basic | derived | Keep as-is. | 6 | 0 | 0 |  |
| rope_pair_hf_yarn_basic | derived | Keep as-is. | 6 | 0 | 0 |  |
| relative_bias_t5 | alias | Deprecated facade; migrate to `Positions.relative_position_bias_basic` where possible. | 1 | 0 | 0 | Kept for compatibility wrappers. |
| relative_bias_disentangled | alias | Keep as wrapper path for DeBERTa until a Derived replacement exists. | 2 | 2 | 0 | Models now call this alias (not primitive form). |

## Prelude.axon

| Op | Type | Suggestion | Use in builtins | Use in models | Use in operators | Comment |
| --- | --- | --- | ---: | ---: | ---: | --- |
| linear | wrapper | Keep as-is. | 3 | 2806 | 0 |  |
| param | alias | Keep as-is. | 0 | 0 | 0 |  |
| param_scale | wrapper | Deprecated facade; use `Params.param_scale`. | 0 | 7 | 0 |  |
| embedding | wrapper | Keep as-is. | 10 | 640 | 0 |  |
| layernorm | wrapper | Keep as-is. | 0 | 274 | 0 |  |
| rmsnorm | wrapper | Keep as-is. | 0 | 586 | 0 |  |
| split | wrapper | Keep as-is. | 0 | 63 | 0 |  |
| chunk | wrapper | Keep as-is. | 1 | 54 | 0 |  |
| split_qkv_heads | alias | Deprecated facade; migrate callers to derived equivalents. | 0 | 18 | 0 |  |
| split_qkv_grouped | alias | Deprecated facade; migrate callers to derived equivalents. | 0 | 3 | 0 |  |
| reshape_heads | alias | Deprecated facade; migrate callers to derived equivalents. | 0 | 1029 | 0 |  |
| merge_heads | alias | Deprecated facade; migrate callers to derived equivalents. | 0 | 351 | 0 |  |
| permute | wrapper | Keep as-is. | 30 | 789 | 0 |  |
| transpose | wrapper | Keep as-is. | 20 | 526 | 0 |  |
| reshape | wrapper | Keep as-is. | 300 | 7890 | 0 |  |
| expand | wrapper | Keep as-is. | 40 | 1052 | 0 |  |
| arange | wrapper | Keep as-is. | 110 | 2893 | 0 |  |
| sinusoidal_positions | wrapper | Keep as-is or consider to derive from other primitives or keep it primitive for caching? | 10 | 269 | 0 |  |
| cast | wrapper | Keep as-is. | 150 | 3945 | 0 |  |
| cast_like | alias | Keep as-is. | 40 | 1052 | 0 |  |
| tensor_like | wrapper | Keep as-is. | 40 | 1052 | 0 |  |
| cumsum | wrapper | Keep as-is. | 10 | 263 | 0 |  |
| causal_mask | alias | Deprecated facade; migrate callers to derived equivalents. | 0 | 248 | 0 |  |
| blocksparse_mask | alias | Deprecated facade; migrate callers to derived equivalents. | 0 | 3 | 0 |  |
| bidirectional_mask | alias | Deprecated facade; migrate callers to derived equivalents. | 0 | 84 | 0 |  |
| attention | alias | Deprecated facade; migrate callers to derived equivalents. | 0 | 351 | 0 |  |
| position_ids | wrapper | Keep as-is. | 0 | 245 | 0 | Delegates to derived `Positions.position_ids`. |
| rope_pair_base | alias | Keep as-is. | 0 | 151 | 0 |  |
| rope_pair_proportional | alias | Keep as-is. | 0 | 10 | 0 |  |
| rope_pair_freq_scale | alias | Keep as-is. | 0 | 19 | 0 |  |
| rope_pair_longrope | alias | Keep as-is. | 0 | 12 | 0 |  |
| rope_pair_yarn | alias | Keep as-is. | 0 | 14 | 0 |  |
| rope_pair_hf_yarn | alias | Keep as-is. | 0 | 3 | 0 |  |
| linear_position_bias | alias | Deprecated facade; migrate callers to derived equivalents. | 0 | 14 | 0 |  |
| relative_bias_t5 | alias | Deprecated facade; prefer `Positions.relative_position_bias_basic`. | 0 | 0 | 0 |  |
| relative_bias_disentangled | alias | Keep as-is until a Derived equivalent exists. | 0 | 2 | 0 |  |
| repeat | wrapper | Keep as-is. | 0 | 376 | 0 |  |
| concat | wrapper | Keep as-is. | 74 | 2327 | 0 |  |
| matmul | alias | Keep as-is. | 40 | 1052 | 0 |  |
| eq | alias | Keep as-is. | 2 | 0 | 212 | `Use in operators` counts `==` binary expressions. |
| le | alias | Keep as-is. | 14 | 0 | 10 | `Use in operators` counts `<=` binary expressions. |
| and | alias | Keep as-is. | 3 | 0 | 47 | `Use in operators` counts boolean `and` expressions. |
| where | alias | Keep as-is. | 205 | 5445 | 0 |  |
| masked_fill | derived | Keep as-is. | 0 | 0 | 0 |  |
| slice | wrapper | Keep as-is. | 130 | 3419 | 0 |  |
| softmax | wrapper | Keep as-is. | 22 | 617 | 0 |  |
| topk | wrapper | Keep as-is. | 23 | 654 | 0 |  |
| dtype_value | wrapper | Keep as-is. | 11 | 300 | 0 |  |
| empty_like | wrapper | Keep as-is. | 10 | 263 | 0 |  |
| fill | wrapper | Keep as-is. | 10 | 263 | 0 |  |
| gather | wrapper | Keep as-is. | 2 | 74 | 0 |  |
| scatter | wrapper | Keep as-is. | 1 | 37 | 0 |  |
| zeros_like | alias | Keep as-is. | 20 | 563 | 0 |  |
| min_like | derived | Keep as-is. | 10 | 263 | 0 |  |

## Primitive Inventory (`brainsurgery/synapse/ops/*.py`)

| Primitive op (`_xyz`) | Type | Status | Suggestion | Use in builtins | Use in models | Use in operators | Comment |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| _activation | primitive | active | Keep as-is (or groom incrementally if still coarse). | 0 | 0 | 0 |  |
| _add | primitive | active | Keep as-is (or groom incrementally if still coarse). | 10 | 263 | 6255 | Inflated: mapped from all `+` binary expressions in Axon ASTs. |
| _arange | primitive | active | Keep as-is (or groom incrementally if still coarse). | 10 | 263 | 0 |  |
| _attention | primitive | to be removed | Replace with derived ops and remove primitive once migration is complete. | 10 | 267 | 0 |  |
| _bidirectional_mask | primitive | to be removed | Replace with derived ops and remove primitive once migration is complete. | 10 | 263 | 0 |  |
| _blocksparse_mask | primitive | active | Keep as-is (or groom incrementally if still coarse). | 10 | 263 | 0 |  |
| _cache_seq_len | primitive | to be removed | Remove after alias/tooling cleanup. | 0 | 0 | 0 |  |
| _cache_update | primitive | to be removed | Remove after alias/tooling cleanup. | 0 | 0 | 0 |  |
| _cast | primitive | active | Keep as-is (or groom incrementally if still coarse). | 10 | 263 | 0 |  |
| _cast_like | primitive | active | Keep as-is (or groom incrementally if still coarse). | 10 | 263 | 0 |  |
| _causal_conv1d | primitive | active | Keep as-is (or groom incrementally if still coarse). | 0 | 0 | 0 |  |
| _causal_mask | primitive | to be removed | Replace with derived ops and remove primitive once migration is complete. | 10 | 263 | 0 |  |
| _chunk | primitive | active | Keep as-is (or groom incrementally if still coarse). | 10 | 263 | 0 |  |
| _clamp | primitive | active | Keep as-is. | 0 | 0 | 0 |  |
| _concat | primitive | active | Keep as-is (or groom incrementally if still coarse). | 10 | 263 | 0 |  |
| _config_float | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 77 | 0 |  |
| _config_has | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 77 | 0 |  |
| _config_int | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 77 | 0 |  |
| _config_str | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 77 | 0 |  |
| _config_value | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 77 | 0 |  |
| _cos | primitive | active | Keep as-is (or groom incrementally if still coarse). | 10 | 263 | 0 |  |
| _cumsum | primitive | active | Keep as-is (or groom incrementally if still coarse). | 10 | 263 | 0 |  |
| _disentangled_relative_bias | primitive | to be removed | Replace with derived ops and remove primitive once migration is complete. | 1 | 0 | 0 |  |
| _div | primitive | active | Keep as-is (or groom incrementally if still coarse). | 10 | 263 | 16652 | Inflated: mapped from all `/` binary expressions, including non-tensor arithmetic. |
| _dtype_value | primitive | active | Keep as-is (or groom incrementally if still coarse). | 10 | 263 | 0 |  |
| _embedding | primitive | active | Keep as-is (or groom incrementally if still coarse). | 10 | 263 | 0 |  |
| _empty_like | primitive | active | Keep as-is (or groom incrementally if still coarse). | 10 | 263 | 0 |  |
| _eq | primitive | active | Keep as-is (or groom incrementally if still coarse). | 10 | 263 | 7612 | Inflated: mapped from all `==` binary expressions in Axon ASTs. |
| _exp | primitive | active | Keep as-is (or groom incrementally if still coarse). | 10 | 263 | 0 |  |
| _expand | primitive | active | Keep as-is (or groom incrementally if still coarse). | 10 | 263 | 0 |  |
| _fill | primitive | active | Keep as-is (or groom incrementally if still coarse). | 10 | 263 | 0 |  |
| _floor | primitive | active | Keep as-is (or groom incrementally if still coarse). | 10 | 263 | 0 |  |
| _gather | primitive | active | Keep as-is (or groom incrementally if still coarse). | 10 | 263 | 0 |  |
| _gemma4_moe_experts | primitive | active | Keep as-is (or groom incrementally if still coarse). | 0 | 0 | 0 |  |
| _gemma4_per_layer_input_at | primitive | active | Keep as-is (or groom incrementally if still coarse). | 0 | 0 | 0 |  |
| _gemma4_per_layer_inputs | primitive | active | Keep as-is (or groom incrementally if still coarse). | 0 | 0 | 0 |  |
| _gemma4_router | primitive | active | Keep as-is (or groom incrementally if still coarse). | 0 | 0 | 0 |  |
| _glm4_router | primitive | active | Keep as-is (or groom incrementally if still coarse). | 0 | 0 | 0 |  |
| _ir_alias | primitive | internal | Internal compiler op; keep hidden. | 0 | 0 | 0 |  |
| _ir_const | primitive | internal | Internal compiler op; keep hidden. | 0 | 0 | 0 |  |
| _l2norm | primitive | active | Keep as-is (or groom incrementally if still coarse). | 0 | 0 | 0 |  |
| _layernorm | primitive | active | Keep as-is (or groom incrementally if still coarse). | 10 | 263 | 0 |  |
| _le | primitive | active | Keep as-is (or groom incrementally if still coarse). | 10 | 263 | 1642 | Inflated: mapped from all `<=` binary expressions in Axon ASTs. |
| _linear | primitive | active | Keep as-is (or groom incrementally if still coarse). | 10 | 263 | 0 |  |
| _linear_position_bias | primitive | to be removed | Replace with derived ops and remove primitive once migration is complete. | 11 | 263 | 0 |  |
| _list_append | primitive | active | Keep as-is (or groom incrementally if still coarse). | 3 | 216 | 0 |  |
| _list_index | primitive | active | Keep as-is (or groom incrementally if still coarse). | 3 | 216 | 0 |  |
| _list_init | primitive | active | Keep as-is (or groom incrementally if still coarse). | 3 | 216 | 0 |  |
| _log | primitive | active | Keep as-is (or groom incrementally if still coarse). | 10 | 263 | 0 |  |
| _logical_and | primitive | active | Keep as-is (or groom incrementally if still coarse). | 0 | 0 | 319 | Inflated: mapped from boolean `and` expressions. |
| _mamba2_scan | primitive | active | Keep as-is (or groom incrementally if still coarse). | 0 | 0 | 0 |  |
| _mamba_scan | primitive | active | Keep as-is (or groom incrementally if still coarse). | 0 | 0 | 0 |  |
| _matmul | primitive | active | Keep as-is (or groom incrementally if still coarse). | 10 | 263 | 0 |  |
| _merge_heads | primitive | to be removed | Replace with derived ops and remove primitive once migration is complete. | 0 | 0 | 0 |  |
| _moe_grouped_ffn | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 37 | 0 |  |
| _moe_grouped_swiglu_ffn | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 37 | 0 |  |
| _moe_scatter_add | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 37 | 0 |  |
| _moe_select | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 37 | 0 |  |
| _mul | primitive | active | Keep as-is (or groom incrementally if still coarse). | 10 | 263 | 18076 | Inflated: mapped from all `*` binary expressions in Axon ASTs. |
| _nemotron_moe | primitive | active | Keep as-is (or groom incrementally if still coarse). | 0 | 0 | 0 |  |
| _param | primitive | active | Keep as-is (or groom incrementally if still coarse). | 10 | 263 | 0 |  |
| _param_scale | primitive | active | Keep as-is (or groom incrementally if still coarse). | 20 | 526 | 0 |  |
| _params_has_root | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 41 | 0 |  |
| _params_root | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 41 | 0 |  |
| _permute | primitive | active | Keep as-is (or groom incrementally if still coarse). | 10 | 263 | 0 |  |
| _position_ids | primitive | deprecated (to be removed) | Remove after migration to `Positions.position_ids` is complete. | 1 | 0 | 0 |  |
| _pow | primitive | active | Keep as-is (or groom incrementally if still coarse). | 0 | 0 | 3 | Used by expression `pow(...)` evaluators in runtime/codegen/materialization paths. |
| _repeat | primitive | active | Keep as-is (or groom incrementally if still coarse). | 10 | 263 | 0 |  |
| _reshape | primitive | active | Keep as-is (or groom incrementally if still coarse). | 10 | 263 | 0 |  |
| _reshape_heads | primitive | to be removed | Replace with derived ops and remove primitive once migration is complete. | 0 | 0 | 0 |  |
| _rmsnorm | primitive | active | Keep as-is (or groom incrementally if still coarse). | 10 | 263 | 0 |  |
| _rope_pair | primitive | to be removed | Replace with derived ops and remove primitive once migration is complete. | 0 | 0 | 0 |  |
| _scatter | primitive | active | Keep as-is (or groom incrementally if still coarse). | 10 | 263 | 0 |  |
| _select | primitive | active | Keep as-is (or groom incrementally if still coarse). | 0 | 0 | 0 |  |
| _sigmoid_topk_router | primitive | active | Keep as-is (or groom incrementally if still coarse). | 0 | 0 | 0 |  |
| _sin | primitive | active | Keep as-is (or groom incrementally if still coarse). | 10 | 263 | 0 |  |
| _sinusoidal_positions | primitive | active | Keep as-is (or groom incrementally if still coarse). | 10 | 263 | 0 |  |
| _slice | primitive | active | Keep as-is (or groom incrementally if still coarse). | 10 | 263 | 0 |  |
| _softmax | primitive | active | Keep as-is (or groom incrementally if still coarse). | 10 | 263 | 0 |  |
| _softmax_topk_router | primitive | active | Keep as-is (or groom incrementally if still coarse). | 0 | 0 | 0 |  |
| _split | primitive | active | Keep as-is (or groom incrementally if still coarse). | 10 | 263 | 0 |  |
| _split_qkv_grouped | primitive | to be removed | Replace with derived ops and remove primitive once migration is complete. | 10 | 263 | 0 |  |
| _split_qkv_heads | primitive | to be removed | Replace with derived ops and remove primitive once migration is complete. | 10 | 263 | 0 |  |
| _sqrt | primitive | active | Keep as-is (or groom incrementally if still coarse). | 10 | 263 | 0 |  |
| _t5_relative_position_bias | primitive | to be removed | Replace with derived ops and remove primitive once migration is complete. | 1 | 0 | 0 |  |
| _tensor_like | primitive | active | Keep as-is (or groom incrementally if still coarse). | 10 | 263 | 0 |  |
| _topk | primitive | active | Keep as-is (or groom incrementally if still coarse). | 10 | 263 | 0 |  |
| _transpose | primitive | active | Keep as-is (or groom incrementally if still coarse). | 10 | 263 | 0 |  |
| _unsqueeze | primitive | active | Keep as-is (or groom incrementally if still coarse). | 0 | 0 | 0 |  |
| _where | primitive | active | Keep as-is (or groom incrementally if still coarse). | 10 | 263 | 0 |  |
| _zeros_like | primitive | active | Keep as-is (or groom incrementally if still coarse). | 10 | 263 | 0 |  |
