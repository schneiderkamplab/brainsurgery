# Op Buckets Overview (Full Inventory)

This file inventories builtins surfaces and primitives (`_xyz`) with an explicit `Type` classification.

Type values: `primitive`, `derived`, `alias`, `wrapper` (non-alias wrapper), `namespace export` (Prelude namespace re-export).

Rule update: direct primitive calls are now enforced as `_xyz` syntax and only from builtins (`*.axon` in builtin namespaces). Model code should call builtins wrappers/aliases/derived ops.
Import policy update: `Prelude` now re-exports namespaces (`NN`, `Math`, `Tensor`) instead of wrapper symbols.

## Counting Methodology

- Scope: counts are computed from parsed Axon ASTs in `brainsurgery/synapse/builtins/*.axon` and `brainsurgery/synapse/models/**/*.axon`.
- `Use in builtins` / `Use in models`: count call expressions to each op, aggregating both unqualified (`foo`) and qualified (`Module.foo`) forms.
- `Use in operators`: counts infix operators in AST binary expressions, mapped to primitive/operator aliases:
  - `+` -> `_add` / `Math.add`
  - `/` -> `_div` / `Math.div`
  - `*` -> `_mul` / `Math.mul`
  - `==` -> `_eq` / `Tensor.eq`
  - `<=` -> `_le` / `Tensor.le`
  - `and` -> `_and` / `Tensor.and`
- Prelude namespace rows (`NN`, `Math`, `Tensor`) count qualified namespace calls (`NN.*`, `Math.*`, `Tensor.*`) across the same AST scope.
- Notes:
  - These are static call-site/operator counts, not runtime execution counts.
  - Files that intentionally do not parse as module programs (for example `Prelude.axon`) are excluded from call-expression traversal and handled by namespace counting.

## To Be Removed Primitives

| Primitive op (`_xyz`) | Use in builtins | Use in models | Status |
| --- | ---: | ---: | --- |
| _attention | 2 | 0 | to be removed (still active via `Compat.attention`) |
| _glm4_router | 1 | 0 | to be removed (still active via `MoE.glm4_router`) |
| _nemotron_moe | 1 | 0 | to be removed (still active via `MoE.nemotron_moe`) |
| _sigmoid_topk_router | 0 | 0 | to be removed |
| _moe_grouped_ffn | 1 | 0 | to be removed (still active via `MoE.grouped_ffn`) |
| _disentangled_relative_bias | 0 | 0 | to be removed |
| _gemma4_router | 0 | 0 | to be removed |
| _gemma4_moe_experts | 0 | 0 | to be removed |
| _moe_select | 0 | 0 | to be removed |
| _moe_scatter_add | 0 | 0 | to be removed |
| _softmax_topk_router | 0 | 0 | to be removed |
| _moe_grouped_swiglu_ffn | 0 | 0 | to be removed |
| _param_scale | 0 | 0 | to be removed |

Notes:
- `select` is intentionally excluded from this list: it is still used internally for lowering `?:` and `if ... then ... else`.

## Activations.axon

| Op | Type | Suggestion | Use in builtins | Use in models | Use in operators | Comment |
| --- | --- | --- | ---: | ---: | ---: | --- |
| gelu | alias | Keep as-is. | 1 | 21 | 0 |  |
| gelu_new | alias | Keep as-is. | 0 | 2 | 0 |  |
| gelu_pytorch_tanh | alias | Keep as-is. | 0 | 24 | 0 |  |
| gegelu | wrapper | Keep as-is. | 0 | 3 | 0 |  |
| relu | alias | Keep as-is. | 1 | 14 | 0 |  |
| relu2 | alias | Keep as-is. | 0 | 0 | 0 |  |
| sigmoid | alias | Keep as-is. | 0 | 0 | 0 |  |
| tanh | alias | Keep as-is. | 1 | 11 | 0 |  |
| silu | alias | Keep as-is. | 2 | 6 | 0 |  |
| swiglu | alias | Keep as-is. | 0 | 0 | 0 | No direct Axon call sites; keep for activation-name compatibility. |
| xielu | alias | Keep as-is. | 0 | 3 | 0 |  |

## Attention.axon

| Op | Type | Suggestion | Use in builtins | Use in models | Use in operators | Comment |
| --- | --- | --- | ---: | ---: | ---: | --- |
| mask_to_additive | derived | Keep as-is. | 1 | 12 | 0 |  |
| reshape_heads | derived | Keep as-is. | 3 | 1029 | 0 |  |
| merge_heads | derived | Keep as-is. | 0 | 0 | 0 |  |
| split_qkv_heads | derived | Keep as-is. | 0 | 17 | 0 |  |
| split_qkv_grouped | derived | Keep as-is. | 0 | 3 | 0 |  |
| attention_core | derived | Keep as-is. | 4 | 0 | 0 | Internal helper used by exported attention variants. |
| attention | derived | Keep as-is. | 0 | 353 | 0 |  |
| attention_gemma2 | derived | Keep as-is. | 0 | 4 | 0 |  |
| attention_hf | derived | Deprecated; remove. | 0 | 0 | 0 |  |
| attention_with_sinks | derived | Keep as-is. | 0 | 3 | 0 |  |

## Cache.axon

| Op | Type | Suggestion | Use in builtins | Use in models | Use in operators | Comment |
| --- | --- | --- | ---: | ---: | ---: | --- |
| update | wrapper | Keep as-is. | 0 | 191 | 0 |  |
| init | wrapper | Keep as-is. | 0 | 195 | 0 |  |
| index | wrapper | Keep as-is. | 0 | 205 | 0 |  |
| append | wrapper | Keep as-is. | 0 | 205 | 0 |  |
| past_length | wrapper | Keep as-is. | 0 | 178 | 0 |  |

## Compat.axon

| Op | Type | Suggestion | Use in builtins | Use in models | Use in operators | Comment |
| --- | --- | --- | ---: | ---: | ---: | --- |
| attention | alias | Deprecated; remove after migration. | 0 | 353 | 0 |  |
| attention_causal_scaled | wrapper | Deprecated; remove after migration. | 0 | 2 | 0 |  |

## Config.axon

| Op | Type | Suggestion | Use in builtins | Use in models | Use in operators | Comment |
| --- | --- | --- | ---: | ---: | ---: | --- |
| has_key | wrapper | Keep as-is. | 0 | 52 | 0 |  |
| has_value | wrapper | Keep as-is. | 0 | 24 | 0 |  |
| int | wrapper | Keep as-is. | 0 | 838 | 0 |  |
| float | wrapper | Keep as-is. | 0 | 281 | 0 |  |
| str | wrapper | Keep as-is. | 0 | 5 | 0 |  |
| bool | wrapper | Keep as-is. | 0 | 19 | 0 |  |
| list | wrapper | Keep as-is. | 0 | 8 | 0 |  |
| value | wrapper | Keep as-is. | 1 | 0 | 0 |  |

## List.axon

| Op | Type | Suggestion | Use in builtins | Use in models | Use in operators | Comment |
| --- | --- | --- | ---: | ---: | ---: | --- |
| init | alias | Keep as-is. | 1 | 0 | 0 |  |
| index | alias | Keep as-is. | 5 | 0 | 0 |  |
| append | alias | Keep as-is. | 1 | 0 | 0 |  |

## Masking.axon

| Op | Type | Suggestion | Use in builtins | Use in models | Use in operators | Comment |
| --- | --- | --- | ---: | ---: | ---: | --- |
| causal_mask_keep | derived | Keep as-is. | 2 | 0 | 0 |  |
| causal_mask_masked | derived | Keep as-is. | 1 | 0 | 0 |  |
| causal_mask | derived | Keep as-is. | 0 | 251 | 0 |  |
| bidirectional_mask_keep | derived | Keep as-is. | 2 | 0 | 0 |  |
| bidirectional_mask_masked | derived | Keep as-is. | 1 | 0 | 0 |  |
| bidirectional_mask | derived | Keep as-is. | 0 | 89 | 0 |  |
| blocksparse_mask | derived | Keep as-is. | 0 | 3 | 0 |  |

## Math.axon

| Op | Type | Suggestion | Use in builtins | Use in models | Use in operators | Comment |
| --- | --- | --- | ---: | ---: | ---: | --- |
| add | alias | Keep as-is. | 3 | 10 | 1047 | `Use in operators` counts `+` binary expressions. |
| div | alias | Keep as-is. | 0 | 0 | 548 | `Use in operators` counts `/` binary expressions (including shape/symbol math). |
| mul | alias | Keep as-is. | 1 | 233 | 498 | `Use in operators` counts `*` binary expressions. |
| pow | alias | Keep as-is. | 0 | 0 | 0 | No Axon call sites; backend expression evaluators still handle `pow(...)` (runtime/codegen/materialization). |
| exp | alias | Keep as-is. | 16 | 0 | 0 |  |
| log | alias | Keep as-is. | 24 | 8 | 0 |  |
| floor | alias | Keep as-is. | 13 | 8 | 0 |  |
| sin | alias | Keep as-is. | 3 | 0 | 0 |  |
| cos | alias | Keep as-is. | 3 | 0 | 0 |  |
| sqrt | alias | Keep as-is. | 0 | 52 | 0 |  |
| clamp | alias | Keep as-is. | 0 | 0 | 0 |  |

## MoE.axon

| Op | Type | Suggestion | Use in builtins | Use in models | Use in operators | Comment |
| --- | --- | --- | ---: | ---: | ---: | --- |
| gemma4_router | wrapper | Keep as-is. | 0 | 2 | 0 |  |
| gemma4_moe_experts | wrapper | Keep as-is. | 0 | 2 | 0 |  |
| glm4_router | wrapper | Keep as-is. | 0 | 6 | 0 |  |
| nemotron_moe | wrapper | Keep as-is. | 0 | 2 | 0 |  |
| select | derived | Keep as-is. | 2 | 27 | 0 |  |
| scatter_add | derived | Keep as-is. | 2 | 27 | 0 |  |
| softmax_topk_router | derived | Keep as-is. | 1 | 8 | 0 |  |
| sigmoid_topk_router | derived | Keep as-is. | 0 | 3 | 0 |  |
| grouped_ffn | alias | Keep as-is. | 0 | 3 | 0 |  |
| grouped_swiglu_ffn_basic | wrapper | Keep as-is. | 1 | 5 | 0 |  |
| grouped_swiglu_ffn_basic_granite | wrapper | Keep as-is. | 0 | 0 | 0 |  |
| sparsemixer_router | wrapper | Keep as-is. | 0 | 1 | 0 |  |

## NN.axon

| Op | Type | Suggestion | Use in builtins | Use in models | Use in operators | Comment |
| --- | --- | --- | ---: | ---: | ---: | --- |
| linear | alias | Keep as-is. | 3 | 2696 | 0 |  |
| embedding | wrapper | Keep as-is. | 1 | 379 | 0 |  |
| layernorm | alias | Keep as-is. | 0 | 270 | 0 |  |
| rmsnorm | wrapper | Keep as-is. | 1 | 589 | 0 |  |

## Params.axon

| Op | Type | Suggestion | Use in builtins | Use in models | Use in operators | Comment |
| --- | --- | --- | ---: | ---: | ---: | --- |
| has_root | wrapper | Keep as-is. | 0 | 44 | 0 |  |
| param | wrapper | Keep as-is. | 12 | 3 | 0 |  |
| param_scale | wrapper | Keep as canonical home for parameter scaling. | 1 | 7 | 0 |  |

## Positions.axon

| Op | Type | Suggestion | Use in builtins | Use in models | Use in operators | Comment |
| --- | --- | --- | ---: | ---: | ---: | --- |
| position_ids_masked | derived | Keep as-is. | 1 | 0 | 0 |  |
| position_ids_nomask | derived | Keep as-is. | 1 | 0 | 0 | Consider caching a base arange with `##` scope and slicing/rebasing with `#` scope for decode steps. |
| position_ids | derived | Keep as-is. | 0 | 248 | 0 | Derived from basic ops (`cast/cumsum/where/slice/arange/reshape/expand`). |
| position_ids_scale | derived | Keep as-is. | 1 | 0 | 0 |  |
| sinusoidal_positions | derived | Keep as-is. | 1 | 6 | 0 |  |
| linear_position_bias | derived | Keep as-is. | 0 | 13 | 0 | Migrated from `Compat.linear_position_bias` to canonical `Positions.linear_position_bias`. |
| rope_rotate_half_noninterleaved | derived | Keep as-is. | 1 | 0 | 0 |  |
| rope_rotate_half_interleaved | derived | Keep as-is. | 1 | 0 | 0 |  |
| rope_rotate_half | derived | Keep as-is. | 4 | 0 | 0 |  |
| rope_expand_half | derived | Keep as-is. | 4 | 0 | 0 |  |
| rope_expand_half_interleaved | derived | Keep as-is. | 4 | 0 | 0 |  |
| rope_apply_raw | derived | Keep as-is. | 3 | 0 | 0 |  |
| rope_apply_prefix | derived | Keep as-is. | 1 | 0 | 0 |  |
| rope_apply_with_partial | derived | Keep as-is. | 1 | 0 | 0 |  |
| rope_apply | derived | Keep as-is. | 2 | 0 | 0 |  |
| rope_pair_base | derived | Keep as-is. | 0 | 151 | 0 |  |
| rope_pair_proportional | derived | Keep as-is. | 0 | 10 | 0 |  |
| rope_inv_freq_base | derived | Keep as-is. | 1 | 0 | 0 |  |
| rope_inv_freq_freq_scale | derived | Keep as-is. | 1 | 0 | 0 |  |
| rope_apply_inv_freq | derived | Keep as-is. | 10 | 0 | 0 |  |
| rope_apply_inv_freq_prefix | derived | Keep as-is. | 1 | 0 | 0 |  |
| rope_apply_inv_freq_with_partial | derived | Keep as-is. | 2 | 0 | 0 |  |
| rope_pair_freq_scale | derived | Keep as-is. | 0 | 20 | 0 |  |
| rope_max_pos | derived | Keep as-is. | 1 | 0 | 0 |  |
| rope_longrope_use_long | derived | Keep as-is. | 2 | 0 | 0 |  |
| rope_inv_freq_longrope | derived | Keep as-is. | 1 | 0 | 0 |  |
| rope_attention_factor_longrope | derived | Keep as-is. | 1 | 0 | 0 |  |
| rope_pair_longrope | derived | Keep as-is. | 0 | 12 | 0 |  |
| rope_inv_freq_yarn | derived | Keep as-is. | 1 | 0 | 0 |  |
| rope_attention_factor_yarn | derived | Keep as-is. | 1 | 0 | 0 |  |
| rope_pair_yarn | derived | Keep as-is. | 0 | 14 | 0 |  |
| rope_inv_freq_hf_yarn | derived | Keep as-is. | 1 | 0 | 0 |  |
| rope_pair_hf_yarn | derived | Keep as-is. | 0 | 3 | 0 |  |
| relative_position_buckets | derived | Keep as-is. | 1 | 0 | 0 |  |
| relative_position_bias | derived | Keep as-is. | 0 | 52 | 0 |  |

## Prelude.axon

| Op | Type | Suggestion | Use in builtins | Use in models | Use in operators | Comment |
| --- | --- | --- | ---: | ---: | ---: | --- |
| NN | namespace export | Keep as-is. | 5 | 3934 | 0 | Re-exported namespace; wrappers live in `NN.axon`. |
| Math | namespace export | Keep as-is. | 10 | 26 | 0 | Re-exported namespace; aliases/wrappers live in `Math.axon`. |
| Tensor | namespace export | Keep as-is. | 327 | 609 | 0 | Re-exported namespace; wrappers/aliases/derived ops live in `Tensor.axon`. |

## SSM.axon

| Op | Type | Suggestion | Use in builtins | Use in models | Use in operators | Comment |
| --- | --- | --- | ---: | ---: | ---: | --- |
| activate | derived | Keep as-is. | 1 | 0 | 0 |  |
| causal_conv1d_step | derived | Keep as-is. | 2 | 0 | 0 |  |
| causal_conv1d | derived | Keep as-is. | 0 | 8 | 0 |  |
| mamba_scan_step | derived | Keep as-is. | 2 | 0 | 0 |  |
| mamba_scan | derived | Keep as-is. | 0 | 6 | 0 |  |
| mamba2_expand_groups | derived | Keep as-is. | 4 | 0 | 0 |  |
| mamba2_scan_step | derived | Keep as-is. | 2 | 0 | 0 |  |
| mamba2_scan | wrapper | Keep as-is. | 0 | 2 | 0 |  |

## Tensor.axon

| Op | Type | Suggestion | Use in builtins | Use in models | Use in operators | Comment |
| --- | --- | --- | ---: | ---: | ---: | --- |
| split | wrapper | Keep as-is. | 0 | 63 | 0 |  |
| chunk | wrapper | Keep as-is. | 2 | 19 | 0 |  |
| permute | wrapper | Keep as-is. | 9 | 0 | 0 |  |
| transpose | wrapper | Keep as-is. | 2 | 12 | 0 |  |
| reshape | wrapper | Keep as-is. | 82 | 9 | 0 |  |
| shape | wrapper | Keep as-is. | 3 | 0 | 0 |  |
| size | derived | Keep as-is. | 6 | 0 | 0 | Built from `shape + List.index`. |
| expand | wrapper | Keep as-is. | 9 | 0 | 0 |  |
| arange | wrapper | Keep as-is. | 21 | 0 | 0 |  |
| arange_step | derived | Keep as-is. | 1 | 0 | 0 |  |
| cast | wrapper | Keep as-is. | 21 | 0 | 0 |  |
| cast_like | alias | Keep as-is. | 5 | 0 | 0 |  |
| tensor_like | wrapper | Keep as-is. | 4 | 0 | 0 |  |
| cumsum | wrapper | Keep as-is. | 2 | 0 | 0 |  |
| unsqueeze | alias | Keep as-is. | 28 | 16 | 0 |  |
| repeat | wrapper | Keep as-is. | 0 | 376 | 0 |  |
| concat | wrapper | Keep as-is. | 17 | 28 | 0 |  |
| matmul | alias | Keep as-is. | 4 | 12 | 0 |  |
| eq | alias | Keep as-is. | 5 | 0 | 240 |  |
| le | alias | Keep as-is. | 20 | 0 | 11 |  |
| and | alias | Keep as-is. | 9 | 0 | 44 |  |
| where | alias | Keep as-is. | 28 | 0 | 0 |  |
| masked_fill | derived | Keep as-is. | 0 | 0 | 0 | Built from `where`. |
| slice | wrapper | Keep as-is. | 45 | 3 | 0 |  |
| softmax | wrapper | Keep as-is. | 4 | 17 | 0 |  |
| topk | wrapper | Keep as-is. | 5 | 17 | 0 |  |
| dtype_value | wrapper | Keep as-is. | 3 | 0 | 0 |  |
| empty_like | wrapper | Keep as-is. | 1 | 0 | 0 |  |
| fill | wrapper | Keep as-is. | 1 | 0 | 0 |  |
| gather | wrapper | Keep as-is. | 2 | 0 | 0 |  |
| scatter | wrapper | Keep as-is. | 1 | 0 | 0 |  |
| where_indices | alias | Keep as-is. | 1 | 0 | 0 |  |
| index_add | wrapper | Keep as-is. | 1 | 0 | 0 |  |
| zeros_like | alias | Keep as-is. | 3 | 37 | 0 |  |
| min_like | derived | Keep as-is. | 1 | 0 | 0 | Built from `empty_like + dtype_value + fill`. |

## Primitive Inventory (`brainsurgery/synapse/ops/*.py`)

| Primitive op (`_xyz`) | Type | Status | Suggestion | Use in builtins | Use in models | Use in operators | Comment |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| _activation | primitive | active | Keep as-is (or groom incrementally if still coarse). | 0 | 0 | 0 |  |
| _add | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 0 | 1047 | Inflated: mapped from all `+` binary expressions in Axon ASTs. |
| _arange | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 0 | 0 |  |
| _attention | primitive | to be removed | Replace with derived ops and remove primitive once migration is complete. | 2 | 0 | 0 |  |
| _cast | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 0 | 0 |  |
| _cast_like | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 0 | 0 |  |
| _chunk | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 0 | 0 |  |
| _clamp | primitive | active | Keep as-is. | 1 | 0 | 0 |  |
| _concat | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 0 | 0 |  |
| _config_bool | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 0 | 0 |  |
| _config_float | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 0 | 0 |  |
| _config_has | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 0 | 0 |  |
| _config_int | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 0 | 0 |  |
| _config_list | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 0 | 0 |  |
| _config_str | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 0 | 0 |  |
| _config_value | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 0 | 0 |  |
| _cos | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 0 | 0 |  |
| _cumsum | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 0 | 0 |  |
| _disentangled_relative_bias | primitive | to be removed | Replace with derived ops and remove primitive once migration is complete. | 1 | 0 | 0 |  |
| _div | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 0 | 548 | Inflated: mapped from all `/` binary expressions, including non-tensor arithmetic. |
| _dtype_value | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 0 | 0 |  |
| _embedding | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 0 | 0 |  |
| _empty_like | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 0 | 0 |  |
| _eq | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 0 | 240 | Inflated: mapped from all `==` binary expressions in Axon ASTs. |
| _exp | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 0 | 0 |  |
| _expand | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 0 | 0 |  |
| _fill | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 0 | 0 |  |
| _floor | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 0 | 0 |  |
| _gather | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 0 | 0 |  |
| _gemma4_moe_experts | primitive | to be removed | Replaced by derived `MoE.gemma4_moe_experts`; remove primitive after migration clean-up. | 0 | 0 | 0 |  |
| _gemma4_per_layer_input_at | primitive | active | Keep as-is (or groom incrementally if still coarse). | 0 | 0 | 0 |  |
| _gemma4_per_layer_inputs | primitive | active | Keep as-is (or groom incrementally if still coarse). | 0 | 0 | 0 |  |
| _gemma4_router | primitive | to be removed | Replaced by derived `MoE.gemma4_router`; remove primitive after migration clean-up. | 0 | 0 | 0 |  |
| _glm4_router | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 0 | 0 |  |
| _ir_alias | primitive | internal | Internal compiler op; keep hidden. | 0 | 0 | 0 |  |
| _ir_expr | primitive | internal | Internal compiler op; keep hidden. | 0 | 0 | 0 |  |
| _l2norm | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 0 | 0 |  |
| _layernorm | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 0 | 0 |  |
| _le | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 0 | 11 | Inflated: mapped from all `<=` binary expressions in Axon ASTs. |
| _linear | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 0 | 0 |  |
| _list_append | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 0 | 0 |  |
| _list_index | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 0 | 0 |  |
| _list_init | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 0 | 0 |  |
| _log | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 0 | 0 |  |
| _and | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 0 | 44 | Inflated: mapped from boolean `and` expressions. |
| _matmul | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 0 | 0 |  |
| _moe_grouped_ffn | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 0 | 0 |  |
| _moe_grouped_swiglu_ffn | primitive | active | Keep as-is (or groom incrementally if still coarse). | 0 | 0 | 0 |  |
| _moe_scatter_add | primitive | to be removed | Replaced by derived `MoE.scatter_add`; remove primitive after migration clean-up. | 0 | 0 | 0 |  |
| _moe_select | primitive | to be removed | Replaced by derived `MoE.select`; remove primitive after migration clean-up. | 0 | 0 | 0 |  |
| _mul | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 0 | 498 | Inflated: mapped from all `*` binary expressions in Axon ASTs. |
| _nemotron_moe | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 0 | 0 |  |
| _params_param | primitive | active | Keep as-is. | 1 | 0 | 0 | Canonical parameter lookup primitive (`Params.param`). |
| _param_scale | primitive | active | Keep as-is (or groom incrementally if still coarse). | 0 | 0 | 0 |  |
| _params_has_root | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 0 | 0 |  |
| _permute | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 0 | 0 |  |
| _pow | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 0 | 0 | Used by expression `pow(...)` evaluators in runtime/codegen/materialization paths. |
| _repeat | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 0 | 0 |  |
| _reshape | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 0 | 0 |  |
| _rmsnorm | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 0 | 0 |  |
| _scatter | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 0 | 0 |  |
| _select | primitive | active | Keep as-is (or groom incrementally if still coarse). | 0 | 0 | 0 |  |
| _shape | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 0 | 0 |  |
| _sigmoid_topk_router | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 0 | 0 |  |
| _sin | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 0 | 0 |  |
| _slice | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 0 | 0 |  |
| _softmax | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 0 | 0 |  |
| _softmax_topk_router | primitive | to be removed | Replaced by derived `MoE.softmax_topk_router`; remove primitive after migration clean-up. | 0 | 0 | 0 |  |
| _split | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 0 | 0 |  |
| _sqrt | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 0 | 0 |  |
| _tensor_like | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 0 | 0 |  |
| _topk | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 0 | 0 |  |
| _transpose | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 0 | 0 |  |
| _unsqueeze | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 0 | 0 |  |
| _where | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 0 | 0 |  |
| _where_indices | primitive | active | Keep as-is. | 1 | 0 | 0 | Used by derived MoE token routing selection. |
| _index_add | primitive | active | Keep as-is. | 1 | 0 | 0 | Used by derived MoE scatter accumulation. |
| _zeros_like | primitive | active | Keep as-is (or groom incrementally if still coarse). | 1 | 0 | 0 |  |
