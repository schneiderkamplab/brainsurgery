# Op Buckets Overview (Full Inventory)

This file inventories builtins surfaces and primitives (`_xyz`) with an explicit `Type` classification.

Type values: `primitive`, `derived`, `alias`, `wrapper` (non-alias wrapper), `namespace export` (Prelude namespace re-export).

Rule update: direct primitive calls are now enforced as `_xyz` syntax and only from builtins (`*.axon` in builtin namespaces). Model code should call builtins wrappers/aliases/derived ops.
Import policy update: `Prelude` now re-exports namespaces (`NN`, `Math`, `Tensor`) instead of wrapper symbols.
Last refreshed: 2026-04-19 (AST recount over `builtins/*.axon` and split `models/**/*.axon`).

## Counting Methodology

- Scope: counts are computed from parsed Axon ASTs in `brainsurgery/synapse/builtins/*.axon` and `brainsurgery/synapse/models/**/*.axon`.
- `Use in builtins` / `Use in generic models` / `Use in materialized models`: count call expressions to each op, aggregating both unqualified (`foo`) and qualified (`Module.foo`) forms.
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

## Activations.axon

| Op | Type | Suggestion | Use in builtins | Use in generic models | Use in materialized models | Use in operators | Comment |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gelu | alias | Keep as-is. | 1 | 6 | 15 | 0 |  |
| gelu_new | alias | Keep as-is. | 2 | 1 | 1 | 0 |  |
| gelu_pytorch_tanh | alias | Keep as-is. | 0 | 7 | 17 | 0 |  |
| gegelu | wrapper | Keep as-is. | 0 | 1 | 2 | 0 |  |
| relu | alias | Keep as-is. | 2 | 3 | 11 | 0 |  |
| relu2 | alias | Keep as-is. | 0 | 0 | 0 | 0 |  |
| sigmoid | alias | Keep as-is. | 6 | 0 | 0 | 0 |  |
| tanh | alias | Keep as-is. | 2 | 4 | 7 | 0 |  |
| silu | alias | Keep as-is. | 3 | 3 | 3 | 0 |  |
| swiglu | alias | Keep as-is. | 0 | 0 | 0 | 0 | No direct Axon call sites; keep for activation-name compatibility. |
| xielu | alias | Keep as-is. | 0 | 1 | 2 | 0 |  |

## Attention.axon

| Op | Type | Suggestion | Use in builtins | Use in generic models | Use in materialized models | Use in operators | Comment |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mask_to_additive | derived | Keep as-is. | 1 | 0 | 0 | 0 |  |
| reshape_heads | derived | Keep as-is. | 3 | 292 | 737 | 0 |  |
| merge_heads | derived | Keep as-is. | 0 | 0 | 0 | 0 |  |
| split_qkv_heads | derived | Keep as-is. | 0 | 3 | 14 | 0 |  |
| split_qkv_grouped | derived | Keep as-is. | 0 | 1 | 2 | 0 |  |
| attention | derived | Keep as-is. | 0 | 102 | 261 | 0 |  |

## Cache.axon

| Op | Type | Suggestion | Use in builtins | Use in generic models | Use in materialized models | Use in operators | Comment |
| --- | --- | --- | --- | --- | --- | --- | --- |
| update | wrapper | Keep as-is. | 0 | 52 | 139 | 0 |  |
| init | wrapper | Keep as-is. | 0 | 53 | 142 | 0 |  |
| index | wrapper | Keep as-is. | 0 | 57 | 148 | 0 |  |
| append | wrapper | Keep as-is. | 0 | 57 | 148 | 0 |  |
| past_length | wrapper | Keep as-is. | 0 | 50 | 128 | 0 |  |

## Config.axon

| Op | Type | Suggestion | Use in builtins | Use in generic models | Use in materialized models | Use in operators | Comment |
| --- | --- | --- | --- | --- | --- | --- | --- |
| has_key | wrapper | Keep as-is. | 0 | 52 | 0 | 0 |  |
| has_value | wrapper | Keep as-is. | 0 | 24 | 0 | 0 |  |
| int | wrapper | Keep as-is. | 0 | 840 | 0 | 0 |  |
| float | wrapper | Keep as-is. | 0 | 281 | 0 | 0 |  |
| str | wrapper | Keep as-is. | 0 | 5 | 0 | 0 |  |
| bool | wrapper | Keep as-is. | 0 | 19 | 0 | 0 |  |
| list | wrapper | Keep as-is. | 0 | 8 | 0 | 0 |  |
| value | wrapper | Keep as-is. | 1 | 0 | 0 | 0 |  |

## List.axon

| Op | Type | Suggestion | Use in builtins | Use in generic models | Use in materialized models | Use in operators | Comment |
| --- | --- | --- | --- | --- | --- | --- | --- |
| init | alias | Keep as-is. | 1 | 0 | 0 | 0 |  |
| index | alias | Keep as-is. | 2 | 0 | 0 | 0 |  |
| append | alias | Keep as-is. | 1 | 0 | 0 | 0 |  |

## Masking.axon

| Op | Type | Suggestion | Use in builtins | Use in generic models | Use in materialized models | Use in operators | Comment |
| --- | --- | --- | --- | --- | --- | --- | --- |
| causal_mask_keep | derived | Keep as-is. | 2 | 0 | 0 | 0 |  |
| causal_mask_masked | derived | Keep as-is. | 1 | 0 | 0 | 0 |  |
| causal_mask | derived | Keep as-is. | 0 | 70 | 183 | 0 |  |
| bidirectional_mask_keep | derived | Keep as-is. | 2 | 0 | 0 | 0 |  |
| bidirectional_mask_masked | derived | Keep as-is. | 1 | 0 | 0 | 0 |  |
| bidirectional_mask | derived | Keep as-is. | 0 | 34 | 83 | 0 |  |
| blocksparse_mask | derived | Keep as-is. | 0 | 1 | 2 | 0 |  |

## Math.axon

| Op | Type | Suggestion | Use in builtins | Use in generic models | Use in materialized models | Use in operators | Comment |
| --- | --- | --- | --- | --- | --- | --- | --- |
| add | alias | Keep as-is. | 0 | 2 | 8 | 1054 | `Use in operators` counts `+` binary expressions. |
| div | alias | Keep as-is. | 0 | 0 | 0 | 582 | `Use in operators` counts `/` binary expressions (including shape/symbol math). |
| mul | alias | Keep as-is. | 3 | 67 | 166 | 548 | `Use in operators` counts `*` binary expressions. |
| pow | alias | Keep as-is. | 0 | 0 | 0 | 0 | No Axon call sites; backend expression evaluators still handle `pow(...)` (runtime/codegen/materialization). |
| exp | alias | Keep as-is. | 17 | 0 | 0 | 0 |  |
| log | alias | Keep as-is. | 27 | 3 | 5 | 0 |  |
| floor | alias | Keep as-is. | 13 | 3 | 5 | 0 |  |
| sin | alias | Keep as-is. | 3 | 0 | 0 | 0 |  |
| cos | alias | Keep as-is. | 3 | 0 | 0 | 0 |  |
| sqrt | alias | Keep as-is. | 2 | 46 | 17 | 0 |  |
| clamp | alias | Keep as-is. | 8 | 0 | 0 | 0 |  |

## MoE.axon

| Op | Type | Suggestion | Use in builtins | Use in generic models | Use in materialized models | Use in operators | Comment |
| --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_sigmoid_router | wrapper | Keep as-is. | 1 | 2 | 4 | 0 |  |
| grouped_swiglu_ffn_basic | wrapper | Keep as-is. | 0 | 2 | 3 | 0 |  |
| scatter_add | derived | Keep as-is. | 2 | 13 | 19 | 0 |  |
| select | derived | Keep as-is. | 2 | 13 | 19 | 0 |  |
| sigmoid_topk_router | derived | Keep as-is. | 0 | 1 | 2 | 0 |  |
| softmax_topk_router | derived | Keep as-is. | 0 | 4 | 6 | 0 |  |
| sparsemixer_router | wrapper | Keep as-is. | 0 | 1 | 0 | 0 |  |

## NN.axon

| Op | Type | Suggestion | Use in builtins | Use in generic models | Use in materialized models | Use in operators | Comment |
| --- | --- | --- | --- | --- | --- | --- | --- |
| linear | alias | Keep as-is. | 0 | 774 | 1915 | 0 |  |
| embedding | wrapper | Keep as-is. | 0 | 113 | 264 | 0 |  |
| layernorm | alias | Keep as-is. | 0 | 88 | 176 | 0 |  |
| rmsnorm | wrapper | Keep as-is. | 0 | 147 | 431 | 0 |  |

## Params.axon

| Op | Type | Suggestion | Use in builtins | Use in generic models | Use in materialized models | Use in operators | Comment |
| --- | --- | --- | --- | --- | --- | --- | --- |
| has_root | wrapper | Keep as-is. | 0 | 33 | 12 | 0 |  |
| param | wrapper | Keep as-is. | 23 | 1 | 2 | 0 |  |
| param_scale | wrapper | Keep as canonical home for parameter scaling. | 1 | 3 | 4 | 0 |  |

## Positions.axon

| Op | Type | Suggestion | Use in builtins | Use in generic models | Use in materialized models | Use in operators | Comment |
| --- | --- | --- | --- | --- | --- | --- | --- |
| position_ids_masked | derived | Keep as-is. | 1 | 0 | 0 | 0 |  |
| position_ids_nomask | derived | Keep as-is. | 1 | 0 | 0 | 0 | Consider caching a base arange with `##` scope and slicing/rebasing with `#` scope for decode steps. |
| position_ids | derived | Keep as-is. | 0 | 73 | 175 | 0 | Derived from basic ops (`cast/cumsum/where/slice/arange/reshape/expand`). |
| position_ids_scale | derived | Keep as-is. | 1 | 0 | 0 | 0 |  |
| sinusoidal_positions | derived | Keep as-is. | 1 | 1 | 5 | 0 |  |
| linear_position_bias | derived | Keep as-is. | 0 | 2 | 11 | 0 | Migrated from `Compat.linear_position_bias` to canonical `Positions.linear_position_bias`. |
| rope_rotate_half_noninterleaved | derived | Keep as-is. | 1 | 0 | 0 | 0 |  |
| rope_rotate_half_interleaved | derived | Keep as-is. | 1 | 0 | 0 | 0 |  |
| rope_rotate_half | derived | Keep as-is. | 4 | 0 | 0 | 0 |  |
| rope_expand_half | derived | Keep as-is. | 4 | 0 | 0 | 0 |  |
| rope_expand_half_interleaved | derived | Keep as-is. | 4 | 0 | 0 | 0 |  |
| rope_apply_raw | derived | Keep as-is. | 3 | 0 | 0 | 0 |  |
| rope_apply_prefix | derived | Keep as-is. | 1 | 0 | 0 | 0 |  |
| rope_apply_with_partial | derived | Keep as-is. | 1 | 0 | 0 | 0 |  |
| rope_apply | derived | Keep as-is. | 2 | 0 | 0 | 0 |  |
| rope_pair_base | derived | Keep as-is. | 0 | 40 | 111 | 0 |  |
| rope_pair_proportional | derived | Keep as-is. | 0 | 4 | 6 | 0 |  |
| rope_inv_freq_base | derived | Keep as-is. | 1 | 0 | 0 | 0 |  |
| rope_inv_freq_freq_scale | derived | Keep as-is. | 1 | 0 | 0 | 0 |  |
| rope_apply_inv_freq | derived | Keep as-is. | 10 | 0 | 0 | 0 |  |
| rope_apply_inv_freq_prefix | derived | Keep as-is. | 1 | 0 | 0 | 0 |  |
| rope_apply_inv_freq_with_partial | derived | Keep as-is. | 2 | 0 | 0 | 0 |  |
| rope_pair_freq_scale | derived | Keep as-is. | 0 | 5 | 15 | 0 |  |
| rope_max_pos | derived | Keep as-is. | 1 | 0 | 0 | 0 |  |
| rope_longrope_use_long | derived | Keep as-is. | 2 | 0 | 0 | 0 |  |
| rope_inv_freq_longrope | derived | Keep as-is. | 1 | 0 | 0 | 0 |  |
| rope_attention_factor_longrope | derived | Keep as-is. | 1 | 0 | 0 | 0 |  |
| rope_pair_longrope | derived | Keep as-is. | 0 | 4 | 8 | 0 |  |
| rope_inv_freq_yarn | derived | Keep as-is. | 1 | 0 | 0 | 0 |  |
| rope_attention_factor_yarn | derived | Keep as-is. | 1 | 0 | 0 | 0 |  |
| rope_pair_yarn | derived | Keep as-is. | 0 | 5 | 9 | 0 |  |
| rope_inv_freq_hf_yarn | derived | Keep as-is. | 1 | 0 | 0 | 0 |  |
| rope_pair_hf_yarn | derived | Keep as-is. | 0 | 1 | 2 | 0 |  |
| relative_position_buckets | derived | Keep as-is. | 1 | 0 | 0 | 0 |  |
| relative_position_bias | derived | Keep as-is. | 0 | 12 | 40 | 0 |  |

## Prelude.axon

| Op | Type | Suggestion | Use in builtins | Use in generic models | Use in materialized models | Use in operators | Comment |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NN | namespace export | Keep as-is. | 0 | 1129 | 2796 | 0 | Re-exported namespace; wrappers live in `NN.axon`. |
| Math | namespace export | Keep as-is. | 0 | 2 | 8 | 0 | Re-exported namespace; aliases/wrappers live in `Math.axon`. |
| Tensor | namespace export | Keep as-is. | 42 | 183 | 411 | 0 | Re-exported namespace; wrappers/aliases/derived ops live in `Tensor.axon`. |

## SSM.axon

| Op | Type | Suggestion | Use in builtins | Use in generic models | Use in materialized models | Use in operators | Comment |
| --- | --- | --- | --- | --- | --- | --- | --- |
| activate | derived | Keep as-is. | 1 | 0 | 0 | 0 |  |
| causal_conv1d_step | derived | Keep as-is. | 2 | 0 | 0 | 0 |  |
| causal_conv1d | derived | Keep as-is. | 0 | 4 | 4 | 0 |  |
| mamba_scan_step | derived | Keep as-is. | 2 | 0 | 0 | 0 |  |
| mamba_scan | derived | Keep as-is. | 0 | 3 | 3 | 0 |  |
| mamba2_expand_groups | derived | Keep as-is. | 4 | 0 | 0 | 0 |  |
| mamba2_scan_step | derived | Keep as-is. | 2 | 0 | 0 | 0 |  |
| mamba2_scan | wrapper | Keep as-is. | 0 | 1 | 1 | 0 |  |

## Tensor.axon

| Op | Type | Suggestion | Use in builtins | Use in generic models | Use in materialized models | Use in operators | Comment |
| --- | --- | --- | --- | --- | --- | --- | --- |
| split | wrapper | Keep as-is. | 0 | 24 | 39 | 0 |  |
| chunk | wrapper | Keep as-is. | 1 | 7 | 12 | 0 |  |
| permute | wrapper | Keep as-is. | 8 | 0 | 0 | 0 |  |
| transpose | wrapper | Keep as-is. | 1 | 0 | 0 | 0 |  |
| reshape | wrapper | Keep as-is. | 10 | 4 | 8 | 0 |  |
| shape | wrapper | Keep as-is. | 3 | 0 | 0 | 0 |  |
| size | derived | Keep as-is. | 2 | 0 | 0 | 0 | Built from `shape + List.index`. |
| expand | wrapper | Keep as-is. | 22 | 0 | 0 | 0 |  |
| arange | wrapper | Keep as-is. | 25 | 0 | 0 | 0 |  |
| arange_step | derived | Keep as-is. | 1 | 0 | 0 | 0 |  |
| cast | wrapper | Keep as-is. | 2 | 0 | 0 | 0 |  |
| cast_like | alias | Keep as-is. | 10 | 0 | 0 | 0 |  |
| tensor_like | wrapper | Keep as-is. | 4 | 0 | 0 | 0 |  |
| cumsum | wrapper | Keep as-is. | 2 | 0 | 0 | 0 |  |
| unsqueeze | alias | Keep as-is. | 0 | 6 | 10 | 0 |  |
| repeat | wrapper | Keep as-is. | 0 | 101 | 275 | 0 |  |
| concat | wrapper | Keep as-is. | 3 | 9 | 19 | 0 |  |
| matmul | alias | Keep as-is. | 2 | 0 | 0 | 0 |  |
| eq | alias | Keep as-is. | 0 | 1 | 2 | 248 |  |
| le | alias | Keep as-is. | 33 | 0 | 0 | 11 |  |
| and | alias | Keep as-is. | 11 | 0 | 0 | 30 |  |
| where | alias | Keep as-is. | 2 | 0 | 0 | 0 |  |
| masked_fill | derived | Keep as-is. | 0 | 0 | 0 | 0 | Built from `where`. |
| slice | wrapper | Keep as-is. | 7 | 2 | 4 | 0 |  |
| softmax | wrapper | Keep as-is. | 2 | 7 | 10 | 0 |  |
| topk | wrapper | Keep as-is. | 0 | 7 | 10 | 0 |  |
| dtype_value | wrapper | Keep as-is. | 3 | 0 | 0 | 0 |  |
| empty_like | wrapper | Keep as-is. | 1 | 0 | 0 | 0 |  |
| fill | wrapper | Keep as-is. | 1 | 0 | 0 | 0 |  |
| gather | wrapper | Keep as-is. | 9 | 0 | 0 | 0 |  |
| scatter | wrapper | Keep as-is. | 2 | 0 | 0 | 0 |  |
| where_indices | alias | Keep as-is. | 0 | 0 | 0 | 0 |  |
| index_add | wrapper | Keep as-is. | 1 | 0 | 0 | 0 |  |
| zeros_like | alias | Keep as-is. | 1 | 15 | 22 | 0 |  |
| min_like | derived | Keep as-is. | 1 | 0 | 0 | 0 | Built from `empty_like + dtype_value + fill`. |

## Primitive Inventory (`brainsurgery/synapse/ops/*.py`)

| Primitive op (`_xyz`) | Type | Status | Suggestion | Use in builtins | Use in generic models | Use in materialized models | Use in operators | Comment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| _activation | primitive | active | 0 | 0 | 0 | 0 | 0 |  |
| _add | primitive | active | 1 | 1 | 0 | 0 | 1054 | Inflated: mapped from all `+` binary expressions in Axon ASTs. |
| _arange | primitive | active | 1 | 1 | 0 | 0 | 0 |  |
| _cast | primitive | active | 1 | 1 | 0 | 0 | 0 |  |
| _cast_like | primitive | active | 1 | 1 | 0 | 0 | 0 |  |
| _chunk | primitive | active | 1 | 1 | 0 | 0 | 0 |  |
| _clamp | primitive | active | 1 | 1 | 0 | 0 | 0 |  |
| _concat | primitive | active | 1 | 1 | 0 | 0 | 0 |  |
| _config_bool | primitive | active | 1 | 1 | 0 | 0 | 0 |  |
| _config_float | primitive | active | 1 | 1 | 0 | 0 | 0 |  |
| _config_has | primitive | active | 1 | 1 | 0 | 0 | 0 |  |
| _config_int | primitive | active | 1 | 1 | 0 | 0 | 0 |  |
| _config_list | primitive | active | 1 | 1 | 0 | 0 | 0 |  |
| _config_str | primitive | active | 1 | 1 | 0 | 0 | 0 |  |
| _config_value | primitive | active | 1 | 1 | 0 | 0 | 0 |  |
| _cos | primitive | active | 1 | 1 | 0 | 0 | 0 |  |
| _cumsum | primitive | active | 1 | 1 | 0 | 0 | 0 |  |
| _div | primitive | active | 1 | 1 | 0 | 0 | 582 | Inflated: mapped from all `/` binary expressions, including non-tensor arithmetic. |
| _dtype_value | primitive | active | 1 | 1 | 0 | 0 | 0 |  |
| _embedding | primitive | active | 1 | 1 | 0 | 0 | 0 |  |
| _empty_like | primitive | active | 1 | 1 | 0 | 0 | 0 |  |
| _eq | primitive | active | 1 | 1 | 0 | 0 | 248 | Inflated: mapped from all `==` binary expressions in Axon ASTs. |
| _exp | primitive | active | 1 | 1 | 0 | 0 | 0 |  |
| _expand | primitive | active | 1 | 1 | 0 | 0 | 0 |  |
| _fill | primitive | active | 1 | 1 | 0 | 0 | 0 |  |
| _floor | primitive | active | 1 | 1 | 0 | 0 | 0 |  |
| _gather | primitive | active | 1 | 1 | 0 | 0 | 0 |  |
| _gemma4_per_layer_input_at | primitive | active | 0 | 0 | 0 | 0 | 0 |  |
| _gemma4_per_layer_inputs | primitive | active | 0 | 0 | 0 | 0 | 0 |  |
| _ir_alias | primitive | internal | 0 | 0 | 0 | 0 | 0 |  |
| _ir_expr | primitive | internal | 0 | 0 | 0 | 0 | 0 |  |
| _l2norm | primitive | active | 1 | 1 | 0 | 0 | 0 |  |
| _layernorm | primitive | active | 1 | 1 | 0 | 0 | 0 |  |
| _le | primitive | active | 1 | 1 | 0 | 0 | 11 | Inflated: mapped from all `<=` binary expressions in Axon ASTs. |
| _linear | primitive | active | 1 | 1 | 0 | 0 | 0 |  |
| _list_append | primitive | active | 1 | 1 | 0 | 0 | 0 |  |
| _list_index | primitive | active | 1 | 1 | 0 | 0 | 0 |  |
| _list_init | primitive | active | 1 | 1 | 0 | 0 | 0 |  |
| _log | primitive | active | 1 | 1 | 0 | 0 | 0 |  |
| _and | primitive | active | 1 | 1 | 0 | 0 | 30 | Inflated: mapped from boolean `and` expressions. |
| _matmul | primitive | active | 1 | 1 | 0 | 0 | 0 |  |
| _mul | primitive | active | 1 | 1 | 0 | 0 | 548 | Inflated: mapped from all `*` binary expressions in Axon ASTs. |
| _params_param | primitive | active | 1 | 1 | 0 | 0 | 0 | Canonical parameter lookup primitive (`Params.param`). |
| _params_has_root | primitive | active | 1 | 1 | 0 | 0 | 0 |  |
| _permute | primitive | active | 1 | 1 | 0 | 0 | 0 |  |
| _pow | primitive | active | 1 | 1 | 0 | 0 | 0 | Used by expression `pow(...)` evaluators in runtime/codegen/materialization paths. |
| _repeat | primitive | active | 1 | 1 | 0 | 0 | 0 |  |
| _reshape | primitive | active | 1 | 1 | 0 | 0 | 0 |  |
| _rmsnorm | primitive | active | 1 | 1 | 0 | 0 | 0 |  |
| _scatter | primitive | active | 1 | 1 | 0 | 0 | 0 |  |
| _select | primitive | active | 0 | 0 | 0 | 0 | 0 |  |
| _shape | primitive | active | 1 | 1 | 0 | 0 | 0 |  |
| _sin | primitive | active | 1 | 1 | 0 | 0 | 0 |  |
| _slice | primitive | active | 1 | 1 | 0 | 0 | 0 |  |
| _softmax | primitive | active | 1 | 1 | 0 | 0 | 0 |  |
| _split | primitive | active | 1 | 1 | 0 | 0 | 0 |  |
| _sqrt | primitive | active | 1 | 1 | 0 | 0 | 0 |  |
| _tensor_like | primitive | active | 1 | 1 | 0 | 0 | 0 |  |
| _topk | primitive | active | 1 | 1 | 0 | 0 | 0 |  |
| _transpose | primitive | active | 1 | 1 | 0 | 0 | 0 |  |
| _unsqueeze | primitive | active | 1 | 1 | 0 | 0 | 0 |  |
| _where | primitive | active | 1 | 1 | 0 | 0 | 0 |  |
| _where_indices | primitive | active | 1 | 1 | 0 | 0 | 0 | Used by derived MoE token routing selection. |
| _index_add | primitive | active | 1 | 1 | 0 | 0 | 0 | Used by derived MoE scatter accumulation. |
| _zeros_like | primitive | active | 1 | 1 | 0 | 0 | 0 |  |
