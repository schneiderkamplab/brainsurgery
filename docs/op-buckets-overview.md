# Op Buckets With Active Calls (Transposed)

Counts are textual call-site counts from:
- `brainsurgery/synapse/models/**/*.axon` (models)
- `brainsurgery/synapse/builtins/*.axon` (builtins)

## Groomed primitive ops

| Defining .axon | Op | Models call sites | Builtins call sites | Suggestions |
|---|---|---:|---:|---|
| Prelude.axon | linear | 2695 | 6 | Keep as-is. |
| Prelude.axon | layernorm | 274 | 3 | Keep as-is. |
| Prelude.axon | embedding | 381 | 4 | Consider `_to_idx` follow-up for stricter index typing. |
| Prelude.axon | repeat | 376 | 3 | Keep as-is. |
| Prelude.axon | rmsnorm | 586 | 3 | Keep as-is. |
| Prelude.axon | split | 63 | 3 | Keep as-is. |
| Prelude.axon | chunk | 17 | 4 | Keep as-is. |
| Prelude.axon | slice | 0 | 16 | Keep as-is. |
| Prelude.axon | reshape | 0 | 33 | Keep as-is. |
| Prelude.axon | permute | 0 | 6 | Keep as-is. |
| Prelude.axon | transpose | 57 | 15 | Keep as-is. |
| Prelude.axon | expand | 0 | 7 | Keep as-is. |
| Prelude.axon | arange | 0 | 14 | Keep as-is. |
| Prelude.axon | cast | 0 | 18 | Keep as-is. |
| Prelude.axon | cumsum | 0 | 4 | Keep as-is. |
| Prelude.axon | softmax | 17 | 7 | Keep as-is. |
| Prelude.axon | topk | 17 | 8 | Keep as-is. |
| Math.axon | add | 12 | 4 | Keep as-is. |
| Math.axon | div | 0 | 0 | Keep as-is. |
| Math.axon | exp | 0 | 8 | Keep as-is. |
| Math.axon | floor | 8 | 11 | Keep as-is. |
| Math.axon | log | 8 | 18 | Keep as-is. |
| Prelude.axon | matmul | 0 | 7 | Keep as-is. |
| Math.axon | mul | 231 | 4 | Keep as-is. |
| Math.axon | pow | 0 | 3 | Keep as-is. |
| Math.axon | sin | 0 | 7 | Keep as-is. |
| Math.axon | cos | 0 | 7 | Keep as-is. |
| Math.axon | clamp | 1 | 0 | Keep as-is. |
| Math.axon | sqrt | 54 | 5 | Keep as-is. |
| Prelude.axon | eq | 0 | 5 | Keep as-is. |
| Prelude.axon | le | 0 | 17 | Keep as-is. |
| Prelude.axon | and | 47 | 6 | Keep as-is. |
| Prelude.axon | where | 1 | 28 | Keep as-is. |
| Prelude.axon | zeros_like | 37 | 5 | Keep as-is. |

## Deprecated operations

| Defining .axon | Op | Models call sites | Builtins call sites | Suggestions |
|---|---|---:|---:|---|
| Prelude.axon | bidirectional_mask | 84 | 3 | Migrate to `bidirectional_mask_basic`. |
| Prelude.axon | causal_mask | 248 | 3 | Migrate to `causal_mask_basic` + `mask_to_additive_basic`. |
| Prelude.axon | attention | 432 | 3 | Migrate to `attention_basic(_hf)`. |
| Prelude.axon | reshape_heads | 1029 | 2 | Migrate to Derived/Prelude `reshape_head_basic`. |
| Prelude.axon | merge_heads | 351 | 2 | Migrate to `merge_heads_basic`. |
| Prelude.axon | position_ids | 245 | 36 | Migrate to `position_ids_basic`. |
| Prelude.axon | split_qkv_heads | 18 | 3 | Replace with derived split helpers. |
| Prelude.axon | split_qkv_grouped | 3 | 3 | Replace with derived grouped split helper. |
| Prelude.axon | linear_position_bias | 14 | 6 | Replace with `relative_position_bias_basic`. |
| Prelude.axon | blocksparse_mask | 3 | 3 | Replace with derived sparse-mask helper. |
| - | t5_relative_position_bias | 48 | 1 | Replace with `relative_position_bias_basic`. |
| - | disentangled_relative_bias | 2 | 1 | Replace with dedicated Derived helper. |
| MoE.axon | moe_select | 0 | 1 | Keep deprecated; remove when model callers are migrated. |
| MoE.axon | moe_scatter_add | 0 | 1 | Keep deprecated; remove when model callers are migrated. |
| MoE.axon | moe_grouped_ffn | 0 | 1 | Keep deprecated; remove when model callers are migrated. |
| MoE.axon | moe_grouped_swiglu_ffn | 0 | 1 | Keep deprecated; remove when model callers are migrated. |
| - | softmax_topk_router | 8 | 0 | Move to derived router module. |
| - | sigmoid_topk_router | 3 | 0 | Move to derived router module. |
| - | nemotron_moe | 2 | 0 | Move to derived nemotron module. |
| - | gemma4_router | 2 | 0 | Move to derived gemma4 router path. |
| - | gemma4_moe_experts | 2 | 0 | Move to derived gemma4 experts path. |
| - | gemma4_per_layer_inputs | 3 | 0 | Move to derived gemma4 helpers. |
| - | gemma4_per_layer_input_at | 3 | 0 | Move to derived gemma4 helpers. |
| - | glm4_router | 6 | 0 | Move to derived glm4 helper. |
| - | mamba_scan | 6 | 0 | Move to derived mamba helper. |
| - | mamba2_scan | 2 | 0 | Move to derived mamba2 helper. |
| - | causal_conv1d | 8 | 0 | Move to derived causal conv helper. |

## To be removed ops

| Defining .axon | Op | Models call sites | Builtins call sites | Suggestions |
|---|---|---:|---:|---|
| - | rope_pair | 0 | 0 | Delete primitive after final migration confirmation. |
| - | cache_seq_len | 0 | 2 | Remove after compiler/tooling alias metadata is cleaned up. |
| - | cache_update | 0 | 1 | Remove after compiler/tooling alias metadata is cleaned up. |
| Prelude.axon | param_scale (primitive) | 0 | 0 | Keep during migration; now path-base normalized (`scale`) and used by wrapper-level scale paths. Remove only after wrapper/runtime path handling is fully stable. |

## Derived ops (current built-in definitions)

| Defining .axon | Op | Models call sites | Builtins call sites | Suggestions |
|---|---|---:|---:|---|
| Derived.axon | reshape_head_basic | 0 | 4 | Candidate to replace deprecated primitive everywhere. |
| Derived.axon | merge_heads_basic | 13 | 4 | Keep adoption in progress. |
| Derived.axon | mask_to_additive_basic | 0 | 4 | Keep and reuse from mask builders. |
| Prelude.axon | min_like | 0 | 3 | Move toward `empty_like` + `fill` replacement if parity is unchanged. |
| Prelude.axon | param_scale (derived helper) | 7 | 2 | Keep until primitive removal finalizes. |
| Derived.axon | position_ids_basic | 6 | 4 | Continue migrating callers; remove deprecated `position_ids` uses. |
| Derived.axon | position_ids_basic_nomask | 0 | 3 | Keep internal helper. |
| Derived.axon | position_ids_basic_masked | 0 | 3 | Keep internal helper. |
| Derived.axon | causal_mask_basic | 7 | 2 | Continue migrating deprecated callers. |
| Derived.axon | causal_mask_basic_nomask | 0 | 4 | Keep internal helper. |
| Derived.axon | causal_mask_basic_masked | 0 | 3 | Keep internal helper. |
| Derived.axon | bidirectional_mask_basic | 7 | 2 | Continue replacing deprecated primitive. |
| Derived.axon | bidirectional_mask_basic_nomask | 0 | 4 | Keep internal helper. |
| Derived.axon | bidirectional_mask_basic_masked | 0 | 3 | Keep internal helper. |
| Derived.axon | attention_basic | 13 | 2 | Use as default derived attention where fidelity proven. |
| Derived.axon | attention_basic_hf | 0 | 2 | Keep as optional HF-aligned variant. |
| Derived.axon | rope_pair_base_basic | 0 | 4 | Keep and continue rope migration. |
| Derived.axon | rope_pair_proportional_basic | 0 | 4 | Keep and continue rope migration. |
| Derived.axon | rope_pair_freq_scale_basic | 2 | 4 | Keep and continue rope migration. |
| Derived.axon | rope_pair_longrope_basic | 0 | 4 | Keep and continue rope migration. |
| Derived.axon | rope_pair_yarn_basic | 0 | 4 | Keep and continue rope migration. |
| Derived.axon | rope_pair_hf_yarn_basic | 0 | 4 | Keep and continue rope migration. |
| Derived.axon | relative_position_bias_basic | 5 | 2 | Keep and continue T5 migration. |
| Derived.axon | relative_position_buckets_basic | 0 | 3 | Keep helper internal. |

## Unbucketed primitives (added)

| Defining .axon | Op | Models call sites | Builtins call sites | Suggestions |
|---|---|---:|---:|---|
| Prelude.axon | cast_like | 0 | 7 | Bucket and groom. |
| Prelude.axon | concat | 28 | 12 | Bucket and groom. |
| - | config_float | 0 | 1 | Bucket and groom. |
| - | config_has | 0 | 1 | Bucket and groom. |
| - | config_int | 0 | 1 | Bucket and groom. |
| - | config_str | 0 | 1 | Bucket and groom. |
| - | config_value | 0 | 1 | Bucket and groom. |
| Prelude.axon | dtype_value | 0 | 5 | Bucket and groom. |
| Prelude.axon | empty_like | 0 | 4 | Bucket and groom. |
| Prelude.axon | fill | 0 | 4 | Bucket and groom. |
| Prelude.axon | gather | 0 | 5 | Bucket and groom. |
| - | _ir_alias | 0 | 0 | Internal; keep hidden. |
| - | _ir_expr | 0 | 0 | Internal; keep hidden. |
| - | l2norm | 6 | 0 | Bucket and groom. |
| List.axon | list_append | 0 | 1 | Merge `_list_*` into list built-in counts and groom API. |
| List.axon | list_index | 0 | 1 | Merge `_list_*` into list built-in counts and groom API. |
| List.axon | list_init | 0 | 1 | Merge `_list_*` into list built-in counts and groom API. |
| Prelude.axon | param | 0 | 4 | Bucket and groom. |
| Params.axon | params_has_root | 0 | 1 | Bucket and groom. |
| Params.axon | params_root | 0 | 1 | Bucket and groom. |
| - | scatter | 0 | 4 | Bucket and groom. |
| MoE.axon | select | 25 | 3 | Bucket and groom. |
| Prelude.axon | sinusoidal_positions | 6 | 4 | Bucket and groom (candidate for Derived decomposition later). |
| Prelude.axon | tensor_like | 0 | 7 | Bucket and groom. |
| - | unsqueeze | 16 | 1 | Bucket and groom. |
