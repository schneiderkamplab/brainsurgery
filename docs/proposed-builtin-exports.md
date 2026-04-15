# Proposed Builtin Export Surface (for review)

Derived from actual cross-file imports and namespaced calls in `brainsurgery/synapse/**/*.axon`.

| Builtin | Proposed exports | Count | Plain import count |
|---|---|---:|---:|
| Activations | gegelu, gelu, gelu_new, gelu_pytorch_tanh, relu, relu2, sigmoid, silu, tanh, xielu | 10 | 262 |
| Attention | attention, mask_to_additive, merge_heads, reshape_heads | 4 | 241 |
| Cache | append, index, init, past_length, update | 5 | 192 |
| Compat | attention, attention_causal_scaled, causal_conv1d, gemma4_per_layer_input_at, gemma4_per_layer_inputs, glm4_router, linear_position_bias, mamba_scan, relative_bias_disentangled, sinusoidal_positions, split_qkv_heads, t5_relative_position_bias | 12 | 242 |
| Config | bool, float, has_key, has_value, int, list, str, value | 8 | 77 |
| List | append, index, init | 3 | 3 |
| Masking | bidirectional_mask, causal_mask | 2 | 241 |
| Math | PI, add, cos, exp, floor, log, mul, sin, sqrt | 9 | 4 |
| MoE | grouped_ffn, grouped_swiglu_ffn_basic, grouped_swiglu_ffn_basic_granite, scatter_add, select, sparsemixer_router | 6 | 37 |
| NN | embedding, layernorm, linear, rmsnorm | 4 | 1 |
| Params | has_root, param, param_scale | 3 | 44 |
| Positions | position_ids, relative_position_bias, rope_pair_base, rope_pair_freq_scale, rope_pair_hf_yarn, rope_pair_longrope, rope_pair_proportional, rope_pair_yarn | 8 | 2 |
| Prelude | | 0 | 1 |
| Tensor | and, arange, cast, cast_like, chunk, concat, cumsum, dtype_value, eq, expand, gather, le, matmul, min_like, permute, repeat, reshape, scatter, size, slice, softmax, split, tensor_like, topk, transpose, unsqueeze, where, zeros_like | 28 | 5 |

## Notes

- This is usage-driven, not design-driven. It captures what is currently referenced.
- Extraction includes both top-level constants and module bodies.
- Namespace self-exports (for example `Compat`) are intentionally omitted from this proposal.
- `Plain import count` shows how many files currently contain `import <Builtin>` (no member list).
