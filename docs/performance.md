# Synapse Codegen Performance Review

This note covers the Axon-to-Synapse lowering and Synapse Python code generation paths, with emphasis on improvements that should increase generated-model performance without changing numerical fidelity.

The review was grounded in:

- [brainsurgery/synapse/codegen.py](/work/training/brainsurgery/brainsurgery/synapse/codegen.py)
- [brainsurgery/synapse/axon/lowering_core.py](/work/training/brainsurgery/brainsurgery/synapse/axon/lowering_core.py)
- all modules in [brainsurgery/synapse/ops](/work/training/brainsurgery/brainsurgery/synapse/ops)
- representative generated output in [examples/gpt2_synapse.py](/work/training/brainsurgery/examples/gpt2_synapse.py) and [examples/gemma3_270m_synapse.py](/work/training/brainsurgery/examples/gemma3_270m_synapse.py)

## Summary

The main performance opportunities are systemic codegen issues rather than isolated slow ops:

1. Repeated runtime parameter-path resolution and dictionary lookup in hot blocks.
2. Missed constant folding and dead-branch elimination during lowering and emission.
3. Re-creation of helper tensors such as `arange`, ramps, masks, and scalar tensors on every call.
4. Repeated weight casting and dtype-alignment work.
5. Python-side allocation and control-flow overhead in sequence kernels and MoE paths.

Simple elementwise and structural ops are mostly fine already. Most gains come from improving what is emitted around the kernels.

## Highest-Impact Opportunities

### 1. Hoist or pre-resolve parameter paths and parameter lookups

Generated code currently resolves many parameter references at runtime by composing strings from node scope and then indexing into `_state`.

Relevant code:

- [brainsurgery/synapse/codegen.py:865](/work/training/brainsurgery/brainsurgery/synapse/codegen.py#L865)
- [brainsurgery/synapse/codegen.py:117](/work/training/brainsurgery/brainsurgery/synapse/codegen.py#L117)
- [brainsurgery/synapse/codegen.py:153](/work/training/brainsurgery/brainsurgery/synapse/codegen.py#L153)
- [brainsurgery/synapse/ops/linear.py:203](/work/training/brainsurgery/brainsurgery/synapse/ops/linear.py#L203)
- [examples/gpt2_synapse.py:75](/work/training/brainsurgery/examples/gpt2_synapse.py#L75)
- [examples/gemma3_270m_synapse.py:110](/work/training/brainsurgery/examples/gemma3_270m_synapse.py#L110)

Observed pattern:

- `_infer_param_expr()` emits calls to `_scope_of()`, `_join_scope()`, `_pick_param_from_single()`, and `_pick_param_path()`.
- Generated ops then call `_param(...)` or `_state.get(...)` repeatedly, often multiple times inside a single node.
- Generated examples show the same path string being rebuilt for weight lookup, bias lookup, and empty-tensor fallback shape lookup.

Why it matters:

- String construction and dictionary access happen in the forward hot path.
- The cost compounds in per-layer and per-token loops.
- This overhead is pure bookkeeping and does not contribute to actual tensor math.

Suggested improvements:

- When scope and parameter roots are statically known, emit direct literal paths instead of runtime path builders.
- Within a node, emit a local path variable once and reuse it for all accesses.
- For static parameters, resolve and cache tensor references in `__init__` or on first use.
- Keep dynamic fallback only for cases that truly depend on `_scope`, `_param_root_expr`, or dynamic block dispatch.

Expected impact:

- Broad reduction in Python overhead across almost every generated model.
- Especially valuable for transformer blocks with many tiny parameter lookups.

### 2. Fold constants and remove dead branches before codegen

A substantial amount of generated Python is validating or branching on values that are already compile-time constants.

Relevant code:

- [brainsurgery/synapse/axon/lowering_core.py:516](/work/training/brainsurgery/brainsurgery/synapse/axon/lowering_core.py#L516)
- [brainsurgery/synapse/codegen.py:926](/work/training/brainsurgery/brainsurgery/synapse/codegen.py#L926)
- [brainsurgery/synapse/ops/reshape_heads.py:96](/work/training/brainsurgery/brainsurgery/synapse/ops/reshape_heads.py#L96)
- [brainsurgery/synapse/ops/select.py:88](/work/training/brainsurgery/brainsurgery/synapse/ops/select.py#L88)
- [brainsurgery/synapse/ops/config_int.py:63](/work/training/brainsurgery/brainsurgery/synapse/ops/config_int.py#L63)
- [brainsurgery/synapse/ops/params_root.py:78](/work/training/brainsurgery/brainsurgery/synapse/ops/params_root.py#L78)

Observed pattern:

- `reshape_heads` emits runtime branches for `heads is None` and `head_dim is None` even when one is a literal constant.
- `select` always emits both branches and nested graph compilation.
- Config and params ops often do runtime parsing or scanning when values may already be statically known after lowering.

Why it matters:

- It expands the generated Python and increases runtime branching.
- It obscures the actual hot tensor path.
- It duplicates checks across repeated invocations of the same block.

Suggested improvements:

- Extend lowering to constant-fold numeric and boolean expressions into direct literals or `_ir_const`.
- If a `select` condition is known, emit only the taken branch.
- If `reshape_heads` gets constant `heads`, emit the straight-line reshape with only the necessary validation.
- Fold `Config.*` and `Params.*` calls when their inputs are fully known at emission time.

Expected impact:

- Smaller generated source.
- Lower Python dispatch overhead.
- Easier downstream optimization by TorchDynamo or other tracing systems.

### 3. Cache helper tensors by device, dtype, and shape

Several ops rebuild shape-dependent helper tensors every call.

Relevant code:

- [brainsurgery/synapse/ops/causal_mask.py:179](/work/training/brainsurgery/brainsurgery/synapse/ops/causal_mask.py#L179)
- [brainsurgery/synapse/ops/rope_pair.py:451](/work/training/brainsurgery/brainsurgery/synapse/ops/rope_pair.py#L451)
- [brainsurgery/synapse/ops/rope_pair.py:495](/work/training/brainsurgery/brainsurgery/synapse/ops/rope_pair.py#L495)
- [brainsurgery/synapse/ops/position_ids.py:171](/work/training/brainsurgery/brainsurgery/synapse/ops/position_ids.py#L171)
- [brainsurgery/synapse/ops/linear_position_bias.py:129](/work/training/brainsurgery/brainsurgery/synapse/ops/linear_position_bias.py#L129)

Observed pattern:

- `causal_mask` constructs `arange`, boolean masks, and scalar tensors when a cache miss occurs.
- `rope_pair` repeatedly constructs `inv_freq`, ramps, and other helper tensors.
- `position_ids` rebuilds `torch.arange(...)` for common decode shapes.
- `linear_position_bias` rebuilds ALiBi slopes.

Why it matters:

- These tensors are often deterministic functions of shape and config.
- In generation, the same shapes recur many times.
- Allocation and initialization overhead becomes noticeable when sequence kernels are otherwise efficient.

Suggested improvements:

- Generalize the existing `_causal_mask_cache` approach to:
  - RoPE inverse-frequency tensors
  - YaRN ramps and interpolation factors
  - ALiBi slope tensors
  - decode-time position ranges
- Key caches by stable tuples such as `(device, dtype, dim, mode, relevant_config)`.
- Store reusable scalar tensors where tensor-typed scalars are required.

Expected impact:

- Lower allocator pressure.
- Less repeated setup work in decode loops.
- Large benefit for small-batch autoregressive generation.

### 4. Add a casted-parameter cache for dtype alignment

Multiple ops repeatedly cast weights and biases to match input dtype or FP32 accumulation rules.

Relevant code:

- [brainsurgery/synapse/ops/linear.py:232](/work/training/brainsurgery/brainsurgery/synapse/ops/linear.py#L232)
- [brainsurgery/synapse/ops/layernorm.py:135](/work/training/brainsurgery/brainsurgery/synapse/ops/layernorm.py#L135)
- [brainsurgery/synapse/ops/rmsnorm.py:136](/work/training/brainsurgery/brainsurgery/synapse/ops/rmsnorm.py#L136)
- [brainsurgery/synapse/ops/mamba_scan.py:341](/work/training/brainsurgery/brainsurgery/synapse/ops/mamba_scan.py#L341)

Observed pattern:

- `linear` conditionally casts weight and bias on every execution.
- norm ops call `.float()` on parameters inside every forward.
- `mamba_scan` casts all working tensors into `work_dtype` every run.

Why it matters:

- Parameter dtype conversion is deterministic for a given state dict and requested dtype.
- Repeating it in the hot path wastes memory bandwidth and adds allocations.

Suggested improvements:

- Add a small cache on the generated model:
  - key: `(param_path, dtype)`
  - value: cast tensor
- Invalidate it in `load_state_dict_tensors()`.
- Use it only for immutable model parameters, not activations.
- Keep exact same accumulation rules and output casts.

Expected impact:

- Reduced overhead for repeated inference calls.
- Most visible on CPU and small-batch GPU inference where Python overhead matters more.

## Medium-Impact Opportunities

### 5. Preallocate outputs in recurrent kernels instead of building Python lists

Relevant code:

- [brainsurgery/synapse/ops/mamba_scan.py:353](/work/training/brainsurgery/brainsurgery/synapse/ops/mamba_scan.py#L353)

Observed pattern:

- `mamba_scan` appends one tensor per time step to a Python list and stacks at the end.

Why it matters:

- Python list append and final `torch.stack` create extra overhead and temporary memory traffic.

Suggested improvements:

- Emit `y_out = torch.empty((batch, seq, dim), ...)` once.
- Write directly into `y_out[:, t, :]` inside the loop.
- Preserve existing `work_dtype` and final cast behavior.

Expected impact:

- Lower Python overhead and lower temporary allocation count.

### 6. Replace tiny tensor allocations with scalar math or fused tensor ops

Relevant code:

- [brainsurgery/synapse/ops/embedding.py:107](/work/training/brainsurgery/brainsurgery/synapse/ops/embedding.py#L107)
- [brainsurgery/synapse/ops/causal_mask.py:245](/work/training/brainsurgery/brainsurgery/synapse/ops/causal_mask.py#L245)

Observed pattern:

- `embedding` multiplies by `torch.tensor(float(scale), dtype=..., device=...)`.
- `causal_mask` uses `torch.where` with `torch.zeros(())` and `torch.full(())` scalar tensors.

Why it matters:

- These are tiny allocations but they occur in common paths.

Suggested improvements:

- Emit multiplication by Python float where type promotion is already correct.
- Use `masked_fill`, `new_zeros`, or `new_full` patterns to avoid recreating scalar tensors.

Expected impact:

- Small per-op gain, but broad across common models.

### 7. Simplify repeated backend checks and fallback loops in MoE code

Relevant code:

- [brainsurgery/synapse/ops/moe_grouped_ffn.py:435](/work/training/brainsurgery/brainsurgery/synapse/ops/moe_grouped_ffn.py#L435)
- [brainsurgery/synapse/ops/moe_grouped_ffn.py:490](/work/training/brainsurgery/brainsurgery/synapse/ops/moe_grouped_ffn.py#L490)
- [brainsurgery/synapse/ops/moe_grouped_ffn.py:461](/work/training/brainsurgery/brainsurgery/synapse/ops/moe_grouped_ffn.py#L461)
- [brainsurgery/synapse/ops/moe_grouped_ffn.py:516](/work/training/brainsurgery/brainsurgery/synapse/ops/moe_grouped_ffn.py#L516)

Observed pattern:

- `grouped_mm` backend checks are re-evaluated every call.
- Alignment checks are lengthy and repeated twice.
- Fallback code converts offsets to Python lists and loops in Python.

Why it matters:

- MoE paths are already complex and often latency-sensitive.

Suggested improvements:

- Cache backend choice per `(device.type, dtype, transpose, availability)`.
- Precompute transposed expert weights if that layout is repeatedly needed.
- Replace `histc` with exact integer counting such as `bincount` where supported.
- Keep Python fallback, but move invariant checks out of the hot path.

Expected impact:

- Moderate benefit on MoE-heavy models, especially CPU and small-batch inference.

### 8. Reduce generator-emitted bookkeeping in `forward()` hot paths

Relevant code:

- [brainsurgery/synapse/codegen.py:422](/work/training/brainsurgery/brainsurgery/synapse/codegen.py#L422)
- [brainsurgery/synapse/codegen.py:664](/work/training/brainsurgery/brainsurgery/synapse/codegen.py#L664)
- [brainsurgery/synapse/codegen.py:320](/work/training/brainsurgery/brainsurgery/synapse/codegen.py#L320)

Observed pattern:

- `forward()` recreates `input_specs` inline on each call.
- `_compile_graph()` emits node-path strings for every node, even when tracing and runtime capture are disabled.
- Runtime value recording recursively clones outputs when enabled.

Why it matters:

- The default inference path pays some cost to support optional tracing features.

Suggested improvements:

- Emit `INPUT_SPECS` and similar metadata as class constants rather than rebuilding them in `forward()`.
- Only emit node-path and trace-record lines under a runtime guard that is checked once per block or graph.
- Consider a codegen option for a lean inference-only class with tracing removed entirely.

Expected impact:

- Lower Python overhead in large graphs.
- Cleaner generated code.

## Lower-Impact Areas

These ops are generally already close to optimal, and do not appear to justify bespoke rewrites unless profiling shows otherwise:

- `add`
- `mul`
- `clamp`
- `concat`
- `split`
- `topk`
- `zeros_like`
- list ops
- most config ops

For these, the main gains still come from:

- constant folding
- removal of dead validation branches
- shared helper caching

## Concrete Evidence in Generated Output

Representative generated files show the same patterns clearly:

- [examples/gpt2_synapse.py:75](/work/training/brainsurgery/examples/gpt2_synapse.py#L75) repeatedly rebuilds node paths, resolves parameters, and duplicates `reshape_heads` validation code.
- [examples/gpt2_synapse.py:167](/work/training/brainsurgery/examples/gpt2_synapse.py#L167) shows `causal_mask` setup logic and cache-key construction in the hot path.
- [examples/gemma3_270m_synapse.py:110](/work/training/brainsurgery/examples/gemma3_270m_synapse.py#L110) shows repeated `.float()`, `.type_as(...)`, and parameter lookups in norm-heavy blocks.

## Recommended Implementation Order

If these changes are pursued, the best order is:

1. Static parameter-path hoisting and per-node lookup deduplication.
2. Lowering-time constant folding and dead-branch elimination.
3. Cached helper tensors for masks, RoPE, and positional helpers.
4. Casted-parameter cache for aligned weights.
5. Mamba and MoE micro-optimizations.

That order should deliver the largest performance gains with the lowest risk of affecting output fidelity.
