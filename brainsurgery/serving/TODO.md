# Serving Optimizations (vLLM-competitive roadmap)

## Done

### Chunked Prefill ✅
- Slice long prefills into chunks, interleave decode tokens between chunks.
- **How**: `Sequence.prefill_pos` tracks progress; `schedule()` returns chunk-size inputs;
  `advance_prefill()` advances position for intermediate chunks; engine skips sampling
  on intermediate chunks and calls `advance_prefill` instead.
- **Files**: `scheduler/base.py`, `scheduler/continuous.py`, `engine.py`.
- **Config**: `Engine(prefill_chunk_size=N)` — defaults to 0 (disabled).
- **Benefit**: Biggest latency win for mixed-length workloads.

### Prefix Caching ✅
- **Problem**: Common prefixes (system prompt, conversation history) recompute KV cache per request.
- **Solution**: Token-hash block IDs (`_hash_to_block`), refcounting (`_block_refcount`),
  copy-on-write in `append_layer_tokens` / `append`, `register_blocks` for incremental
  hash registration across chunks.
- **Where**: `cache/paged.py` — `init_entry(seq_id, prompt_tokens)` finds cached prefix,
  pre-populates `block_table`; `register_blocks` fills hash map after prefill.
- **How**: `init_entry` collects all full-block hashes, decouples cache-hit checking from
  hash collection; `register_blocks` uses `_next_block_to_register` counter for correct
  incremental registration across chunked prefill.
- **Files**: `cache/{base,paged,mlx_paged,tinygrad_paged}.py`, `engine.py`,
  `scheduler/{base,continuous}.py`.
- **Tests**: `TestPrefixCaching` — 4 tests: identical, shared prefix, +chunking, background loop.
- **Benefit**: Big throughput win for chat workloads with shared prefixes.

### Copy-on-Write for Shared Blocks ✅
- Built into prefix caching implementation. When `append_layer_tokens` / `append` writes to
  a block with refcount > 1, `_cow_block` clones the data into a newly allocated block and
  updates the sequence's block table.
- **Where**: `cache/paged.py` — `_cow_block` method.

## High Priority

### `F.scaled_dot_product_attention` (torch backend only — low effort, high reward)
- **Problem**: `paged.py` gather + vanilla attention is much slower than fused kernels.
- **Solution**: Replace manual attention score computation in the torch paged path with
  `torch.nn.functional.scaled_dot_product_attention`. This is a drop-in change — PyTorch 2.x
  automatically selects the best fused kernel (FlashAttention, memory-efficient attention, or
  a fallback) with no downsides for standard causal attention. The serving model's attention
  is already compatible.
- **Where**: `cache/paged.py` (the attention/score step) — torch backend only.
- **MLX note**: No action needed — MLX's default attention already uses fused Metal kernels
  analogous to FA2 via `mx.fast.scaled_dot_product_attention`.
- **Benefit**: ~2-4x attention speedup AND O(N²)→O(N) memory (no materialized attention matrix),
  zero new deps, no regressions.

### `torch.compile` / CUDA Graphs (engine hot loop)
- **Problem**: Python dispatch overhead per decode step in the serving engine.
- **Precedent**: Axon models are already `torch.compile` compatible in the
  test/validation pipeline (`synapse/axon_test.py:_maybe_compile_model` wraps HF
  or Axon models with `torch.compile`). The serving engine just doesn't do it yet.
- **Solution**: Wrap the model's forward call in `_execute_plan_sequential` /
  `_execute_plan_batched` with `torch.compile`, or capture the full decode step
  as a CUDA graph. Model architecture is compatible — wiring is what's missing.
- **Where**: `engine.py` forward path.
- **Benefit**: Reduces per-step overhead, most noticeable at small batch sizes.

## Lower Priority

### Block-level Memory Management
- **Problem**: Current allocation may OOM under high concurrency.
- **Solution**: Add CPU swap, pre-allocation watermarks, LRU eviction for prefix cache entries.
- **Where**: `cache/paged.py` block manager.
- **Benefit**: Robustness under load.
