# T2 self-report

- **Final artifact path:** `out/T2/solution.py` (output `out/T2/model.safetensors`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why:** none.
- **Pitfalls or surprises you hit:**
  - `mlp.c_proj.weight` shares the suffix `c_proj.weight` with the attention output projection; matched on the structured key path (`h.<i>.attn.<name>`) instead of a suffix pattern to avoid over-matching.
  - `attn.bias` is the causal mask buffer, not a projection bias — it must not be touched despite the name.
  - Conv1D `[in, out]` layout means heads are columns in `c_attn` but rows in `c_proj`; the two get different axes.
  - `c_attn` is fused `[q|k|v]`, so the same 64-wide hole is cut at three offsets (0, 768, 1536), which reproduces the required index list.
  - `index_select` returns fresh tensors, but I still call `.contiguous()` before `save_file` since safetensors rejects non-contiguous/shared storage.
- **Anything unclear:** nothing; the task spelled out the exact kept index ranges, which made verification straightforward.
- **Tools used (condition F):** `safetensors` 0.5.3 (load/save, preserving key order and metadata) and `torch` 2.14.0 (`index_select`). I deliberately did **not** use transformers `prune_heads`: it operates on a live `nn.Module`, rewrites `config.n_head`/pruned-heads bookkeeping, and its Conv1D pruning path gives no direct guarantee of bit-exact preservation of the remaining columns in the required order. A ~35-line script over the raw state dict is smaller, exact, and directly checkable. mergekit/peft/torch-state-bridge are key- or layer-level tools and cannot express an intra-tensor head slice.
- **Approximate time spent:** ~5 minutes.
