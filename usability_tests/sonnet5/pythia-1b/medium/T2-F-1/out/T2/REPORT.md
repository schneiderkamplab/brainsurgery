# Participant self-report — T2

- Final artifact path: `out/T2/solution.py` (produces `out/T2/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why: none — the single execution succeeded.
- Pitfalls or surprises you hit:
  - transformers 5.12's `GPTNeoXAttention` does not expose a `prune_heads`
    method (unlike BERT-family models), so the "route via `prune_heads`"
    suggested in `F-allowed.md` was not actually available for this
    architecture; fell back to a plain script operating directly on the
    safetensors state dict.
  - The qkv row layout is per-head interleaved (q,k,v within each 768-row
    block), not the `[q_all | k_all | v_all]` layout some fused-QKV
    checkpoints use, so pruning a head is a single contiguous 768-row
    deletion per layer rather than three separate 256-row deletions.
  - `dense.weight` heads are column blocks (input side), matching the head
    order/width of the qkv output, so the same head index maps directly to
    columns `256*h .. 256*h+255`.
- Anything in the task text or documentation that was unclear: none; the
  row/column ranges to keep were given explicitly in TASK.md and matched
  what the layout description implied.
- Tools used (condition F): `torch` 2.14.0 and `safetensors` 0.5.3 only, via
  a plain Python script (`out/T2/solution.py`) doing `index_select` on the
  loaded state dict. Considered `transformers.prune_heads` first but it is
  not implemented for GPT-NeoX in this version, and `mergekit`/
  `torch-state-bridge` are aimed at cross-checkpoint merging/renaming rather
  than intra-tensor slicing, so a direct script was the most reliable route
  and the shapes/values were verified against a hand-derived reference
  slicing in a follow-up check.
- Approximate time spent, if you can tell: a few minutes (single pass, no
  retries needed).
