# Participant self-report — T2 (condition F)

- Final artifact path: `out/T2/solution.py` (invoked via `out/T2/run.sh`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - None major. The task spec fully pins down the row/column block boundaries
    (head 5, rows/cols 640..767 removed, keep 0..639 and 768..2047), so the
    only care needed was matching `q/k/v` (row blocks) vs `o_proj` (column
    blocks) and leaving the three MLP matrices untouched.
- Anything in the task text or documentation that was unclear: none.
- Tools used (condition F): name, version, and why:
  - `safetensors` 0.5.3 — direct shard read/write; chosen over `transformers`
    `prune_heads` because it operates on raw tensors with no model
    construction, dtype casting, or config mutation in the loop, which keeps
    the transform an auditable slice-and-concat and makes bit-exact output
    straightforward to reason about and verify.
  - `torch` 2.14.0 — `torch.cat` for the row/column slicing.
  - Plain Python script (`solution.py`) reading `model.safetensors.index.json`
    to enumerate all 114 tensors across the two shards, rather than a
    mergekit/torch-state-bridge config, since the transform (drop 128
    contiguous rows/cols per head-bearing tensor) is a few lines and needed
    exact control over slice order.
- Approximate time spent, if you can tell: a few minutes (single write-run-verify pass).
