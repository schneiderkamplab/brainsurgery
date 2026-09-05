## Participant self-report

- Final artifact path: `out/T2/solution.py`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none, the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - The qkv row layout is interleaved per head (768-row blocks of q/k/v), not
    separate q/k/v segments across the whole tensor, so the head-to-prune
    block boundaries are `768*h .. 768*h+767`, not `256*h` within a q-only
    segment; the task text spelled this out clearly so no rework was needed.
  - `dense.weight` needed a column (dim=1) slice since it's `nn.Linear`
    `[out, in]` and heads live on the input side, opposite of the qkv weight
    which is sliced on dim=0.
- Anything in the task text or documentation that was unclear: none; the
  exact row/column ranges to keep were given explicitly, which made
  the required checks straightforward to write directly from the spec.
- Tools used (condition F): n/a (condition P).
- Approximate time spent, if you can tell: a few minutes.
