## Participant self-report

- Final artifact path: `out/T2/solution.py`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none, the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - The query_key_value rows are interleaved per head (each head's own
    768-row q/k/v block), not grouped as one big q segment then k then k
    then v across all heads — had to keep that straight when computing the
    keep-ranges for head 5.
  - `dense.weight` heads are column blocks (input side, since dense is
    `[out, in]` and consumes concatenated per-head outputs), so pruning
    there means dropping columns, not rows, unlike the qkv tensor.
- Anything in the task text or documentation that was unclear: none, the
  row/column ranges to keep were given explicitly in "Required result" and
  matched exactly what I derived from the layout description, which let me
  cross-check them with plain assertions before slicing.
- Tools used (condition F): n/a (condition P).
- Approximate time spent, if you can tell: ~5 minutes.
