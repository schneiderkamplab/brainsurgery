## Participant self-report

- Final artifact path: `out/T2/solution.py`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none, the single execution succeeded.
- Pitfalls or surprises you hit (one line each): none in execution; the main
  care point was reading the spec closely for the interleaved qkv layout
  (768-row per-head block = q/k/v of 256 each, not a `[q|k|v]` global split)
  and for `dense.weight` heads being column blocks rather than row blocks.
- Anything in the task text or documentation that was unclear: no, the row
  and column ranges to keep were given explicitly in "Required result", which
  made it easy to cross-check against the derived ranges in code.
- Tools used (condition F): N/A (condition P).
- Approximate time spent, if you can tell: a few minutes.
