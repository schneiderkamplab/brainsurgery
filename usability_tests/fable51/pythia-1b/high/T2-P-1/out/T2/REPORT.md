# Participant self-report: T2 (Pythia-1B head pruning), condition P

- Final artifact path: `out/T2/solution.py` (output checkpoint at `out/T2/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - The fused `query_key_value` projection is interleaved per head (768-row blocks of q, k, v), so removing a head is one contiguous row-block drop rather than three separate segment slices; the task text stated this explicitly, which avoided the usual guesswork.
  - `dense.weight` prunes along columns (input side), while `query_key_value` prunes along rows (output side); both use `index_select` followed by `.contiguous()` so `save_file` accepts them.
- Anything in the task text or documentation that was unclear: nothing; the row/column ranges to keep were spelled out.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3)
- Approximate time spent, if you can tell: about 2 minutes
