# Participant self-report: T2 (Pythia-1B head pruning), condition P

- Final artifact path: `out/T2/solution.py` (output checkpoint `out/T2/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - Interleaved per-head q|k|v layout in `query_key_value` means one 768-row block per head, not three separate 256-row slices; the task text stated this explicitly so it was not a problem.
  - `index_select` results are made contiguous before saving to avoid safetensors rejecting non-contiguous tensors.
- Anything in the task text or documentation that was unclear: nothing; the row/column ranges were fully specified.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3)
- Approximate time spent, if you can tell: about 2 minutes
