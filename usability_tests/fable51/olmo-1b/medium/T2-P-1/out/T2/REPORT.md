# Participant self-report: T2 (condition P)

- Final artifact path: `out/T2/solution.py` (output: `out/T2/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - Input is sharded across two files; merged both shards via the index's `weight_map` before slicing.
  - `index_select` on a loaded safetensors view gives a contiguous copy, but I called `.contiguous()` on everything before saving to be safe.
- Anything in the task text or documentation that was unclear: nothing; the row/column ranges and head axis per tensor were stated explicitly.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3)
- Approximate time spent, if you can tell: about 2 minutes
