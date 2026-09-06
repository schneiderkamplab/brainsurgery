# T2 self-report (condition P)

- Final artifact path: `out/T2/solution.py` (output `out/T2/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - The input is sharded, so the shards have to be merged via `model.safetensors.index.json` before slicing and written back as one file.
  - `q/k/v_proj` are row blocks but `o_proj` is a column block; using the same axis for all four would have produced a loadable but wrong checkpoint.
  - `index_select` returns views/strided results in the column case, so tensors are made contiguous (and cloned) before `save_file` to avoid shared-storage rejection.
  - Name matching is done on split path components rather than a substring/regex so nothing in `mlp.*` can be caught accidentally.
- Anything in the task text or documentation that was unclear: nothing; the kept row/column ranges and the required shapes were stated explicitly.
- Tools used (condition F): n/a (condition P).
- Approximate time spent, if you can tell: about 5 minutes.
