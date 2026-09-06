# T2 participant self-report (condition F)

- Final artifact path: `out/T2/solution.py` (output `out/T2/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - Sliced tensors are non-contiguous views; called `.contiguous()` before `save_file` to avoid a safetensors save error.
- Anything in the task text or documentation that was unclear: nothing; row/column ranges were stated explicitly.
- Tools used (condition F): name, version, and why:
  - `safetensors` 0.5.3: load the two shards via the index and write the single output file.
  - `torch` 2.14.0: index-select rows (q/k/v) and columns (o_proj) with one shared keep-index vector.
  - Did not use `transformers.prune_heads`: it would require loading the full model and it also rescales/renumbers config; direct slicing is simpler and bit-exact.
- Approximate time spent, if you can tell: about 2 minutes
