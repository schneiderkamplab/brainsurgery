# T2 self-report (condition F, Pythia-1B)

- Final artifact path: `out/T2/solution.py` (output `out/T2/model.safetensors`)
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - Execution 1: `failed_assertion`. An extra check I added (all tensors float16) fired because the 16 `attention.bias` mask buffers are uint8. Not a task requirement; removed it.
- Pitfalls or surprises you hit (one line each):
  - The `attention.bias` buffers are U8, not float16, despite "the checkpoint is stored in float16".
- Anything in the task text or documentation that was unclear: nothing; the row/column ranges were spelled out explicitly.
- Tools used (condition F): name, version, and why:
  - `torch` 2.14.0: index-select rows/columns with a precomputed kept-head index.
  - `safetensors` 0.5.3: `load_file` / `save_file`, single-file output with `format: pt` metadata.
  - Did not use `transformers.prune_heads`: it re-initialises a model, reorders nothing helpful here, and risks dtype or buffer changes; a direct slice is bit-exact by construction.
- Approximate time spent, if you can tell: about 3 minutes.
