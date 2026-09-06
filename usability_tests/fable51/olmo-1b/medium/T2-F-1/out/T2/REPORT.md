# T2 participant self-report (condition F)

- Final artifact path: `out/T2/solution.py` (output: `out/T2/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - None. The task text already states the layout (rows for q/k/v, columns for o_proj, head 5 = indices 640..767), so no discovery was needed.
  - Advanced indexing yields non-contiguous views in some cases; I called `.contiguous()` before `save_file` to avoid a safetensors save error.
- Anything in the task text or documentation that was unclear: nothing.
- Tools used (condition F): name, version, and why:
  - `safetensors` 0.5.3: read the two input shards via `safe_open`, write the single output file with `save_file`.
  - `torch` 2.14.0: index-based row/column slicing of the projection weights (bit-exact, no arithmetic).
  - Not used: `transformers` `prune_heads` was considered but rejected. It requires instantiating the model, changes the config, and its pruning path goes through `nn.Linear` copies rather than a straight state-dict slice; a direct slice is simpler, keeps tensor names and dtypes untouched, and is trivially bit-exact. `mergekit`/`peft`/`torch-state-bridge` do not address head slicing.
- Approximate time spent, if you can tell: about 3 minutes, including a separate verification pass comparing every output tensor against the sliced input.
