# T4 self-report (condition P)

- Final artifact path: `out/T4/solution.py` -> `out/T4/model.safetensors`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - The base is sharded, so the state dict had to be assembled from
    `model.safetensors.index.json` plus both shard files rather than read from one file.
  - Task-vector ordering: both deltas are taken against the untouched base tensor, so
    the base value is read once into a local before either delta is subtracted.
  - `save_file` rejects tensors that share storage, so every passthrough tensor is
    cloned and made contiguous before writing.
  - The MLP regex is anchored and escapes the dots to avoid matching anything outside
    the 48 `gate_proj`/`up_proj`/`down_proj` weights; the count is asserted at 48.
- Anything in the task text or documentation that was unclear: nothing; the required
  checks and the output path were explicit.
- Tools used (condition F): n/a (condition P: torch + safetensors only).
- Approximate time spent, if you can tell: about 5 minutes.
