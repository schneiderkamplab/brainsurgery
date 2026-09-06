# T4 participant self-report

- Final artifact path: `out/T4/solution.py` (output: `out/T4/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - The base is sharded, so it needs the index file to locate each tensor; the fine-tunes are single files. Handled with a small sharded view over `safe_open`.
  - Ordering hazard is avoided by never mutating the base tensor: both task vectors are computed from the same read-only `base[X]` before summing.
- Anything in the task text or documentation that was unclear: nothing significant. The `inputs/lora/` directory is present but unused by T4.
- Tools used (condition F): name, version, and why:
  - `safetensors` 0.5.3: lazy per-tensor reads via `safe_open` (keeps memory low on a 5 GB checkpoint) and `save_file` for the single-file output.
  - `torch` 2.14.0: float32 arithmetic and `torch.equal` for bit-exact shared-tensor verification.
  - Not used: `mergekit` task_arithmetic. It could compute the merge, but it does not enforce the abort-if-shared-tensors-differ precondition, the exact merged-tensor count, or the single-file 114-tensor output, so the required checks would still need a wrapper script; a direct script was simpler and fully verifiable.
- Approximate time spent, if you can tell: about 2 minutes, most of it the ~1 minute the script needs to read three checkpoints and write 5 GB.

## Checks enforced by `solution.py`

- Identical key sets across base/ft1/ft2, and exactly 114 keys.
- Exactly 48 keys match the MLP regex; every other (66) tensor bit-identical (shape, dtype, values) in all three checkpoints, else abort.
- Merged-tensor counter must equal 48; output dict must have 114 entries.
- After writing, the file is re-opened: key set equals base, 114 tensors, unchanged tensors bit-exact vs base, merged tensors float32 with base shapes.
- Refuses to overwrite an existing output file.

Independent spot check after the run: for `model.layers.3.mlp.down_proj.weight`, the written tensor matched a freshly computed `base + 0.4*(ft1-base) + 0.4*(ft2-base)` with relative error 0.0.
