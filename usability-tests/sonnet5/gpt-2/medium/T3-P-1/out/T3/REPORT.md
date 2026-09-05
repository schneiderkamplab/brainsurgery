## Participant self-report

- Final artifact path: `out/T3/solution.py`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - Used explicit per-tensor regex matches for the 4 projection-weight names per layer rather than a broad `.*weight` pattern, since that would also catch `wte.weight`, `wpe.weight`, layer-norm weights, and it would not distinguish weights from biases anyway.
  - Made sure to `.contiguous()` every tensor before `save_file`, since safetensors rejects non-contiguous/shared-memory tensors.
  - The oversized `wte.weight` (154 MB, over the 64 MiB budget) needed to go into its own shard rather than trigger an error; handled with a size check before packing.
- Anything in the task text or documentation that was unclear: none — the tensor names, shapes and buffer name were specified explicitly, so no exploration of the checkpoint was needed to find naming risks.
- Tools used (condition F): n/a (condition P)
- Approximate time spent, if you can tell: a few minutes; single script write and one execution.
