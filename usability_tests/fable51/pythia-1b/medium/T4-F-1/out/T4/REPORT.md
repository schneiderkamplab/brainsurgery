# T4 participant self-report (condition F)

- Final artifact path: `out/T4/solution.py` (output: `out/T4/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - None. Inputs matched the spec exactly (244 names, 180 shared tensors bit-identical, 64 MLP tensors matched by an anchored regex bounded to layers 0..15).
- Anything in the task text or documentation that was unclear:
  - Nothing material. The spec does not say whether the merged output needs safetensors `metadata`; I wrote `{"format": "pt"}` so `transformers` can load it.
- Tools used (condition F): name, version, and why:
  - `safetensors` 0.5.3: `load_file` / `save_file` for checkpoint I/O.
  - `torch` 2.14.0: `torch.equal` for the bit-exact shared-tensor check and float32 arithmetic for the merge.
  - I did not use `mergekit` task arithmetic: it does not enforce the required precondition (shared tensors identical across all three checkpoints) or the exact merged/total tensor counts, and it would have needed extra wrapping to fail loudly on those. A ~70-line script enforces all three required checks directly and computes each task vector against the unmodified base.
- Approximate time spent, if you can tell: about 3 minutes.
