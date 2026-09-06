# T1 self-report

- Final artifact path: `out/T1/solution.py` (output `out/T1/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - Anchored regex `^h\.(\d+)\.` so that `attn.bias` mask buffers and `mlp.c_proj` are handled per block with no overreach.
  - Built the new dict fresh (rather than renaming in place) so renumbering order cannot cause a collision; still guarded with an explicit collision check.
- Anything in the task text or documentation that was unclear: nothing significant.
- Tools used (condition F): safetensors 0.5.3 (`load_file`/`save_file`) and torch 2.14.0, plain script. A script is the shortest, fully transparent route for a pure rename/drop; mergekit passthrough would need a config and HF-directory plumbing, and torch-state-bridge still needs custom persistence.
- Approximate time spent, if you can tell: about 2 minutes.
