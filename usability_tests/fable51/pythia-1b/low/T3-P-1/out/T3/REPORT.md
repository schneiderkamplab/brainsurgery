# Participant self-report: T3 (Pythia-1B, condition P)

- Final artifact path: `out/T3/solution.py` (output shards + `model.safetensors.index.json` in `out/T3/`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - Buffers `attention.bias` / `attention.masked_bias` share the `.bias` suffix with real projection biases, so deletion used an anchored regex on the exact buffer names rather than a suffix match.
  - Projection matrices were matched by anchored full-name regex so `.*weight` could not reach embeddings or norms.
  - The embedding matrices are 412 MB in float32 (task text quoted 206 MB, the float16 size), so each exceeds the 256 MiB budget and sits alone in its own shard.
- Anything in the task text or documentation that was unclear: the 206 MB figure for the embeddings refers to the float16 input, not the float32 output; otherwise clear.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 2 minutes
