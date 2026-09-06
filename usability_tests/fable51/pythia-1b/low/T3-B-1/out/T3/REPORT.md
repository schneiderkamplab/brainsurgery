# Participant self-report: T3 (Pythia-1B), condition B

- Final artifact path: `out/T3/plan.yaml` (output: `out/T3/model-0000*-of-00009.safetensors` + `out/T3/model.safetensors.index.json`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single run passed all asserts and wrote 9 shards, 196 tensors (64 bfloat16, 132 float32).
- Pitfalls or surprises you hit (one line each):
  - `.*weight` would overreach onto embeddings and norms, so the bfloat16 cast uses an explicit regex for the four projection names per layer; the same regex is reused in a `count: 64` assert.
  - Buffer names `attention.bias` / `attention.masked_bias` must be deleted first, otherwise `cast_: '.*'` to float32 would also touch the uint8 mask.
  - The two embedding matrices are 412 MB each in float32, above the 256 MiB budget; the writer put each alone in its own shard as the task requires.
  - Ordering: delete buffers, upcast everything to float32, then downcast the 64 projections; float16 -> float32 -> bfloat16 equals a direct float16 -> bfloat16 cast.
- Anything in the task text or documentation that was unclear: the README does not state explicitly how oversized single tensors are sharded; I verified after the run that they landed alone in a shard. Shard sizes use binary units (`256MB` = 256 MiB), which the README documents.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 3 minutes
