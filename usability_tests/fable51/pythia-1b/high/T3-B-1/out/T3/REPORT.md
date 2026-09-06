# T3 participant self-report (condition B, Pythia-1B)

- Final artifact path: `out/T3/plan.yaml` (output: `out/T3/model-0000N-of-00009.safetensors` + `out/T3/model.safetensors.index.json`, executed-plan summary in `out/T3/summary.yaml`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - The embeddings become 412 MB each after the float32 upcast (TASK.md quotes 206 MB, their float16 size); they still exceed the 256 MiB budget and each landed alone in its own shard, as required.
  - "Exactly 64 tensors are bfloat16" cannot be expressed with a single dtype-filter count, so it is encoded as: projection pattern counts 64 and is bfloat16, complement pattern (negative lookahead) counts 132 and is float32, total counts 196.
  - Deleting the uint8 mask buffers before the blanket `cast_ '.*' -> float32` avoids casting non-float buffers.
- Anything in the task text or documentation that was unclear:
  - The `dtype`/`shape` assert help says "the tensor", singular; it was not explicit that `of` may match many tensors (the `dimensions`/`reads` examples suggested it does, and it worked).
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 3 minutes.
