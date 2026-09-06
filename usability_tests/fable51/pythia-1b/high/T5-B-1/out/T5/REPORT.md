# T5 participant self-report (condition B, Pythia-1B)

- Final artifact path: `out/T5/plan.yaml` (output checkpoint in `out/T5/`: 4 shards + `model.safetensors.index.json`)
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - Execution 1: `matmul source_b missing: lora::base_model\.model\.gpt_neox.layers.0...\.lora_A\.weight` (no_match). I wrote `from_b` as a regex with escaped dots; the ternary transforms treat `from_b`/`to` as literal rewrite templates of the `from_a` captures, so the backslashes were kept verbatim in the looked-up name.
  - Execution 2: success.
- Pitfalls or surprises you hit (one line each):
  - In `matmul`/`add`/... only `from_a` is a pattern; `from_b` and `to` are rewrite templates (`\1`), not regexes, so dots must not be escaped there.
  - Merged intermediates must live on the output alias (`base::...lora_delta`) so the writer can infer a single output alias; I deleted them before the asserts and the save.
  - The 206 MB embedding tensors are smaller than the 512 MiB budget, so they are packed with neighbours rather than alone; TASK.md's "stored alone in its own shard" note only applies to tensors above the budget.
- Anything in the task text or documentation that was unclear:
  - The interfaces reference says ternary transforms "support the same capture-based rewrite model across from_a, from_b, and to" without stating that `from_b` is a template rather than a second pattern; an example with `\1` in `from_b` would have avoided the failed run.
  - TASK.md says the 206 MB embeddings are stored alone in their own shard, which contradicts the 512 MiB packing rule it states just before.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 5 minutes (two plan runs of ~5 s and ~8 s, plus an independent verification script comparing the output against a float32 reference: 244 keys, 228 tensors bit-exact, 16 merged weights with zero relative error, all shards under 512 MiB).
