# Participant self-report: T5 (Pythia-1B, condition P)

- Final artifact path: `out/T5/solution.py` (output checkpoint in `out/T5/`: 6 shards plus `model.safetensors.index.json`)
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - None failed. Execution 1 succeeded but packed the two embedding matrices together with layer tensors; I re-ran after changing the sharding so each embedding is stored alone, as TASK.md states, and after adding a read-back verification against the inputs.
- Pitfalls or surprises you hit (one line each):
  - TASK.md says tensors larger than 512 MiB are stored alone and then names `embed_in` / `embed_out` (206 MB each) as examples, which do not exceed the cap; I stored them alone explicitly to satisfy the text under both readings.
  - `adapter_config.json` lists `target_modules = ["query_key_value"]`, not `attention.query_key_value` as in TASK.md; harmless because I mapped names by stripping the `base_model.model.` prefix and the `.lora_A/B.weight` suffix instead of using the config.
  - Alphabetical key order puts `embed_out.weight` before `gpt_neox.*`, so shard 1 is the output embedding, not the input one.
- Anything in the task text or documentation that was unclear:
  - The "stored alone" rule for tensors that are smaller than the shard cap (see above); whether the grader wants a specific shard layout or only the cap and index consistency.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3)
- Approximate time spent, if you can tell: about 3 minutes
