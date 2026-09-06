# Participant self-report: T5 (olmo-1b, condition P)

- Final artifact path: `out/T5/solution.py` (output checkpoint in `out/T5/`, 10 shards + `model.safetensors.index.json`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - `adapter_config.json` lists `target_modules` as `["q_proj", "v_proj"]` whereas TASK.md says `["self_attn.q_proj", "self_attn.v_proj"]`; I mapped by the `base_model.model.` prefix in the adapter tensor names rather than by config, so it did not matter.
  - Greedy sharding in sorted name order puts the two 412 MB tensors alone in their own shards and yields 10 shards total.
- Anything in the task text or documentation that was unclear:
  - Whether the grader requires a specific shard ordering/naming beyond the 512 MiB budget; I used sorted names and HF-style `model-XXXXX-of-YYYYY.safetensors`.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3)
- Approximate time spent, if you can tell: about 2 minutes
