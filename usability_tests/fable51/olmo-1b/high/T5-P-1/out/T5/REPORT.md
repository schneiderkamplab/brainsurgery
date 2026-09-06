# Participant self-report: T5 (OLMo-1B-0724-hf, condition P)

- Final artifact path: `out/T5/solution.py` (output checkpoint in `out/T5/`, 10 shards plus `model.safetensors.index.json`)
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - None failed. Execution 1 produced a valid checkpoint but packed `model.embed_tokens.weight` (412 MB) together with a 67 MB layer tensor under greedy 512 MiB packing; I re-ran after changing the sharding so tensors of 256 MiB or more get a shard of their own, matching the task text that says the two big matrices are stored alone.
- Pitfalls or surprises you hit (one line each):
  - The base checkpoint's own sharding (113 tensors in shard 1, about 4.7 GB) violates the 512 MiB budget, so the output must be resharded rather than copied shard by shard.
  - The task text says a tensor "larger than 512 MiB" is stored alone, then names two 412 MB tensors as examples; plain greedy packing does not isolate them, so I added an explicit lower bound for standalone tensors.
  - `adapter_config.json` lists `target_modules` as `q_proj`/`v_proj` while TASK.md says `self_attn.q_proj`/`self_attn.v_proj`; I derived base names from the adapter tensor names (strip `base_model.model.`, replace `.lora_A/B.weight` with `.weight`) instead of relying on the config list.
- Anything in the task text or documentation that was unclear:
  - Whether "sharding rules" in grading requires an exact shard layout or only the budget and standalone constraints; I chose the layout that satisfies both readings.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3)
- Approximate time spent, if you can tell: about 5 minutes
