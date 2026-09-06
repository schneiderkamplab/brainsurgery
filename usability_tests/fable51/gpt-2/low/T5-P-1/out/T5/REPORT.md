# Participant self-report: T5 (GPT-2 124M), condition P

- Final artifact path: `out/T5/solution.py` (output checkpoint in `out/T5/`, 5 shards + `model.safetensors.index.json`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - `wte.weight` (154 MB) exceeds the 100 MiB shard budget, so the greedy packer places it alone in its own shard.
  - Adapter names carry the `base_model.model.` PEFT prefix that must be stripped to reach `h.<i>.attn.c_attn.weight`.
  - Conv1D `[in, out]` base vs Linear `[out, in]` factors: `fan_in_fan_out = true` means `(B @ A).T` before adding.
- Anything in the task text or documentation that was unclear: nothing; the task spelled out the transpose and sharding rules.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3)
- Approximate time spent, if you can tell: about 2 minutes
