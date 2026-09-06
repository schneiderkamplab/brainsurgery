# Participant self-report: T5 (GPT-2 124M, condition P)

- Final artifact path: `out/T5/solution.py` (output checkpoint in `out/T5/`, 5 shards + `model.safetensors.index.json`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - Base `c_attn.weight` is Conv1D `[in, out]`, so the `B @ A` product (`[out, in]`) had to be transposed before adding, as `fan_in_fan_out = true` signals.
  - The `wte.weight` tensor (154 MB) exceeds the 100 MiB shard budget and needed its own shard; greedy packing with an "oversized tensor goes alone" rule handled it.
  - Sorted key order put `wte.weight` last, so it landed as the final single-tensor shard.
- Anything in the task text or documentation that was unclear: the adapter_config lists `target_modules = ["c_attn"]` while TASK.md says `["attn.c_attn"]`; harmless, since names were derived from the adapter tensor keys rather than the config.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3)
- Approximate time spent, if you can tell: about 2 minutes
