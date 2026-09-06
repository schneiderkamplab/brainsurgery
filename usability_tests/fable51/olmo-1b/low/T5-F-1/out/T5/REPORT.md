# T5 participant self-report

- Final artifact path: `out/T5/solution.py` (output checkpoint in `out/T5/`: 10 shards + `model.safetensors.index.json`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - `adapter_config.json` lists `target_modules` as `q_proj`/`v_proj` (no `self_attn.` prefix), unlike TASK.md; irrelevant since I mapped names by stripping the `base_model.model.` prefix rather than via target_modules.
  - The 512 MiB shard budget is a limit on tensor data, so the 412 MB embedding fits in a shard with a smaller tensor; I used greedy packing in base key order.
- Anything in the task text or documentation that was unclear:
  - "A single tensor larger than that ... 412 MB each is stored alone": 412 MB is not larger than 512 MiB, so the sentence is contradictory. I followed the byte budget literally (greedy fill); `lm_head` happened to land alone anyway.
- Tools used (condition F): name, version, and why:
  - torch 2.14.0: `B @ A` in float32 and the add.
  - safetensors 0.5.3: `safe_open` to read shards/adapter, `save_file` to write shards.
  - Plain script instead of peft `merge_and_unload`: avoids instantiating the 1B model and gives direct control over sharding, name mapping and the required checks.
- Approximate time spent, if you can tell: about 2 minutes.
