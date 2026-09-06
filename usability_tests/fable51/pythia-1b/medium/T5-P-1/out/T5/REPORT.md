# Participant self-report: T5 (condition P, Pythia-1B)

- Final artifact path: `out/T5/solution.py` (output checkpoint in `out/T5/`: 4 shards + `model.safetensors.index.json`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - `adapter_config.json` lists `target_modules = ["query_key_value"]` rather than `attention.query_key_value` as TASK.md states; irrelevant since names are taken from the adapter tensors, not the config.
  - Merge is done in float32 and cast back to float16 (`W.float() + scale * B @ A`), then made contiguous before saving.
  - Shard budget is enforced on tensor bytes only (numel * element_size), greedy in base key order.
- Anything in the task text or documentation that was unclear:
  - The sharding clause says a tensor "larger than 512 MiB is stored alone", but names the 206 MB embeddings as examples; they are below the budget, so I applied the plain greedy rule and did not force them into solo shards.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3)
- Approximate time spent, if you can tell: about 2 minutes
