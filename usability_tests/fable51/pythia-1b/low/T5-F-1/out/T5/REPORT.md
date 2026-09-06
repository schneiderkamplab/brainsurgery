# T5 participant self-report

- Final artifact path: `out/T5/solution.py`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - The PEFT adapter names carry a `base_model.model.` prefix that must be stripped to reach the base key; handled with a regex.
  - `fan_in_fan_out=false` so `B @ A` is added directly; the script asserts the flag rather than silently assuming it.
- Anything in the task text or documentation that was unclear:
  - The task says the two 206 MB embedding tensors are "larger than" the 512 MiB budget and stored alone; they are not larger, so I packed shards greedily under the 512 MiB cap (4 shards). If the grader requires the embeddings to be isolated, that rule as written is contradictory.
- Tools used (condition F): name, version, and why:
  - `torch` 2.14.0: float32 matmul, dtype cast.
  - `safetensors` 0.5.3: load base/adapter, save shards.
  - Skipped `peft.merge_and_unload` because it requires instantiating the model and the task asks for a checkpoint-file-level merge; a script is simpler and the sharding rule is custom anyway.
- Approximate time spent, if you can tell: about 3 minutes.
