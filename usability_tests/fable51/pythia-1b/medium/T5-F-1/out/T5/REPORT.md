# T5 participant self-report (condition F)

- Final artifact path: `out/T5/solution.py` (run with `.venv/bin/python out/T5/solution.py`); output shards and `model.safetensors.index.json` in `out/T5/`.
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - None failed. Execution 1 produced a valid 4-shard output (all shards under 512 MiB, values correct); I reran after changing the sharder so that `gpt_neox.embed_in.weight` and `embed_out.weight` are each stored alone, as the task text states. Final output: 6 shards.
- Pitfalls or surprises you hit (one line each):
  - The base contains `uint8` `attention.bias` mask buffers (`[1,1,2048,2048]`), so a shard-size estimate that assumes float16 everywhere overcounts; the script itself uses `numel * element_size`, so it was unaffected.
  - The task says the 206 MB embeddings are "larger than" the 512 MiB budget and stored alone; they are not larger, so a pure greedy packer legitimately packs them with other tensors. I added an explicit solo rule for those two names to match the stated layout.
  - `adapter_config.json` lists `target_modules = ["query_key_value"]`, not `attention.query_key_value` as TASK.md says; irrelevant since I paired keys by the `lora_A`/`lora_B` suffix and stripped the `base_model.model.` prefix rather than using `target_modules`.
- Anything in the task text or documentation that was unclear:
  - Whether the grader expects the two embeddings to be alone (as the prose says) or greedy packing (as the stated rule implies). I followed the prose; both satisfy the 512 MiB rule.
- Tools used (condition F): name, version, and why:
  - `torch` 2.14.0 for the float32 `B @ A` merge and dtype cast; `safetensors` 0.5.3 (`safe_open`, `save_file`) for I/O. Plain script chosen over `peft.merge_and_unload` / `transformers.save_pretrained` because the sharding rule (custom budget plus solo embeddings) and the required pre-write checks are easier to enforce directly on the state dict, and it avoids instantiating the model.
- Approximate time spent, if you can tell: about 3 minutes.
