# T5 participant self-report

- Final artifact path: `out/T5/solution.py` (run with `.venv/bin/python out/T5/solution.py`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - `adapter_config.json` lists `target_modules` as `q_proj`/`v_proj` while TASK.md writes `self_attn.q_proj`; irrelevant for a file-level merge since the names are derived from the adapter keys themselves.
  - TASK.md says `embed_tokens`/`lm_head` are "larger than" the 512 MiB budget, but 412 MB is below it; I followed the stated intent and gave each its own shard explicitly, then packed the rest greedily in index order.
- Anything in the task text or documentation that was unclear: the exact shard packing the hidden reference uses (greedy fill vs. any other rule) is not specified beyond the budget and the two standalone tensors.
- Tools used (condition F): name, version, and why:
  - `safetensors` 0.5.3: read base shards and the adapter, write output shards directly, no model instantiation.
  - `torch` 2.14.0: float32 `B @ A` and the scaled add.
  - Not used: `peft` `merge_and_unload` (would instantiate the model and re-export via transformers, adding load time and leaving sharding to `save_pretrained`, whose shard rule differs from the task's).
- Approximate time spent, if you can tell: about 3 minutes.
