## Participant self-report

- Final artifact path: `out/T5/solution.py`
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - Run 1: `AssertionError: base tensor gpt_neox.layers.0.attention.query_key_value not found` — the regex captured the base module name without the trailing `.weight` suffix, so lookups into the base state dict missed.
- Pitfalls or surprises you hit (one line each):
  - Needed to strip the `base_model.model.` PEFT prefix and re-add `.weight` to map adapter names to base tensor names.
  - Had to bin-pack tensors into shards in key order (not sorted by size) while special-casing the two embedding tensors that individually exceed the 512 MiB shard budget.
- Anything in the task text or documentation that was unclear: None; the spec fully determined the merge formula, scale, and sharding rule.
- Tools used (condition F): n/a (condition P)
- Approximate time spent, if you can tell: ~10 minutes
