## Participant self-report

- Final artifact path: `out/T5/solution.py`
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - Execution 1 failed with `AssertionError: unexpected target_modules in adapter_config.json` — I initially compared `adapter_config.json`'s `target_modules` (`["q_proj", "v_proj"]`) against the full dotted names (`["self_attn.q_proj", "self_attn.v_proj"]`) instead of the leaf module names.
- Pitfalls or surprises you hit (one line each):
  - PEFT's `target_modules` in `adapter_config.json` lists bare leaf names (`q_proj`, `v_proj`), not the full attribute path used in the adapter tensor names or in the base checkpoint keys — a naive equality check against the full path fails.
  - The 512 MiB shard budget is over tensor payload bytes only, not file size (safetensors header adds a small fixed overhead per file), so I sized shards using `numel * dtype_size` rather than on-disk file size.
- Anything in the task text or documentation that was unclear:
  - None; the layout convention (`fan_in_fan_out=False` → no transpose, `B @ A` directly) was stated explicitly and matched what PEFT actually does.
- Tools used (condition F): n/a (condition P)
- Approximate time spent, if you can tell: ~10 minutes
