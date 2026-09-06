# T5 report

- Final artifact path: `out/T5/solution.py` (invoked via `out/T5/run.sh`)
- Number of times you executed the script or plan: 2 (first run failed a
  required check, second run succeeded)
- Which executions failed, and why (one line each):
  - Run 1: `failed_assertion` — "expected to merge 32 weights, merged 0";
    the code compared the full module name (`self_attn.q_proj`) against
    `adapter_config.json`'s `target_modules`, which lists the short names
    (`q_proj`, `v_proj`), so the membership check never matched.
- Pitfalls or surprises you hit (one line each):
  - `target_modules` in the adapter config uses short module names
    (`q_proj`/`v_proj`) while the adapter tensor names and base weight names
    use the full path (`self_attn.q_proj`); needed to strip to the last
    path component before comparing.
  - The 512 MiB-per-shard budget is tight against the tensor sizes here, so
    several shards land essentially exactly at the limit (536,870,912
    bytes) — used a simple greedy bin-packing by declaration order rather
    than anything fancier, and it happens to fit within budget for every
    shard.
- Anything in the task text or documentation that was unclear: none;
  the shape/layout/scale spec (`fan_in_fan_out = false`, `scale = alpha/r`)
  was stated precisely enough to implement directly.
- Tools used (condition F): plain Python on top of `torch` 2.14.0 and
  `safetensors` 0.5.3 only. Chose this over `peft.merge_and_unload` because
  the merge here is a pure tensor operation on 32 known name pairs — running
  it directly on the checkpoint files avoids instantiating the full HF model
  just to get the same `W + scale * B @ A` update, and it gives direct
  control over the required checks (pair count, no `lora_` leakage, shape,
  tensor count) and the sharding rule (512 MiB per shard, oversized tensors
  alone in their own shard), which is easier to guarantee explicitly than to
  configure through a higher-level merge/export API.
- Approximate time spent, if you can tell: ~10 minutes.
