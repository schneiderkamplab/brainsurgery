# T3 self-report (condition B: BrainSurgery plan)

- Final artifact path: `out/T3/plan.yaml` (output written to `out/T3/`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single run passed all asserts and wrote the output.
- Pitfalls or surprises you hit (one line each):
  - Obvious trap avoided by reading the task: `.*weight` would have caught `wte.weight`, `wpe.weight` and the layer-norm weights, so I anchored the pattern to `h\.<i>\.(attn\.(c_attn|c_proj)|mlp\.(c_fc|c_proj))\.weight`.
  - `h.<i>.attn.bias` is a causal-mask buffer, not a projection bias; deleting by `h\.\d+\.attn\.bias` leaves the real `*.c_*.bias` parameters alone.
  - "exactly 48 bfloat16" is not directly expressible as a dtype-histogram assert, so I paired `count is: 48` with a negative-lookahead `dtype ... is: float32` over the complement, which pins both sides.
  - Shard units in BrainSurgery are binary (`64MB` = 67,108,864 bytes), which matches the task budget exactly; `wte.weight` (154 MB) exceeds it and the writer put it alone in shard 4, as required.
- Anything in the task text or documentation that was unclear:
  - The README documents `output.shard`/`--shard-size` as binary units, but the task states the budget in bytes; the two agreed here, though a task stating a decimal-MB budget would be ambiguous.
  - Whether `assert: dtype` applies to every match of a multi-match `of` pattern or only the first is not stated in the help text; I relied on it applying to all, and the complement check makes the result correct either way.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: ~5 minutes.

## Verification performed

- Index `weight_map` has 148 entries across 4 shards.
- dtype histogram of the written checkpoint: 100 float32, 48 bfloat16.
- `wte.weight` is the only tensor in its shard.
