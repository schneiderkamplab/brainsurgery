# T3 participant self-report (condition B, GPT-2 124M)

- Final artifact path: `out/T3/plan.yaml` (output checkpoint in `out/T3/`: 4 shards + `model.safetensors.index.json`, 148 tensors)
- Number of times you executed the script or plan: 1 (plus one read-only verification plan, `out/T3/verify.yaml`, with no `output` section, run once against `out/T3`)
- Which executions failed, and why (one line each): none; the first execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - The docs offer no direct "count tensors of dtype X" assertion, so "exactly 48 tensors are bfloat16" was expressed as: the 48-match projection pattern is bfloat16 AND the complementary pattern (negative lookahead, `(?!...projection...$).*`) is float32; the verification plan confirmed the complement matches exactly 100 tensors.
  - The mask buffer is named `h.<i>.attn.bias`, which a loose `.*bias` pattern would confuse with projection biases; targeted it with an anchored regex before casting.
  - Shard size `64MB` in `output.shard` is binary (67,108,864 bytes) as documented; `wte.weight` (154 MB) landed alone in its own shard as expected.
- Anything in the task text or documentation that was unclear:
  - Whether `assert: dtype` with a pattern checks every match (it does, per the run) is not stated explicitly in the help text.
  - `help.txt` lists `cast_` allowed keys as only `target`, though `to` is required and works; minor doc inconsistency.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 3 minutes.
