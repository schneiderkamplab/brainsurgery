# Participant self-report: T3 (olmo-1b, condition B)

- Final artifact path: `out/T3/plan.yaml` (output checkpoint in `out/T3/`, 10 shards plus `model.safetensors.index.json`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - The `cast_` help example uses `'.*weight'`, which would have overreached onto `model.embed_tokens.weight` and `lm_head.weight`; I used an escaped, anchored regex over the seven projection names instead and asserted its match count was 112 before casting.
  - "Exactly 112 tensors are bfloat16" has no single operator; I expressed it as count(projections)=112, dtype(projections)=bfloat16, plus count(complement via negative lookahead)=2 and dtype(complement)=float32, inside one `assert: all`.
  - The shard budget is given in binary units, so `shard: 256MB` maps to exactly 268,435,456 bytes; the resulting non-oversized shards land at exactly that byte count of tensor data.
- Anything in the task text or documentation that was unclear:
  - The `dtype` assert help does not say explicitly whether a multi-match reference checks every match; it does (verified by the output dtypes), but a sentence in the help would have saved a doubt.
  - Whether the plan's own output directory may also hold `plan.yaml` and `REPORT.md` next to the shards (the task requires both) was implicit; it worked.
- Tools used (condition F): not applicable (condition B). After the run I read the output with the `safetensors` library only to verify shard sizes and dtypes; the edit itself is entirely in the plan.
- Approximate time spent, if you can tell: about 5 minutes.
