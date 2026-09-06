# Participant self-report

- Final artifact path: `out/T3/plan.yaml`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none, the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - `cast_` (in-place) was needed rather than `cast` so tensor names stay unchanged, and it
    conveniently also enforces "destination tensors must not already exist" being moot.
  - The projection-matrix pattern has to explicitly enumerate `self_attn.(q|k|v|o)_proj` and
    `mlp.(gate|up|down)_proj` to avoid also matching `embed_tokens.weight` / `lm_head.weight`,
    which also end in `.weight` but must stay float32.
  - `output.shard: 256MB` maps to exactly 268,435,456 bytes (binary units), matching the
    task's byte budget directly, and oversized tensors (`embed_tokens`, `lm_head`, 412MB
    each) are automatically placed alone in their own shard by brainsurgery's packing rule.
- Anything in the task text or documentation that was unclear: none; the README's
  "Output behavior" section documented shard packing and oversized-tensor handling clearly
  enough to rely on the default packer without extra transforms.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: a few minutes, single attempt.
