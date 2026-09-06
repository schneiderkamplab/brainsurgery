## Participant self-report

- Final artifact path: `out/T3/plan.yaml`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none, the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - The mask buffer is named `h.<i>.attn.bias`, while the real bias parameters
    are `h.<i>.attn.c_attn.bias` / `h.<i>.attn.c_proj.bias` / etc. — distinct
    enough that a `.*bias` pattern would have wrongly deleted parameters, so
    the delete target had to be the fully-qualified `attn.bias` pattern.
  - `assert.dtype`'s `of` accepts a regex matching many tensors and checks all
    of them, which let one assert cover "all 48 are bfloat16" instead of
    needing 48 separate checks; combined with a negative-lookahead regex
    (`(?!h\.\d+\.(...)\.weight).+`) it also let me assert "everything else is
    float32" in one shot.
  - `cast_` (in place) was preferable to `cast` (copy to new name) since the
    task requires tensor names to stay unchanged and no leftover fp32 copies.
  - Shard sizing worked as documented: with `shard: 64MB`, `wte.weight`
    (154 MB) alone exceeded the budget and was placed alone in its own shard
    (shard 4 of 4) automatically, no special-casing needed in the plan.
- Anything in the task text or documentation that was unclear:
  - None; the README's tensor-reference/regex and output-sharding sections
    were sufficient to write the plan without trial and error beyond one
    dump to confirm the tensor names/shapes up front.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: ~10 minutes.
