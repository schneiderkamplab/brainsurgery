# Participant self-report

- Final artifact path: out/T3/plan.yaml
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - Execution 1 failed the `dtype` assert on `model.layers.0.mlp.down_proj.weight` because the first attempt used a structured-path tensor reference (`["model","layers","$i","mlp",~p::"gate_proj|up_proj|down_proj","weight"]`) whose alternation inside the `~p::` capture apparently did not match as intended; the `cast_` for the mlp projections matched nothing.
- Pitfalls or surprises you hit (one line each):
  - Structured list-form tensor references with a `~name::REGEX` alternation capture didn't behave as expected for `cast_`'s `target`; switching to a plain anchored regex string (`model\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)\.weight`) worked immediately and is easier to verify by eye.
  - Used two `assert: count` calls with a negative lookahead (`(?!...).*`) to check the complement set (the 2 non-projection tensors) without naming `embed_tokens`/`lm_head` explicitly, since the assert regex is full-match and lookaheads work fine there.
- Anything in the task text or documentation that was unclear:
  - None; the README's shard-budget and structured-expression sections were sufficient once the structured-token issue was worked around with a plain regex.
- Tools used (condition F): name, version, and why: N/A (condition B)
- Approximate time spent, if you can tell: ~5 minutes
