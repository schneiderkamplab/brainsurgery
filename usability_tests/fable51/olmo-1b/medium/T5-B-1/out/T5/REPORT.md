# T5 participant self-report (condition B, OLMo-1B-0724-hf)

- Final artifact path: `out/T5/plan.yaml` (output checkpoint in `out/T5/`, 10 shards + `model.safetensors.index.json`)
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - Execution 1: `matmul source_b missing: lora::base_model\.model\.model.layers.0.self_attn.q_proj\.lora_A\.weight`. I had regex-escaped the dots in `from_b`, but `from_b` is a rewrite template of the `from_a` captures (like `to`), so the backslashes were kept literally. Fixed by writing `from_b` with plain dots.
- Pitfalls or surprises you hit (one line each):
  - In ternary transforms only `from_a` is a regex; `from_b` and `to` are rewrite templates, so escaping there breaks the name.
  - With two inputs the output alias is inferred from write destinations, so the `B @ A` intermediates had to be created on the `base` alias (as `__delta.*`) and deleted before saving, rather than placed on the `lora` alias.
  - `adapter_config.json` cannot be read by a plan, so `scale = 2` is hard-coded from the task text and the shape/count asserts guard the `r = 16` assumption.
- Anything in the task text or documentation that was unclear:
  - The docs say ternary transforms "support the same capture-based rewrite model across from_a, from_b, and to" but do not state that `from_b` is a template rather than a pattern; an example with a backreference in `from_b` would have avoided the failed attempt.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 5 minutes
