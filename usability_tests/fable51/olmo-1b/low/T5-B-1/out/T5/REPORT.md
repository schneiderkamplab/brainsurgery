# Participant self-report: T5 (OLMo-1B-0724-hf, condition B)

- Final artifact path: `out/T5/plan.yaml` (output checkpoint in `out/T5/`, 10 shards + `model.safetensors.index.json`)
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - Execution 1: `matmul source_b missing` (no_match). I wrote `from_b` with regex escapes (`base_model\.model\.\1\.lora_A\.weight`); `from_b` is a rewrite template of the `from_a` match, so the backslashes were taken literally and the name did not resolve.
- Pitfalls or surprises you hit (one line each):
  - In ternary transforms only `from_a` is a regex; `from_b` and `to` are rewrite templates and must be written as plain names with `\1` captures, no regex escaping.
  - Output alias inference requires every edit on one alias, so the B@A intermediate was written into `base::` and deleted there rather than living in `lora::`.
  - Shard size units are binary, so `512MB` in the plan equals the 536,870,912-byte budget required.
- Anything in the task text or documentation that was unclear:
  - The `matmul` help does not show a pattern example, and nothing states explicitly that `from_b` is a rewrite rather than a second regex; the `add` example (`from_b: '.*.delta'`) even suggests otherwise.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 3 minutes
