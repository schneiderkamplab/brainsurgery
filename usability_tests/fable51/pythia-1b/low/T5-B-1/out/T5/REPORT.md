# Participant self-report: T5 (condition B, Pythia-1B)

- Final artifact path: `out/T5/plan.yaml` (output checkpoint in `out/T5/`, 4 shards + `model.safetensors.index.json`)
- Number of times you executed the script or plan: 3
- Which executions failed, and why (one line each):
  - #1: crash (plan did not compile): `assert: { not: { exists: { of: ... } } }` — `exists` takes a bare string reference, not `{ of: ... }` like `count`/`shape`/`dtype`.
  - #2: no_match: `matmul` `from_b` is a rewrite template, not a regex, so regex-escaped dots (`\.`) were kept literally and the lora_A name was not found.
- Pitfalls or surprises you hit (one line each):
  - Argument shape of assert operators is inconsistent (`exists: 'ref'` vs `count: { of, is }`), and the README/help.txt do not show an `exists` example.
  - In ternary transforms (`matmul`), `from_a` is a regex but `from_b` and `to` are rewrite templates; the help text only says "references may be regex".
  - Output alias inference requires all writes on one alias, so the B@A intermediate had to be created on `base::` and deleted afterwards instead of living on `lora::`.
  - `add_` needs matching dtypes, so merging in float32 needed `cast_` to float32, `add_`, `cast_` back to float16 (which also matches the task's requested numerics).
- Anything in the task text or documentation that was unclear:
  - Whether the intermediate cast_/add_/cast_ round trip counts as an "intermediate tensor" (it does not, since it stays under the original name).
  - No documented way to check "no name containing lora_" other than `not: exists`.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 5 minutes
