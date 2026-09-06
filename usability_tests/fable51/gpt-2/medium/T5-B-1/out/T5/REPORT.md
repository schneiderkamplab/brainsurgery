# Participant self-report: T5 (condition B, GPT-2 124M)

- Final artifact path: `out/T5/plan.yaml` (output checkpoint in `out/T5/`, 5 shards + `model.safetensors.index.json`, 160 tensors)
- Number of times you executed the script or plan: 3
- Which executions failed, and why (one line each):
  - Execution 1: `matmul source_b missing: lora::base_model\.model\.h\.0\...` — I wrote `from_b` with regex-escaped dots, but `from_b` and `to` are rewrite templates of the `from_a` match (literal names with `\1`), not regexes, so the backslashes were kept literally (no_match).
  - Execution 2: `cannot infer output model uniquely` — the matmul result and its `scale_` lived on the `lora` alias while `permute`/`add_`/`delete` wrote to `base`; the output alias is inferred from write targets, so all intermediates had to be moved onto `base` (other).
  - Execution 3: success.
- Pitfalls or surprises you hit (one line each):
  - In ternary transforms (`matmul`), only `from_a` is a regex; `from_b` and `to` are substitution templates, so dots must not be escaped there.
  - Output alias inference counts every write, including intermediates and in-place `scale_`, so scratch tensors must be created on the output alias and deleted before saving.
  - No `transpose` transform; `permute` with `order: [1, 0]` does the job for the `fan_in_fan_out` case.
  - Shard budget `100MB` is binary (104,857,600 bytes), matching the task; `wte.weight` (154 MB) correctly got its own shard.
- Anything in the task text or documentation that was unclear:
  - The README's "Mapping note" says ternary transforms support the capture-based rewrite model but does not say explicitly that `from_b`/`to` are literal templates rather than patterns; the `equal` assert docs do say this for `right`, and the same holds for transforms.
- Tools used (condition F): n/a (condition B)
- Approximate time spent, if you can tell: about 5 minutes
