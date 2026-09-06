# T5 (GPT-2 124M), condition B: participant self-report

- Final artifact path: `out/T5/plan.yaml` (output checkpoint in `out/T5/`, 5 shards + `model.safetensors.index.json`)
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - Execution 1: `TransformError: matmul source_b missing: lora::base_model\.model\.h\.0\.attn\.c_attn\.lora_A\.weight` (no_match). I wrote `from_b` as an escaped regex with `\1`; it is a plain rewrite template of the `from_a` match, so the backslashes became literal characters in the looked-up name.
  - Execution 2: success.
- Pitfalls or surprises you hit (one line each):
  - In ternary transforms (`matmul`), only `from_a` is a regex; `from_b` and `to` are rewrite templates where dots must not be escaped. The README states this for `assert equal` (`right` is a rewrite of `left`) and the interfaces reference mentions it for `from_a`/`from_b`/`to`, but neither shows a multi-key example with `\1` in `from_b`.
  - With two inputs, intermediate tensors must be created inside the output alias (`base::`) and deleted afterwards; creating them in `lora::` would make the output alias ambiguous.
  - `h.<i>.attn.bias` (mask buffer) exists in the base; the c_attn regex was kept exact so it was never a problem.
  - The transpose is done with `permute` (`order: [1, 0]`); there is no dedicated transpose transform.
  - Sharding with `shard: 100MB` packed 5 shards (4 under 100 MiB, `wte.weight` alone in the last); no CLI option was needed.
- Anything in the task text or documentation that was unclear:
  - Whether the `count` assert accepts `is: 0` for "no matches"; I used `not: { exists: ... }` instead.
  - The `help.txt` box for `assert` does not show the payload keys of `count` (`of`, `is`); I took them from the README.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 5 minutes
