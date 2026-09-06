# T5 self-report (condition B)

- Final artifact path: `out/T5/plan.yaml` (output checkpoint in `out/T5/`, 4 shards + `model.safetensors.index.json`)
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - Execution 1: `no_match` — `matmul source_b missing`: I wrote `from_b` as a regex with escaped dots (`\.`), but `from_b`/`to` are rewrite templates of the `from_a` captures, so the backslashes were kept literally and the adapter A name did not resolve.
- Pitfalls or surprises you hit (one line each):
  - In ternary transforms (`matmul`) only `from_a` is a regex; `from_b` and `to` are rewrite templates, so use plain dots there and `\1` for the captured layer index.
  - The merge in float32 needs a `cast_` of the base weight to float32 before `add_` and a `cast_` back to float16 after; there is no dtype-promoting `add_` documented.
  - `output.shard: 512MB` is a binary unit (512 MiB) counting tensor data only, which matches the task's 536,870,912-byte budget exactly.
- Anything in the task text or documentation that was unclear:
  - The doc pack says ternary transforms "support the same capture-based rewrite model across from_a, from_b, and to" but does not state explicitly that `from_b` is a template rather than a second regex; an example with a `\1` in `from_b` would have avoided the failed attempt.
  - Whether `add_` promotes or rejects mixed fp16/fp32 operands is not documented; I sidestepped it with explicit casts.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 5 minutes, most of it reading `help.txt` and the README.

Post-run verification (outside the plan, read-only): 244 tensors, same key set, shapes and dtypes as the base; the 228 non-adapted tensors are bit-exact; the 16 merged weights match `fp16(W + 2·B@A)` computed in float32 exactly; each shard file is under 536,870,912 bytes.
