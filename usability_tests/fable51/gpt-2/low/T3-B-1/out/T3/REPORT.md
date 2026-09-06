# Participant self-report: T3 (GPT-2 124M), condition B

- Final artifact path: `out/T3/plan.yaml` (output: `out/T3/model-0000{1..4}-of-00004.safetensors` + `out/T3/model.safetensors.index.json`)
- Number of times you executed the script or plan: 2 (first run produced the output; second run was only to confirm the exit status is 0 and rewrote identical output)
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - The CLI prints the whole plan back on success, so the exit code, not the output, is the signal that the run passed; I re-ran once just to check it.
  - There is no "count tensors by dtype" assert, so "exactly 48 bfloat16" was expressed as count(projection regex)==48 + dtype(projection regex)==bfloat16 + dtype(negative-lookahead complement)==float32.
  - The `attn.bias` causal mask shares the `.bias` suffix with real parameters, so the delete pattern must be anchored to `h\.\d+\.attn\.bias` (full-match regex makes this safe).
- Anything in the task text or documentation that was unclear:
  - The README says a directory-like output path defaults to sharding; it was not explicit whether `output.shard: 64MB` (binary units) is required or the CLI `--shard-size` default (5GB) would apply, so I set it in the plan.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: ~3 minutes
