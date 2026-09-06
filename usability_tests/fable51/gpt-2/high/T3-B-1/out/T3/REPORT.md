# Participant self-report: T3 (GPT-2 124M), condition B

- Final artifact path: `out/T3/plan.yaml` (output checkpoint in `out/T3/`: 4 shards + `model.safetensors.index.json`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution passed all asserts and wrote the output.
- Pitfalls or surprises you hit (one line each):
  - There is no assert that counts tensors by dtype, so "exactly 48 tensors are bfloat16" is expressed as: the 48-tensor projection pattern is all bfloat16 and the 100-tensor complement (negative-lookahead regex) is all float32.
  - Tensor references are full-match regexes, so dots must be escaped (`h\.\d+\....`) and the buffer pattern `h\.\d+\.attn\.bias` does not touch `h.<i>.attn.c_proj.bias`.
  - `dtype` accepted a pattern in `of` and checked every match, which the help text does not state explicitly.
- Anything in the task text or documentation that was unclear:
  - Whether `assert: dtype` with a multi-match pattern checks all matches or only the first is not documented in help.txt; it checked all (verified by the complement check passing with 100 tensors).
  - The README documents shard units as binary (64MB = 64 MiB), which matched the task's 67,108,864 bytes budget; verified shard data sizes are 66,259,968 / 66,256,896 / 40,983,552 and `wte.weight` alone in shard 4.
- Tools used (condition F): n/a (condition B)
- Approximate time spent, if you can tell: about 5 minutes (reading the doc pack, one plan execution, output inspection).
