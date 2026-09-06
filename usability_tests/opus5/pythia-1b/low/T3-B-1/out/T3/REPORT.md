# T3 self-report (condition B)

- Final artifact path: `out/T3/plan.yaml` (output checkpoint in `out/T3/`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - The buffer `gpt_neox.layers.<i>.attention.bias` shares a suffix with the real
    parameter biases (`attention.dense.bias`, `query_key_value.bias`), so the delete
    pattern had to be an exact alternation of the three buffer names, not `.*bias`.
  - After deleting the uint8/scalar buffers, a blanket `cast_: {target: '.*', to: float32}`
    is safe and simpler than enumerating the float32 keys; the bfloat16 cast is applied
    afterwards on top of it.
  - "exactly 64 tensors are bfloat16" needs two asserts: `dtype` over the projection
    pattern plus a `dtype: float32` assert over the negative-lookahead complement,
    since there is no count-by-dtype expression.
  - The 256 MiB shard budget is expressed as `shard: 256MB` (binary units); the two
    412 MB float32 embedding matrices exceed it and are written alone, as specified.
- Anything in the task text or documentation that was unclear:
  - The task lists the embeddings as "206 MB each", which is their float16 input size;
    in the float32 output they are 412 MB. Not ambiguous once noticed.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: ~10 minutes, most of it reading `help.txt`.
