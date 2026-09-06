# Participant self-report

- Final artifact path: `out/T3/solution.py`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - After upcasting `embed_in.weight`/`embed_out.weight` from float16 to float32 they grow to ~393 MiB each, over the 256 MiB shard budget, so each must sit alone in its own shard even though the task text quotes their float16 size (206 MB) — sizing has to be computed post-cast, not from the input dtype.
- Anything in the task text or documentation that was unclear: none; the buffer names, projection list, and required checks were unambiguous enough to implement directly.
- Tools used (condition F): n/a (condition P)
- Approximate time spent, if you can tell: ~10 minutes
