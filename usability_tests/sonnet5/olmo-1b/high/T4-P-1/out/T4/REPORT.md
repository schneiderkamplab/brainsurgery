# Participant self-report

- Final artifact path: `out/T4/solution.py`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none, the single execution succeeded.
- Pitfalls or surprises you hit (one line each): the base checkpoint is sharded (two files plus an index) while the fine-tunes are single files, so the loader needed to handle both layouts; used `safetensors.safe_open` per shard/file to avoid materializing all three ~5GB checkpoints in memory at once.
- Anything in the task text or documentation that was unclear: none — the merge formula, the 48-tensor list, and the required checks were unambiguous.
- Tools used (condition F): n/a (condition P).
- Approximate time spent, if you can tell: ~10 minutes to write, run and verify the script.
