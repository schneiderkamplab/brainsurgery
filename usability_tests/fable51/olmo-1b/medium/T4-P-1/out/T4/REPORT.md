# T4 participant self-report (condition P)

- Final artifact path: `out/T4/solution.py` (output: `out/T4/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - Base is sharded (two shards plus an index) while the fine-tunes are single files, so the loader had to merge shards and check for duplicate names.
  - Each task vector must be taken against the original base tensor, so the merge computes `b + l*(ft1-b) + l*(ft2-b)` from an untouched `base` dict rather than updating in place.
- Anything in the task text or documentation that was unclear: nothing; the formula, tensor list and counts were explicit.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3)
- Approximate time spent, if you can tell: about 3 minutes, of which the single run took 13 s wall clock.
