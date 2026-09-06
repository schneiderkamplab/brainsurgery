## Participant self-report

- Final artifact path: `out/T4/solution.py` (output written to `out/T4/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none, first run succeeded.
- Pitfalls or surprises you hit (one line each):
  - The base checkpoint is sharded (index JSON + two shard files) while the
    fine-tunes are each a single `model.safetensors`, so the base needed its
    own loader that resolves the weight map across shards before the
    equality checks could run.
  - There is an `inputs/lora` directory (adapter config + adapter weights)
    that TASK.md never mentions; ignored it since the spec only references
    `base`, `ft1`, `ft2`.
- Anything in the task text or documentation that was unclear: no, the spec
  (verify shared tensors, merge formula, required checks) was unambiguous.
- Tools used (condition F): n/a (condition P).
- Approximate time spent, if you can tell: a few minutes.
