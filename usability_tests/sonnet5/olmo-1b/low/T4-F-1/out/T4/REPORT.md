# T4 self-report

## Participant self-report

- Final artifact path: `out/T4/model.safetensors`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single run succeeded.
- Pitfalls or surprises you hit (one line each): base checkpoint is sharded (index.json + 2 shard files) while ft1/ft2 are single-file, so loading had to branch on that; none of the merge logic itself was surprising.
- Anything in the task text or documentation that was unclear: none.
- Tools used (condition F): name, version, and why: plain script on `torch` 2.14.0 and `safetensors` 0.5.3 only. Chose this over `mergekit`'s task-arithmetic YAML because the required precondition check (bit-exact identity of all 66 non-MLP tensors across three checkpoints, aborting loudly otherwise) and exact-count checks (48 merged, 114 output) are easier to make explicit and auditable in a short script than to express/verify through a merge-config abstraction.
- Approximate time spent, if you can tell: a few minutes.
