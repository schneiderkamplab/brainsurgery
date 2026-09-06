## Participant self-report

- Final artifact path: `out/T3/solution.py`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - Had to distinguish projection weights (regex on the 4 layer-local module names) from embeddings/norms/biases so a broad `.*weight` pattern wouldn't overmatch onto `embed_in`/`embed_out`/layer norms.
  - The two embedding tensors are float32 in the output (412 MB each, larger than the 256 MiB shard limit) and each landed alone in its own shard, matching the spec's oversized-tensor rule.
- Anything in the task text or documentation that was unclear: none.
- Tools used (condition F): n/a (condition P).
- Approximate time spent, if you can tell: a few minutes.
