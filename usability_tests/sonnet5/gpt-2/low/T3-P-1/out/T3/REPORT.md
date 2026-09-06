# Participant self-report

- Final artifact path: `out/T3/solution.py`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none, first run succeeded.
- Pitfalls or surprises you hit (one line each):
  - Had to be careful to only cast the 4 named projection-weight suffixes per
    layer, not biases or any `attn.bias` buffer, so used an explicit name set
    instead of a regex.
  - Bin-packed shards greedily in original tensor order, giving `wte.weight`
    (154 MB, over the 64 MiB limit) its own shard automatically since it never
    fits alongside anything else.
- Anything in the task text or documentation that was unclear: none.
- Tools used (condition F): n/a (condition P).
- Approximate time spent, if you can tell: ~5 minutes.
