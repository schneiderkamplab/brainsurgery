## Participant self-report

- Final artifact path: `out/T1/solution.py`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none, first run succeeded.
- Pitfalls or surprises you hit (one line each): none of note; the block-index
  regex (`model\.layers\.(\d+)\.`) cleanly separates block tensors from the
  two non-block tensors, and building an explicit old-to-new index map from
  the sorted surviving indices avoids any renumbering collision.
- Anything in the task text or documentation that was unclear: none.
- Tools used (condition F): n/a (condition P).
- Approximate time spent, if you can tell: a few minutes.
