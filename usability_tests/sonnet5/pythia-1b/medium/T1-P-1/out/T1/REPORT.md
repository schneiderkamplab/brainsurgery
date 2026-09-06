## Participant self-report

- Final artifact path: `out/T1/solution.py`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none, the single execution succeeded
- Pitfalls or surprises you hit (one line each):
  - Had to compute the old-index -> new-index mapping from the sorted list of
    surviving indices rather than trust the explicit list in the task text,
    to make the script generic and self-checking rather than hardcoding the
    12-entry table.
  - Built the new dict before checking for name collisions, so a stray bug
    in the mapping would be caught by an explicit collision check rather
    than silently overwriting a tensor.
- Anything in the task text or documentation that was unclear: none; the
  per-block tensor names, the drop set, and the old->new mapping were all
  spelled out explicitly.
- Tools used (condition F): n/a (condition P)
- Approximate time spent, if you can tell: a few minutes
