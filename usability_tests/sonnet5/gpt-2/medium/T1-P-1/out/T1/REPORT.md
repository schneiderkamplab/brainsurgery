# Participant self-report

- Final artifact path: `out/T1/solution.py`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none, the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - Had to be careful that the "no tensor of blocks 9, 10, 11 remains" check is about the
    *output* keys (post-renumbering), since old blocks 9/10/11 are legitimately reassigned
    to new indices 6/7/8 rather than literally forbidden as source blocks.
  - Renumbering was done via a single dict built in ascending old-index order, avoiding any
    old-index/new-index collision (e.g. old 3 -> 2 could never clash with a not-yet-moved
    old 2, since block 2 is dropped, not renumbered).
- Anything in the task text or documentation that was unclear: the phrase "no tensor of
  blocks 9, 10, 11 remains" is slightly ambiguous between "no output tensor keeps a 9/10/11
  index" (what I implemented) vs. "no tensor originally from blocks 9/10/11 remains
  anywhere" (which would be wrong, since those blocks are kept and renumbered, not removed).
- Tools used (condition F): n/a (condition P).
- Approximate time spent, if you can tell: a few minutes.
