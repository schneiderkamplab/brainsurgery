## Participant self-report

- Final artifact path: `out/T1/solution.py`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each): none — mapped old block indices to new ones via a simple list of surviving indices in order, then applied the renumbering with a regex on the `h.<i>.` prefix so it can't accidentally touch `wte`/`wpe`/`ln_f`.
- Anything in the task text or documentation that was unclear: no
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: a few minutes
