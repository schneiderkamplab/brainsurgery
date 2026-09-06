# Participant self-report

- Final artifact path: `out/T2/plan.yaml`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none, the plan succeeded on the first run.
- Pitfalls or surprises you hit (one line each):
  - `concat`/`copy` require the destination to not already exist, and `move`/`concat` don't support in-place overwrite of the source name, so each projection was rebuilt under a `.tmp` name, the original deleted, then `move`d back to the original name to keep tensor names unchanged.
  - `concat`'s `from` list only takes exact references (each must resolve to one tensor), so the plan needed one `concat`/`delete`/`move` triplet per (layer, projection) rather than a single regex-driven rule; generated the 16x4 repetitions with a short local script since the plan itself has no loop construct.
- Anything in the task text or documentation that was unclear: none; the row/column block boundaries for q/k/v vs. o_proj were stated explicitly enough to slice directly (`[:640,:]`+`[768:,:]` for rows, `[:, :640]`+`[:, 768:]` for columns).
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: a few minutes.
