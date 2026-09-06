## Participant self-report

- Final artifact path: `out/T2/plan.yaml` (output written to `out/T2/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none, the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - `concat`'s destination must not already exist, so each pruned tensor was built under a
    `_pruned` temp name, the original deleted, then the temp renamed (`move`) back onto the
    original name, since tensor names must stay unchanged.
  - Had to convert the head-based row ranges (drop head 5 of 8, 768 rows/head in the fused
    QKV) and column ranges (256 cols/head in `dense.weight`) into explicit slice bounds by hand:
    keep `[:3840]`+`[4608:]` for QKV rows/bias, `[:, :1280]`+`[:, 1536:]` for dense columns.
- Anything in the task text or documentation that was unclear: none; the row/column layout and
  block boundaries were spelled out explicitly enough to compute the slice bounds directly.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: a few minutes, mostly reading `help.txt` for
  `concat`/`move`/`delete`/`assert` syntax before writing the plan.
