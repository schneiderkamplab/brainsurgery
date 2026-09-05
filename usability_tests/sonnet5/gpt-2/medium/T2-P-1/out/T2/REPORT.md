# Participant self-report

- Final artifact path: out/T2/solution.py
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - `c_attn.weight` is `[in, out]` Conv1D layout, so heads are column blocks (not row blocks like a `nn.Linear` weight would be).
  - The fused QKV bias/weight has three separate 768-wide segments (q, k, v), each needing the same head-block removed independently, then concatenated back together in q/k/v order.
  - `c_proj.weight` is the output projection with heads as *row* blocks (input side), unlike `c_attn.weight`'s column blocks.
- Anything in the task text or documentation that was unclear: none; the required-result section gave exact column/row ranges to keep, which made this a mechanical slicing task once the per-tensor layout (column-blocked vs row-blocked) was identified.
- Tools used (condition F): n/a (condition P)
- Approximate time spent, if you can tell: a few minutes
