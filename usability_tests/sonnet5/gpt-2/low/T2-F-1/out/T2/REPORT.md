## Participant self-report

- Final artifact path: `out/T2/solution.py` (run via `out/T2/run.sh`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single run succeeded.
- Pitfalls or surprises you hit (one line each):
  - None major; the task spec fully spelled out the column/row ranges to keep, so it was a matter of translating them into slices in the right axis (columns for `c_attn.weight`/segment order, rows for `c_attn.bias`, rows for `c_proj.weight`) and concatenating per-segment (q, k, v) rather than treating `c_attn.weight` as one flat 2304-wide block.
- Anything in the task text or documentation that was unclear: none; the required kept-column/row ranges were given explicitly per tensor, which removed any ambiguity about Conv1D layout or segment order.
- Tools used (condition F): `torch` 2.14.0 and `safetensors` 0.5.3 for tensor slicing and safetensors load/save. Chose plain slicing over `transformers` `prune_heads` because the task specifies exact kept-column ranges per q/k/v segment and a direct script gives full, auditable control over concatenation order and lets the required shape/count checks be asserted explicitly before writing, rather than trusting a library helper's internal head-pruning logic to match the byte-for-byte layout required for grading.
- Approximate time spent, if you can tell: a few minutes.
