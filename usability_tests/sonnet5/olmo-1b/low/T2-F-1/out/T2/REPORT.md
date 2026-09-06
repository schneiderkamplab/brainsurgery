## Participant self-report

- Final artifact path: `out/T2/solution.py` (invoked by `out/T2/run.sh`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each): none — the task spec fully pins down the row/column
  block boundaries and slice order (keep `0..639`, `768..2047`), so no exploration of the checkpoint
  layout was needed beyond confirming shard count/tensor count via the index JSON.
- Anything in the task text or documentation that was unclear: no.
- Tools used (condition F): `torch` 2.14.0 and `safetensors` 0.5.3 only, via a plain script. This is
  a fixed-layout slice-and-reassemble operation on two tensors' worth of row/column blocks per layer;
  `index_select` on the row/column dims is a direct, dependency-free expression of exactly what the
  spec asks for. I considered `transformers`' `prune_heads`, but that API prunes heads in a live model
  (and permanently renumbers/tracks pruned indices in the module config) rather than emitting a
  standalone edited safetensors file with the exact required key set, so a plain script kept the
  output shape (114 tensors, unchanged names, only q/k/v/o resized) explicit and easy to check against
  the "Required checks" before writing.
- Approximate time spent, if you can tell: a few minutes.
