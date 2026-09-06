## Participant self-report

- Final artifact path: `out/T4/solution.py` (output written to `out/T4/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none, first execution succeeded.
- Pitfalls or surprises you hit (one line each): none — the MLP tensor name
  list given in the task matched the checkpoint exactly, and the non-MLP
  tensors were indeed bit-identical across all three checkpoints, so the
  precondition check passed on the first try.
- Anything in the task text or documentation that was unclear: none.
- Tools used (condition F): n/a (condition P).
- Approximate time spent, if you can tell: a few minutes; single script write
  and run, no debugging needed.
