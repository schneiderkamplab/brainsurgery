# T4 self-report

- Final artifact path: `out/T4/solution.py` (output `out/T4/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - The merge order hazard is easy to avoid by never writing into the base dict: both task vectors are taken from the loaded, untouched `base` and only the new `out` dict is written.
  - Non-MLP tensors are cloned from the base rather than referenced, so safetensors does not see shared storage.
- Anything in the task text or documentation that was unclear: no; the 64 MLP names, lambda, dtype handling and required checks were fully specified.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3)
- Approximate time spent, if you can tell: ~5 minutes.
