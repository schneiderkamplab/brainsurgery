# Participant self-report

- Final artifact path: out/T4/solution.py
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none, the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - None significant; the main care point was verifying non-MLP tensors bit-exact across all three checkpoints (base vs ft1 and base vs ft2, not just ft1 vs ft2) before computing anything.
  - Made sure to compute the task vectors against the unmodified `base` array only, not against an already-merged output, to avoid the ordering hazard called out in the task.
  - Cast tensors to float32 before the arithmetic and called `.contiguous()` before saving to avoid safetensors errors on non-contiguous/shared storage.
- Anything in the task text or documentation that was unclear: no, the spec (tensor names, formula, required checks) was fully unambiguous.
- Tools used (condition F): N/A (condition P — plain PyTorch + safetensors only).
- Approximate time spent, if you can tell: a few minutes to write and run the script once.
