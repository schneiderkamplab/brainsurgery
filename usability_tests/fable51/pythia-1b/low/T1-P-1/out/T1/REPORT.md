# Participant self-report: T1 (condition P)

- Final artifact path: `out/T1/solution.py` (output `out/T1/model.safetensors`, 184 tensors)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - Renumbering was done by building a fresh dict from an old->new index map rather than renaming in place, which removes the collision hazard entirely; an explicit collision check was still added.
  - Tensors were made contiguous before `save_file` as a precaution against safetensors rejecting views.
- Anything in the task text or documentation that was unclear: nothing; the block-count and tensor-count checks were unambiguous.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 2 minutes
