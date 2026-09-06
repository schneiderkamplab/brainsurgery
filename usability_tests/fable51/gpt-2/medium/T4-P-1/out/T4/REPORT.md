# Participant self-report: T4 (condition P)

- Final artifact path: `out/T4/solution.py` (output: `out/T4/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - Selected the 48 MLP tensors with an anchored regex and cross-checked the set against the expected names so `c_proj` in `attn` could not leak in.
  - Computed both task vectors from the untouched `base` dict rather than from an accumulating output tensor, and made outputs contiguous before saving.
- Anything in the task text or documentation that was unclear: nothing; the formula and the check list were explicit.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3)
- Approximate time spent, if you can tell: about 2 minutes
