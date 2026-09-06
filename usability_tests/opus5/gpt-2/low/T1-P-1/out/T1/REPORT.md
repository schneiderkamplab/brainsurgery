# T1 self-report (condition P)

- Final artifact path: `out/T1/model.safetensors` (script: `out/T1/solution.py`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - Renumbering collision hazard: avoided entirely by building a fresh output dict
    keyed by new names instead of renaming in place, plus an explicit collision check.
  - `attn.bias` is a mask buffer, not a parameter, but it is part of the 13 tensors
    per block, so a block-index-based rule (not a parameter-name list) is the safe filter.
  - Anchored the regex at `^h\.(\d+)\.` so non-block tensors (`wte`, `wpe`, `ln_f.*`)
    can never be touched.
- Anything in the task text or documentation that was unclear: nothing; the
  old->new index mapping was given explicitly, which removed the main ambiguity.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: ~3 minutes
