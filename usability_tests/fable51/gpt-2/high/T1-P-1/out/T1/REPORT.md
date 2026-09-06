# Participant self-report: T1 (GPT-2 124M), condition P

- Final artifact path: `out/T1/solution.py` (output checkpoint: `out/T1/model.safetensors`, 121 tensors)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - Avoided the renumbering-collision hazard by building a fresh output dict from the source instead of renaming in place, and by asserting on duplicate destination names.
  - Anchored the block regex to `^h\.(\d+)\.` so nothing else (e.g. `wte`, `ln_f`) can be caught by it.
  - Called `.contiguous()` before `save_file` to be safe; the source tensors were already contiguous so this was a no-op.
- Anything in the task text or documentation that was unclear:
  - The check "no tensor of blocks 9, 10, 11 remains" refers to the *new* numbering (i.e. no stale high indices after the shift); I implemented it as "no block index >= 9 remains" plus "block indices are exactly 0..8".
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3, stdlib `re`)
- Approximate time spent, if you can tell: about 2 minutes
