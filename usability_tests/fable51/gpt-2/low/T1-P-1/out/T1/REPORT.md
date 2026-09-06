# T1 self-report (condition P)

- Final artifact path: `out/T1/solution.py` (output `out/T1/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - Avoided renumbering collisions by building a fresh dict from an old->new index map rather than renaming in place.
  - Anchored the block regex to `^h\.(\d+)\.` so `attn.bias` / `mlp.c_proj` names could not be mis-parsed.
- Anything in the task text or documentation that was unclear: the "Required checks" say no tensor of blocks 9, 10, 11 may remain, which refers to post-renumbering indices; I checked both that and that the removed blocks 2, 5, 8 are gone (via the exact-count and value checks).
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 2 minutes
