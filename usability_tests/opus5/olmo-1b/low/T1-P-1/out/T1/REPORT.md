# T1 participant self-report

- Final artifact path: `out/T1/solution.py` (output `out/T1/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - Input is sharded, so the state dict had to be assembled from `model.safetensors.index.json` rather than a single file.
  - Renumbering collisions are the real hazard here; I avoided in-place shifting entirely by building a fresh output dict from an explicit old->new map and failing loudly on any duplicate destination key.
  - Regex had to anchor on `^model\.layers\.(\d+)\.` with escaped dots so `lm_head.weight` and `model.embed_tokens.weight` pass through untouched.
- Anything in the task text or documentation that was unclear:
  - The task says the output is "a single file `out/T1/model.safetensors`" while grading compares "`out/T1`"; I wrote only the one safetensors file and no index/config, which matches the Required result.
- Tools used (condition F): n/a (condition P)
- Approximate time spent, if you can tell: ~5 minutes.
