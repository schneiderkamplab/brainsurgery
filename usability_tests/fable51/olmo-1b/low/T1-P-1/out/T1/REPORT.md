# Participant self-report: T1 (OLMo-1B-0724-hf), condition P

- Final artifact path: `out/T1/solution.py` (output: `out/T1/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - Avoided renumbering collisions by building a fresh destination dict rather than renaming in place; a collision check on destination keys guards it anyway.
  - Input is sharded; read the index's weight_map to enumerate shard files rather than hardcoding names.
- Anything in the task text or documentation that was unclear: the "Required checks" list says "no tensor of blocks 12, 13, 14, 15 remains", which refers to post-renumbering indices; I also checked the removed original blocks by construction.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 2 minutes
