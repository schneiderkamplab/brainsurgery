# T1 self-report (condition P)

- Final artifact path: `out/T1/solution.py` (output: `out/T1/model.safetensors`, 184 tensors)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - Avoided the renumbering collision hazard by building a fresh dict from an old->new index map instead of renaming in place; an explicit collision check guards it anyway.
  - The three non-parameter buffers (`attention.bias`, `masked_bias`, `rotary_emb.inv_freq`) are part of each block and must be dropped/renumbered with it; the regex on `gpt_neox.layers.<i>.` covers them.
- Anything in the task text or documentation that was unclear: the "Required checks" say "no tensor of blocks 12, 13, 14, 15 remains", which refers to post-renumbering indices, not the removed blocks 2, 6, 10, 14; I checked both (no index >= 12, and contiguity 0..11).
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 2 minutes
