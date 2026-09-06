# T1 self-report

- Final artifact path: `out/T1/solution.py` (output: `out/T1/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none.
- Pitfalls or surprises you hit (one line each):
  - The renumbering collision hazard is real if you rename in place; I avoided it entirely by building a fresh output dict keyed by the new names and asserting no destination key is written twice.
  - The three non-parameter buffers (`attention.bias`, `attention.masked_bias`, `attention.rotary_emb.inv_freq`) are part of a block and must move with it; a regex anchored on `gpt_neox.layers.<i>.` picks them up naturally.
- Anything in the task text or documentation that was unclear: nothing; the explicit old->new mapping removed all ambiguity.
- Tools used (condition F): `safetensors` 0.5.3 (`safe_open` / `save_file`) and `torch` 2.14.0 for the tensors, plus the Python stdlib `re`. I did not use mergekit or torch-state-bridge: mergekit's passthrough slicing works on contiguous layer ranges and would need four slices plus a full model re-export through transformers, and torch-state-bridge would still leave saving, key-set validation and the collision check to me. The task is a pure key-space rewrite of one file, so a ~50-line script over safetensors is the smallest thing that does exactly the required work and enforces the required checks itself.
- Checks enforced by the script (all before any output is written, `sys.exit(1)` otherwise): input has 244 tensors; no destination key written twice; no tensor of blocks 12-15 remains; exactly 12 `attention.query_key_value.weight` tensors; block indices are exactly 0..11; output has exactly 184 tensors.
- Approximate time spent, if you can tell: about 5 minutes.
