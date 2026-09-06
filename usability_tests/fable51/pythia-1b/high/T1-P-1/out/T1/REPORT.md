# Participant self-report: T1 (Pythia-1B depth pruning), condition P

- Final artifact path: `out/T1/solution.py` (writes `out/T1/model.safetensors`, 184 tensors, 12 blocks)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution passed all checks and wrote the output.
- Pitfalls or surprises you hit (one line each):
  - Avoided the renumbering collision hazard by building a fresh output dict keyed by new names (old->new map from the ordered survivor list) instead of renaming in place, and by failing on any duplicate new name.
  - Anchored the layer regex (`^gpt_neox\.layers\.(\d+)\.`) with escaped dots so `embed_in`, `embed_out` and `final_layer_norm` pass through untouched.
  - Checks run before anything is written; the file is written to a `.tmp` path and renamed only on success, so a failure leaves no output.
- Anything in the task text or documentation that was unclear: the "Required checks" say "no tensor of blocks 12, 13, 14, 15 remains", which refers to post-renumbering indices, not the dropped blocks 2, 6, 10, 14; that took a moment to read correctly. Also unclear whether safetensors metadata must match the reference; I preserved the input's metadata header.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3)
- Approximate time spent, if you can tell: about 2 minutes.
