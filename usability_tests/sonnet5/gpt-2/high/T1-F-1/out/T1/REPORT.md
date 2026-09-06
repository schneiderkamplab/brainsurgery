# T1 self-report

## Participant self-report

- Final artifact path: `out/T1/solution.py` (invoked via `out/T1/run.sh`), output at `out/T1/model.safetensors`.
- Number of times you executed the script or plan: 1 (the single successful execution that produced `out/T1/model.safetensors`). I additionally ran an ad hoc negative-control copy of the script outside `out/` with `DROP_BLOCKS = {2, 5}` (missing block 8) to confirm the required checks actually fail loudly on a wrong input; that control run is not one of the counted attempts on this task's real artifact and did not touch `out/`.
- Which executions failed, and why (one line each): none — the single real execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - None specific to this task; the main hazard the task calls out (shifting blocks in the wrong order and colliding indices) is avoided by computing the old->new mapping from the full sorted list of surviving indices up front, then writing to a fresh dict rather than renaming in place.
  - Double-checked the `attn.bias` causal-mask buffer is treated like any other per-block tensor (moved and renumbered, not filtered out) since it still belongs to its block.
- Anything in the task text or documentation that was unclear: no.
- Tools used (condition F): `safetensors` 0.5.3 (`safe_open`/`get_tensor` to load, `safetensors.torch.save_file` to write) and `torch` 2.14.0 (tensor container) via a plain Python script. Chose a plain script over `mergekit` layer-slicing or `torch-state-bridge` regex rewriting because the required transformation is a single bulk-rename with an explicit drop set and a hand-built old->new index map — expressing it directly in Python was less indirection than authoring a merge config for a rename-only op, and made it straightforward to add the exact required checks (residual-block check, 9-block count check, exact tensor-count check) as hard `sys.exit` assertions before any file is written, so the run fails closed and leaves no partial output.
- Approximate time spent, if you can tell: a few minutes (single pass: write script, run, verify against a bit-exact/shape/dtype re-check derived from the spec's explicit renumbering table).
