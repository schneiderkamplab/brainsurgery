# T1 participant self-report (condition F, Pythia-1B)

- Final artifact path: `out/T1/solution.py` (output: `out/T1/model.safetensors`, 184 tensors)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - None material. Collision hazard avoided by building a fresh dict from an old->new index map (never renaming in place), with an explicit duplicate-name check.
  - Output is written to a temp file and renamed only after all checks pass, so a failing check leaves no output.
- Anything in the task text or documentation that was unclear:
  - The "Required checks" say "no tensor of blocks 12, 13, 14, 15 remains", which reads like the last four indices rather than the dropped set (2, 6, 10, 14); I enforced both (no index >= 12, and exact per-block tensor counts for 0..11).
- Tools used (condition F): name, version, and why:
  - `safetensors` 0.5.3: `load_file` / `save_file` for bit-exact tensor I/O.
  - `torch` 2.14.0: tensor container only.
  - Python `re` for regex capture of the block index.
  - Considered `mergekit` passthrough slicing and `torch-state-bridge` key rewriting, but a 40-line script is smaller, needs no YAML/config, keeps the non-parameter buffers (`attention.bias`, `masked_bias`, `inv_freq`) which mergekit's HF-model route would drop, and guarantees bit-exact values.
  - Post-hoc verification (not part of the artifact): an independent rebuild of the expected key map with `torch.equal` on every tensor, plus a strict `load_state_dict` into a 12-layer `GPTNeoXForCausalLM` from `transformers` 5.12.1.
- Approximate time spent, if you can tell: about 3 minutes.
