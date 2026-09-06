# T1 self-report (condition F, OLMo-1B-0724-hf)

- Final artifact path: `out/T1/solution.py` (writes `out/T1/model.safetensors`, 86 tensors, 12 blocks)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - None during the run. The known hazard (renumbering collisions when shifting blocks in place) was avoided by building a fresh destination dict from an explicit old->new index map and failing on any duplicate destination name.
- Anything in the task text or documentation that was unclear:
  - The "Required checks" say "no tensor of blocks 12, 13, 14, 15 remains"; I read this as "no block index >= 12 remains in the output" (checked generically), in addition to verifying the four removed blocks contributed exactly 28 dropped tensors.
- Tools used (condition F): name, version, and why:
  - `safetensors` 0.5.3: load the two input shards and save the single output file; bit-exact, no framework model instantiation needed.
  - `torch` 2.14.0: tensor equality checks and `.contiguous()` before saving.
  - Python stdlib `re`/`json`: parse the shard index and match `model.layers.<i>.<rest>` names.
  - Considered and rejected: `mergekit` passthrough slicing (would need a YAML with four slice ranges and produces a sharded HF directory plus config, not a single file; more moving parts than a 90-line script) and `torch-state-bridge` (regex rewriting is fine, but persistence and the checks would still be my own code). `transformers` loading was unnecessary since only key renaming is involved.
- Approximate time spent, if you can tell: about 2 minutes wall clock, including a second independent verification pass comparing the output against the expected mapping.
