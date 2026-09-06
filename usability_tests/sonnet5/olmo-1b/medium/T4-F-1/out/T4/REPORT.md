# T4 Participant self-report

- **Final artifact path:** `out/T4/solution.py` (invoked via `out/T4/run.sh`), output at `out/T4/model.safetensors`
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why (one line each):** none — the single execution succeeded.
- **Pitfalls or surprises you hit (one line each):**
  - The base checkpoint is sharded (`model.safetensors.index.json` + two shard files) while the fine-tunes are single-file, so loading had to branch on whether an index file is present.
  - Had to be careful to compute both task vectors (`ft1 - base`, `ft2 - base`) against the *original* base tensor, not against a base already mutated by the first addition — did this by building a fresh `out[k]` tensor per key rather than mutating `base[k]` in place.
- **Anything in the task text or documentation that was unclear:** No — inputs, formula, and required checks were unambiguous.
- **Tools used (condition F): name, version, and why:**
  - `torch` 2.14.0 — tensor arithmetic in float32.
  - `safetensors` 0.5.3 — loading the sharded base and single-file fine-tunes, and saving the merged output.
  - Chose a plain script over `mergekit`'s task-arithmetic YAML config because the task additionally requires a strict pre-merge equality check (all non-MLP tensors bit-identical across all three checkpoints) that isn't something mergekit's config expresses directly; a script makes that check, the abort-on-mismatch behavior, and the exact-count assertions (48 merged, 114 total) explicit and easy to verify.
- **Approximate time spent, if you can tell:** ~10 minutes.
