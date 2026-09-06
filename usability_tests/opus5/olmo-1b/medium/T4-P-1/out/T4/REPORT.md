# T4 participant self-report (condition P)

- **Final artifact path:** `out/T4/solution.py` (output: `out/T4/model.safetensors`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why (one line each):** none; the single execution succeeded.
- **Pitfalls or surprises you hit (one line each):**
  - The base is sharded, so key discovery has to go through `model.safetensors.index.json` while ft1/ft2 are single files; I wrote one small checkpoint view to hide that difference.
  - The ordering hazard is easy to avoid but easy to get wrong: both task vectors must be taken against the base tensor as read, so I never write back into the base view.
  - The checkpoint has an untied `lm_head.weight` (114 = 16*7 + embed + norm + lm_head), so no shared-storage problem on save; I still `clone()`/`contiguous()` copied tensors before `save_file`.
  - Keeping three ~4.8 GB float32 checkpoints open lazily (`safe_open` per shard, tensor at a time) avoids loading all three at once; only the output dict is fully resident.
- **Anything in the task text or documentation that was unclear:**
  - The task says the output is a single file with 114 tensors but does not say whether an index/config should accompany it; I wrote only `model.safetensors`.
  - "identical in all three" was taken as bit-exact (`torch.equal`), which also implies shape/dtype equality; I check shape and dtype explicitly first for a clearer error.
- **Tools used (condition F):** n/a (condition P: torch 2.14.0, safetensors 0.5.3 only).
- **Approximate time spent, if you can tell:** ~5 minutes, of which 16 s was the script run.
