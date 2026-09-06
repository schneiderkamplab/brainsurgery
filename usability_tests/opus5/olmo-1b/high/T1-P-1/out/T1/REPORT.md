# T1 participant self-report

- Final artifact path: `out/T1/model.safetensors` (script: `out/T1/solution.py`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none — the single execution passed all checks.
- Pitfalls or surprises you hit (one line each):
  - Renumbering collision hazard: avoided by building a fresh output dict keyed by the new names instead of renaming in place, plus an explicit collision check.
  - Sharded input: had to read both shards via `model.safetensors.index.json` rather than assuming one file, and assert the loaded key set matches the weight map.
  - Tied/shared storage: `model.embed_tokens.weight` and `lm_head.weight` can share memory, which `safetensors.save_file` rejects, so tensors are cloned and made contiguous before writing.
- Anything in the task text or documentation that was unclear:
  - The task says output "a single file `out/T1/model.safetensors`" but grading compares the directory `out/T1`; I left `solution.py` and this report in that directory, assuming grading only inspects the safetensors file.
  - Whether the HF config/tokenizer files (and a `num_hidden_layers: 12` edit) should be copied alongside the checkpoint is not stated; taking the required-result list literally, I wrote only the safetensors file.
- Tools used (condition F): n/a — condition P (torch 2.14.0, safetensors 0.5.3).
- Approximate time spent, if you can tell: ~5 minutes.
