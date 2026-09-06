# Participant self-report: T2 (condition P, OLMo-1B-0724-hf)

- Final artifact path: `out/T2/solution.py` (output: `out/T2/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - The input is sharded, so the script merges both shards via `model.safetensors.index.json` and checks for duplicate keys before writing a single file.
  - `index_select` results were made contiguous explicitly so safetensors accepts them.
- Anything in the task text or documentation that was unclear: nothing; the row/column ranges (`0..639`, `768..2047`) matched head 5 at 128 dims per head exactly.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3)
- Approximate time spent, if you can tell: about 2 minutes.
