# Participant self-report: T2 (Pythia-1B head pruning), condition B

- Final artifact path: `out/T2/plan.yaml` (output checkpoint `out/T2/model.safetensors`, 244 tensors)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - `concat` requires each source to resolve to exactly one tensor, so the plan cannot use one regex over all 16 layers; the 48 concat/delete/move triples were generated with a shell loop.
  - `concat` and `copy` cannot write to an existing name, so each tensor is rebuilt under a `tmp.*` name, the original is deleted, and the temporary is moved back to the original name.
  - The shell used `=` as a special prefix, which broke a separator echo while reading docs; unrelated to the tool.
- Anything in the task text or documentation that was unclear:
  - `help.txt` examples for `split` and `concat` show an empty `to:`/`from:` list (`from: , to: a::xy`), which looks like a rendering bug.
  - The README does not state whether `concat`/`copy` may overwrite an existing destination; the per-transform help does.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 5 minutes
