# Participant self-report: T1 (GPT-2 124M), condition B

- Final artifact path: `out/T1/model.safetensors` (121 tensors), plan at `out/T1/plan.yaml`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none failed.
- Pitfalls or surprises you hit (one line each):
  - `move` refuses existing destinations, so survivors had to be renumbered in ascending order (old 3->2, 4->3, ...) after the delete so each target index is already vacant; that ordering is also what makes collisions impossible.
  - Regex `from` with a capture group and `\1` in `to` handled a whole block (13 tensors) per `move`.
- Anything in the task text or documentation that was unclear:
  - The Required checks say "no tensor of blocks 9, 10, 11 remains", which is a post-renumbering check; the deleted blocks are 2, 5, 8. I asserted both (deleted blocks absent right after `delete`, 9-11 absent at the end).
  - Help for `move`/`delete` does not state explicitly that `target`/`from` accept regex patterns with captures; inferred from the README tensor-reference section and the `equal` docs.
- Tools used (condition F): n/a (condition B, plan only)
- Approximate time spent, if you can tell: about 3 minutes
