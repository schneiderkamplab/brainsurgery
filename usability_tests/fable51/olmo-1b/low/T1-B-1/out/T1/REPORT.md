# Participant self-report: T1 (OLMo-1B-0724-hf depth pruning), condition B

- Final artifact path: `out/T1/plan.yaml` (output checkpoint `out/T1/model.safetensors`, 86 tensors)
- Number of times you executed the script or plan: 2 (both succeeded; the second run was only to re-read the log, the plan was unchanged)
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - `move` refuses existing destinations, so the renumbering had to be ordered ascending (old 3->2 before 4->3, ...) so every destination index is freed by a delete or a prior move before it is used.
  - Regex references are full-match, so `model\.layers\.(2|6|10|14)\..*` needs the trailing `\..*` and the escaped dot to avoid matching e.g. layer 12 or 14 when deleting 1 or 2.
  - No CLI option was needed; the sharded input directory with `model.safetensors.index.json` loaded directly as the `model` alias and a `.safetensors` output path produced a single file.
- Anything in the task text or documentation that was unclear:
  - `help.txt` does not explicitly say that `move` supports regex capture rewrites in `to`; this had to be inferred from `assign`/`copy` examples and the interfaces reference note on "destination synthesis".
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 3 minutes
