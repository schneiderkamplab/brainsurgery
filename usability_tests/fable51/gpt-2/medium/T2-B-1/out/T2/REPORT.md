# Participant self-report: T2 (GPT-2 124M, condition B)

- Final artifact path: `out/T2/plan.yaml` (output checkpoint `out/T2/model.safetensors`, 160 tensors)
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each): none failed. The first run succeeded; the second was a re-run only to read the log lines that the first run's plan echo had scrolled past.
- Pitfalls or surprises you hit (one line each):
  - `concat` cannot write over an existing tensor and `move` refuses existing destinations, so each pruned tensor needs concat-to-temp, delete original, move temp back (3 transforms x 3 tensors x 12 layers).
  - `concat` sources must each resolve to exactly one tensor, so no regex/`$i` batching across layers; the plan was generated with a shell loop instead.
  - Regex references are full-match with unescaped dots (`h.0.attn.c_attn.weight`); safe here because no other name matches, but it would overmatch in general.
  - The successful run prints the whole normalized plan after the log, which hides the "Preserved 160 tensors" line unless you filter the output.
- Anything in the task text or documentation that was unclear: nothing about the task. The docs don't state explicitly whether `concat`'s `to` may name an existing tensor; I assumed not, given `copy`/`move` semantics.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 5 minutes
