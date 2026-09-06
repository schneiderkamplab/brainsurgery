# T1 participant self-report (condition B, OLMo-1B-0724-hf)

- Final artifact path: `out/T1/plan.yaml` (plan); output checkpoint `out/T1/model.safetensors` (86 tensors); executed-plan summary in `out/T1/summary.yaml`.
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution passed all asserts and wrote the output.
- Pitfalls or surprises you hit (one line each):
  - Renumbering collisions were avoided by moving surviving blocks one at a time in ascending source order (3->2, 4->3, ..., 15->11); every destination index is lower than its source and already vacated, and `move` refuses existing destinations anyway.
  - Regex dots must be escaped (`model\.layers\.3\.(.*)`) so `layers.3.` cannot match e.g. `layers.13.`; the `(2|6|10|14)` alternation is anchored by the surrounding `\.` so 1/11/12 etc. are not touched.
  - The zsh shell treats a bare `=====` echo argument as an equals-expansion; unrelated to the tool, but cost one noisy command.
- Anything in the task text or documentation that was unclear:
  - The README does not say in which order a single pattern `move` processes multiple matches, so I used one `move` per block rather than a single regex over all surviving layers.
  - `help.txt` does not show the payload form for `not`/`exists`; the doc-pack example plan (`assert: { not: { exists: ... } }`) resolved it.
- Tools used (condition F): n/a (condition B, `brainsurgery` only).
- Approximate time spent, if you can tell: about 3 minutes of reading and one plan run (about 15 s, dominated by writing the 4 GB output).
