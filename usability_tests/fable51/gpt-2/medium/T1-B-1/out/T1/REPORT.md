# Participant self-report: T1 (GPT-2 124M), condition B

- Final artifact path: `out/T1/plan.yaml` (output checkpoint: `out/T1/model.safetensors`, 121 tensors)
- Number of times you executed the script or plan: 2 (first run produced the final output; the second was an identical re-run only to capture the exit code explicitly, since the first invocation was piped through `tail`)
- Which executions failed, and why (one line each): none; both runs exited 0 with all asserts passing
- Pitfalls or surprises you hit (one line each):
  - Tensor references are full-match regexes, so `.` must be escaped (`h\.3\.(.*)`) to avoid `h.10.*`-style overreach; `h\.(2|5|8)\..*` deletes exactly 39 tensors.
  - Renumbering collisions are avoided by moving in ascending source order (3->2, 4->3, 6->4, ...): every destination index is lower than its source and already vacated; `move` also refuses existing destinations, so a wrong order would fail loudly rather than overwrite.
  - `help.txt` does not show the payload syntax for `not`/`exists` (key metadata "unavailable"), but the README example form `{ not: { exists: '...' } }` worked.
- Anything in the task text or documentation that was unclear:
  - The "Required checks" ask that no tensor of blocks 9, 10, 11 remains, which only holds after renumbering; I placed the check after the moves. The task's "Required result" removes blocks 2, 5, 8, so the two lists refer to different stages, which took a moment to reconcile.
- Tools used (condition F): n/a (condition B)
- Approximate time spent, if you can tell: about 5 minutes
