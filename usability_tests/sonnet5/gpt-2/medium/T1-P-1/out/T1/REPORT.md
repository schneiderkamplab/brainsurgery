# Participant self-report

- Final artifact path: `out/T1/solution.py` (output written to `out/T1/model.safetensors`)
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - Execution 1: ran without crashing but wrote the output to `out/model.safetensors` instead of `out/T1/model.safetensors` due to a path-join bug (`OUTPUT_DIR` computed one directory too high); fixed the path and re-ran, then deleted the misplaced file.
- Pitfalls or surprises you hit (one line each):
  - The block-index regex has to be anchored (`^h\.(\d+)\.`) so it doesn't also match things like `h.1.` as a prefix of `h.11.`.
  - Needed to derive the old-block -> new-block mapping generically (sorted surviving old indices enumerated in order) rather than hardcoding the 9 pairs, then cross-checked it against the mapping given in the task text.
- Anything in the task text or documentation that was unclear: None; the required renumbering mapping was given explicitly, which made verification straightforward.
- Tools used (condition F): n/a (condition P)
- Approximate time spent, if you can tell: ~5 minutes
