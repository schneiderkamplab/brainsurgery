# T1 self-report

- Final artifact path: `out/T1/solution.py` (invoked via `out/T1/run.sh`), producing `out/T1/model.safetensors`.
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none.
- Pitfalls or surprises you hit (one line each):
  - Block-index regex must anchor on `h.<i>.` with a literal dot to avoid `h.1.` matching as a prefix of `h.11.`; used `^h\.(\d+)\.` and reconstructed the new key from the match end, so no risk of `h.1` also grabbing `h.10`/`h.11` text.
  - Renumbering must be computed as a single old->new mapping and applied in one pass building a fresh dict, rather than renaming in place, to avoid overwrite collisions (e.g. writing new block 3 before old block 3's slot is read).
  - GPT-2's Conv1D `[in, out]` weight layout didn't matter here since transform is a pure rename/copy — no transpose needed for this task.
- Anything in the task text or documentation that was unclear: none; the renumbering table in TASK.md fully specified the mapping.
- Tools used (condition F): `safetensors` 0.5.3 (`safe_open`/`save_file`) for load/save. Chose a plain script over `mergekit`/`torch-state-bridge` because the task is a pure key-rename + subset copy with no need for merging, arithmetic, or regex-capture rewriting machinery — a direct script keeps the collision-avoidance logic (build full new dict before writing) explicit and easy to verify, and lets the required checks (block count, contiguity, no blocks 9/10/11, exact 121-tensor count) be asserted inline before any file is written.
- Approximate time spent, if you can tell: a few minutes.
