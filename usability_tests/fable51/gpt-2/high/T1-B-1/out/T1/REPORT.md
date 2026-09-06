# T1 (GPT-2 124M, condition B) — Participant self-report

- Final artifact path: `out/T1/model.safetensors` (plan: `out/T1/plan.yaml`,
  executed-transform summary: `out/T1/summary.yaml`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution
  passed all asserts and wrote 121 tensors.
- Pitfalls or surprises you hit (one line each):
  - Renumbering collision hazard: moves are done in ascending source order
    (3→2, 4→3, 6→4, 7→5, 9→6, 10→7, 11→8) after deleting 2/5/8, so each
    destination is either deleted or already vacated; `move` refusing to
    overwrite existing destinations is a second safety net.
  - Regex references are full-match, so `h\.(2|5|8)\..*` cannot overmatch
    e.g. block 12 or `h.20.*` (there are none, but the anchor matters);
    structured lists `["h", "3", "*rest"]` were used for the moves to avoid
    regex escaping and to match the block index as a whole segment.
  - The `count`/`exists`/`not` asserts run against the in-memory state, so the
    "output has 121 tensors" check is expressed as `count: { of: '.*', is: 121 }`
    right before `output`.
  - After the run I inspected the result read-only with the safetensors
    library (key set, shapes, dtypes, bit-exact values vs the input); the edit
    itself is entirely in the plan.
- Anything in the task text or documentation that was unclear:
  - `help.txt` lists `count` keys (`of`, `is`) but the README only mentions
    them in passing; a worked example of `not: { exists: ... }` and `count`
    in the README would have saved a lookup.
  - `move` help does not state explicitly whether pattern moves within one
    transform are atomic with respect to collisions, so I ordered the moves
    conservatively as separate transforms.
- Tools used (condition F): n/a (condition B)
- Approximate time spent, if you can tell: about 3 minutes (sandbox created
  03:55, plan executed 03:56).
