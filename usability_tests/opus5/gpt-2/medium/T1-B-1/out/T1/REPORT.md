# T1 self-report (condition B, BrainSurgery plan)

- Final artifact path: `out/T1/plan.yaml` (output: `out/T1/model.safetensors`,
  121 tensors). `out/T1/verify.yaml` is an extra read-only cross-check plan,
  not part of the solution.
- Number of times you executed the script or plan: 1 (plus 1 execution of the
  separate `verify.yaml` cross-check plan).
- Which executions failed, and why (one line each): none; the plan passed on
  the first execution.
- Pitfalls or surprises you hit (one line each):
  - Renumbering collisions: `move` refuses an existing destination, so the
    seven block moves are ordered by ascending destination (3->2, 4->3, 6->4,
    7->5, 9->6, 10->7, 11->8) and the deletions of blocks 2/5/8 run first;
    any other order would either abort or clobber a surviving block.
  - Regex references are full-match, so dots must be escaped (`h\.3\.(.*)`)
    to avoid `h.3x...`-style overreach; the replacement side uses `\1`.
  - The causal-mask buffer `attn.bias` is just another `h.<i>.*` tensor and is
    carried along by the same pattern, so no special handling was needed.
- Anything in the task text or documentation that was unclear:
  - The README documents that `to` in `move` is a rewrite of each `from` match
    but does not spell out whether a pattern-move is applied atomically; I
    ordered the moves defensively so correctness does not depend on that.
  - Nothing else; the required checks map directly onto `assert.count`,
    `assert.not` + `assert.exists`.
- Tools used (condition F): n/a.
- Approximate time spent, if you can tell: about 5 minutes.
