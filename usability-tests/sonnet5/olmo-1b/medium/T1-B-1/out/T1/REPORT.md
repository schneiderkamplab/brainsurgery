## Participant self-report

- Final artifact path: `out/T1/plan.yaml`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single run succeeded.
- Pitfalls or surprises you hit (one line each):
  - The old→new layer mapping isn't a uniform shift (blocks are dropped at 2, 6, 10, 14), so a single regex/capture-based `move` can't express it; each surviving layer needed its own explicit `move` with literal old/new indices.
  - Move destinations must not already exist, so the ten per-layer `move` transforms had to be ordered so each destination was already vacated (by an earlier delete or an earlier move) before that move ran; verified the whole chain by hand before running (3→2, 4→3, 5→4, 7→5, 8→6, 9→7, 11→8, 12→9, 13→10, 15→11).
  - Used structured path patterns (`["model", "layers", "<i>", "*rest"]`) instead of regex so each `move`/`delete` grabs the whole block (all 7 tensors) in one transform without having to enumerate `self_attn.q_proj`, `mlp.gate_proj`, etc.
- Anything in the task text or documentation that was unclear: none; the README's structured-expression section and the worked MoE example were enough to figure out the wildcard capture (`*rest`) trick.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: under 10 minutes, one plan draft, one execution.
