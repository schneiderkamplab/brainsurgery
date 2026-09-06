# T4 self-report (condition B)

- Final artifact path: `out/T4/plan.yaml` (output `out/T4/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - Output alias inference forces every write onto one alias, so the scaled ft1/ft2
    copies had to be staged as scratch tensors (`tv1.*`, `tv2.*`) inside `base::`.
  - Ordering hazard: both scaled copies are created before `scale_` touches the base,
    so each task vector is taken against the unmodified base.
  - Rewrote the merge as `0.2*base + 0.4*ft1 + 0.4*ft2` because `add_` has no scale
    factor and `subtract` needs a pre-existing destination.
  - Dots in tensor names must be escaped in the regexes (`h\.\d+\.mlp\.`), and the
    negative lookahead `(?!h\.\d+\.mlp\.).+` from the README was the clean way to
    select the 112 shared tensors.
- Anything in the task text or documentation that was unclear:
  - The README does not state whether `scale`/`add_` destination rewrites accept a
    different alias than the source; the copy example suggested yes and it worked.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: ~10 minutes, most of it reading `help.txt`.
