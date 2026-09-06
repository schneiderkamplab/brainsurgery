# Participant self-report: T4 (Pythia-1B, condition B)

- Final artifact path: `out/T4/plan.yaml` (output checkpoint `out/T4/model.safetensors`)
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - Execution 1: failed_assertion. My own post-merge check `writes: is: 1` on the merged tensors fired; the tool reports writes=2 for a tensor that was loaded and then assigned once. All verification, arithmetic and cleanup steps before it had passed; no output was written. Fixed by asserting `is: 2` for merged tensors and `le: 1` for untouched ones.
- Pitfalls or surprises you hit (one line each):
  - Write counters include more than the plan's explicit writes (load or assign counts as one extra), so `writes` thresholds must be calibrated empirically.
  - `equal` has no "names/shapes only" mode; I used a huge `eps` to verify that MLP tensor names, shapes and dtypes coincide across checkpoints without comparing values.
  - Binary math transforms (`add`, `subtract`) need pre-existing destinations, so the merge was done with `cast` into temporaries plus in-place `subtract_`/`scale_`/`add_`, then `cast_` and `assign` back into the original slots.
  - Output alias inference requires all edits on one alias; temporaries were created under `base::tmp.*` and deleted at the end.
- Anything in the task text or documentation that was unclear:
  - The docs do not say what counts toward `reads`/`writes` access counters (e.g. whether loading counts), which is needed to use them as "exactly N tensors were merged" checks.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 5 minutes
