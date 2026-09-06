# Participant self-report: T4 (Pythia-1B), condition B

- Final artifact path: `out/T4/plan.yaml` (output checkpoint `out/T4/model.safetensors`, 244 tensors; executed-plan summary in `out/T4/summary.yaml`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; execution 1 succeeded.
- Pitfalls or surprises you hit (one line each):
  - The output alias is inferred from where the transforms write, so all float32 working copies (`work.acc.*`, `work.tv1.*`, `work.tv2.*`) had to be created under the `base` alias even when cast from `ft1`/`ft2`.
  - Whether `to` in `add_`/`subtract_` is a capture rewrite of `from` or an independently matched pattern is not stated in `help.txt`; the interfaces reference (section on binary mapping transforms) confirms it is a rewrite when both are regex strings.
  - `shape`/`dtype` assert help says "the tensor" (singular), so I did not rely on them with multi-match patterns; shape/dtype agreement of the MLP tensors is enforced by the in-place arithmetic instead.
  - I avoided `count ... is: 0` on a pattern that matches nothing, since it was unclear whether a no-match is an error; the 244 total count covers it.
  - The name-set check for the 64 MLP tensors is indirect: counts of 64 in every checkpoint plus in-place ops whose rewritten destinations must exist (so a missing or renamed MLP tensor in ft1/ft2 aborts the run).
  - Ordering hazard handled by taking both task vectors against the float32 base copy before any `add_` into it.
  - A scripted edit of the plan file with an empty search string corrupted it (my own mistake, before any execution); rewrote the plan from scratch.
- Anything in the task text or documentation that was unclear:
  - README/help do not say whether `shape`, `dtype` and `count` accept multi-match patterns and what a zero-match pattern does in `count`.
  - The `add` example `to: '.*.weight'` in `help.txt` looks like independent pattern matching rather than a rewrite, which contradicts the interfaces reference; the reference was right for `add_`/`subtract_`.
- Tools used (condition F): not applicable (condition B).
- Approximate time spent, if you can tell: about 10 minutes reading the doc pack and writing the plan; the run itself took about 11 seconds.
