# T1 report

- Final artifact path: `out/T1/solution.py` (invoked via `out/T1/run.sh`), output at `out/T1/model.safetensors`.
- Number of times you executed the script or plan: 1.
- Which executions failed, and why (one line each): none; the single run succeeded.
- Pitfalls or surprises you hit (one line each):
  - Renumbering collisions are only a hazard for in-place/sequential rename schemes; building a fresh dict keyed by target names and asserting no key is produced twice sidesteps it entirely, so I didn't need mergekit's ordering guarantees for this.
  - Verified bit-exactness end to end (e.g. old block 3 -> new block 2, old block 15 -> new block 11) and that the 4 non-block tensors are untouched, beyond just the required tensor-count checks.
- Anything in the task text or documentation that was unclear: no, the renumbering map and required checks were fully specified.
- Tools used (condition F): `safetensors` 0.5.3 for load/save, `torch` 2.14.0 for tensor equality/contiguity checks. Chose a plain script over mergekit/torch-state-bridge because the transform is a single fully-specified regex-based key rewrite plus block drop with a known target index map — a general merge-config or rule-based rewriting layer added indirection without buying anything, and building a fresh output dict (rather than renaming in place) already eliminates the collision hazard the task is designed to probe.
- Approximate time spent, if you can tell: ~10 minutes.
