# T2 self-report (condition B, Pythia-1B)

- Final artifact path: `out/T2/plan.yaml` -> output `out/T2/model.safetensors`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution
  succeeded with all asserts passing and 244 tensors written.
- Pitfalls or surprises you hit (one line each):
  - `concat` requires each `from` reference to resolve to exactly one tensor, so the
    plan cannot be written pattern-wise over the 16 layers; I generated the 16
    explicit per-layer blocks (192 transforms) instead.
  - `copy` destinations must not already exist and cannot be sliced, so restoring the
    original tensor name needs the order copy-slices -> `delete` original -> `concat`
    back into the same name.
  - Scratch tensors had to be named without dots (`tmp_qkvw_a_<i>`) and deleted with
    `delete: { target: 'tmp_.*' }` before the 244-tensor count check.
  - `assert: shape` accepts a pattern in `of`, so one assert per tensor kind covers all
    16 layers; I used that in addition to the four required layer-0/count checks.
- Anything in the task text or documentation that was unclear:
  - The README does not state explicitly whether an `assert: shape` with a multi-match
    pattern checks every match or only the first; I relied on it checking all, and added
    a `count` assert alongside it.
  - The `concat`/`split` help examples in `help.txt` have empty `from:`/`to:` lists
    (rendering artifact), so the exact list syntax had to be inferred from the second
    `concat` example.
- Tools used (condition F): n/a (condition B, only `brainsurgery`).
- Approximate time spent, if you can tell: ~3 minutes.
