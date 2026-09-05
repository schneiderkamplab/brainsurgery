# Participant self-report

- Final artifact path: `out/T1/plan.yaml` (output written to `out/T1/model.safetensors`)
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - Execution 1: `assert: { count: { of: ..., is: 0 } }` used to check that no
    tensor of blocks 12/13/14/15 remained; `count.of` raises "matched zero
    tensors" as an error whenever the pattern has zero matches, so it can
    never be used to assert a count of exactly 0 — the run aborted with a
    `TransformError` before writing output.
- Pitfalls or surprises you hit (one line each):
  - Fixed the above by asserting `not: { exists: <pattern> } }` instead of
    `count: { is: 0 }` for the "no blocks 12-15 remain" check.
  - The renumbering has a genuine collision hazard: moving blocks in the
    wrong order can overwrite a surviving block. Solved by processing moves
    in increasing old-index order (3,4,5,7,8,9,11,12,13,15), each move
    always targeting a slot already vacated by the previous move or by the
    initial deletion, and `move`'s destination-must-not-exist rule as a
    backstop.
  - Used structured tensor references (`["gpt_neox","layers","<n>","*rest"]`)
    with a literal block index and a `*rest` splice so one `move` per block
    relocates all 15 of its tensors at once, rather than pattern-matching
    per tensor kind.
- Anything in the task text or documentation that was unclear:
  - The README's `assert.count` help doesn't mention that a zero-match
    reference raises before the count comparison even runs; had to discover
    this empirically.
- Tools used (condition F): name, version, and why: n/a (condition B)
- Approximate time spent, if you can tell: ~10 minutes
