# Participant self-report

- Final artifact path: `out/T1/plan.yaml` (output written to `out/T1/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none, first run succeeded.
- Pitfalls or surprises you hit (one line each):
  - Renumbering order matters: moves are only safe if each destination block
    index is already vacated by an earlier delete or an earlier move in the
    same list, so I processed old indices in ascending order
    (3->2, 4->3, 6->4, 7->5, 9->6, 10->7, 11->8), each of which lands on a
    slot just freed by the delete step or the immediately preceding move.
  - Regex refs use `re.fullmatch`, so anchoring block numbers with literal
    dots (`h\.10\.(.+)`) is enough to avoid `h.1.` prefix collisions with
    `h.10.`/`h.11.`; no extra care beyond escaping the dots was needed.
  - `move` renames by regex + backreference (`to: h.2.\1`) and moves every
    matching tensor of a block in one transform, so no per-tensor listing
    was required.
- Anything in the task text or documentation that was unclear: none; the
  README's tensor-reference and assert-expression sections had everything
  needed (regex full-match refs with capture groups, `count`/`not exists`
  asserts).
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: well under 10 minutes; one plan,
  one execution, no retries.
