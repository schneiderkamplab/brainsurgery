# T1 participant self-report

- **Final artifact path:** `out/T1/plan.yaml` (plan); output `out/T1/model.safetensors`
  (single file, 86 tensors).

- **Number of times you executed the script or plan:** 1

- **Which executions failed, and why (one line each):** None; the single execution
  passed every assert and wrote the output.

- **Pitfalls or surprises you hit (one line each):**
  - Renumbering collision hazard: the ten `move` transforms must be emitted in
    ascending source order (3->2, 4->3, ..., 15->11) so each destination index is
    already vacant; I relied on `move`'s "destination must not exist" rule as the
    safety net, so a wrong ordering would abort rather than clobber a block.
  - Tensor references are full-match regexes, so unescaped `.` is a wildcard;
    I escaped every literal dot in `from`/`of` patterns (the `to` side is a
    replacement template, where dots are literal and `\1` carries the rest of
    the name).
  - `model\.layers\.1\..*` vs. layer 12/13 was a non-issue precisely because of
    full-match plus the escaped trailing dot, but the alternation for the deleted
    blocks needed `(?:2|6|10|14)` rather than a character class.
  - `count: { of: 'model::.*', is: N }` turned out to be the simplest way to
    assert the total tensor count before and after the delete.
  - Output path with a `.safetensors` suffix writes one file; a directory-like
    path would have sharded instead.

- **Anything in the task text or documentation that was unclear:**
  - The README documents `assert` operators on singular wording ("Succeeds if the
    tensor has the given shape"), so it was not obvious from the docs alone that
    `shape`/`dtype` apply to every match of a multi-match pattern; the worked
    example in `docpack/examples/` was what confirmed it.
  - The doc pack's README links point at absolute paths on the author's machine
    (`/Users/petersk/...`), which are not present in the sandbox.
  - Otherwise the task text was unambiguous: the explicit old->new index table
    removed any doubt about the renumbering.

- **Tools used (condition F):** N/A (condition B).

- **Approximate time spent, if you can tell:** ~10 minutes; the plan run itself
  took about 10 seconds.
