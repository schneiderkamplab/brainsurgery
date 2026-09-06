# T1 self-report (condition B: BrainSurgery plan)

- **Final artifact path:** `out/T1/plan.yaml` (output: `out/T1/model.safetensors`,
  86 tensors, float32, blocks 0..11).

- **Number of times you executed the script or plan:** 2 executions of
  `brainsurgery`. 1 execution of `out/T1/plan.yaml` itself (succeeded on the
  first try), plus 1 deliberate negative control: a throwaway copy of the plan
  with the `count: 86` check changed to `85`, run to confirm that a violated
  check really exits non-zero instead of passing silently. It failed as
  intended (`TransformError: count failed: model::.* matched 86 tensors,
  expected 85`, exit 1) and the copy was deleted.

- **Which executions failed, and why (one line each):**
  - None of the real plan. The only failure was the intentional negative
    control described above (`failed_assertion`, self-inflicted).

- **Pitfalls or surprises you hit (one line each):**
  - The renumbering collision hazard the task warns about is real but easy to
    disarm: delete blocks 2/6/10/14 *first*, then move survivors in ascending
    old-index order, so every destination index is free at the moment of the
    move (freed either by a delete or by the preceding move). `move` refusing
    an existing destination then acts as a second safety net.
  - Counts alone cannot distinguish a correct renumbering from a wrong one —
    a plan that maps old 4 -> 2 and old 3 -> 3 passes every count check in the
    task. I loaded the input a second time under a `ref::` alias and asserted
    `model::model.layers.<new>.(.*) == ref::model.layers.<old>.\1` for all 12
    blocks. That is the check that actually pins down the mapping.
  - Two inputs plus the rule that the output alias is inferred from what the
    transforms write to: this works only because `ref` is read by asserts only.
    Had I touched `ref` with any writing transform, the run would have failed
    with "cannot infer output model uniquely".
  - Dots must be escaped in the `from`/`of` regexes but the `to` side is a
    replacement template, so it takes plain dots and `\1`. Easy to get
    backwards; the example plan in the doc pack shows the asymmetry clearly.
  - `model\.layers\.1\.(.*)` full-matches, so it does not leak onto layers
    10-15. Worth confirming rather than assuming, since a prefix match here
    would silently mangle the checkpoint.
  - Output as a single file needed only a path with a `.safetensors` suffix —
    sharding is the default for directory-like paths, not for file paths.

- **Anything in the task text or documentation that was unclear:**
  - The task's "Required checks" are all cardinality checks, which a wrong-but-
    plausible renumbering satisfies. I read them as a minimum, not a
    sufficient set, and added the per-block value comparison. Worth stating
    explicitly in the task text which of the two is meant.
  - The README documents output-alias inference well, but nothing says whether
    an assert-only alias counts as a write; I inferred "no" from the sentence
    listing `assert`, `diff`, `dump`, `help` as non-counting and it held.
  - `count` is documented with `is` only, so "at most N" style checks are not
    directly expressible; I worked around it with exact counts, which was fine
    here.

- **Tools used (condition F):** n/a — condition B, only `brainsurgery` and the
  doc pack. One short read-only `python -c` snippet to inspect the output's key
  set after the run, outside the plan and not part of the solution.

- **Approximate time spent, if you can tell:** roughly 10 minutes, most of it
  reading `help.txt` and the worked example; the plan itself ran in ~19 s.
