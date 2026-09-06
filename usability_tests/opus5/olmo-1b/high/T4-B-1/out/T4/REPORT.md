# T4 self-report (condition B, BrainSurgery plan)

- **Final artifact path:** `out/T4/plan.yaml` (plan), `out/T4/model.safetensors`
  (output, 114 tensors, float32), `out/T4/summary.yaml` (executed-plan summary).

- **Number of times you executed the script or plan:** 2.

- **Which executions failed, and why (one line each):**
  1. `no_match` — `assert: { count: { of: base::tv1\..*, is: 0 } }`, my
     post-cleanup "the scratch tensors are gone" check, aborted with
     `count.of matched zero tensors`: reference resolution raises when a
     pattern matches nothing, so a count of 0 can never succeed. The merge
     itself had already completed at that point; no output was written.
  2. (second execution passed all 38 transforms and wrote the output)

- **Pitfalls or surprises you hit (one line each):**
  - Absence cannot be asserted with `count: is: 0`; every reference resolution
    goes through the same helper that raises on zero matches, so the idiom is
    `assert: { not: { exists: <pattern> } }` (`not` catches the error).
  - The ordering hazard is handled by materialising *both* task vectors as
    scratch tensors first (`copy` ft -> `base::tv{1,2}.<name>`, then
    `subtract_` the base out of them), and only then `add_`-ing them into the
    base; doing one fine-tune end-to-end first would take the second delta
    against an already-merged base.
  - All writes have to land on one alias or the run fails with
    `cannot infer output model uniquely`, so the scratch deltas live inside the
    `base` alias under a `tv1.`/`tv2.` name prefix rather than in their own
    alias. The prefix also keeps them out of the `model\.layers\...` patterns.
  - `assert: equal` resolves `right` as a rewrite of each `left` match, which
    makes one assertion do double duty: it proves the 66 non-MLP names exist in
    the fine-tune *and* that the values are identical.
  - Name-set equality across three checkpoints is not directly expressible.
    I got it from a partition argument instead: 114 total, 48 matching the
    exact gate/up/down MLP pattern and 66 matching `(?!.*\.mlp\..*).*` in each
    of the three; the `equal` assertion pins the 66 non-MLP names; and driving
    the `subtract_` from the *base* side (`from: base::<mlp>`,
    `to: base::tv1.\1`) makes the run fail if a base MLP name is missing from
    a fine-tune, which with 48 == 48 forces the MLP name sets to coincide too.
  - `count`/`shape`/`dtype` on a multi-match pattern: I used multi-match only
    for `count` and `equal`, where the docs guarantee it, and pinned shape and
    dtype on single named tensors, to avoid depending on unspecified behaviour.
  - Writing to a path with a `.safetensors` suffix produces one file even
    though the model (4.8 GiB of tensor data) is close to the 5GB default
    shard budget; sharding only kicks in for directory-like output paths.

- **Anything in the task text or documentation that was unclear:**
  - The README lists the assert operators but does not say that a reference
    matching zero tensors is an error rather than an empty match set. That is
    the one thing that cost me an execution, and it is not visible from
    `help: { assert: count }` either.
  - The task says the merge must be "computed in float32". Everything is
    already float32 end to end, so I read this as "do not lose precision"
    rather than as a request for an explicit `cast`.
  - "Exactly 48 tensors were merged" has no direct expression; I asserted the
    count of scratch delta tensors (48 per fine-tune) immediately after the
    copies, which is the closest observable proxy.

- **Tools used (condition F):** n/a (condition B).

- **Approximate time spent, if you can tell:** roughly 10 minutes, most of it
  reading `help.txt` and the two example plans; each execution took ~10-17 s.
