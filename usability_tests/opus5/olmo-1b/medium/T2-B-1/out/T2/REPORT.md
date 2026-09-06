# T2 self-report (condition B, BrainSurgery plan)

- **Final artifact path:** `out/T2/plan.yaml` (output: `out/T2/model.safetensors`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why (one line each):** none; the single execution
  passed all asserts and wrote the checkpoint.
- **Pitfalls or surprises you hit (one line each):**
  - `copy`/`move` refuse existing destinations and `move` refuses slices, so the
    pruned tensors had to be built under scratch names (`....weight.pruned`) and
    swapped in with `delete` + `move` rather than written in place.
  - `concat` requires each source ref to resolve to exactly one tensor, which
    rules it out for a regex over all 16 layers; the workable route was to make a
    correctly *shaped* scratch tensor with a sliced `copy` and then fill it with
    two sliced `assign`s (both `from` and `to` accept slices).
  - Regexes are full-match, so the `.pruned` scratch names could not be hit
    accidentally by the `delete` of the original q/k/v/o names.
  - Row blocks vs column blocks: q/k/v are `[out, in]` so heads are rows
    (`[768:, :]`), while `o_proj` consumes heads on its input side so the same
    head is a column block (`[:, 768:]`).
- **Anything in the task text or documentation that was unclear:** nothing
  blocking. The README lists `concat`'s per-ref single-tensor restriction only in
  `help.txt`, so the first plan sketch had to be revised before running.
- **Tools used (condition F):** n/a
- **Approximate time spent, if you can tell:** ~5 minutes.
