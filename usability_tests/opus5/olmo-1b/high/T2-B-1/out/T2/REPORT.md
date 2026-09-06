# T2 self-report (condition B: BrainSurgery plan)

- **Final artifact path:** `out/T2/plan.yaml` (output written to `out/T2/model.safetensors`)
- **Number of times you executed the script or plan:** 3

## Which executions failed, and why

1. `failed_assertion` — `count failed: model::tmpkeep\..* matched 128 tensors, expected 80`. My own sanity check on the number of snapshot tensors; I miscounted (16 layers x 3 qkv x 2 blocks = 96, plus 16 x 2 o_proj blocks = 32, so 128). The transforms were correct; only my expected number was wrong.
2. `crash` — `count.of matched zero tensors: model::model\.layers\.\d+\.self_attn\..*`. I had written `assert: { count: { of: <pattern>, is: 0 } }` to confirm the originals were gone after `delete`. Reference resolution raises "matched zero tensors" *before* the count is compared, so `count: is: 0` can never succeed. Replaced with `assert: { not: { exists: ... } }`.
3. Success.

- **first_execution_success:** no; **executions_until_first_success:** 3

## Approach

Head 5 of 16 (128 dims) means dropping rows 640:768 of `q/k/v_proj` and columns
640:768 of `o_proj`. `concat` requires each source ref to resolve to exactly one
tensor, so a concat-based solution would have needed 64 hand-written transforms.
Instead I kept everything pattern-based (one rule covers all 16 layers, ~19
transforms total for the edit plus checks):

1. snapshot the two kept blocks per tensor (`[:640, :]` / `[768:, :]`, and the
   column equivalents) into `tmpkeep.*` — `copy` clones, so these are
   independent of the source;
2. `assign` the tail snapshot back over rows `[640:1920, :]`, closing the gap.
   Using the snapshot rather than a self-slice avoids an overlapping copy;
3. `copy` the leading `[:1920, :]` into a fresh contiguous `tmpnew.*` tensor;
4. `delete` the originals, `move` `tmpnew.*` into their names;
5. assert shapes, then assert every kept block still equals its pre-edit
   snapshot, then delete the snapshots and assert the final count of 114.

## Pitfalls or surprises

- `assert: { count: { of: X, is: 0 } }` is unusable: zero matches is a
  resolution error, not a count of zero. `not: { exists: X }` is the idiom.
- `concat`/`split` are per-tensor only (each `from` must resolve to exactly one
  tensor), so they do not scale to a 16-layer pattern rule; `copy` with slices +
  `assign` with a sliced destination does, since both support pattern-based
  destination synthesis with capture groups.
- Sliced `copy` clones, which is what makes the result safe to serialise; a
  plan that left non-contiguous views in the state dict would risk a save error.
- Two-group regexes (`([qkv])`) work in `from`/`to`/`left`/`right` rewrites, which
  collapses q, k and v into a single rule.
- Temporary tensors must live under the same alias as the edit, otherwise output
  alias inference sees writes to more than one alias.
- I deliberately kept the snapshots alive until after the value assertions so
  the plan verifies its own result block by block, not just the shapes.

## Anything in the task text or documentation that was unclear

- The task text is precise; no ambiguity about which axis holds heads.
- The docs do not say that `count` cannot express zero, nor that `concat` sources
  cannot be patterns that expand per match — both cost me an execution or a
  redesign. The `README` transform list says "concatenate multiple source refs
  into one new tensor" without the one-tensor-per-ref restriction that is only
  in `help: concat`.

- **Tools used (condition F):** n/a
- **Approximate time spent:** ~15 minutes, of which ~30 s of plan execution.
