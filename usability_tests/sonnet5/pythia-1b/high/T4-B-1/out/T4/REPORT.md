## Participant self-report

- Final artifact path: `out/T4/plan.yaml` (run with `brainsurgery out/T4/plan.yaml`, output at `out/T4/model.safetensors`).
- Number of times you executed the script or plan: 1.
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - `add`/`subtract`/`assign` require the destination tensor to already exist, and
    `copy`/`cast`/`scale`/`matmul` require it not to exist, so the multi-step
    float32 arithmetic needed explicit `copy` steps to pre-create each
    intermediate destination before `add_`/`subtract_` could write into it.
  - With multiple inputs, brainsurgery infers the single output alias from
    which alias the transforms write to (destinations of `copy`/`move`/in-place
    ops/`delete`); writing scratch tensors to a new alias would have made that
    inference ambiguous, so all intermediate float32/float16 scratch tensors
    were kept as extra tensor names under the `base` alias (`scratch.<stage>.*`)
    and deleted again before the final checks and save.
  - `cast` (out-of-place) takes `dtype`, while `cast_` (in-place) takes `to` for
    the target dtype key — easy to mix up.
  - The ordering hazard called out in the task (each task vector must be taken
    against the unmodified base) is handled by casting `base`'s MLP tensors to
    a float32 snapshot (`scratch.b32.*`) once, before either fine-tune's diff is
    computed, and only ever reading from that snapshot for both diffs.
- Anything in the task text or documentation that was unclear: no; the
  README's `equal`/`copy`/`cast` capture-and-rewrite semantics (`\1`, `\g<0>`,
  differing aliases) and the "which alias gets written" rule were enough to
  build the plan without guessing.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: one pass of reading the doc pack,
  writing the plan, and one successful run; no debugging iterations were
  needed.
