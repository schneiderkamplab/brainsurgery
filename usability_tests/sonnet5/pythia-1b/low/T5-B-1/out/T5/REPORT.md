## Participant self-report

- Final artifact path: `out/T5/plan.yaml`
- Number of times you executed the script or plan: 2 (first attempt failed on
  `load`, second succeeded)
- Which executions failed, and why (one line each):
  - Attempt 1: tried to `load` the LoRA checkpoint into the already-loaded
    `model` alias to keep everything on one alias for output-alias inference;
    the provider rejects loading into an alias that already exists
    (`LoadTransformError: load alias already exists: 'model'`), a `crash`.
- Pitfalls or surprises you hit (one line each):
  - Output-alias inference requires all writing transforms to target exactly
    one alias, but `matmul`/`scale`/`cast`/`add_`/`assign`/`delete` each count
    as a write to whichever alias their destination names; with two input
    checkpoints (`model`, `lora`) I had to route every destination through
    `model::` explicitly and only ever read from `lora::`, and skip deleting
    the adapter tensors from the `lora` alias (they are never saved anyway,
    since only `model` gets written).
  - `add` (non-underscore) requires the destination to already exist, so the
    fp32 accumulation had to use `add_` in-place instead.
  - `cast`/`matmul`/`scale` all require the destination to not already exist,
    so putting the merged fp16 values back onto the original weight name
    needed a final `assign` step (which does allow overwriting an existing
    tensor of matching shape/dtype) rather than a direct cast back onto the
    original name.
- Anything in the task text or documentation that was unclear:
  - The README's alias-inference rule ("the alias the transforms write to")
    doesn't spell out that reads from a second alias are fine as long as
    nothing is written or deleted there; that took one failed run to
    discover empirically.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: ~15 minutes
