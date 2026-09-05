## Participant self-report

- Final artifact path: `out/T5/plan.yaml`
- Number of times you executed the script or plan: 1 (final run; the merge
  logic and regex pairing were first validated against small synthetic
  checkpoints outside `out/T5` before running against the real inputs)
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - With two input aliases (`model`, `lora`) there is no default alias, so
    every reference in every transform after the cross-alias `matmul` needed
    an explicit `model::` prefix, even for later steps that only touch the
    base checkpoint.
  - `add`/`matmul`/`scale`/`cast` all support regex capture groups with `\1`
    backreferences on both sides, so all 16 layers could be merged with one
    transform each instead of 16 unrolled copies; confirmed this against the
    doc pack's `assign`/`equal` backreference examples before relying on it.
  - `add`/`add_` require the destination to already exist and (implicitly)
    matching dtype, so the merge has to go: `matmul` (float32) -> `scale`
    (float32) -> `cast` base to float32 -> `add_` in place -> `cast` back to
    float16 -> `assign` into the original tensor name -> `delete` every
    intermediate (`delta_raw`, `delta_scaled`, `basef32`, `merged`) and both
    adapter tensors, so only the 244 original names remain.
  - `output.shard: 512MB` in this tool means MiB (1024-based), matching the
    512 MiB / 536,870,912-byte budget in the task exactly.
- Anything in the task text or documentation that was unclear:
  - The task says a tensor "larger than" the 512 MiB shard cap "is stored
    alone in its own shard," citing `embed_in`/`embed_out` at 206 MB each as
    the example, but 206 MB is well under the 512 MiB cap, so it isn't
    actually a tensor that exceeds the limit. Left the built-in sharder to
    pack shards up to the size cap rather than forcing those two tensors into
    solo shards; the produced shards (4 total, largest 529,611,452 bytes,
    smallest tensor-data total 511,955,272 bytes) all satisfy the "at most
    512 MiB per shard" rule and cover all 244 tensor names once.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: ~25 minutes, most of it exploring
  `help.txt`/README for the pairwise-regex and sharding semantics and
  validating them on small synthetic checkpoints before touching the real
  inputs.
