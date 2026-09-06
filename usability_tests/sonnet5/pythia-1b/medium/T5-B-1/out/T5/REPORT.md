# Participant self-report

- Final artifact path: `out/T5/plan.yaml`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - `matmul`/`scale`/`cast` all require the destination to not already exist, so the
    `B @ A` product, its `alpha/r` scaling, and its cast to float16 each need their
    own new intermediate tensor name (per layer) before the final `add_` into the
    base weight; those intermediates then have to be deleted before saving so they
    don't show up in the output or throw off the 244-tensor count.
  - `add_` (in-place) requires the destination to already exist and, implicitly,
    matching dtype/shape with the source, so the delta has to be cast to float16
    before adding it to the float16 base weight, even though the matmul/scale work
    is done in float32.
  - `output.shard: 512MB` uses the binary-MB convention from the README (512 x 1024
    x 1024 = 536,870,912 bytes), which lines up exactly with the byte budget stated
    in the task.
- Anything in the task text or documentation that was unclear: the task text says
  the two 206 MB embedding tensors are "stored alone in their own shard" for a
  512 MiB budget, but 206 MB is well under that budget; I left sharding to
  brainsurgery's default packing behavior (`shard: 512MB`) rather than trying to
  force those two tensors into solo shards, since the README describes packing
  tensors up to the budget rather than isolating any tensor smaller than it.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: about 15 minutes reading the doc pack,
  writing the plan (16 layers x 4 transforms plus asserts and cleanup), and
  verifying the output.
