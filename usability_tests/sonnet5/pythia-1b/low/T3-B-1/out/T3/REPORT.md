## Participant self-report

- Final artifact path: `out/T3/plan.yaml`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none, the plan succeeded on the first run.
- Pitfalls or surprises you hit (one line each):
  - Upcasting everything to float32 first (including the buffers) before deleting them, rather
    than trying to skip the buffers during the upcast, was simpler and harmless since they're
    deleted afterward anyway.
  - After the required float32 upcast, `embed_in.weight`/`embed_out.weight` grow to ~412 MB
    (vs. ~206 MB in the float16 input), so they still land alone in their own shard under the
    256 MiB budget — consistent with the spec's sharding rule, just a bigger number than the
    task text's float16-based example.
- Anything in the task text or documentation that was unclear: none; the `cast_`/`delete`/`assert`
  docs and the sharding notes in the README were sufficient to write the plan directly.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: ~10 minutes.
