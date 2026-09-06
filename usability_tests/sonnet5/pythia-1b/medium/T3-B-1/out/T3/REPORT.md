## Participant self-report

- Final artifact path: `out/T3/plan.yaml`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single run succeeded.
- Pitfalls or surprises you hit (one line each):
  - There is no dtype-filtered tensor reference, so "exactly 64 tensors are
    bfloat16" had to be checked indirectly: assert the projection regex
    matches 64 tensors and is bfloat16, then assert every tensor *not*
    matching that regex (via a negative lookahead) is float32 — together
    these pin the bfloat16 count at exactly 64 rather than "at least 64".
  - Ordered the transforms as delete buffers -> cast_ everything to float32
    -> cast_ only the 64 projection weights to bfloat16, so the broad
    float32 cast never has to special-case the buffers or the projections.
  - `cast_` (in place) was the right choice over `cast` (which creates a new
    named tensor) since tensor names must not change.
  - `output.shard: 256MB` matched the 268,435,456-byte budget exactly (binary
    MB per the README), and the two oversized embedding tensors were each
    placed alone in their own shard automatically, as documented.
- Anything in the task text or documentation that was unclear: none; the
  README's notes on shard budgeting and full-match regex tensor references
  were sufficient.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: ~10 minutes.
