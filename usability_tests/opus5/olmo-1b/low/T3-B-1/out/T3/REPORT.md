# T3 self-report (condition B)

- Final artifact path: `out/T3/plan.yaml` (output checkpoint in `out/T3/`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded
- Pitfalls or surprises you hit (one line each):
  - The obvious hazard is a `.*weight` pattern, so I anchored the cast pattern to
    `model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)\.weight` and
    escaped every dot.
  - I asserted the match count is 112 *before* the cast as well as after, so an
    over- or under-matching pattern fails loudly rather than silently casting the wrong set.
  - `shard: 256MB` is a binary unit (268,435,456 B) in this tool, which is exactly the
    budget the task asks for; the two 412 MB embedding/lm_head tensors were placed alone
    in their own shards automatically.
  - I added a negative-lookahead `dtype` assert to confirm everything outside the
    projection set stayed float32, which the required checks only cover for `embed_tokens`.
- Anything in the task text or documentation that was unclear:
  - The task mentions dropping non-parameter buffers and upcasting norms/biases, but this
    checkpoint has none of those; the Input section says so explicitly, so I deleted nothing.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: a few minutes
