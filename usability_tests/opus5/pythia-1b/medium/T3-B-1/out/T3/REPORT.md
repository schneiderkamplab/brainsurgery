# T3 self-report (condition B: BrainSurgery plan)

- **Final artifact path:** `out/T3/plan.yaml` (output checkpoint in `out/T3/`,
  9 shards + `model.safetensors.index.json`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why (one line each):** none; the single
  execution passed all in-plan asserts and wrote the output.
- **Pitfalls or surprises you hit (one line each):**
  - The over-broad-pattern hazard is real: `.*weight` would have hit
    `gpt_neox.embed_in.weight`, `embed_out.weight` and every layer-norm, so the
    bfloat16 cast uses an explicit alternation over the four projection names,
    guarded by `assert count is 64`.
  - `gpt_neox.layers.<i>.attention.bias` is a uint8 buffer whose name ends in
    `.bias`, so any "biases stay float32" pattern must not be `.*\.bias`; the
    buffers are deleted first, then a blanket `cast_ .*` to float32 is safe
    because nothing non-float remains.
  - Ordering matters: delete buffers -> cast everything to float32 -> cast the
    64 projections down to bfloat16. Doing the blanket float32 cast last would
    undo the bfloat16 step.
  - Shard sizes are binary units per the README, so `shard: 256MB` is exactly
    the 268,435,456-byte budget the task asks for; the two 206 MB embedding
    tensors each landed alone in their own shard automatically.
  - "Exactly 64 tensors are bfloat16" is not directly expressible (there is no
    dtype-based selector), so I expressed it as: count of the projection
    pattern is 64, its dtype is bfloat16, and the complement pattern
    (negative lookahead) is float32.
- **Anything in the task text or documentation that was unclear:** nothing
  blocking. The README notes that `assert` does not count as a write for output
  alias inference, which was worth confirming since the plan is assert-heavy;
  with a single input it is moot. It was not stated whether `assert: dtype`
  applies to every match of a multi-match pattern (it does), so I paired each
  dtype check with a count check.
- **Tools used (condition F):** n/a (condition B).
- **Approximate time spent, if you can tell:** ~5 minutes, mostly reading
  `docpack/README.md` and the `help.txt` entries for `cast_`, `delete` and the
  assert operators.
