# T3 participant self-report (condition B, OLMo-1B-0724-hf)

- **Final artifact path:** `out/T3/plan.yaml` (output checkpoint in `out/T3/`:
  10 shards + `model.safetensors.index.json`)

- **Number of times you executed the script or plan:** 1

- **Which executions failed, and why (one line each):** none — the single
  execution succeeded.

- **Pitfalls or surprises you hit (one line each):**
  - The obvious over-broad target `.*weight` would hit `model.embed_tokens.weight`
    and `lm_head.weight` too, so the cast target is anchored on the projection
    names: `model\.layers\.\d+\.(?:self_attn\.[qkvo]_proj|mlp\.(?:gate|up|down)_proj)\.weight`.
  - "Exactly 112 tensors are bfloat16" is not directly expressible as one
    assert, since `dtype` only checks the tensors a reference matches. I encoded
    it as three asserts that together pin it down: the projection pattern matches
    exactly 112 tensors and all of them are bfloat16; the complement pattern
    (same regex behind a negative lookahead) is all float32; and the state dict
    still holds exactly 114 tensors.
  - References are full-match regexes, so dots must be escaped and the negative
    lookahead needs its own `$` anchor (`(?!...\.weight$).*`) to mean
    "the whole name is not a projection".
  - Shard sizes are binary units, so the 268,435,456-byte budget in the task is
    written `256MB` in the plan, not `268435456` or `256MiB`.
  - `cast_` (in-place) rather than `cast` (out-of-place, new destination) is the
    right transform here: names must not change, and in-place keeps the
    state-dict order, which is what the shard packer walks.
  - The task mentions dropping non-parameter buffers, but this checkpoint has
    none, so the plan contains no `delete` — deleting anything would have been
    wrong.

- **Anything in the task text or documentation that was unclear:**
  - The objective paragraph talks about norms, biases and buffers, while the
    "Input" section says this checkpoint has none of them. The requirements
    section resolves it, but the two read as contradictory at first.
  - The docs say `dtype` "Succeeds if the tensor has the given dtype" without
    stating whether a multi-match reference requires *every* match to have that
    dtype (as `reads`/`writes` explicitly do). I relied on the every-match
    reading and cross-checked with `count`.

- **Tools used (condition F):** n/a (condition B).

- **Approximate time spent, if you can tell:** ~10 minutes, most of it reading
  `docpack/help.txt` for the assert operators; the plan ran in ~9 s.
