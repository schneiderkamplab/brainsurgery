# T5 self-report (Condition B: BrainSurgery plan)

- Final artifact path: `out/T5/` (5 shards + `model.safetensors.index.json`),
  plan at `out/T5/plan.yaml`.
- Number of times you executed the script or plan: 2 (plus one throwaway
  `dump`-only plan run from `/tmp` to confirm the base/adapter tensor names).
- Which executions failed, and why (one line each):
  - Run 1: `TransformError: matmul source_b missing: lora::base_model\.model\.h\.0\....`
    — I had written the `from_b` / `to` rewrite templates with regex-escaped
    dots; rewrite targets are literal strings with only `\1` interpolated, so
    the backslashes ended up in the requested tensor name.
- Pitfalls or surprises you hit (one line each):
  - Match patterns and rewrite templates are asymmetric: `from_a` (and any
    `target`/`of`) is a regex where `\.` is required, but `from_b`/`to` are
    literal names where `\.` is wrong. That is the whole content of run 1.
  - Output alias inference: temporaries had to be created on the `base` alias
    (`base::tmp_merge.<i>.ba`), not on `lora`, or the run would have written to
    two aliases and failed with `cannot infer output model uniquely`.
  - `100MB` in `output.shard` is binary (104,857,600 bytes), which matches the
    task's limit exactly, and `wte.weight` (154 MB) is automatically given its
    own shard — no special handling needed.
  - `permute` with `order: [1, 0]` is the transpose; it creates a new tensor,
    so the un-transposed product had to be deleted too, which the single
    `delete: { target: 'base::tmp_merge\..*' }` handles.
- Anything in the task text or documentation that was unclear:
  - The README documents capture/rewrite for `assert.equal`'s `right`, and
    `interfaces-reference.md` says ternary transforms use "the same
    capture-based rewrite model", but neither shows a ternary example with
    captures, so the escaping rule for `from_b` had to be learned from the
    failure. One worked `matmul` example with `\1` would have prevented run 1.
  - `assert: { count: ... }` counts matches of a pattern; there is no direct
    "these two families pair up 1:1" assertion, so "exactly 12 adapter pairs
    were found and merged" is expressed as three counts (12 A, 12 B, and 12
    products actually produced by the `matmul`), which is what I did.
- Tools used (condition F): n/a.
- Approximate time spent, if you can tell: ~10 minutes.

## Verification performed

Beyond the in-plan asserts, I checked the written checkpoint against a direct
recomputation: 160 tensors with the base's exact key set, the 148 untouched
tensors bit-identical to the base, the 12 `h.<i>.attn.c_attn.weight` equal to
`base + 2 * (B @ A).T` to a relative Frobenius error below 1e-6, all float32
and `[768, 2304]`, no `lora_`/`tmp_merge` name in the index, and every shard's
tensor data within 104,857,600 bytes except `wte.weight`, which is alone in
shard 5.
