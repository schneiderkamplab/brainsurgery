# T3 (Condition B) — self-report

- **Final artifact path:** `out/T3/` (10 shard files + `model.safetensors.index.json`);
  plan at `out/T3/plan.yaml`.

- **Number of times you executed the script or plan:** 1.

- **Which executions failed, and why (one line each):** None; the single run
  passed all asserts and wrote the output.

- **Pitfalls or surprises you hit (one line each):**
  - "exactly 112 tensors are bfloat16" is not directly expressible: `assert:
    count` counts name matches and `assert: dtype` checks a dtype, so I asserted
    it as a pair — 112 matches of the projection pattern are bfloat16, and the
    complement (a negative-lookahead regex, 2 tensors) is still float32, which
    together rule out a 113th bfloat16 tensor.
  - `cast_` is the right transform: `cast` writes to a *new* destination name,
    which would have violated the "tensor names do not change" requirement.
  - `output.shard` uses binary units, so `256MB` is exactly 268,435,456 bytes;
    the two 412 MB float32 tensors were placed alone in their own shards
    automatically, no special handling needed.
  - Negative lookahead needs a trailing `$` (`(?!...weight$).*`) since the
    reference is full-matched; without it the lookahead does not anchor as
    intended.
  - `assert: { count: { of: '.*' } }` resolves against the default alias, so the
    verification alias `orig` had to be deleted before the final 114-tensor count.

- **Anything in the task text or documentation that was unclear:**
  - The task text mentions dropping non-parameter buffers and upcasting norms and
    biases, but this checkpoint has neither (114 = 2 + 16x7 projections); the
    "Input" section says so explicitly, so the objective paragraph reads as
    generic boilerplate that contradicts the concrete instance.
  - The README lists transforms but does not state how shard sizes are counted;
    that detail is only in `interfaces-reference.md` ("binary units, tensor data
    only, oversized tensor gets its own shard"). It would help in the README's
    output section.
  - Documentation does not say whether `assert: dtype` with a multi-match `of`
    checks every match or just the first. I relied on it checking all (behaviour
    confirmed by the run), but it is not written down.

- **Tools used (condition F): name, version, and why:** N/A (condition B).

- **Approximate time spent, if you can tell:** ~15 minutes, most of it reading
  `docpack/README.md`, the `cast`/`cast_` and assert entries in `help.txt`, and
  the sharding notes in `interfaces-reference.md`.
